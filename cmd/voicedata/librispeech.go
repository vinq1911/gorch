//go:build darwin

package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model"
)

// libriUtt is one LibriSpeech utterance.
type libriUtt struct {
	spk, chap, id string // id = "spk-chap-nnnn"
	wavPath       string
	flacPath      string
	transcript    string
	sec           float64
}

// scanLibriSpeech enumerates utterances of the converted speakers in
// the deterministic subset order: speaker ids numeric ascending,
// chapters numeric ascending, utterance ids lexical (zero-padded =
// numeric) ascending. Transcripts come from the source corpus
// *.trans.txt files. Durations are NOT filled in (read lazily).
func scanLibriSpeech(srcRoot, wavRoot string) ([]libriUtt, error) {
	spkDirs, err := numericDirs(wavRoot)
	if err != nil {
		return nil, err
	}
	if len(spkDirs) == 0 {
		return nil, fmt.Errorf("no converted speakers in %s (run audio/voicedata/convert_librispeech.sh)", wavRoot)
	}
	var utts []libriUtt
	for _, spk := range spkDirs {
		chaps, err := numericDirs(filepath.Join(wavRoot, spk))
		if err != nil {
			return nil, err
		}
		for _, chap := range chaps {
			trans, err := readTrans(filepath.Join(srcRoot, spk, chap, spk+"-"+chap+".trans.txt"))
			if err != nil {
				return nil, err
			}
			ids := make([]string, 0, len(trans))
			for id := range trans {
				ids = append(ids, id)
			}
			sort.Strings(ids)
			for _, id := range ids {
				utts = append(utts, libriUtt{
					spk: spk, chap: chap, id: id,
					wavPath:    filepath.Join(wavRoot, spk, chap, id+".wav"),
					flacPath:   filepath.Join(srcRoot, spk, chap, id+".flac"),
					transcript: trans[id],
				})
			}
		}
	}
	return utts, nil
}

// numericDirs lists numeric-named subdirectories sorted numerically.
func numericDirs(root string) ([]string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if _, err := strconv.Atoi(e.Name()); err == nil {
			out = append(out, e.Name())
		}
	}
	sort.Slice(out, func(i, j int) bool {
		a, _ := strconv.Atoi(out[i])
		b, _ := strconv.Atoi(out[j])
		return a < b
	})
	return out, nil
}

// readTrans parses a LibriSpeech *.trans.txt ("ID TEXT" per line).
func readTrans(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		sp := strings.IndexByte(line, ' ')
		if sp <= 0 {
			return nil, fmt.Errorf("%s: malformed line %q", path, line)
		}
		out[line[:sp]] = line[sp+1:]
	}
	return out, sc.Err()
}

// buildLibriSpeech builds the LISTEN shard (deterministic ~10 h
// subset) plus the held-out eval manifest (see main.go doc).
func buildLibriSpeech(c cliConfig) error {
	if c.src == "" {
		home, _ := os.UserHomeDir()
		c.src = filepath.Join(home, "speech-corpora/LibriSpeech/train-clean-100")
	}
	if c.wavDir == "" {
		home, _ := os.UserHomeDir()
		c.wavDir = filepath.Join(home, "speech-corpora/LibriSpeech-wav16k")
	}
	tokDir, err := tokenizerDir(c)
	if err != nil {
		return err
	}
	tok, err := model.LoadQwenTokenizer(tokDir)
	if err != nil {
		return err
	}
	mimiPath, err := findMimiCheckpoint(c)
	if err != nil {
		return err
	}
	b := newSampleBuilder(tok)

	allUtts, err := scanLibriSpeech(c.src, c.wavDir)
	if err != nil {
		return err
	}
	fmt.Printf("scanned %d utterances from %d converted speakers\n", len(allUtts), countSpeakers(allUtts))

	// Reserve the eval-speakers highest-numbered speakers for the
	// held-out eval manifest; everything else is the training pool.
	spkList := speakerList(allUtts)
	if len(spkList) <= c.evalSpeakers {
		return fmt.Errorf("only %d speakers converted; need > %d (eval reserve)", len(spkList), c.evalSpeakers)
	}
	evalSpkSet := map[string]bool{}
	for _, s := range spkList[len(spkList)-c.evalSpeakers:] {
		evalSpkSet[s] = true
	}
	var utts, evalUtts []libriUtt
	for _, u := range allUtts {
		if evalSpkSet[u.spk] {
			evalUtts = append(evalUtts, u)
		} else {
			utts = append(utts, u)
		}
	}
	fmt.Printf("reserved %d eval speakers (%s..%s), %d eval-pool utterances\n",
		c.evalSpeakers, spkList[len(spkList)-c.evalSpeakers], spkList[len(spkList)-1], len(evalUtts))

	// Selection pass: walk in deterministic order, keep clips that pass
	// the duration cap and the conservative length estimate, until the
	// kept audio reaches target-hours.
	var (
		selected                     []libriUtt
		keptSec                      float64
		nTooLong, nEstLong, nScanned int
		lastTrainSpeaker             string
		targetSec                    = c.targetHours * 3600
		speakerSet                   = map[string]bool{}
	)
	for i := range utts {
		if keptSec >= targetSec {
			break
		}
		if c.limit > 0 && len(selected) >= c.limit {
			break
		}
		u := &utts[i]
		nScanned++
		sec, err := wavSeconds(u.wavPath)
		if err != nil {
			return fmt.Errorf("%s: %w (rerun convert_librispeech.sh?)", u.wavPath, err)
		}
		u.sec = sec
		if sec > c.maxClipSec {
			nTooLong++
			continue
		}
		tTok := len(tok.Encode(u.transcript))
		if b.listenEstLen(sec, tTok) > c.maxSeq {
			nEstLong++
			continue
		}
		selected = append(selected, *u)
		keptSec += sec
		lastTrainSpeaker = u.spk
		speakerSet[u.spk] = true
	}
	if keptSec < targetSec && c.limit == 0 {
		fmt.Printf("WARNING: converted speakers exhausted at %.2f h (< target %.2f h) — convert more speakers\n",
			keptSec/3600, c.targetHours)
	}
	fmt.Printf("selected %d clips (%.2f h kept) from %d speakers (last %s); scanned %d, dropped %d >%.1fs, %d est>%d tok\n",
		len(selected), keptSec/3600, len(speakerSet), lastTrainSpeaker, nScanned, nTooLong, c.maxClipSec, nEstLong, c.maxSeq)

	// Encode pass.
	paths := make([]string, len(selected))
	for i, u := range selected {
		paths[i] = u.wavPath
	}
	results, err := encodeClips(paths, mimiPath, c.workers)
	if err != nil {
		return err
	}

	// Assembly pass.
	var samples []data.ShardSample
	var manifest [][]string
	var nEncErr, nDropSeq int
	var audioSec float64
	for i, u := range selected {
		r := results[i]
		if r.err != nil {
			fmt.Printf("  skip %s: %v\n", u.id, r.err)
			nEncErr++
			continue
		}
		s := b.listen(r.ids, u.transcript)
		if len(s.Tokens) > c.maxSeq {
			nDropSeq++
			continue
		}
		samples = append(samples, s)
		audioSec += r.sec
		manifest = append(manifest, []string{
			u.id, fmt.Sprintf("%.3f", r.sec), strconv.Itoa(r.frames),
			strconv.Itoa(len(s.Tokens)), tsvSafe(u.transcript),
		})
	}
	binPath := filepath.Join(c.out, "listen_librispeech.bin")
	if err := data.WriteTokenShard(binPath, samples); err != nil {
		return err
	}
	st := statsOf(samples)
	fmt.Printf("wrote %s: %d samples, %.2f h, %d tokens (%d audio), longest %d\n",
		binPath, st.Samples, audioSec/3600, st.TotalTokens, st.AudioTokens, st.MaxLen)

	// Held-out eval manifest from the reserved speakers (dev-clean
	// substitute; see main.go doc): ≤ eval-max-sec, at most 5
	// utterances per speaker for speaker diversity.
	var evalRows [][]string
	perSpeaker := map[string]int{}
	for i := range evalUtts {
		if len(evalRows) >= c.evalCount {
			break
		}
		u := &evalUtts[i]
		if perSpeaker[u.spk] >= 5 {
			continue
		}
		sec, err := wavSeconds(u.wavPath)
		if err != nil {
			return err
		}
		if sec > c.evalMaxSec {
			continue
		}
		perSpeaker[u.spk]++
		evalRows = append(evalRows, []string{
			u.id, u.flacPath, u.wavPath, fmt.Sprintf("%.3f", sec), tsvSafe(u.transcript),
		})
	}
	if len(evalRows) < c.evalCount {
		return fmt.Errorf("only %d/%d eval utterances available from %d reserved speakers — raise -eval-speakers",
			len(evalRows), c.evalCount, c.evalSpeakers)
	}

	trainManifest := filepath.Join(c.manifestDir, "listen_train_manifest.tsv")
	if err := writeManifest(trainManifest,
		[]string{"utt_id", "sec", "frames", "sample_tokens", "transcript"}, manifest); err != nil {
		return err
	}
	evalManifest := filepath.Join(c.manifestDir, "listen_eval_manifest.tsv")
	if err := writeManifest(evalManifest,
		[]string{"utt_id", "flac_path", "wav_path", "sec", "transcript"}, evalRows); err != nil {
		return err
	}

	speakers := make([]string, 0, len(speakerSet))
	for s := range speakerSet {
		speakers = append(speakers, s)
	}
	sort.Slice(speakers, func(i, j int) bool {
		a, _ := strconv.Atoi(speakers[i])
		bn, _ := strconv.Atoi(speakers[j])
		return a < bn
	})
	return writeStats(filepath.Join(c.manifestDir, "listen_stats.json"), map[string]any{
		"corpus":             "LibriSpeech train-clean-100",
		"selection":          "speakers numeric ascending, chapters numeric, utterances lexical; kept until target_hours",
		"eval_note":          "dev-clean absent on disk; eval draws from reserved held-out train-clean-100 speakers (highest speaker ids)",
		"eval_speakers":      spkList[len(spkList)-c.evalSpeakers:],
		"target_hours":       c.targetHours,
		"max_clip_sec":       c.maxClipSec,
		"max_seq":            c.maxSeq,
		"kept_hours":         audioSec / 3600,
		"clips_scanned":      nScanned,
		"dropped_over_cap":   nTooLong,
		"dropped_est_seq":    nEstLong,
		"dropped_post_seq":   nDropSeq,
		"encode_errors":      nEncErr,
		"train_speakers":     speakers,
		"last_train_speaker": lastTrainSpeaker,
		"eval_count":         len(evalRows),
		"eval_max_sec":       c.evalMaxSec,
		"shard":              binPath,
		"tokens":             st,
	})
}

// speakerList returns the distinct speakers of utts in numeric order.
func speakerList(utts []libriUtt) []string {
	set := map[string]bool{}
	var out []string
	for _, u := range utts {
		if !set[u.spk] {
			set[u.spk] = true
			out = append(out, u.spk)
		}
	}
	return out
}

func countSpeakers(utts []libriUtt) int {
	set := map[string]bool{}
	for _, u := range utts {
		set[u.spk] = true
	}
	return len(set)
}
