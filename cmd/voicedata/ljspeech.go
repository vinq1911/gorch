//go:build darwin

package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/model/qwen"
)

// ljClip is one LJSpeech metadata.csv row.
type ljClip struct {
	id   string
	text string // normalized transcript (numbers spelled out)
	sec  float64
}

// readLJMetadata parses metadata.csv (id|original|normalized) in file
// order — the file is sorted by id, which fixes the deterministic
// even/odd train/held-out split.
func readLJMetadata(path string) ([]ljClip, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []ljClip
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 3)
		if len(parts) < 2 {
			return nil, fmt.Errorf("%s: malformed line %q", path, line)
		}
		text := parts[len(parts)-1]
		if strings.TrimSpace(text) == "" {
			text = parts[1]
		}
		out = append(out, ljClip{id: parts[0], text: text})
	}
	return out, sc.Err()
}

// buildLJSpeech builds the SPEAK shard from the even-index half of
// LJSpeech (~12 h) plus the held-out eval manifest (50 sentences
// evenly strided over the odd-index half).
func buildLJSpeech(c cliConfig) error {
	if c.src == "" {
		home, _ := os.UserHomeDir()
		c.src = filepath.Join(home, "speech-corpora/LJSpeech-1.1")
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

	clips, err := readLJMetadata(filepath.Join(c.src, "metadata.csv"))
	if err != nil {
		return err
	}
	fmt.Printf("metadata: %d clips\n", len(clips))

	// Deterministic split: even indices train, odd indices held out.
	var train, heldOut []ljClip
	for i, cl := range clips {
		if i%2 == 0 {
			train = append(train, cl)
		} else {
			heldOut = append(heldOut, cl)
		}
	}

	// Selection pass over the training half: duration + conservative
	// assembled-length estimate.
	var selected []ljClip
	var textTok []int
	var nEstLong int
	var keptSec float64
	for i := range train {
		if c.limit > 0 && len(selected) >= c.limit {
			break
		}
		cl := &train[i]
		sec, err := wavSeconds(filepath.Join(c.src, "wavs", cl.id+".wav"))
		if err != nil {
			return err
		}
		cl.sec = sec
		tTok := len(tok.Encode(cl.text))
		if b.speakEstLen(sec, tTok) > c.maxSeq {
			nEstLong++
			continue
		}
		selected = append(selected, *cl)
		textTok = append(textTok, tTok)
		keptSec += sec
	}
	fmt.Printf("selected %d/%d train clips (%.2f h kept, %d dropped by est>%d tok)\n",
		len(selected), len(train), keptSec/3600, nEstLong, c.maxSeq)

	paths := make([]string, len(selected))
	for i, cl := range selected {
		paths[i] = filepath.Join(c.src, "wavs", cl.id+".wav")
	}
	results, err := encodeClips(paths, mimiPath, c.workers)
	if err != nil {
		return err
	}

	var samples []data.ShardSample
	var manifest [][]string
	var nEncErr, nDropSeq int
	var audioSec float64
	for i, cl := range selected {
		r := results[i]
		if r.err != nil {
			fmt.Printf("  skip %s: %v\n", cl.id, r.err)
			nEncErr++
			continue
		}
		s := b.speak(cl.text, qwen.TokVoiceLj, r.ids)
		if len(s.Tokens) > c.maxSeq {
			nDropSeq++
			continue
		}
		samples = append(samples, s)
		audioSec += r.sec
		manifest = append(manifest, []string{
			cl.id, fmt.Sprintf("%.3f", r.sec), strconv.Itoa(r.frames),
			strconv.Itoa(len(s.Tokens)), tsvSafe(cl.text),
		})
	}
	binPath := filepath.Join(c.out, "speak_ljspeech.bin")
	if err := data.WriteTokenShard(binPath, samples); err != nil {
		return err
	}
	st := statsOf(samples)
	fmt.Printf("wrote %s: %d samples, %.2f h, %d tokens (%d audio), longest %d\n",
		binPath, st.Samples, audioSec/3600, st.TotalTokens, st.AudioTokens, st.MaxLen)

	// Eval manifest: evenly strided over the held-out (odd-index) half.
	stride := len(heldOut) / c.evalCount
	if stride < 1 {
		stride = 1
	}
	var evalRows [][]string
	for k := 0; k < c.evalCount && k*stride < len(heldOut); k++ {
		cl := heldOut[k*stride]
		wavPath := filepath.Join(c.src, "wavs", cl.id+".wav")
		sec, err := wavSeconds(wavPath)
		if err != nil {
			return err
		}
		evalRows = append(evalRows, []string{
			cl.id, wavPath, fmt.Sprintf("%.3f", sec), tsvSafe(cl.text),
		})
	}
	if len(evalRows) < c.evalCount {
		return fmt.Errorf("only %d/%d eval sentences available", len(evalRows), c.evalCount)
	}

	if err := writeManifest(filepath.Join(c.manifestDir, "speak_train_manifest.tsv"),
		[]string{"clip_id", "sec", "frames", "sample_tokens", "text"}, manifest); err != nil {
		return err
	}
	if err := writeManifest(filepath.Join(c.manifestDir, "speak_eval_manifest.tsv"),
		[]string{"clip_id", "wav_path", "sec", "text"}, evalRows); err != nil {
		return err
	}
	return writeStats(filepath.Join(c.manifestDir, "speak_stats.json"), map[string]any{
		"corpus":           "LJSpeech-1.1",
		"selection":        "metadata.csv order; even indices train (~12 h half), odd indices held out; eval strided over held-out half",
		"voice_token":      "<|voice:lj|>",
		"max_seq":          c.maxSeq,
		"kept_hours":       audioSec / 3600,
		"train_half_clips": len(train),
		"dropped_est_seq":  nEstLong,
		"dropped_post_seq": nDropSeq,
		"encode_errors":    nEncErr,
		"eval_count":       len(evalRows),
		"eval_stride":      stride,
		"shard":            binPath,
		"tokens":           st,
	})
}
