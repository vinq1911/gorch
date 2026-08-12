//go:build darwin

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vinq1911/gorch/audio"
	"github.com/vinq1911/gorch/audio/mimi"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/model/qwen"
)

// mimiSampleRate and mimiFrameHz fix the audio-token arithmetic:
// 24 kHz PCM in, 12.5 latent frames (= 100 audio tokens at 8
// codebooks) per second out.
const (
	mimiSampleRate = 24000
	mimiFrameHz    = 12.5
)

// tokenizerDir resolves the Qwen tokenizer file directory
// (cmd/qwenvoice-train precedent): -tokenizer-dir, else
// QWEN_TOKENIZER_DIR, else the checkpoint snapshot dir.
func tokenizerDir(c cliConfig) (string, error) {
	if c.tokDir != "" {
		return c.tokDir, nil
	}
	if d := os.Getenv("QWEN_TOKENIZER_DIR"); d != "" {
		return d, nil
	}
	ckpt, err := qwen.FindCheckpoint()
	if err != nil {
		return "", err
	}
	return filepath.Dir(ckpt), nil
}

// findMimiCheckpoint locates the kyutai/mimi model.safetensors: the
// MIMI_MODEL env var, else the HF hub cache.
func findMimiCheckpoint(c cliConfig) (string, error) {
	if c.mimiPath != "" {
		return c.mimiPath, nil
	}
	if p := os.Getenv("MIMI_MODEL"); p != "" {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	pattern := filepath.Join(home,
		".cache/huggingface/hub/models--kyutai--mimi/snapshots/*/model.safetensors")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("no mimi checkpoint found (set MIMI_MODEL or populate the HF cache): %s", pattern)
	}
	return matches[0], nil
}

// sampleBuilder assembles ChatML-framed training samples with spliced
// audio ids and §3.3 supervised spans (cmd/qwenvoice-train overfit
// builder conventions: the sample formats the M1 trainer already
// consumes).
type sampleBuilder struct {
	tok     *model.QwenTokenizer
	userPre []int // "<|im_start|>user\n"
	mid     []int // "<|im_end|>\n<|im_start|>assistant\n"
}

func newSampleBuilder(tok *model.QwenTokenizer) *sampleBuilder {
	return &sampleBuilder{
		tok:     tok,
		userPre: tok.Encode("<|im_start|>user\n"),
		mid:     tok.Encode("<|im_end|>\n<|im_start|>assistant\n"),
	}
}

// assemble builds user span + assistant span into one sample.
// supStartTok indexes INTO the assistant slice: supervision grades
// predictions of assistant[supStartTok..] plus the final <|im_end|>
// (SPEAK uses 2 to skip the <|speak|><|voice:*|> prefix).
func (b *sampleBuilder) assemble(task string, user, assistant []int, supStartTok int) data.ShardSample {
	toks := make([]int, 0, len(b.userPre)+len(user)+len(b.mid)+len(assistant)+1)
	toks = append(toks, b.userPre...)
	toks = append(toks, user...)
	toks = append(toks, b.mid...)
	aStart := len(toks)
	toks = append(toks, assistant...)
	toks = append(toks, qwen.StopTokenImEnd)
	return data.ShardSample{
		Tokens: toks,
		Task:   task,
		Supervised: []data.Span{{
			Start: aStart + supStartTok - 1,
			End:   len(toks) - 1,
		}},
	}
}

// listen: user = <|listen|> + audio + <|audio_end|>, assistant =
// transcript; the whole assistant span is supervised.
func (b *sampleBuilder) listen(audioIDs []int, transcript string) data.ShardSample {
	user := make([]int, 0, len(audioIDs)+2)
	user = append(user, qwen.TokListen)
	user = append(user, audioIDs...)
	user = append(user, qwen.TokAudioEnd)
	return b.assemble("listen", user, b.tok.Encode(transcript), 0)
}

// speak: user = text, assistant = <|speak|><|voice|> + audio +
// <|audio_end|>; supervision starts at the first audio token.
func (b *sampleBuilder) speak(text string, voiceTok int, audioIDs []int) data.ShardSample {
	assistant := make([]int, 0, len(audioIDs)+3)
	assistant = append(assistant, qwen.TokSpeak, voiceTok)
	assistant = append(assistant, audioIDs...)
	assistant = append(assistant, qwen.TokAudioEnd)
	return b.assemble("speak", b.tok.Encode(text), assistant, 2)
}

// overheadTokens is the assembled-length overhead outside text+audio:
// userPre + mid + task specials. Used with estFrames for the
// conservative pre-encode length estimate.
func (b *sampleBuilder) listenEstLen(sec float64, transcriptTok int) int {
	return len(b.userPre) + 2 + estFrames(sec)*qwen.AudioNumCodebooks + len(b.mid) + transcriptTok + 1
}

func (b *sampleBuilder) speakEstLen(sec float64, textTok int) int {
	return len(b.userPre) + textTok + len(b.mid) + 3 + estFrames(sec)*qwen.AudioNumCodebooks + 1
}

// estFrames conservatively over-estimates the 12.5 Hz frame count for
// sec seconds of audio (encoder padding can add a frame).
func estFrames(sec float64) int {
	return int(sec*mimiFrameHz) + 2
}

// encResult is one clip's Mimi encoding.
type encResult struct {
	ids    []int // frame-major interleaved extended-vocab audio ids
	frames int   // 12.5 Hz frame count
	sec    float64
	err    error
}

// encodeClips runs the WAV → 24 kHz mono → mimi.EncodeWindowed →
// Quantizer.Encode(·, 8) → AudioFrameIDs pipeline over paths with
// `workers` parallel workers, each owning its own Encoder+Quantizer
// instance. Results are index-aligned with paths.
func encodeClips(paths []string, mimiPath string, workers int) ([]encResult, error) {
	if workers < 1 {
		workers = 1
	}
	if workers > len(paths) {
		workers = len(paths)
	}
	results := make([]encResult, len(paths))
	jobs := make(chan int)
	var wg sync.WaitGroup
	var done, errs int64
	var mu sync.Mutex
	start := time.Now()

	workerErr := make([]error, workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			enc, quant, err := mimi.LoadWithQuantizer(mimiPath)
			if err != nil {
				workerErr[w] = err
				for range jobs { // drain
				}
				return
			}
			for i := range jobs {
				results[i] = encodeOne(enc, quant, paths[i])
				mu.Lock()
				done++
				if results[i].err != nil {
					errs++
				}
				if done%200 == 0 {
					rate := float64(done) / time.Since(start).Seconds()
					fmt.Printf("  encoded %d/%d clips (%.1f clips/s, eta %s)\n",
						done, len(paths), rate,
						time.Duration(float64(len(paths)-int(done))/rate*float64(time.Second)).Round(time.Second))
				}
				mu.Unlock()
			}
		}(w)
	}
	for i := range paths {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	for _, err := range workerErr {
		if err != nil {
			return nil, fmt.Errorf("mimi load: %w", err)
		}
	}
	fmt.Printf("  encoded %d clips in %s (%d errors)\n", len(paths), time.Since(start).Round(time.Second), errs)
	return results, nil
}

func encodeOne(enc *mimi.Encoder, quant *mimi.Quantizer, path string) encResult {
	w, err := audio.ReadWAV(path)
	if err != nil {
		return encResult{err: err}
	}
	mono := w.Mono()
	sec := float64(len(mono)) / float64(w.SampleRate)
	pcm := audio.Resample(mono, w.SampleRate, mimiSampleRate)
	latent := enc.EncodeWindowed(pcm)
	codes := quant.Encode(latent, qwen.AudioNumCodebooks)
	return encResult{
		ids:    qwen.AudioFrameIDs(codes),
		frames: len(codes[0]),
		sec:    sec,
	}
}

// wavSeconds returns the duration of a WAV file in seconds (full
// decode; selection-time cost is acceptable at corpus scale).
func wavSeconds(path string) (float64, error) {
	w, err := audio.ReadWAV(path)
	if err != nil {
		return 0, err
	}
	return float64(len(w.Samples)) / float64(w.SampleRate*w.Channels), nil
}

// writeManifest writes a TSV manifest (header + rows).
func writeManifest(path string, header []string, rows [][]string) error {
	var b strings.Builder
	b.WriteString(strings.Join(header, "\t"))
	b.WriteByte('\n')
	for _, r := range rows {
		b.WriteString(strings.Join(r, "\t"))
		b.WriteByte('\n')
	}
	return os.WriteFile(path, []byte(b.String()), 0644)
}

// writeStats writes a stats/config JSON document.
func writeStats(path string, v any) error {
	raw, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0644)
}

// tsvSafe flattens text for a single TSV cell.
func tsvSafe(s string) string {
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.ReplaceAll(s, "\n", "\\n")
	return strings.ReplaceAll(s, "\r", "")
}

// shardTokenStats sums token counts over samples: total, audio-token,
// and supervised-position counts.
type shardTokenStats struct {
	Samples          int   `json:"samples"`
	TotalTokens      int   `json:"total_tokens"`
	AudioTokens      int   `json:"audio_tokens"`
	TextTokens       int   `json:"text_tokens"` // total - audio - specials/framing
	SupervisedTokens int   `json:"supervised_positions"`
	MaxLen           int   `json:"max_len"`
	BinBytes         int64 `json:"bin_bytes"`
}

func statsOf(samples []data.ShardSample) shardTokenStats {
	var st shardTokenStats
	st.Samples = len(samples)
	for _, s := range samples {
		st.TotalTokens += len(s.Tokens)
		if len(s.Tokens) > st.MaxLen {
			st.MaxLen = len(s.Tokens)
		}
		for _, t := range s.Tokens {
			if _, _, ok := qwen.AudioTokenOf(t); ok {
				st.AudioTokens++
			} else if t < qwen.BaseVocabSize {
				st.TextTokens++
			}
		}
		for _, sp := range s.Supervised {
			st.SupervisedTokens += sp.End - sp.Start
		}
	}
	st.BinBytes = int64(st.TotalTokens) * 4
	return st
}
