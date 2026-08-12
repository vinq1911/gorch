//go:build darwin

package main

import (
	"fmt"
	"path/filepath"
	"strconv"
	"time"

	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/model/qwen"
)

// buildTextReplay builds the §4.3 TEXT-replay shard: each seed prompt
// is rendered through the ChatML template (enable_thinking=False
// prologue) and answered by the frozen base Qwen3-0.6B with greedy
// decoding; the sample supervises exactly the base model's own answer
// tokens (plus the closing <|im_end|>), guarding frozen-brain behavior
// through the active adapters.
func buildTextReplay(c cliConfig) error {
	tokDir, err := tokenizerDir(c)
	if err != nil {
		return err
	}
	tok, err := model.LoadQwenTokenizer(tokDir)
	if err != nil {
		return err
	}
	fmt.Println("loading Qwen3-0.6B (full depth — replay targets must come from the real frozen brain) ...")
	t0 := time.Now()
	m, err := qwen.LoadPretrained()
	if err != nil {
		return err
	}
	fmt.Printf("model loaded in %.1fs\n", time.Since(t0).Seconds())

	prompts := buildPrompts(c.numSamples)
	if c.limit > 0 && len(prompts) > c.limit {
		prompts = prompts[:c.limit]
	}
	fmt.Printf("generating %d greedy answers (max %d new tokens each)\n", len(prompts), c.maxNew)

	var samples []data.ShardSample
	var manifest [][]string
	var nEmpty int
	start := time.Now()
	for i, p := range prompts {
		rendered := qwen.RenderChatML([]qwen.Message{{Role: "user", Content: p}}, true)
		promptIDs := tok.Encode(rendered)
		full := qwen.Generate(m, promptIDs, qwen.Greedy(c.maxNew))
		gen := full[len(promptIDs):]
		if len(gen) == 0 {
			nEmpty++
			continue
		}
		toks := make([]int, 0, len(promptIDs)+len(gen)+1)
		toks = append(toks, promptIDs...)
		toks = append(toks, gen...)
		toks = append(toks, qwen.StopTokenImEnd)
		samples = append(samples, data.ShardSample{
			Tokens: toks,
			Task:   "text",
			// Position len(promptIDs)-1 predicts gen[0]; the last
			// supervised position predicts the closing <|im_end|>.
			Supervised: []data.Span{{Start: len(promptIDs) - 1, End: len(toks) - 1}},
		})
		manifest = append(manifest, []string{
			strconv.Itoa(i), strconv.Itoa(len(toks)), strconv.Itoa(len(gen)),
			tsvSafe(p), tsvSafe(tok.Decode(gen)),
		})
		if (i+1)%25 == 0 {
			rate := float64(i+1) / time.Since(start).Seconds()
			fmt.Printf("  %d/%d prompts (%.2f prompts/s, eta %s)\n", i+1, len(prompts), rate,
				time.Duration(float64(len(prompts)-i-1)/rate*float64(time.Second)).Round(time.Second))
		}
	}

	binPath := filepath.Join(c.out, "text_replay.bin")
	if err := data.WriteTokenShard(binPath, samples); err != nil {
		return err
	}
	st := statsOf(samples)
	fmt.Printf("wrote %s: %d samples, %d tokens, longest %d (%d empty answers skipped)\n",
		binPath, st.Samples, st.TotalTokens, st.MaxLen, nEmpty)

	if err := writeManifest(filepath.Join(c.manifestDir, "text_replay_manifest.tsv"),
		[]string{"idx", "sample_tokens", "answer_tokens", "prompt", "answer"}, manifest); err != nil {
		return err
	}
	return writeStats(filepath.Join(c.manifestDir, "text_replay_stats.json"), map[string]any{
		"generator":     "Qwen3-0.6B frozen base, greedy, ChatML enable_thinking=False",
		"prompts":       len(prompts),
		"max_new":       c.maxNew,
		"empty_skipped": nEmpty,
		"shard":         binPath,
		"tokens":        st,
	})
}
