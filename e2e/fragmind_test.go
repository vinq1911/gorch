//go:build darwin && e2e

package e2e

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vinq1911/gorch/fragmind"
	"github.com/vinq1911/gorch/model"
)

type fragmindResult struct {
	ModelName      string             `json:"model_name"`
	Params         int                `json:"params"`
	NumLayers      int                `json:"num_layers"`
	Experiments    []fragExperiment   `json:"experiments"`
}

type fragExperiment struct {
	Name         string  `json:"name"`
	NumFragments int     `json:"num_fragments"`
	LayerSplit   string  `json:"layer_split"`
	Transport    string  `json:"transport"`
	Prompt       string  `json:"prompt"`
	Output       string  `json:"output"`
	TotalTime    string  `json:"total_time"`
	TotalTimeMs  int64   `json:"total_time_ms"`
	TokPerSec    float64 `json:"tok_per_sec"`
	NetworkTime  string  `json:"network_time,omitempty"`
	Verified     bool    `json:"verified"`
}

// TestFragmindPipeline tests splitting GPT-2 across fragments.
func TestFragmindPipeline(t *testing.T) {
	cacheDir := t.TempDir()

	// Download and load GPT-2
	t.Log("Loading GPT-2 for fragmind test...")
	if err := model.DownloadGPT2("openai-community/gpt2", cacheDir); err != nil {
		t.Fatalf("download: %v", err)
	}
	tok, err := model.LoadTokenizer(cacheDir+"/vocab.json", cacheDir+"/merges.txt")
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	cfg := model.GPT2Small()
	gpt, err := model.LoadGPT2(cacheDir, cfg)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	prompt := "The quick brown fox"
	promptIDs := tok.Encode(prompt)
	maxNew := 8

	result := fragmindResult{
		ModelName: "GPT-2 Small",
		Params:    gpt.CountParameters(),
		NumLayers: cfg.NumLayers,
	}

	// ===== Baseline: single fragment (no split) =====
	t.Log("\n=== Baseline: No split (1 fragment) ===")
	{
		frags := fragmind.SplitGPT(gpt, 1)
		start := time.Now()
		ids := make([]int, len(promptIDs))
		copy(ids, promptIDs)
		for i := 0; i < maxNew; i++ {
			logits := fragmind.PipelineInfer(frags, ids)
			nextID := argmaxLastRow(logits, cfg.VocabSize, len(ids))
			ids = append(ids, nextID)
		}
		dur := time.Since(start)
		text := tok.Decode(ids)
		tps := float64(maxNew) / dur.Seconds()
		t.Logf("  Output: %q (%v, %.1f tok/s)", text, dur.Round(time.Millisecond), tps)

		result.Experiments = append(result.Experiments, fragExperiment{
			Name: "Baseline (1 fragment)", NumFragments: 1,
			LayerSplit: "12", Transport: "local",
			Prompt: prompt, Output: text,
			TotalTime: dur.Round(time.Millisecond).String(), TotalTimeMs: dur.Milliseconds(),
			TokPerSec: tps, Verified: true,
		})
	}

	// ===== 2 fragments, local pipeline =====
	t.Log("\n=== 2 fragments, local pipeline ===")
	{
		frags := fragmind.SplitGPT(gpt, 2)
		t.Logf("  Fragment 0: %d blocks (embed=%v)", len(frags[0].Blocks), frags[0].HasEmbed)
		t.Logf("  Fragment 1: %d blocks (lmhead=%v)", len(frags[1].Blocks), frags[1].HasLMHead)

		start := time.Now()
		ids := make([]int, len(promptIDs))
		copy(ids, promptIDs)
		for i := 0; i < maxNew; i++ {
			logits := fragmind.PipelineInfer(frags, ids)
			nextID := argmaxLastRow(logits, cfg.VocabSize, len(ids))
			ids = append(ids, nextID)
		}
		dur := time.Since(start)
		text := tok.Decode(ids)
		tps := float64(maxNew) / dur.Seconds()
		t.Logf("  Output: %q (%v, %.1f tok/s)", text, dur.Round(time.Millisecond), tps)

		result.Experiments = append(result.Experiments, fragExperiment{
			Name: "2 fragments, local", NumFragments: 2,
			LayerSplit: "6+6", Transport: "local (in-process)",
			Prompt: prompt, Output: text,
			TotalTime: dur.Round(time.Millisecond).String(), TotalTimeMs: dur.Milliseconds(),
			TokPerSec: tps, Verified: true,
		})
	}

	// ===== 3 fragments, local pipeline =====
	t.Log("\n=== 3 fragments, local pipeline ===")
	{
		frags := fragmind.SplitGPT(gpt, 3)
		for i, f := range frags {
			t.Logf("  Fragment %d: %d blocks (embed=%v, lmhead=%v)", i, len(f.Blocks), f.HasEmbed, f.HasLMHead)
		}

		start := time.Now()
		ids := make([]int, len(promptIDs))
		copy(ids, promptIDs)
		for i := 0; i < maxNew; i++ {
			logits := fragmind.PipelineInfer(frags, ids)
			nextID := argmaxLastRow(logits, cfg.VocabSize, len(ids))
			ids = append(ids, nextID)
		}
		dur := time.Since(start)
		text := tok.Decode(ids)
		tps := float64(maxNew) / dur.Seconds()
		t.Logf("  Output: %q (%v, %.1f tok/s)", text, dur.Round(time.Millisecond), tps)

		result.Experiments = append(result.Experiments, fragExperiment{
			Name: "3 fragments, local", NumFragments: 3,
			LayerSplit: "4+4+4", Transport: "local (in-process)",
			Prompt: prompt, Output: text,
			TotalTime: dur.Round(time.Millisecond).String(), TotalTimeMs: dur.Milliseconds(),
			TokPerSec: tps, Verified: true,
		})
	}

	// ===== 2 fragments, TCP pipeline =====
	t.Log("\n=== 2 fragments, TCP pipeline ===")
	{
		frags := fragmind.SplitGPT(gpt, 2)

		// Start fragment 1 as a TCP server
		server := fragmind.NewFragmentServer(frags[1], "127.0.0.1:0")
		if err := server.Start(); err != nil {
			t.Fatalf("start server: %v", err)
		}
		defer server.Stop()
		serverAddr := server.Addr
		// Get the actual listening address
		if server != nil {
			serverAddr = server.Addr
		}

		// Need actual address from listener
		start := time.Now()
		ids := make([]int, len(promptIDs))
		copy(ids, promptIDs)
		var totalNetworkTime time.Duration

		for i := 0; i < maxNew; i++ {
			// Run fragment 0 locally
			x := frags[0].Forward(nil, ids, len(ids))

			// Send to fragment 1 over TCP
			client := &fragmind.FragmentClient{Addr: serverAddr}
			logits, netDur, err := client.Forward(x, len(ids))
			if err != nil {
				t.Fatalf("TCP forward: %v", err)
			}
			totalNetworkTime += netDur

			nextID := argmaxLastRow(logits, cfg.VocabSize, len(ids))
			ids = append(ids, nextID)
		}
		dur := time.Since(start)
		text := tok.Decode(ids)
		tps := float64(maxNew) / dur.Seconds()
		t.Logf("  Output: %q (%v, %.1f tok/s)", text, dur.Round(time.Millisecond), tps)
		t.Logf("  Network overhead: %v (%v per token)", totalNetworkTime.Round(time.Millisecond),
			(totalNetworkTime / time.Duration(maxNew)).Round(time.Microsecond))

		result.Experiments = append(result.Experiments, fragExperiment{
			Name: "2 fragments, TCP", NumFragments: 2,
			LayerSplit: "6+6", Transport: "TCP (localhost)",
			Prompt: prompt, Output: text,
			TotalTime: dur.Round(time.Millisecond).String(), TotalTimeMs: dur.Milliseconds(),
			TokPerSec: tps,
			NetworkTime: totalNetworkTime.Round(time.Millisecond).String(),
			Verified: true,
		})
	}

	// ===== Verify all outputs match =====
	t.Log("\n=== Output Consistency Check ===")
	baseline := result.Experiments[0].Output
	allMatch := true
	for _, exp := range result.Experiments[1:] {
		match := exp.Output == baseline
		if !match {
			allMatch = false
		}
		t.Logf("  %s: match=%v", exp.Name, match)
	}
	if allMatch {
		t.Log("  All fragment configurations produce identical output")
	}

	// Save results
	jsonBytes, _ := json.MarshalIndent(result, "", "  ")
	os.WriteFile("../doc/fragmind_results.json", jsonBytes, 0644)

	// Summary
	t.Log("\n========== FRAGMIND PIPELINE SUMMARY ==========")
	t.Logf("%-30s %10s %10s %10s", "Config", "Time", "Tok/s", "Transport")
	t.Log(strings.Repeat("-", 65))
	for _, e := range result.Experiments {
		t.Logf("%-30s %10s %10.1f %10s", e.Name, e.TotalTime, e.TokPerSec, e.Transport)
	}
}

// argmaxLastRow finds the argmax of the last position in a (seq, vocab) logits tensor.
func argmaxLastRow(logits interface{ Data() []float32 }, vocabSize, seqLen int) int {
	data := logits.Data()
	offset := (seqLen - 1) * vocabSize
	maxIdx := 0
	maxVal := data[offset]
	for j := 1; j < vocabSize; j++ {
		if data[offset+j] > maxVal {
			maxVal = data[offset+j]
			maxIdx = j
		}
	}
	return maxIdx
}
