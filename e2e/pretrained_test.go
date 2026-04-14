//go:build darwin && e2e

package e2e

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vinq1911/gorch/model"
)

type pretrainedResult struct {
	ModelName  string         `json:"model_name"`
	Params     int            `json:"params"`
	LoadTime   string         `json:"load_time"`
	LoadTimeMs int64          `json:"load_time_ms"`
	Prompts    []promptResult `json:"prompts"`
}

type promptResult struct {
	Prompt       string  `json:"prompt"`
	Generated    string  `json:"generated"`
	NewTokens    int     `json:"new_tokens"`
	GenTime      string  `json:"gen_time"`
	GenTimeMs    int64   `json:"gen_time_ms"`
	TokensPerSec float64 `json:"tokens_per_sec"`
}

// TestPretrainedGPT2 downloads GPT-2 small (124M params) and runs text generation.
func TestPretrainedGPT2(t *testing.T) {
	cacheDir := t.TempDir()

	// Download model files
	t.Log("Downloading GPT-2 small (124M params, ~131MB)...")
	err := model.DownloadGPT2("openai-community/gpt2", cacheDir)
	if err != nil {
		t.Fatalf("download: %v", err)
	}

	// Load tokenizer
	t.Log("Loading tokenizer...")
	tok, err := model.LoadTokenizer(
		cacheDir+"/vocab.json",
		cacheDir+"/merges.txt",
	)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	t.Logf("Tokenizer vocab size: %d", tok.VocabSize)

	// Load model
	t.Log("Loading GPT-2 model...")
	cfg := model.GPT2Small()
	start := time.Now()
	gpt, err := model.LoadGPT2(cacheDir, cfg)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	loadTime := time.Since(start)
	t.Logf("Model loaded: %d params in %v", gpt.CountParameters(), loadTime.Round(time.Millisecond))

	// Test prompts — generate fewer tokens since GPT-2 is large
	prompts := []string{
		"The meaning of life is",
		"Once upon a time in a land far away",
		"Artificial intelligence will",
	}

	result := pretrainedResult{
		ModelName:  "GPT-2 Small (124M)",
		Params:     gpt.CountParameters(),
		LoadTime:   loadTime.Round(time.Millisecond).String(),
		LoadTimeMs: loadTime.Milliseconds(),
	}

	maxNewTokens := 10 // keep it small — no KV cache means O(n^2) per token

	for _, prompt := range prompts {
		t.Logf("\nPrompt: %q", prompt)
		ids := tok.Encode(prompt)
		t.Logf("  Token IDs (%d tokens): %v", len(ids), ids)

		start := time.Now()
		outputIDs := model.Generate(gpt, ids, maxNewTokens)
		genTime := time.Since(start)

		generated := tok.Decode(outputIDs)
		newText := tok.Decode(outputIDs[len(ids):])
		tokPerSec := float64(maxNewTokens) / genTime.Seconds()

		t.Logf("  Full output: %q", generated)
		t.Logf("  New text: %q", newText)
		t.Logf("  Time: %v (%.2f tok/s)", genTime.Round(time.Millisecond), tokPerSec)

		result.Prompts = append(result.Prompts, promptResult{
			Prompt:       prompt,
			Generated:    generated,
			NewTokens:    maxNewTokens,
			GenTime:      genTime.Round(time.Millisecond).String(),
			GenTimeMs:    genTime.Milliseconds(),
			TokensPerSec: tokPerSec,
		})
	}

	// Save results
	jsonBytes, _ := json.MarshalIndent(result, "", "  ")
	jsonPath := "../doc/pretrained_results.json"
	os.WriteFile(jsonPath, jsonBytes, 0644)

	// Print summary
	t.Log("\n========== PRETRAINED INFERENCE SUMMARY ==========")
	t.Logf("Model: %s (%d params)", result.ModelName, result.Params)
	t.Logf("Load time: %s", result.LoadTime)
	t.Log(strings.Repeat("-", 60))
	for _, p := range result.Prompts {
		t.Logf("Prompt: %q", p.Prompt)
		t.Logf("  → %q", p.Generated)
		t.Logf("  %d tokens in %s (%.2f tok/s)", p.NewTokens, p.GenTime, p.TokensPerSec)
	}
}
