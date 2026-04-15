//go:build darwin && e2e

package e2e

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
)

type metalBenchResult struct {
	Backend   string         `json:"backend"`
	Prompt    string         `json:"prompt"`
	Output    string         `json:"output"`
	Tokens    int            `json:"tokens"`
	Time      string         `json:"time"`
	TimeMs    int64          `json:"time_ms"`
	TokPerSec float64       `json:"tok_per_sec"`
}

type metalReport struct {
	ModelName string              `json:"model_name"`
	Params    int                 `json:"params"`
	LoadTime  string              `json:"load_time"`
	Results   []metalBenchResult  `json:"results"`
}

// TestMetalVsAccelerateInference benchmarks GPT-2 inference on Metal GPU vs Accelerate CPU.
func TestMetalVsAccelerateInference(t *testing.T) {
	cacheDir := t.TempDir()

	// Download and setup
	t.Log("Downloading GPT-2...")
	if err := model.DownloadGPT2("openai-community/gpt2", cacheDir); err != nil {
		t.Fatalf("download: %v", err)
	}
	tok, err := model.LoadTokenizer(cacheDir+"/vocab.json", cacheDir+"/merges.txt")
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}

	cfg := model.GPT2Small()
	prompts := []string{
		"The quick brown fox",
		"In the beginning there was",
		"Machine learning is",
	}
	maxNew := 15

	report := metalReport{ModelName: "GPT-2 Small", Params: 0}

	// ===== Accelerate CPU =====
	t.Log("\n=== Accelerate CPU ===")
	{
		gpt, err := model.LoadGPT2(cacheDir, cfg)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		report.Params = gpt.CountParameters()

		for _, prompt := range prompts {
			ids := tok.Encode(prompt)
			start := time.Now()
			outputIDs := model.Generate(gpt, ids, maxNew)
			dur := time.Since(start)
			text := tok.Decode(outputIDs)
			tps := float64(maxNew) / dur.Seconds()
			t.Logf("  [CPU] %q → %q (%v, %.1f tok/s)", prompt, tok.Decode(outputIDs[len(ids):]), dur.Round(time.Millisecond), tps)

			report.Results = append(report.Results, metalBenchResult{
				Backend: "Accelerate CPU", Prompt: prompt, Output: text,
				Tokens: maxNew, Time: dur.Round(time.Millisecond).String(),
				TimeMs: dur.Milliseconds(), TokPerSec: tps,
			})
		}
	}

	// ===== Metal GPU =====
	t.Log("\n=== Metal GPU ===")
	{
		gpt, err := model.LoadGPT2(cacheDir, cfg)
		if err != nil {
			t.Fatalf("load: %v", err)
		}

		// Init Metal
		gpu, err := g.InitMetal()
		if err != nil {
			t.Fatalf("init metal: %v", err)
		}

		// Move model to GPU
		t.Log("Moving model to Metal GPU...")
		start := time.Now()
		gpt.ToMetal(gpu.Dev)
		moveTime := time.Since(start)
		t.Logf("Model moved to Metal in %v", moveTime.Round(time.Millisecond))

		for _, prompt := range prompts {
			ids := tok.Encode(prompt)
			start := time.Now()
			outputIDs := model.Generate(gpt, ids, maxNew)
			dur := time.Since(start)
			text := tok.Decode(outputIDs)
			tps := float64(maxNew) / dur.Seconds()
			t.Logf("  [GPU] %q → %q (%v, %.1f tok/s)", prompt, tok.Decode(outputIDs[len(ids):]), dur.Round(time.Millisecond), tps)

			report.Results = append(report.Results, metalBenchResult{
				Backend: "Metal GPU", Prompt: prompt, Output: text,
				Tokens: maxNew, Time: dur.Round(time.Millisecond).String(),
				TimeMs: dur.Milliseconds(), TokPerSec: tps,
			})
		}
	}

	// Save results
	jsonBytes, _ := json.MarshalIndent(report, "", "  ")
	os.WriteFile("../doc/metal_inference_results.json", jsonBytes, 0644)

	// Summary
	t.Log("\n========== METAL vs ACCELERATE SUMMARY ==========")
	t.Logf("%-15s %-30s %10s %10s", "Backend", "Prompt", "Time", "Tok/s")
	t.Log(strings.Repeat("-", 70))
	for _, r := range report.Results {
		t.Logf("%-15s %-30s %10s %10.1f", r.Backend, r.Prompt, r.Time, r.TokPerSec)
	}

	// Compute averages
	var cpuAvg, gpuAvg float64
	cpuN, gpuN := 0, 0
	for _, r := range report.Results {
		if r.Backend == "Accelerate CPU" {
			cpuAvg += r.TokPerSec
			cpuN++
		} else {
			gpuAvg += r.TokPerSec
			gpuN++
		}
	}
	if cpuN > 0 { cpuAvg /= float64(cpuN) }
	if gpuN > 0 { gpuAvg /= float64(gpuN) }
	t.Logf("\nAverage: CPU=%.1f tok/s, GPU=%.1f tok/s", cpuAvg, gpuAvg)
	if gpuAvg > cpuAvg {
		t.Logf("Metal GPU is %.1fx faster", gpuAvg/cpuAvg)
	} else {
		t.Logf("Accelerate CPU is %.1fx faster (Metal overhead dominates at this scale)", cpuAvg/gpuAvg)
	}
}
