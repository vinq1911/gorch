//go:build darwin && e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
	"github.com/vinq1911/gorch/metal"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/nn"
)

type matmulResult struct {
	Size       int     `json:"size"`
	Label      string  `json:"label"`
	CPUTimeUs  float64 `json:"cpu_time_us"`
	GPUTimeUs  float64 `json:"gpu_time_us"`
	Speedup    float64 `json:"speedup"`
	CPUGFLOPS  float64 `json:"cpu_gflops"`
	GPUGFLOPS  float64 `json:"gpu_gflops"`
}

type crossoverReport struct {
	Hardware      string            `json:"hardware"`
	MatmulResults []matmulResult    `json:"matmul_results"`
	TransformerResults []transformerResult `json:"transformer_results"`
}

type transformerResult struct {
	Name      string  `json:"name"`
	Dim       int     `json:"dim"`
	Layers    int     `json:"layers"`
	SeqLen    int     `json:"seq_len"`
	CPUTimeMs float64 `json:"cpu_time_ms"`
	GPUTimeMs float64 `json:"gpu_time_ms"`
	Speedup   float64 `json:"speedup"`
}

// TestMetalCrossover finds the matrix size where Metal GPU beats Accelerate CPU.
func TestMetalCrossover(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatalf("no Metal: %v", err)
	}
	queue := dev.NewCommandQueue()

	report := crossoverReport{Hardware: "Apple M4"}

	// ================================================================
	// Part 1: Standalone SGEMM at increasing sizes
	// ================================================================
	t.Log("=== Part 1: Matmul Crossover Point ===")
	t.Logf("%-8s %12s %12s %10s %10s %10s", "Size", "CPU (µs)", "GPU (µs)", "Speedup", "CPU GFLOPS", "GPU GFLOPS")
	t.Log(strings.Repeat("-", 70))

	sizes := []int{256, 512, 768, 1024, 1536, 2048, 3072, 4096}
	warmup := 5
	iters := 20

	for _, n := range sizes {
		flops := 2.0 * float64(n) * float64(n) * float64(n)

		// CPU: Accelerate BLAS
		a := make([]float32, n*n)
		b := make([]float32, n*n)
		c := make([]float32, n*n)
		for i := range a { a[i] = 0.01; b[i] = 0.01 }

		for i := 0; i < warmup; i++ {
			accelerate.Sgemm(n, n, n, 1.0, a, b, 0.0, c)
		}
		cpuStart := time.Now()
		for i := 0; i < iters; i++ {
			accelerate.Sgemm(n, n, n, 1.0, a, b, 0.0, c)
		}
		cpuTotal := time.Since(cpuStart)
		cpuUs := float64(cpuTotal.Microseconds()) / float64(iters)

		// GPU: MPS matmul
		bufA := dev.NewBuffer(n * n * 4)
		bufB := dev.NewBuffer(n * n * 4)
		bufC := dev.NewBuffer(n * n * 4)
		aSlice := bufA.FloatSlice()
		bSlice := bufB.FloatSlice()
		for i := range aSlice { aSlice[i] = 0.01; bSlice[i] = 0.01 }

		for i := 0; i < warmup; i++ {
			queue.MatMul(bufA, bufB, bufC, n, n, n)
		}
		gpuStart := time.Now()
		for i := 0; i < iters; i++ {
			queue.MatMul(bufA, bufB, bufC, n, n, n)
		}
		gpuTotal := time.Since(gpuStart)
		gpuUs := float64(gpuTotal.Microseconds()) / float64(iters)

		speedup := cpuUs / gpuUs
		cpuGFlops := flops / (cpuUs * 1e-6) / 1e9
		gpuGFlops := flops / (gpuUs * 1e-6) / 1e9

		t.Logf("%-8d %10.0f µs %10.0f µs %9.2fx %9.1f %9.1f", n, cpuUs, gpuUs, speedup, cpuGFlops, gpuGFlops)

		report.MatmulResults = append(report.MatmulResults, matmulResult{
			Size: n, Label: fmt.Sprintf("%dx%d", n, n),
			CPUTimeUs: cpuUs, GPUTimeUs: gpuUs, Speedup: speedup,
			CPUGFLOPS: cpuGFlops, GPUGFLOPS: gpuGFlops,
		})

		bufA.Release(); bufB.Release(); bufC.Release()
	}

	// ================================================================
	// Part 2: Synthetic transformer forward pass at different dims
	// ================================================================
	t.Log("\n=== Part 2: Synthetic Transformer Forward Pass ===")

	gpu, err := g.InitMetal()
	if err != nil {
		t.Fatalf("init metal: %v", err)
	}
	_ = gpu

	configs := []struct {
		dim, heads, layers, seq int
		name string
	}{
		{768, 12, 2, 32, "GPT-2 Small dim (768, 2L, seq=32)"},
		{1024, 16, 2, 32, "GPT-2 Medium dim (1024, 2L, seq=32)"},
		{2048, 16, 2, 32, "Large dim (2048, 2L, seq=32)"},
		{2048, 16, 4, 64, "Large dim (2048, 4L, seq=64)"},
		{4096, 32, 2, 32, "XL dim (4096, 2L, seq=32)"},
	}

	t.Logf("%-40s %10s %10s %10s", "Config", "CPU (ms)", "GPU (ms)", "Speedup")
	t.Log(strings.Repeat("-", 75))

	for _, cfg := range configs {
		// Build model
		mdl := nn.NewGPT(1000, cfg.dim, cfg.heads, cfg.layers, 256)
		tokens := make([]int, cfg.seq)
		for i := range tokens { tokens[i] = i % 1000 }

		// CPU timing
		cpuStart := time.Now()
		for i := 0; i < 3; i++ {
			mdl.Forward(tokens)
		}
		cpuMs := float64(time.Since(cpuStart).Milliseconds()) / 3.0

		// GPU: move to Metal and time
		mdl2 := nn.NewGPT(1000, cfg.dim, cfg.heads, cfg.layers, 256)
		// Copy weights from mdl to mdl2 so they're the same
		p1 := mdl.Parameters()
		p2 := mdl2.Parameters()
		for i := range p1 {
			copy(p2[i].Data(), p1[i].Data())
		}
		mdl2.ToMetal(gpu.Dev)

		// Warmup
		mdl2.Forward(tokens)

		gpuStart := time.Now()
		for i := 0; i < 3; i++ {
			mdl2.Forward(tokens)
		}
		gpuMs := float64(time.Since(gpuStart).Milliseconds()) / 3.0

		speedup := cpuMs / gpuMs
		t.Logf("%-40s %8.0f ms %8.0f ms %9.2fx", cfg.name, cpuMs, gpuMs, speedup)

		report.TransformerResults = append(report.TransformerResults, transformerResult{
			Name: cfg.name, Dim: cfg.dim, Layers: cfg.layers, SeqLen: cfg.seq,
			CPUTimeMs: cpuMs, GPUTimeMs: gpuMs, Speedup: speedup,
		})
	}

	// ================================================================
	// Part 3: GPT-2 pretrained at longer sequence
	// ================================================================
	t.Log("\n=== Part 3: GPT-2 Pretrained at Longer Sequence ===")
	{
		cacheDir := t.TempDir()
		model.DownloadGPT2("openai-community/gpt2", cacheDir)
		tok, _ := model.LoadTokenizer(cacheDir+"/vocab.json", cacheDir+"/merges.txt")
		cfg := model.GPT2Small()

		// CPU
		gptCPU, _ := model.LoadGPT2(cacheDir, cfg)
		prompt := "The history of artificial intelligence begins with ancient myths and legends of mechanical beings. In modern times, the field emerged in the 1950s when researchers"
		ids := tok.Encode(prompt)
		t.Logf("Prompt tokens: %d", len(ids))

		cpuStart := time.Now()
		cpuOut := model.Generate(gptCPU, ids, 20)
		cpuDur := time.Since(cpuStart)
		cpuText := tok.Decode(cpuOut[len(ids):])

		// GPU
		gptGPU, _ := model.LoadGPT2(cacheDir, cfg)
		gptGPU.ToMetal(gpu.Dev)

		gpuStart := time.Now()
		gpuOut := model.Generate(gptGPU, ids, 20)
		gpuDur := time.Since(gpuStart)
		gpuText := tok.Decode(gpuOut[len(ids):])

		t.Logf("CPU: %v (%.1f tok/s) → %q", cpuDur.Round(time.Millisecond), 20.0/cpuDur.Seconds(), cpuText)
		t.Logf("GPU: %v (%.1f tok/s) → %q", gpuDur.Round(time.Millisecond), 20.0/gpuDur.Seconds(), gpuText)
		t.Logf("Match: %v", cpuText == gpuText)
	}

	// Save results
	jsonBytes, _ := json.MarshalIndent(report, "", "  ")
	os.WriteFile("../doc/metal_crossover_results.json", jsonBytes, 0644)
}
