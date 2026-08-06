//go:build darwin

package nn

import (
	"testing"

	g "github.com/vinq1911/gorch"
)

// ---------- KV cache benchmarks ----------

// BenchmarkGenerateUncached measures un-cached autoregressive
// generation: each step recomputes attention over the full prefix.
// O(N²) per token in seq length.
func BenchmarkGenerateUncached(b *testing.B) {
	gpt := NewGPT(256, 64, 4, 4, 256)
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	const newTokens = 64

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids := append([]int{}, prompt...)
		for j := 0; j < newTokens; j++ {
			logits := gpt.Forward(ids)
			lastIdx := argmaxRow(logits.Data(), len(ids)-1, gpt.VocabSize)
			ids = append(ids, lastIdx)
		}
	}
}

// BenchmarkGenerateCached measures KV-cache autoregressive generation:
// prefill once, then 1-token steps. O(N) per token (only the new
// query attends to the full cache).
func BenchmarkGenerateCached(b *testing.B) {
	gpt := NewGPT(256, 64, 4, 4, 256)
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	const newTokens = 64

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := NewKVCache(gpt.NumLayers, gpt.Dim)
		ids := append([]int{}, prompt...)
		logits := gpt.ForwardCached(ids, cache)
		lastIdx := argmaxRow(logits.Data(), len(ids)-1, gpt.VocabSize)
		ids = append(ids, lastIdx)
		for j := 1; j < newTokens; j++ {
			step := gpt.ForwardCached([]int{lastIdx}, cache)
			lastIdx = argmaxRow(step.Data(), 0, gpt.VocabSize)
			ids = append(ids, lastIdx)
		}
	}
}

// ---------- GPU autograd benchmark ----------

// BenchmarkLinearTrainStepCPULarge mirrors LinearTrainStepCPU at a
// shape where MPS amortises its dispatch overhead and the GPU bench
// is supposed to win. Useful crossover marker.
func BenchmarkLinearTrainStepCPULarge(b *testing.B) {
	const batch, in, out = 256, 2048, 2048
	l := NewLinear(in, out)
	xData := make([]float32, batch*in)
	for i := range xData {
		xData[i] = float32(i%17) * 0.01
	}
	x := g.NewTensor(xData, batch, in)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Weight.ZeroGrad()
		l.Bias.ZeroGrad()
		y := l.Forward(x)
		loss := g.Sum(y)
		loss.Backward()
	}
}

// BenchmarkLinearTrainStepMetalLarge — large-shape GPU bench.
func BenchmarkLinearTrainStepMetalLarge(b *testing.B) {
	gpu, err := g.InitMetal()
	if err != nil {
		b.Skipf("metal not available: %v", err)
	}
	const batch, in, out = 256, 2048, 2048
	l := NewLinear(in, out)
	l.ToMetal(gpu.Dev)
	xData := make([]float32, batch*in)
	for i := range xData {
		xData[i] = float32(i%17) * 0.01
	}
	x := g.NewTensorOnMetal(gpu.Dev, xData, batch, in)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Weight.ZeroGrad()
		l.Bias.ZeroGrad()
		y := l.Forward(x)
		loss := g.Sum(y)
		loss.Backward()
	}
}

// BenchmarkLinearTrainStepCPU runs one full forward+backward+update
// on a Linear-only model with weights resident on CPU.
func BenchmarkLinearTrainStepCPU(b *testing.B) {
	const batch, in, out = 64, 768, 768
	l := NewLinear(in, out)
	xData := make([]float32, batch*in)
	for i := range xData {
		xData[i] = float32(i%17) * 0.01
	}
	x := g.NewTensor(xData, batch, in)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Weight.ZeroGrad()
		l.Bias.ZeroGrad()
		y := l.Forward(x)
		loss := g.Sum(y) // scalar, requires_grad propagates
		loss.Backward()
	}
}

// BenchmarkLinearTrainStepMetal mirrors the above with weights on
// Metal. Measures whether GPU dispatch shows up as a win at this
// shape on Apple M-series.
func BenchmarkLinearTrainStepMetal(b *testing.B) {
	gpu, err := g.InitMetal()
	if err != nil {
		b.Skipf("metal not available: %v", err)
	}
	const batch, in, out = 64, 768, 768
	l := NewLinear(in, out)
	l.ToMetal(gpu.Dev)
	xData := make([]float32, batch*in)
	for i := range xData {
		xData[i] = float32(i%17) * 0.01
	}
	x := g.NewTensorOnMetal(gpu.Dev, xData, batch, in)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Weight.ZeroGrad()
		l.Bias.ZeroGrad()
		y := l.Forward(x)
		loss := g.Sum(y)
		loss.Backward()
	}
}

// argmaxRow returns the argmax of one row of a (rows, cols) flat
// slice. Helper for the generation benchmarks above.
func argmaxRow(data []float32, row, cols int) int {
	off := row * cols
	maxIdx := 0
	maxVal := data[off]
	for j := 1; j < cols; j++ {
		if data[off+j] > maxVal {
			maxVal = data[off+j]
			maxIdx = j
		}
	}
	return maxIdx
}
