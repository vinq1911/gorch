//go:build darwin

package nn

import (
	"testing"

	g "github.com/vinq1911/gorch"
)

// Diagnostic benchmarks decomposing the GPU train-step regression.
// Running these together pinpoints which phase costs what.

func benchSetup(b *testing.B, metal bool) (*Linear, *g.Tensor) {
	const batch, in, out = 64, 768, 768
	l := NewLinear(in, out)
	xData := make([]float32, batch*in)
	for i := range xData {
		xData[i] = float32(i%17) * 0.01
	}
	if metal {
		gpu, err := g.InitMetal()
		if err != nil {
			b.Skipf("metal not available: %v", err)
		}
		l.ToMetal(gpu.Dev)
		return l, g.NewTensorOnMetal(gpu.Dev, xData, batch, in)
	}
	return l, g.NewTensor(xData, batch, in)
}

// Pure forward, no backward. Measures GPU vs CPU on the matmul + bias.
func BenchmarkLinearForwardOnlyCPU(b *testing.B) {
	l, x := benchSetup(b, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Forward(x)
	}
}

func BenchmarkLinearForwardOnlyMetal(b *testing.B) {
	l, x := benchSetup(b, true)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Forward(x)
	}
}

// Forward + Sum (CPU), no backward. Adds the cross-device read.
func BenchmarkLinearForwardSumCPU(b *testing.B) {
	l, x := benchSetup(b, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y := l.Forward(x)
		_ = g.Sum(y)
	}
}

func BenchmarkLinearForwardSumMetal(b *testing.B) {
	l, x := benchSetup(b, true)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y := l.Forward(x)
		_ = g.Sum(y)
	}
}
