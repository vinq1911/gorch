//go:build darwin

package nn

import (
	"testing"

	g "github.com/vinq1911/gorch"
)

// BenchmarkEncodeWithGrad measures Encode building the full autograd
// graph (the previous default for any model with requires_grad
// parameters — which means every model).
func BenchmarkEncodeWithGrad(b *testing.B) {
	gpt := NewGPT(50257, 768, 12, 12, 1024)
	tokens := make([]int, 64)
	for i := range tokens {
		tokens[i] = (i * 13) % 50257
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gpt.Encode(tokens)
	}
}

// BenchmarkEncodeNoGrad measures the same Encode under g.NoGrad —
// no autograd graph, activations free to GC immediately.
func BenchmarkEncodeNoGrad(b *testing.B) {
	gpt := NewGPT(50257, 768, 12, 12, 1024)
	tokens := make([]int, 64)
	for i := range tokens {
		tokens[i] = (i * 13) % 50257
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.NoGrad(func() {
			_ = gpt.Encode(tokens)
		})
	}
}

// BenchmarkEncodeBatchNoGrad: NoGrad + batched encode — what kgate
// actually wants for inference.
func BenchmarkEncodeBatchNoGrad(b *testing.B) {
	gpt := NewGPT(50257, 768, 12, 12, 1024)
	batch := make([][]int, 16)
	for j := range batch {
		seq := make([]int, 64)
		for i := range seq {
			seq[i] = ((j+1)*i*13) % 50257
		}
		batch[j] = seq
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.NoGrad(func() {
			_ = gpt.EncodeBatch(batch)
		})
	}
}
