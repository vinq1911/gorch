//go:build darwin

package nn

import "testing"

// BenchmarkEncodeLinearAlloc measures the impact of Linear.Forward's
// output-buffer alloc strategy on GPT-2-small encode time. Compare
// against pre-fix main to see the win.
func BenchmarkEncodeLinearAlloc(b *testing.B) {
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
