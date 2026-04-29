//go:build darwin

package nn

import "testing"

// BenchmarkEncode is the per-sequence baseline.
func BenchmarkEncode(b *testing.B) {
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

// BenchmarkEncodeBatch16 batches 16 same-length sequences in one
// forward — what kgate's NLI cross-encoder workload looks like.
func BenchmarkEncodeBatch16(b *testing.B) {
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
		_ = gpt.EncodeBatch(batch)
	}
}

// BenchmarkEncodeSerial16 runs 16 sequential Encode calls — the
// pre-batched-API baseline. Compare against EncodeBatch16.
func BenchmarkEncodeSerial16(b *testing.B) {
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
		for _, ids := range batch {
			_ = gpt.Encode(ids)
		}
	}
}
