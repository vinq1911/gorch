//go:build darwin

package qwen

import (
	"fmt"
	"testing"
)

// benchIDs returns n deterministic prompt token ids (the long fixture
// prompt tiled — real text statistics, no tokenizer dependency).
func benchIDs(n int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = promptLong[i%len(promptLong)]
	}
	return ids
}

// BenchmarkQwenDecode measures cached single-token decode throughput
// at fixed context lengths — plan 0008 §2.7's THE-unknown measurement
// (the demo's speak rate: real-time audio needs ≥100 tok/s, §5.2).
// The cache is prefilled to the context length once; each iteration
// decodes one token and trims the cache back so every measured step
// sees the same context.
func BenchmarkQwenDecode(b *testing.B) {
	m := loadModelCached(b)
	for _, ctx := range []int{128, 512, 1024, 2048} {
		b.Run(fmt.Sprintf("ctx%d", ctx), func(b *testing.B) {
			cache := m.NewCache()
			m.ForwardCached(benchIDs(ctx), cache)
			keep := ctx * cache.Dim
			tok := promptLong[0]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.ForwardCached([]int{tok}, cache)
				for l := range cache.Keys {
					cache.Keys[l] = cache.Keys[l][:keep]
					cache.Values[l] = cache.Values[l][:keep]
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// BenchmarkQwenPrefill measures multi-token prefill throughput
// (staircase-masked chunk through the cached path, fresh cache per
// iteration) at 512 and 1024 tokens.
func BenchmarkQwenPrefill(b *testing.B) {
	m := loadModelCached(b)
	for _, n := range []int{512, 1024} {
		b.Run(fmt.Sprintf("tokens%d", n), func(b *testing.B) {
			ids := benchIDs(n)
			cache := m.NewCache()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.Reset()
				m.ForwardCached(ids, cache)
			}
			b.StopTimer()
			b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
