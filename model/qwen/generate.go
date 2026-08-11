//go:build darwin

package qwen

import (
	"math"
	"math/rand"
	"sort"
)

// GenerateConfig controls generation (thin Qwen-typed port of
// model.GenerateConfig; the sampling helpers mirror model/generate.go).
type GenerateConfig struct {
	MaxNewTokens int
	Temperature  float32 // 0 = greedy
	TopK         int     // 0 = disabled
	TopP         float32 // 0 = disabled
	StopTokens   []int   // stop when one of these is generated (not appended)
	Seed         int64   // sampling RNG seed (ignored for greedy)
}

// Greedy returns a deterministic greedy-decoding config with the
// Qwen ChatML stop tokens.
func Greedy(maxNewTokens int) GenerateConfig {
	return GenerateConfig{
		MaxNewTokens: maxNewTokens,
		StopTokens:   []int{151645, 151643}, // <|im_end|>, <|endoftext|>
	}
}

// Generate produces new token ids after prompt using KV-cached
// incremental decoding: one staircase-masked prefill pass over the
// prompt, then one token per step. Returns prompt + generated ids
// (stop token excluded). Inference-only.
func Generate(m *Model, prompt []int, cfg GenerateConfig) []int {
	if len(prompt) == 0 {
		panic("qwen: empty prompt")
	}
	result := make([]int, len(prompt), len(prompt)+cfg.MaxNewTokens)
	copy(result, prompt)

	rng := rand.New(rand.NewSource(cfg.Seed))
	cache := m.NewCache()
	logits := m.ForwardCached(prompt, cache) // prefill → (1, vocab)
	last := logits.Data()

	for i := 0; i < cfg.MaxNewTokens; i++ {
		var next int
		if cfg.Temperature == 0 {
			next = argmaxF32(last)
		} else {
			next = sampleLogits(last, cfg.Temperature, cfg.TopK, cfg.TopP, rng)
		}
		if isStop(next, cfg.StopTokens) {
			break
		}
		result = append(result, next)
		if cache.Len() >= m.Cfg.MaxSeq {
			break
		}
		logits = m.ForwardCached([]int{next}, cache)
		last = logits.Data()
	}
	return result
}

func isStop(tok int, stops []int) bool {
	for _, s := range stops {
		if tok == s {
			return true
		}
	}
	return false
}

// argmaxF32 returns the index of the maximum value.
func argmaxF32(logits []float32) int {
	maxIdx := 0
	maxVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// sampleLogits samples with temperature, top-k, and top-p
// (model/generate.go sample() pattern, with an explicit RNG for
// deterministic seeding).
func sampleLogits(logits []float32, temperature float32, topK int, topP float32, rng *rand.Rand) int {
	n := len(logits)
	scaled := make([]float32, n)
	for i, v := range logits {
		scaled[i] = v / temperature
	}
	if topK > 0 && topK < n {
		threshold := kthLargestF32(scaled, topK)
		for i := range scaled {
			if scaled[i] < threshold {
				scaled[i] = float32(math.Inf(-1))
			}
		}
	}

	maxVal := scaled[0]
	for _, v := range scaled[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	probs := make([]float64, n)
	var sum float64
	for i, v := range scaled {
		probs[i] = math.Exp(float64(v - maxVal))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	if topP > 0 && topP < 1 {
		// Nucleus: keep the smallest prefix of descending-prob tokens
		// whose cumulative mass reaches topP; zero the rest.
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
		var cum float64
		cut := n
		for r, i := range idx {
			cum += probs[i]
			if cum >= float64(topP) {
				cut = r + 1
				break
			}
		}
		var resum float64
		kept := make(map[int]bool, cut)
		for _, i := range idx[:cut] {
			kept[i] = true
			resum += probs[i]
		}
		for i := range probs {
			if !kept[i] {
				probs[i] = 0
			} else {
				probs[i] /= resum
			}
		}
	}

	r := rng.Float64()
	var cumulative float64
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return n - 1
}

// kthLargestF32 returns the k-th largest value (1-indexed) —
// model/generate.go kthLargest pattern.
func kthLargestF32(data []float32, k int) float32 {
	sorted := make([]float32, len(data))
	copy(sorted, data)
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] > sorted[maxIdx] {
				maxIdx = j
			}
		}
		sorted[i], sorted[maxIdx] = sorted[maxIdx], sorted[i]
	}
	return sorted[k-1]
}
