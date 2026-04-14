//go:build darwin

package model

import (
	"math"
	"math/rand"

	"github.com/vinq1911/gorch/nn"
)

// GenerateConfig controls text generation behavior.
type GenerateConfig struct {
	MaxNewTokens int     // maximum tokens to generate
	Temperature  float32 // 0 = greedy, >0 = sample with temperature
	TopK         int     // 0 = disabled, >0 = sample from top-K
	TopP         float32 // 0 = disabled, >0 = nucleus sampling threshold
	StopToken    int     // -1 = disabled, otherwise stop at this token
}

// DefaultGenerateConfig returns sensible defaults for text generation.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxNewTokens: 50,
		Temperature:  0.8,
		TopK:         40,
		TopP:         0.0,
		StopToken:    -1,
	}
}

// GreedyConfig returns config for deterministic greedy decoding.
func GreedyConfig(maxTokens int) GenerateConfig {
	return GenerateConfig{
		MaxNewTokens: maxTokens,
		Temperature:  0,
		TopK:         0,
		TopP:         0,
		StopToken:    -1,
	}
}

// GenerateText produces text from a prompt using the given model and config.
func GenerateText(model *nn.GPT, tok *BPETokenizer, prompt string, cfg GenerateConfig) string {
	ids := tok.Encode(prompt)
	outputIDs := GenerateWithConfig(model, ids, cfg)
	return tok.Decode(outputIDs)
}

// GenerateWithConfig generates tokens with temperature, top-k, and top-p sampling.
func GenerateWithConfig(model *nn.GPT, tokenIDs []int, cfg GenerateConfig) []int {
	result := make([]int, len(tokenIDs))
	copy(result, tokenIDs)

	for i := 0; i < cfg.MaxNewTokens; i++ {
		// Truncate to max sequence length
		input := result
		if len(input) > model.MaxSeq {
			input = input[len(input)-model.MaxSeq:]
		}

		// Forward pass
		logits := model.Forward(input)

		// Get logits for last position
		lastLogits := logits.Data()[(len(input)-1)*model.VocabSize : len(input)*model.VocabSize]

		// Sample next token
		var nextToken int
		if cfg.Temperature == 0 {
			nextToken = argmax(lastLogits)
		} else {
			nextToken = sample(lastLogits, cfg.Temperature, cfg.TopK, cfg.TopP)
		}

		// Check stop token
		if cfg.StopToken >= 0 && nextToken == cfg.StopToken {
			break
		}

		result = append(result, nextToken)
	}

	return result
}

// argmax returns the index of the maximum value.
func argmax(logits []float32) int {
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

// sample samples from logits with temperature, top-k, and top-p.
func sample(logits []float32, temperature float32, topK int, topP float32) int {
	n := len(logits)

	// Apply temperature
	scaled := make([]float32, n)
	for i, v := range logits {
		scaled[i] = v / temperature
	}

	// Top-K filtering
	if topK > 0 && topK < n {
		threshold := kthLargest(scaled, topK)
		for i := range scaled {
			if scaled[i] < threshold {
				scaled[i] = float32(math.Inf(-1))
			}
		}
	}

	// Softmax
	maxVal := scaled[0]
	for _, v := range scaled[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	probs := make([]float64, n)
	for i, v := range scaled {
		probs[i] = math.Exp(float64(v - maxVal))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Top-P (nucleus) filtering
	if topP > 0 && topP < 1 {
		// Simplified top-P: zero out low-probability tokens
		// Sort by descending prob, accumulate, zero rest
		for {
			// Find max remaining prob
			maxP := 0.0
			maxI := -1
			cumProb := 0.0
			for i, p := range probs {
				if p > 0 {
					cumProb += p
				}
				if p > maxP {
					maxP = p
					maxI = i
				}
			}
			_ = maxI
			if cumProb <= float64(topP)*1.5 {
				break // already pruned enough
			}
			// Find and zero the smallest non-zero prob
			minP := math.MaxFloat64
			minI := -1
			for i, p := range probs {
				if p > 0 && p < minP {
					minP = p
					minI = i
				}
			}
			if minI >= 0 {
				probs[minI] = 0
			} else {
				break
			}
		}
		// Renormalize
		var resum float64
		for _, p := range probs {
			resum += p
		}
		for i := range probs {
			probs[i] /= resum
		}
	}

	// Sample from distribution
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return n - 1
}

// kthLargest returns the k-th largest value in the slice (1-indexed).
func kthLargest(data []float32, k int) float32 {
	// Simple approach: partial sort
	// For vocab sizes (~50K), this is fast enough
	sorted := make([]float32, len(data))
	copy(sorted, data)

	// Partial selection: find k-th largest
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

// KVCache stores precomputed key-value pairs for efficient autoregressive generation.
// This avoids recomputing attention for all previous tokens.
type KVCache struct {
	Keys   [][][]float32 // [layer][head] → flat (seqSoFar * headDim)
	Values [][][]float32 // [layer][head] → flat (seqSoFar * headDim)
	Layers int
	Heads  int
	HeadDim int
	SeqLen int // number of tokens cached so far
}

// NewKVCache creates an empty KV cache for a model.
func NewKVCache(numLayers, numHeads, headDim int) *KVCache {
	keys := make([][][]float32, numLayers)
	values := make([][][]float32, numLayers)
	for l := 0; l < numLayers; l++ {
		keys[l] = make([][]float32, numHeads)
		values[l] = make([][]float32, numHeads)
		for h := 0; h < numHeads; h++ {
			keys[l][h] = make([]float32, 0)
			values[l][h] = make([]float32, 0)
		}
	}
	return &KVCache{
		Keys: keys, Values: values,
		Layers: numLayers, Heads: numHeads, HeadDim: headDim,
	}
}

// Append adds new key-value vectors for one token to the cache.
func (kv *KVCache) Append(layer, head int, key, value []float32) {
	kv.Keys[layer][head] = append(kv.Keys[layer][head], key...)
	kv.Values[layer][head] = append(kv.Values[layer][head], value...)
}

// GetKeys returns all cached keys for a layer/head as (seqLen, headDim).
func (kv *KVCache) GetKeys(layer, head int) []float32 {
	return kv.Keys[layer][head]
}

// GetValues returns all cached values for a layer/head.
func (kv *KVCache) GetValues(layer, head int) []float32 {
	return kv.Values[layer][head]
}

// Len returns the number of tokens cached.
func (kv *KVCache) Len() int {
	if len(kv.Keys) == 0 || len(kv.Keys[0]) == 0 {
		return 0
	}
	return len(kv.Keys[0][0]) / kv.HeadDim
}

// Reset clears the cache.
func (kv *KVCache) Reset() {
	for l := 0; l < kv.Layers; l++ {
		for h := 0; h < kv.Heads; h++ {
			kv.Keys[l][h] = kv.Keys[l][h][:0]
			kv.Values[l][h] = kv.Values[l][h][:0]
		}
	}
}
