//go:build darwin

package nn

// KVCache stores the per-layer key/value tensors produced during
// autoregressive decoding so subsequent steps don't re-project the
// entire prefix.
//
// Storage is per-layer flat float32 of length seqLen*dim. New tokens
// are appended in order; the cache grows by one token per step (or
// by len(prompt) on the prefill call). Computation happens on CPU,
// which on Apple Silicon shares physical memory with the GPU so
// there is no copy cost across the boundary.
type KVCache struct {
	Keys   [][]float32 // [layer] → flat seqLen*Dim
	Values [][]float32 // [layer] → flat seqLen*Dim
	Layers int
	Dim    int
}

// NewKVCache allocates an empty cache for a model with numLayers
// transformer blocks of hidden size dim.
func NewKVCache(numLayers, dim int) *KVCache {
	keys := make([][]float32, numLayers)
	values := make([][]float32, numLayers)
	for i := range keys {
		keys[i] = make([]float32, 0)
		values[i] = make([]float32, 0)
	}
	return &KVCache{
		Keys:   keys,
		Values: values,
		Layers: numLayers,
		Dim:    dim,
	}
}

// Append adds new key/value vectors for one or more tokens at the
// given layer. k and v must have length newTokens*Dim.
func (c *KVCache) Append(layer int, k, v []float32) {
	c.Keys[layer] = append(c.Keys[layer], k...)
	c.Values[layer] = append(c.Values[layer], v...)
}

// Len returns the number of tokens currently cached. Always equal
// across layers since they're appended together.
func (c *KVCache) Len() int {
	if len(c.Keys) == 0 || c.Dim == 0 {
		return 0
	}
	return len(c.Keys[0]) / c.Dim
}

// Reset clears all cached tokens but keeps allocated capacity.
func (c *KVCache) Reset() {
	for i := range c.Keys {
		c.Keys[i] = c.Keys[i][:0]
		c.Values[i] = c.Values[i][:0]
	}
}
