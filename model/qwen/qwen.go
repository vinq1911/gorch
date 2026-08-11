//go:build darwin

package qwen

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// Model is the top-level Qwen causal LM:
//
//	tokens → Embedding → NumLayers × Block → RMSNorm → tied LM head
//
// The LM head is tied to the input embedding (HF tie_word_embeddings):
// logits = h @ Embed.Weightᵀ via the inference fast path MatMulTransB,
// re-emitted through MatMul(h, Transpose2D(W)) when autograd is active
// — the exact mythos.go pattern.
type Model struct {
	Cfg    Config
	Embed  *nn.Embedding
	RoPE   *nn.RoPE
	Blocks []*Block
	Norm   *nn.RMSNorm
}

// New builds a randomly initialised model from cfg. The shared RoPE
// table is built once and threaded into every block's attention.
func New(cfg Config) *Model {
	if cfg.VocabSize <= 0 || cfg.NumLayers <= 0 {
		panic("qwen: VocabSize and NumLayers must be > 0")
	}
	rope := nn.NewRoPE(cfg.HeadDim, cfg.MaxSeq, cfg.RopeTheta, nn.RopeLlama)
	blocks := make([]*Block, cfg.NumLayers)
	for i := range blocks {
		blocks[i] = NewBlock(cfg, rope)
	}
	norm := nn.NewRMSNorm(cfg.HiddenSize)
	norm.Eps = cfg.RMSNormEps
	return &Model{
		Cfg:    cfg,
		Embed:  nn.NewEmbedding(cfg.VocabSize, cfg.HiddenSize),
		RoPE:   rope,
		Blocks: blocks,
		Norm:   norm,
	}
}

// Forward runs the full model without a cache. Returns logits of
// shape (seq, VocabSize). startPos is the absolute position of
// tokens[0]; pass 0 for a full-sequence forward.
func (m *Model) Forward(tokens []int, startPos int) *g.Tensor {
	if len(tokens) == 0 {
		panic("qwen: empty token slice")
	}
	h := m.Embed.Forward(tokens) // (seq, hidden)
	for _, blk := range m.Blocks {
		h = blk.Forward(h, startPos)
	}
	h = m.Norm.Forward(h)
	return m.lmHead(h)
}

// ForwardCached runs the model against a KV cache and returns logits
// for the LAST position only, shape (1, VocabSize) — the only row
// generation consumes, and at a 151,936 vocab the full-sequence logits
// of a long prefill would be hundreds of MB of dead weight. Feed the
// whole prompt on the first call (prefill, staircase-masked) and one
// token per subsequent call. Inference-only.
func (m *Model) ForwardCached(tokens []int, cache *nn.KVCache) *g.Tensor {
	if len(tokens) == 0 {
		panic("qwen: empty token slice")
	}
	var logits *g.Tensor
	g.NoGrad(func() {
		posOffset := cache.Len()
		h := m.Embed.Forward(tokens)
		for i, blk := range m.Blocks {
			h = blk.ForwardCached(h, cache, i, posOffset)
		}
		seq := h.Shape()[0]
		dim := h.Shape()[1]
		last := g.NewTensor(h.Data()[(seq-1)*dim:seq*dim], 1, dim)
		last = m.Norm.Forward(last)
		logits = g.MatMulTransB(last, m.Embed.Weight)
	})
	return logits
}

// NewCache allocates a KV cache sized for this model
// (kvDim = numKVHeads·headDim per layer).
func (m *Model) NewCache() *nn.KVCache {
	return nn.NewKVCache(m.Cfg.NumLayers, m.Cfg.KVDim())
}

// lmHead applies the tied LM head with the GradEnabled re-emission
// pattern from mythos.go.
func (m *Model) lmHead(h *g.Tensor) *g.Tensor {
	logits := g.MatMulTransB(h, m.Embed.Weight)
	if g.GradEnabled() && (h.RequiresGrad() || m.Embed.Weight.RequiresGrad()) {
		wT := g.Transpose2D(m.Embed.Weight)
		logits = g.MatMul(h, wT)
	}
	return logits
}

// Parameters returns every learnable tensor in the model.
func (m *Model) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, m.Embed.Parameters()...)
	for _, blk := range m.Blocks {
		params = append(params, blk.Parameters()...)
	}
	params = append(params, m.Norm.Parameters()...)
	return params
}
