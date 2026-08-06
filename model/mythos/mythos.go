//go:build darwin

package mythos

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// Mythos is the top-level recurrent-depth transformer model.
//
// Forward path:
//
//	tokens          ─────► Embedding ─────► h0
//	h0      ─────► Prelude (PreludeLayers blocks) ─────► h_in
//	h_in    ─────► Recurrent (MaxLoopIters iterations of one shared
//	                          block, each followed by LTI mixing) ─────► h_out
//	h_out   ─────► Coda (CodaLayers blocks) ─────► h_final
//	h_final ─────► RMSNorm ─────► LMHead Linear ─────► logits (vocab)
//
// v1 simplifications relative to the OpenMythos paper, captured in
// plan 0001 Phase 2 / 5:
//
//   - LTI: per-channel diagonal damping (sigmoid-of-logit), not the
//     full matrix-A parameterisation.
//   - Recurrent block weights are SHARED across the MaxLoopIters
//     iterations (depth-wise LoRA adapters are deferred — same as the
//     OpenMythos repo's "USE_LORA = False" branch).
//   - No ACT halting: MaxLoopIters is a fixed Config field, the loop
//     runs that many times. ACT's halting probabilities + cumulative
//     surplus are deferred until v1 demonstrates the recurrent-depth
//     benefit on TinyStories.
//
// LM head is tied to the input embedding (HF-style; saves vocab*dim
// parameters and is what GPT-2/Llama do today).
type Mythos struct {
	Cfg     Config
	Embed   *nn.Embedding
	RoPE    *nn.RoPE
	Prelude []*TransformerBlock
	Recur   *TransformerBlock
	LTI     *LTIInjection
	Coda    []*TransformerBlock
	Norm    *nn.RMSNorm
}

// New builds a Mythos model from cfg. The shared RoPE table is built
// once and threaded into every attention sublayer so the same precomputed
// cos/sin pairs are used end-to-end.
func New(cfg Config) *Mythos {
	if cfg.VocabSize <= 0 {
		panic("mythos: VocabSize must be > 0")
	}
	if cfg.PreludeLayers < 1 || cfg.CodaLayers < 1 {
		panic("mythos: PreludeLayers and CodaLayers must be ≥ 1")
	}
	if cfg.MaxLoopIters < 1 {
		panic("mythos: MaxLoopIters must be ≥ 1")
	}

	rope := nn.NewRoPE(cfg.HeadDim(), cfg.MaxSeqLen, cfg.RopeBaseFreq, nn.RopeLlama)

	prelude := make([]*TransformerBlock, cfg.PreludeLayers)
	for i := range prelude {
		prelude[i] = NewTransformerBlock(cfg, rope)
	}
	coda := make([]*TransformerBlock, cfg.CodaLayers)
	for i := range coda {
		coda[i] = NewTransformerBlock(cfg, rope)
	}
	return &Mythos{
		Cfg:     cfg,
		Embed:   nn.NewEmbedding(cfg.VocabSize, cfg.Dim),
		RoPE:    rope,
		Prelude: prelude,
		Recur:   NewTransformerBlock(cfg, rope),
		LTI:     NewLTIInjection(cfg.Dim, cfg.LTIDampInit),
		Coda:    coda,
		Norm:    nn.NewRMSNorm(cfg.Dim),
	}
}

// Forward runs the full model on a flat token-id slice. Returns logits
// of shape (seq, VocabSize). startPos is the absolute position of
// tokens[0] (relevant for KV-cached decoding; pass 0 for full-sequence
// training forward).
//
// loopIters can override the config's MaxLoopIters at inference time
// (the recurrent-depth-ablation core test). Pass -1 to use cfg.MaxLoopIters.
func (m *Mythos) Forward(tokens []int, startPos, loopIters int) *g.Tensor {
	if loopIters < 0 {
		loopIters = m.Cfg.MaxLoopIters
	}
	if len(tokens) == 0 {
		panic("mythos: empty token slice")
	}

	h := m.Embed.Forward(tokens) // (seq, dim)

	// Prelude.
	for _, blk := range m.Prelude {
		h = blk.Forward(h, startPos)
	}

	// Recurrent loop with LTI injection.
	for t := 0; t < loopIters; t++ {
		hBlock := m.Recur.Forward(h, startPos)
		h = m.LTI.Apply(h, hBlock)
	}

	// Coda.
	for _, blk := range m.Coda {
		h = blk.Forward(h, startPos)
	}

	// Final norm + tied LM head: logits = h @ Embed.Weight^T.
	h = m.Norm.Forward(h)
	logits := g.MatMulTransB(h, m.Embed.Weight)

	// MatMulTransB doesn't carry autograd today (used as inference fast
	// path elsewhere). Re-emit the autograd through MatMul + Transpose
	// when training. The CPU path is the only one wired for training
	// today; bf16 / Metal training go through the same f32 graph here.
	if g.GradEnabled() && (h.RequiresGrad() || m.Embed.Weight.RequiresGrad()) {
		// MatMul(h, W^T) — W is (vocab, dim), W^T is (dim, vocab).
		wT := g.Transpose2D(m.Embed.Weight)
		logits = g.MatMul(h, wT)
	}
	return logits
}

// Parameters returns every learnable tensor in the model.
func (m *Mythos) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, m.Embed.Parameters()...)
	for _, blk := range m.Prelude {
		params = append(params, blk.Parameters()...)
	}
	params = append(params, m.Recur.Parameters()...)
	params = append(params, m.LTI.Parameters()...)
	for _, blk := range m.Coda {
		params = append(params, blk.Parameters()...)
	}
	params = append(params, m.Norm.Parameters()...)
	return params
}
