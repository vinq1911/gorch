//go:build darwin

package mythos

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// TransformerBlock is the standard pre-norm OpenMythos block:
//
//	h = h + Attention(RMSNorm(h), startPos)
//	h = h + MoE(RMSNorm(h))
//
// Uses GQA today (UseMLA path is plan 0001's open MLA-completes-autograd
// item; pinned in tests below). Both sublayers add residual connections
// in the standard pre-norm pattern. RoPE is composed inside the
// attention module on Q and K.
//
// One block has:
//   - 2 RMSNorm layers (gamma per layer)
//   - 1 GQA attention with Wq, Wk, Wv, Wo Linear projections
//   - 1 MoE FFN with router + N experts (each 3 Linear projections)
//
// Plan 0001 Phase 2 deliverable.
type TransformerBlock struct {
	NormAttn *nn.RMSNorm
	Attn     *nn.GQA
	NormFFN  *nn.RMSNorm
	FFN      *nn.MoE
}

// NewTransformerBlock builds a block sized to cfg.
func NewTransformerBlock(cfg Config, rope *nn.RoPE) *TransformerBlock {
	if cfg.UseMLA {
		// MLA forward exists (nn/mla.go) but its autograd is partially
		// broken — Slice/Concat helpers strip the chain. The fix is the
		// "MLA full autograd" item in handout.md. Until then, refuse to
		// build an MLA block silently; force callers to opt out
		// explicitly.
		panic("model/mythos: UseMLA=true requires the autograd-aware Slice/Concat ops (not yet shipped); use GQA for v1")
	}
	attn := nn.NewGQA(cfg.Dim, cfg.NumHeads, cfg.NumKVHeads)
	attn.RoPE = rope
	attn.Causal = true
	return &TransformerBlock{
		NormAttn: nn.NewRMSNorm(cfg.Dim),
		Attn:     attn,
		NormFFN:  nn.NewRMSNorm(cfg.Dim),
		FFN:      nn.NewMoE(cfg.Dim, cfg.ExpertDim, cfg.NumExperts, cfg.NumExpertsPerToken),
	}
}

// Forward runs the block with residuals. x is (seq, dim); startPos is
// the absolute position of x[0] for RoPE. Output: (seq, dim).
func (b *TransformerBlock) Forward(x *g.Tensor, startPos int) *g.Tensor {
	// Attention sublayer with residual.
	h1 := b.NormAttn.Forward(x)
	h1 = b.Attn.Forward(h1, startPos)
	h1 = g.Add(x, h1)

	// FFN sublayer with residual.
	h2 := b.NormFFN.Forward(h1)
	h2 = b.FFN.Forward(h2)
	return g.Add(h1, h2)
}

// Parameters returns every learnable tensor in the block.
func (b *TransformerBlock) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, b.NormAttn.Parameters()...)
	params = append(params, b.Attn.Parameters()...)
	params = append(params, b.NormFFN.Parameters()...)
	params = append(params, b.FFN.Parameters()...)
	return params
}
