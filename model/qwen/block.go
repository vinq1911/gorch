//go:build darwin

package qwen

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// Block is the standard Qwen pre-norm decoder block (mythos pattern):
//
//	h = h + Attn(RMSNorm(h), startPos)
//	h = h + FFN(RMSNorm(h))
//
// Attention is GQA with explicit head_dim, optional per-head QK-norm,
// and RoPE on Q/K. The FFN is a dense SwiGLU MLP — exactly nn.Expert
// (gate/up/down + SwiGLU) without the MoE router (plan 0008 §1.2).
type Block struct {
	NormAttn *nn.RMSNorm
	Attn     *nn.GQA
	NormFFN  *nn.RMSNorm
	FFN      *nn.Expert
}

// NewBlock builds one decoder block sized to cfg, sharing the model's
// precomputed RoPE table.
func NewBlock(cfg Config, rope *nn.RoPE) *Block {
	attn := nn.NewGQAConfig(nn.GQAConfig{
		Dim:     cfg.HiddenSize,
		NumQ:    cfg.NumQueryHeads,
		NumKV:   cfg.NumKVHeads,
		HeadDim: cfg.HeadDim,
		Bias:    cfg.AttnBias,
	})
	attn.RoPE = rope
	attn.Causal = true
	if cfg.UseQKNorm {
		attn.QNorm = nn.NewRMSNorm(cfg.HeadDim)
		attn.QNorm.Eps = cfg.RMSNormEps
		attn.KNorm = nn.NewRMSNorm(cfg.HeadDim)
		attn.KNorm.Eps = cfg.RMSNormEps
	}

	normAttn := nn.NewRMSNorm(cfg.HiddenSize)
	normAttn.Eps = cfg.RMSNormEps
	normFFN := nn.NewRMSNorm(cfg.HiddenSize)
	normFFN.Eps = cfg.RMSNormEps

	return &Block{
		NormAttn: normAttn,
		Attn:     attn,
		NormFFN:  normFFN,
		FFN:      nn.NewExpert(cfg.HiddenSize, cfg.IntermediateSize),
	}
}

// Forward runs the block full-sequence (no cache). x is (seq, hidden);
// startPos is the absolute position of x[0] for RoPE.
func (b *Block) Forward(x *g.Tensor, startPos int) *g.Tensor {
	h1 := b.NormAttn.Forward(x)
	h1 = b.Attn.Forward(h1, startPos)
	h1 = g.Add(x, h1)

	h2 := b.NormFFN.Forward(h1)
	h2 = b.FFN.Forward(h2)
	return g.Add(h1, h2)
}

// ForwardCached runs the block against a KV cache (decode/prefill
// path). posOffset must equal cache.Len() before the call.
func (b *Block) ForwardCached(x *g.Tensor, cache *nn.KVCache, layerIdx, posOffset int) *g.Tensor {
	h1 := b.NormAttn.Forward(x)
	h1 = b.Attn.ForwardCached(h1, cache, layerIdx, posOffset)
	h1 = g.Add(x, h1)

	h2 := b.NormFFN.Forward(h1)
	h2 = b.FFN.Forward(h2)
	return g.Add(h1, h2)
}

// Parameters returns every learnable tensor in the block.
func (b *Block) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, b.NormAttn.Parameters()...)
	params = append(params, b.Attn.Parameters()...)
	params = append(params, b.NormFFN.Parameters()...)
	params = append(params, b.FFN.Parameters()...)
	return params
}
