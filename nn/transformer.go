//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
)

// TransformerBlock implements a single pre-norm transformer block:
//   x = x + Attn(LayerNorm(x))
//   x = x + FFN(LayerNorm(x))
type TransformerBlock struct {
	Attn  *MultiHeadAttention
	FFN1  *Linear    // dim → 4*dim
	FFN2  *Linear    // 4*dim → dim
	Norm1 *LayerNorm
	Norm2 *LayerNorm
}

// NewTransformerBlock creates a transformer block with pre-norm architecture.
func NewTransformerBlock(dim, numHeads int) *TransformerBlock {
	return &TransformerBlock{
		Attn:  NewMultiHeadAttention(dim, numHeads),
		FFN1:  NewLinear(dim, 4*dim),
		FFN2:  NewLinear(4*dim, dim),
		Norm1: NewLayerNorm(dim),
		Norm2: NewLayerNorm(dim),
	}
}

// Forward runs the transformer block.
// x: (seq, dim). Returns: (seq, dim).
func (tb *TransformerBlock) Forward(x *g.Tensor, seqLen int) *g.Tensor {
	// Self-attention with residual
	normed := tb.Norm1.Forward(x)
	attnOut := tb.Attn.Forward(normed, seqLen)
	x = g.Add(x, attnOut) // residual

	// FFN with residual
	normed2 := tb.Norm2.Forward(x)
	ffnOut := tb.FFN1.Forward(normed2)
	ffnOut = g.ReLU(ffnOut) // GELU would be better, but ReLU works
	ffnOut = tb.FFN2.Forward(ffnOut)
	x = g.Add(x, ffnOut) // residual

	return x
}

func (tb *TransformerBlock) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, tb.Attn.Parameters()...)
	params = append(params, tb.FFN1.Parameters()...)
	params = append(params, tb.FFN2.Parameters()...)
	params = append(params, tb.Norm1.Parameters()...)
	params = append(params, tb.Norm2.Parameters()...)
	return params
}
