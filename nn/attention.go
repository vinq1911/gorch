//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
)

// MultiHeadAttention implements multi-head self-attention.
// All operations use 2D tensors (flattened batch*seq) with explicit reshaping.
type MultiHeadAttention struct {
	Wq       *Linear
	Wk       *Linear
	Wv       *Linear
	Wo       *Linear
	NumHeads int
	HeadDim  int
	Dim      int
}

// NewMultiHeadAttention creates a multi-head attention module.
// dim must be divisible by numHeads.
func NewMultiHeadAttention(dim, numHeads int) *MultiHeadAttention {
	if dim%numHeads != 0 {
		panic("gorch: dim must be divisible by numHeads")
	}
	headDim := dim / numHeads
	return &MultiHeadAttention{
		Wq:       NewLinear(dim, dim),
		Wk:       NewLinear(dim, dim),
		Wv:       NewLinear(dim, dim),
		Wo:       NewLinear(dim, dim),
		NumHeads: numHeads,
		HeadDim:  headDim,
		Dim:      dim,
	}
}

// Forward computes multi-head self-attention.
// x: (seq, dim) — single sequence, no batch dim.
// seqLen: sequence length.
// Returns: (seq, dim).
func (mha *MultiHeadAttention) Forward(x *g.Tensor, seqLen int) *g.Tensor {
	dim := mha.Dim
	numHeads := mha.NumHeads
	headDim := mha.HeadDim

	// Project: (seq, dim) → (seq, dim) for Q, K, V
	q := mha.Wq.Forward(x) // (seq, dim)
	k := mha.Wk.Forward(x)
	v := mha.Wv.Forward(x)

	// Process each head separately (loop over heads — correct, not optimally batched)
	// Split dim into numHeads chunks of headDim
	headOutputs := make([]*g.Tensor, numHeads)

	for h := 0; h < numHeads; h++ {
		// Extract head h: columns [h*headDim : (h+1)*headDim] from Q, K, V
		qh := extractHead(q, seqLen, dim, h, headDim) // (seq, headDim)
		kh := extractHead(k, seqLen, dim, h, headDim)
		vh := extractHead(v, seqLen, dim, h, headDim)

		// Attention scores: (seq, headDim) @ (headDim, seq) = (seq, seq)
		kT := g.Transpose2D(kh)
		scores := g.ScaledMatMul(qh, kT, float32(headDim))

		// Causal mask
		mask := g.CausalMask(seqLen)
		scores = g.MaskFill(scores, mask, -1e9)

		// Softmax over last dim
		attnWeights := g.Softmax(scores) // (seq, seq)

		// Weighted values: (seq, seq) @ (seq, headDim) = (seq, headDim)
		headOut := g.MatMul(attnWeights, vh)
		headOutputs[h] = headOut
	}

	// Concatenate heads: (seq, numHeads*headDim) = (seq, dim)
	concat := concatHeads(headOutputs, seqLen, numHeads, headDim)

	// Output projection
	return mha.Wo.Forward(concat)
}

// extractHead extracts one attention head's columns from a (seq, dim) tensor.
// Returns (seq, headDim).
func extractHead(x *g.Tensor, seq, dim, headIdx, headDim int) *g.Tensor {
	out := g.Zeros(seq, headDim)
	xData := x.Data()
	outData := out.Data()
	offset := headIdx * headDim
	for i := 0; i < seq; i++ {
		copy(outData[i*headDim:(i+1)*headDim], xData[i*dim+offset:i*dim+offset+headDim])
	}
	if x.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("ExtractHead", []*g.Tensor{x}, func(grad *g.Tensor) []*g.Tensor {
			dx := g.Zeros(seq, dim)
			dxData := dx.Data()
			gData := grad.Data()
			for i := 0; i < seq; i++ {
				for j := 0; j < headDim; j++ {
					dxData[i*dim+offset+j] += gData[i*headDim+j]
				}
			}
			return []*g.Tensor{dx}
		})
	}
	return out
}

// concatHeads concatenates head outputs back into (seq, dim).
func concatHeads(heads []*g.Tensor, seq, numHeads, headDim int) *g.Tensor {
	dim := numHeads * headDim
	out := g.Zeros(seq, dim)
	outData := out.Data()

	for h, head := range heads {
		hData := head.Data()
		offset := h * headDim
		for i := 0; i < seq; i++ {
			copy(outData[i*dim+offset:i*dim+offset+headDim], hData[i*headDim:(i+1)*headDim])
		}
	}

	// Autograd: scatter gradients back to each head
	anyGrad := false
	for _, h := range heads {
		if h.RequiresGrad() {
			anyGrad = true
			break
		}
	}
	if anyGrad {
		out.SetRequiresGrad(true)
		inputs := make([]*g.Tensor, len(heads))
		copy(inputs, heads)
		out.SetGradFn("ConcatHeads", inputs, func(grad *g.Tensor) []*g.Tensor {
			gData := grad.Data()
			grads := make([]*g.Tensor, numHeads)
			for h := 0; h < numHeads; h++ {
				dh := g.Zeros(seq, headDim)
				dhData := dh.Data()
				offset := h * headDim
				for i := 0; i < seq; i++ {
					copy(dhData[i*headDim:(i+1)*headDim], gData[i*dim+offset:i*dim+offset+headDim])
				}
				grads[h] = dh
			}
			return grads
		})
	}
	return out
}

func (mha *MultiHeadAttention) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, mha.Wq.Parameters()...)
	params = append(params, mha.Wk.Parameters()...)
	params = append(params, mha.Wv.Parameters()...)
	params = append(params, mha.Wo.Parameters()...)
	return params
}
