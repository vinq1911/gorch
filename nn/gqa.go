//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// GQA — Grouped-Query Attention (Ainslie et al. 2023).
//
// Standard MHA has numQueryHeads = numKVHeads. GQA decouples them:
// fewer KV heads each shared by a group of query heads. At inference
// time KV-cache memory is proportional to numKVHeads, so dropping
// from 32 KV heads (Llama 2 7B's MHA) to 8 KV heads (Llama 3 8B's
// GQA) is a 4× KV-cache reduction.
//
// Used by Llama 3, Mistral, OpenMythos. The query/KV heads must
// satisfy numQueryHeads % numKVHeads == 0; the ratio is "queries per
// group."
//
// Composition: Linear projections + RepeatInterleave (to broadcast
// each KV head across its query group) + the existing batched MHA
// math. Optional RoPE applied to Q and K before the score matmul.
type GQA struct {
	Wq            *Linear
	Wk            *Linear
	Wv            *Linear
	Wo            *Linear
	NumQueryHeads int
	NumKVHeads    int
	HeadDim       int
	Dim           int
	Causal        bool
	RoPE          *RoPE // optional; nil = no positional encoding
}

// NewGQA builds a Grouped-Query Attention module.
//
//	dim           — hidden size
//	numQueryHeads — Q heads (e.g. 32 for Llama 3 8B)
//	numKVHeads    — K/V heads (e.g. 8 for Llama 3 8B); each shared by
//	                numQueryHeads/numKVHeads queries
//
// headDim = dim / numQueryHeads (standard convention).
func NewGQA(dim, numQueryHeads, numKVHeads int) *GQA {
	if dim%numQueryHeads != 0 {
		panic("gorch/nn: GQA dim must be divisible by numQueryHeads")
	}
	if numQueryHeads%numKVHeads != 0 {
		panic("gorch/nn: GQA numQueryHeads must be divisible by numKVHeads")
	}
	headDim := dim / numQueryHeads
	kvDim := numKVHeads * headDim
	return &GQA{
		Wq:            NewLinear(dim, dim),
		Wk:            NewLinear(dim, kvDim),
		Wv:            NewLinear(dim, kvDim),
		Wo:            NewLinear(dim, dim),
		NumQueryHeads: numQueryHeads,
		NumKVHeads:    numKVHeads,
		HeadDim:       headDim,
		Dim:           dim,
		Causal:        true,
	}
}

// Forward runs GQA on input x of shape (seq, dim). startPos is the
// absolute position of x[0] for RoPE / causal masking; pass 0 for the
// no-cache full-sequence forward.
//
// Inference-only path — does not build the autograd graph. Mirrors
// the existing MultiHeadAttention.Forward batched-CPU pattern but
// with the QHeads → KVHeads expansion folded in via RepeatInterleave.
func (gqa *GQA) Forward(x *g.Tensor, startPos int) *g.Tensor {
	seqLen := x.Shape()[0]
	dim := gqa.Dim
	headDim := gqa.HeadDim
	numQ := gqa.NumQueryHeads
	numKV := gqa.NumKVHeads
	groupSize := numQ / numKV

	// Project: (seq, dim) → Q (seq, dim), K/V (seq, kvDim).
	q := gqa.Wq.Forward(x)
	k := gqa.Wk.Forward(x)
	v := gqa.Wv.Forward(x)

	// Reshape to (seq, numHeads, headDim) view, then permute to
	// (numHeads, seq, headDim) for batched-matmul-friendly layout.
	qH := g.Permute(q.Reshape(seqLen, numQ, headDim), []int{1, 0, 2})
	kH := g.Permute(k.Reshape(seqLen, numKV, headDim), []int{1, 0, 2})
	vH := g.Permute(v.Reshape(seqLen, numKV, headDim), []int{1, 0, 2})

	// Apply RoPE to Q and K (NOT V).
	if gqa.RoPE != nil {
		qH = gqa.RoPE.Apply(qH, startPos)
		kH = gqa.RoPE.Apply(kH, startPos)
	}

	// GQA expansion: each KV head shared across `groupSize` query heads.
	// (numKV, seq, headDim) → (numQ, seq, headDim) via repeat each KV
	// head `groupSize` times along the head dimension.
	if groupSize > 1 {
		// RepeatInterleave operates on second-to-last dim. Reshape KV
		// to (numKV, seq*headDim) → repeat → (numQ, seq*headDim) →
		// back to (numQ, seq, headDim).
		kFlat := kH.Reshape(numKV, seqLen*headDim)
		vFlat := vH.Reshape(numKV, seqLen*headDim)
		// RepeatInterleave needs rank ≥ 2 with the repeated axis at
		// position dim-2; here that's the leading dim of (numKV,
		// inner). Promote to (numKV, 1, inner) so the repeat dim is
		// position 0 (= ndim-2 with inner-trailing convention) and we
		// get (numKV * groupSize, 1, inner) back.
		kRepeat := g.RepeatInterleave(kFlat.Reshape(numKV, 1, seqLen*headDim), groupSize)
		vRepeat := g.RepeatInterleave(vFlat.Reshape(numKV, 1, seqLen*headDim), groupSize)
		kH = kRepeat.Reshape(numQ, seqLen, headDim)
		vH = vRepeat.Reshape(numQ, seqLen, headDim)
	}

	// Batched scores: (numQ, seq, headDim) × (numQ, headDim, seq) → (numQ, seq, seq)
	scores := g.BatchedMatMulTransB(qH, kH, numQ, seqLen, seqLen, headDim)
	invScale := float32(1.0 / math.Sqrt(float64(headDim)))
	scoresData := scores.Data()
	for i := range scoresData {
		scoresData[i] *= invScale
	}

	// Causal mask + softmax per (head, row).
	for h := 0; h < numQ; h++ {
		block := scoresData[h*seqLen*seqLen : (h+1)*seqLen*seqLen]
		if gqa.Causal {
			for i := 0; i < seqLen; i++ {
				for j := i + 1; j < seqLen; j++ {
					block[i*seqLen+j] = -1e9
				}
			}
		}
		softmaxInPlace(block, seqLen)
	}

	// attn @ V: (numQ, seq, seq) × (numQ, seq, headDim) → (numQ, seq, headDim)
	attnOut := g.BatchedMatMul(scores, vH, numQ, seqLen, headDim, seqLen)

	// Permute back: (numQ, seq, headDim) → (seq, numQ, headDim) → (seq, dim).
	concat := g.Permute(attnOut, []int{1, 0, 2}).Reshape(seqLen, dim)

	return gqa.Wo.Forward(concat)
}

// Parameters returns the four projection matrices.
func (gqa *GQA) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, gqa.Wq.Parameters()...)
	params = append(params, gqa.Wk.Parameters()...)
	params = append(params, gqa.Wv.Parameters()...)
	params = append(params, gqa.Wo.Parameters()...)
	return params
}
