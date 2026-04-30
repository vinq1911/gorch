//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// MLA — Multi-head Latent Attention (DeepSeek-V2, Liu et al. 2024).
//
// MLA's distinguishing feature: KV is compressed through a shared
// low-rank "latent" projection of size kvLoraRank, much smaller than
// the per-head KV that GQA caches. At decode time the KV-cache stores
// the latent (per-token kvLoraRank) plus the shared rotated key
// (per-token ropeHeadDim), not the full per-head K/V. For DeepSeek-V2,
// this drops the KV-cache from ~32 GB to ~4 GB at typical context
// length — a key reason DeepSeek-V2 fits where Llama 3 70B doesn't.
//
// Per-head dimensions are split into a "no-positional-encoding"
// (nope) part of size nopeHeadDim and a RoPE-rotated part of size
// ropeHeadDim. Total headDim = nopeHeadDim + ropeHeadDim.
//
// Plan 0001 Phase 1 item 10. The heaviest single piece. Plan flags
// it explicitly: "the most thorough numerical-agreement test suite
// of any module" — this implementation aims for parity with the
// DeepSeek-V2 reference math.
//
// Forward at v1: full-sequence forward with optional RoPE on the
// rope part of Q and the shared K_rope. KV-cache integration is
// scope-deferred (Plan 0001 Phase 2 wires this into mythos blocks
// where the cache lifecycle is owned by the recurrent loop).
type MLA struct {
	// Q-side projections.
	Wq         *Linear // dim → numHeads * headDim (nope + rope concatenated per head)
	// KV LoRA compression.
	WkvDown    *Linear // dim → kvLoraRank + ropeHeadDim (latent + shared rope key)
	WkvUp      *Linear // kvLoraRank → numHeads * (nopeHeadDim + valueHeadDim)
	// Output.
	Wo         *Linear // numHeads * valueHeadDim → dim

	NumHeads     int
	NopeHeadDim  int
	RopeHeadDim  int
	ValueHeadDim int
	KVLoraRank   int
	Dim          int
	Causal       bool
	RoPE         *RoPE // optional, applied to Q rope part and shared K rope key
}

// NewMLA builds a Multi-head Latent Attention module.
//
//	dim          — hidden size
//	numHeads     — number of attention heads
//	nopeHeadDim  — per-head dim that does NOT receive RoPE
//	ropeHeadDim  — per-head dim that DOES receive RoPE
//	               (note: ropeHeadDim must be even for RoPE)
//	valueHeadDim — per-head dim of V (often equal to nopeHeadDim)
//	kvLoraRank   — latent compression dim (e.g. 512 for DeepSeek-V2 7B)
//
// DeepSeek-V2 7B example: dim=4096, numHeads=32, nopeHeadDim=128,
// ropeHeadDim=64, valueHeadDim=128, kvLoraRank=512.
func NewMLA(dim, numHeads, nopeHeadDim, ropeHeadDim, valueHeadDim, kvLoraRank int) *MLA {
	if ropeHeadDim%2 != 0 {
		panic("gorch/nn: MLA ropeHeadDim must be even")
	}
	headDim := nopeHeadDim + ropeHeadDim
	return &MLA{
		Wq:           NewLinear(dim, numHeads*headDim),
		WkvDown:      NewLinear(dim, kvLoraRank+ropeHeadDim),
		WkvUp:        NewLinear(kvLoraRank, numHeads*(nopeHeadDim+valueHeadDim)),
		Wo:           NewLinear(numHeads*valueHeadDim, dim),
		NumHeads:     numHeads,
		NopeHeadDim:  nopeHeadDim,
		RopeHeadDim:  ropeHeadDim,
		ValueHeadDim: valueHeadDim,
		KVLoraRank:   kvLoraRank,
		Dim:          dim,
		Causal:       true,
	}
}

// Forward runs MLA on input x of shape (seq, dim). startPos is the
// absolute position of x[0] for RoPE; pass 0 for full-sequence
// forward without a cache.
//
// Inference-only — does not build the autograd graph. The math
// follows DeepSeek-V2's MLA paper:
//
//	1. Q = Wq(x)  →  reshape (seq, numHeads, nopeDim+ropeDim)
//	   Q_nope = Q[..., :nopeDim], Q_rope = Q[..., nopeDim:] then
//	   apply RoPE to Q_rope.
//
//	2. KV down = WkvDown(x)  →  (seq, kvLoraRank + ropeDim)
//	   c_kv  = KV_down[..., :kvLoraRank]   (latent KV)
//	   k_rope = KV_down[..., kvLoraRank:]  (shared rope key)
//	   apply RoPE to k_rope.
//
//	3. KV up = WkvUp(c_kv)  →  (seq, numHeads * (nopeDim + valueDim))
//	   reshape to (seq, numHeads, nopeDim + valueDim)
//	   K_nope = KV_up[..., :nopeDim],  V = KV_up[..., nopeDim:]
//
//	4. K_full per head h = concat(K_nope[h], k_rope_broadcast)
//	   shape (seq, numHeads, nopeDim + ropeDim)
//
//	5. scores = Q · K_full^T / sqrt(nopeDim + ropeDim) per head
//	   causal mask + softmax
//
//	6. out = scores · V  →  (seq, numHeads, valueDim)
//	   concat heads, project Wo.
func (m *MLA) Forward(x *g.Tensor, startPos int) *g.Tensor {
	seqLen := x.Shape()[0]
	H := m.NumHeads
	nopeDim := m.NopeHeadDim
	ropeDim := m.RopeHeadDim
	valDim := m.ValueHeadDim
	headDim := nopeDim + ropeDim

	// 1. Q projection + reshape + permute.
	q := m.Wq.Forward(x).Reshape(seqLen, H, headDim)
	qPerm := g.Permute(q, []int{1, 0, 2}) // (H, seq, headDim)

	// Split Q into Q_nope and Q_rope along the last dim.
	qNope := mlaSliceLast(qPerm, 0, nopeDim, []int{H, seqLen, nopeDim})
	qRope := mlaSliceLast(qPerm, nopeDim, headDim, []int{H, seqLen, ropeDim})
	if m.RoPE != nil {
		qRope = m.RoPE.Apply(qRope, startPos)
	}

	// 2. KV-down projection: latent + shared rope key.
	kvDown := m.WkvDown.Forward(x) // (seq, kvLoraRank + ropeDim)
	cKV := mlaSliceLast(kvDown, 0, m.KVLoraRank, []int{seqLen, m.KVLoraRank})
	kRope := mlaSliceLast(kvDown, m.KVLoraRank, m.KVLoraRank+ropeDim, []int{seqLen, ropeDim})
	// k_rope is shared across all heads. Apply RoPE on it once
	// (treat as 1 head for the RoPE call).
	if m.RoPE != nil {
		kRope3 := kRope.Reshape(1, seqLen, ropeDim)
		kRope3 = m.RoPE.Apply(kRope3, startPos)
		kRope = kRope3.Reshape(seqLen, ropeDim)
	}

	// 3. KV-up: latent → numHeads * (nopeDim + valueDim).
	kvUp := m.WkvUp.Forward(cKV).Reshape(seqLen, H, nopeDim+valDim)
	kvUpPerm := g.Permute(kvUp, []int{1, 0, 2}) // (H, seq, nopeDim+valueDim)
	kNope := mlaSliceLast(kvUpPerm, 0, nopeDim, []int{H, seqLen, nopeDim})
	v := mlaSliceLast(kvUpPerm, nopeDim, nopeDim+valDim, []int{H, seqLen, valDim})

	// 4. Build full K per head: concat(K_nope, k_rope_broadcast).
	// k_rope is (seq, ropeDim); broadcast to all H heads.
	kFull := mlaConcatRopeKey(kNope, kRope, H, seqLen, nopeDim, ropeDim)

	// 5. Q_full = concat(Q_nope, Q_rope). Then scores = Q · K^T / sqrt(headDim).
	qFull := mlaConcatLast(qNope, qRope, H, seqLen, nopeDim, ropeDim)
	scores := g.BatchedMatMulTransB(qFull, kFull, H, seqLen, seqLen, headDim)

	// Autograd-aware scale + mask + softmax (replaces in-place
	// mutation that broke the chain). Note: MLA's chain is still
	// not fully autograd-trainable upstream of this point because
	// mlaSliceLast/mlaConcatLast/mlaConcatRopeKey are pure-data
	// helpers without autograd. Fixing those needs a Slice and
	// Concat primitive — separate change.
	invScale := float32(1.0 / math.Sqrt(float64(headDim)))
	scaleVec := g.Full(invScale, scores.Shape()...)
	scoredScaled := g.Mul(scores, scaleVec)

	var masked *g.Tensor
	if m.Causal {
		flatScores := scoredScaled.Reshape(H*seqLen, seqLen)
		baseMask := g.CausalMask(seqLen)
		fullMask := make([]bool, H*seqLen*seqLen)
		for h := 0; h < H; h++ {
			copy(fullMask[h*seqLen*seqLen:(h+1)*seqLen*seqLen], baseMask)
		}
		masked = g.MaskFill(flatScores, fullMask, -1e9)
	} else {
		masked = scoredScaled.Reshape(H*seqLen, seqLen)
	}
	softmaxed := g.Softmax(masked)
	scoresOut := softmaxed.Reshape(H, seqLen, seqLen)

	// 6. attn @ V → (H, seq, valueDim) → (seq, H, valueDim) → (seq, H*valueDim)
	attnOut := g.BatchedMatMul(scoresOut, v, H, seqLen, valDim, seqLen)
	concat := g.Permute(attnOut, []int{1, 0, 2}).Reshape(seqLen, H*valDim)
	return m.Wo.Forward(concat)
}

// Parameters returns the four projection weight+bias pairs.
func (m *MLA) Parameters() []*g.Tensor {
	var p []*g.Tensor
	p = append(p, m.Wq.Parameters()...)
	p = append(p, m.WkvDown.Parameters()...)
	p = append(p, m.WkvUp.Parameters()...)
	p = append(p, m.Wo.Parameters()...)
	return p
}

// mlaSliceLast: pure-Go slice along the last dim of an N-D tensor.
// Inclusive start, exclusive end. outShape replaces the last dim of
// the input shape with (end - start). No autograd — MLA forward is
// inference-only here.
func mlaSliceLast(t *g.Tensor, start, end int, outShape []int) *g.Tensor {
	srcShape := t.Shape()
	innerOut := end - start
	innerSrc := srcShape[len(srcShape)-1]
	outer := 1
	for i := 0; i < len(srcShape)-1; i++ {
		outer *= srcShape[i]
	}
	out := g.Zeros(outShape...)
	srcData := t.Data()
	dstData := out.Data()
	for o := 0; o < outer; o++ {
		copy(dstData[o*innerOut:(o+1)*innerOut], srcData[o*innerSrc+start:o*innerSrc+end])
	}
	return out
}

// mlaConcatRopeKey: assemble per-head K_full = concat(K_nope[h, s, :],
// k_rope[s, :]) along the last dim. K_nope is (H, seq, nopeDim);
// k_rope is (seq, ropeDim) shared across heads. Output: (H, seq, nopeDim+ropeDim).
func mlaConcatRopeKey(kNope *g.Tensor, kRope *g.Tensor, H, seq, nopeDim, ropeDim int) *g.Tensor {
	out := g.Zeros(H, seq, nopeDim+ropeDim)
	dst := out.Data()
	srcN := kNope.Data()
	srcR := kRope.Data()
	rowOut := nopeDim + ropeDim
	for h := 0; h < H; h++ {
		for s := 0; s < seq; s++ {
			dstRow := dst[(h*seq+s)*rowOut : (h*seq+s+1)*rowOut]
			copy(dstRow[:nopeDim], srcN[(h*seq+s)*nopeDim:(h*seq+s+1)*nopeDim])
			copy(dstRow[nopeDim:], srcR[s*ropeDim:(s+1)*ropeDim])
		}
	}
	return out
}

// mlaConcatLast: concatenate a (H, seq, A) tensor with a (H, seq, B)
// tensor along the last dim, output (H, seq, A+B).
func mlaConcatLast(a, b *g.Tensor, H, seq, dimA, dimB int) *g.Tensor {
	out := g.Zeros(H, seq, dimA+dimB)
	dst := out.Data()
	srcA := a.Data()
	srcB := b.Data()
	rowOut := dimA + dimB
	for h := 0; h < H; h++ {
		for s := 0; s < seq; s++ {
			dstRow := dst[(h*seq+s)*rowOut : (h*seq+s+1)*rowOut]
			copy(dstRow[:dimA], srcA[(h*seq+s)*dimA:(h*seq+s+1)*dimA])
			copy(dstRow[dimA:], srcB[(h*seq+s)*dimB:(h*seq+s+1)*dimB])
		}
	}
	return out
}
