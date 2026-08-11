//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
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

	// Optional per-head Q/K RMSNorm (Qwen3-style "qk-norm"): applied to
	// the (heads·seq, headDim) view of the projected Q/K after the
	// linear projection and BEFORE RoPE, matching HF modeling_qwen3.py
	// order exactly (project → view heads → q_norm/k_norm → RoPE).
	// nil = disabled (Llama / mythos behaviour, unchanged).
	QNorm *RMSNorm
	KNorm *RMSNorm

	// Optional LoRA adapters (plan 0008 §3.1). When non-nil, the
	// corresponding projection routes through the adapter (whose Base
	// must alias the projection's Linear); nil = plain projection,
	// bit-identical to the unadapted module.
	LoRAQ *LoRALinear
	LoRAK *LoRALinear
	LoRAV *LoRALinear
	LoRAO *LoRALinear
}

// GQAConfig fully parametrises a GQA module. Unlike NewGQA (which pins
// headDim = dim / numQueryHeads), HeadDim is explicit — Qwen3-0.6B has
// hidden 1024 with 16 heads × headDim 128, so the attention inner dim
// (numQ·headDim = 2048) differs from the hidden dim. Bias=false zeroes
// and freezes the Linear bias tensors (Qwen3 has no attention biases;
// gorch Linear always carries a bias tensor, so "no bias" = frozen
// zeros, which is exact).
type GQAConfig struct {
	Dim     int  // hidden size (input/output of the module)
	NumQ    int  // query heads
	NumKV   int  // key/value heads; NumQ % NumKV == 0
	HeadDim int  // per-head dim; q/o inner dim = NumQ*HeadDim, kv dim = NumKV*HeadDim
	Bias    bool // false = zero + freeze all projection biases
}

// NewGQAConfig builds a GQA module from an explicit config. NewGQA
// delegates here with the standard headDim = dim/numQueryHeads
// convention (mythos back-compat).
func NewGQAConfig(cfg GQAConfig) *GQA {
	if cfg.Dim <= 0 || cfg.NumQ <= 0 || cfg.NumKV <= 0 || cfg.HeadDim <= 0 {
		panic("gorch/nn: GQAConfig fields must be positive")
	}
	if cfg.NumQ%cfg.NumKV != 0 {
		panic("gorch/nn: GQA NumQ must be divisible by NumKV")
	}
	innerDim := cfg.NumQ * cfg.HeadDim
	kvDim := cfg.NumKV * cfg.HeadDim
	gqa := &GQA{
		Wq:            NewLinear(cfg.Dim, innerDim),
		Wk:            NewLinear(cfg.Dim, kvDim),
		Wv:            NewLinear(cfg.Dim, kvDim),
		Wo:            NewLinear(innerDim, cfg.Dim),
		NumQueryHeads: cfg.NumQ,
		NumKVHeads:    cfg.NumKV,
		HeadDim:       cfg.HeadDim,
		Dim:           cfg.Dim,
		Causal:        true,
	}
	if !cfg.Bias {
		for _, l := range []*Linear{gqa.Wq, gqa.Wk, gqa.Wv, gqa.Wo} {
			for i := range l.Bias.Data() {
				l.Bias.Data()[i] = 0
			}
			l.Bias.SetRequiresGrad(false)
		}
	}
	return gqa
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
	return NewGQAConfig(GQAConfig{
		Dim:     dim,
		NumQ:    numQueryHeads,
		NumKV:   numKVHeads,
		HeadDim: dim / numQueryHeads,
		Bias:    true,
	})
}

// Forward runs GQA on input x of shape (seq, dim). startPos is the
// absolute position of x[0] for RoPE / causal masking; pass 0 for the
// no-cache full-sequence forward.
//
// Builds the full autograd graph (this is the training path; a stale
// "inference-only" note here predated the autograd-aware
// scale/mask/softmax rework — fixed in plan 0009 X1). Mirrors the
// existing MultiHeadAttention.Forward batched pattern with the
// QHeads → KVHeads expansion folded in via RepeatInterleave.
//
// The causal scale+mask+softmax runs as the fused g.CausalSoftmax op
// (plan 0009 K1): one op, one output tensor, Metal kernel when the
// scores are GPU-resident — replacing the previous Full+Mul scale
// tensor (144 MB/layer at seq 1500), the tiled bool mask, and the
// MaskFill intermediate.
func (gqa *GQA) Forward(x *g.Tensor, startPos int) *g.Tensor {
	seqLen := x.Shape()[0]
	headDim := gqa.HeadDim
	numQ := gqa.NumQueryHeads
	numKV := gqa.NumKVHeads
	groupSize := numQ / numKV
	innerDim := numQ * headDim

	qH, kH, vH := gqa.ProjectQKV(x, startPos)

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
	var scoresOut *g.Tensor
	if gqa.Causal {
		// Fused scale + causal mask + softmax (K1). The mask is a
		// compare against the column index inside the kernel/CPU loop —
		// no mask tensor is built at all (subsumes the per-(heads, seq)
		// cached-mask plan item).
		scoresOut = g.CausalSoftmax(scores, numQ, seqLen, invScale)
	} else {
		// Non-causal: grad-aware scalar scaling (no Full+Mul constant
		// tensor) + plain softmax on the (numQ*seq, seq) view.
		scaled := g.Scale(scores, invScale)
		scoresOut = g.Softmax(scaled.Reshape(numQ*seqLen, seqLen)).Reshape(numQ, seqLen, seqLen)
	}

	// attn @ V: (numQ, seq, seq) × (numQ, seq, headDim) → (numQ, seq, headDim)
	attnOut := g.BatchedMatMul(scoresOut, vH, numQ, seqLen, headDim, seqLen)

	// Permute back: (numQ, seq, headDim) → (seq, numQ, headDim) → (seq, innerDim).
	// innerDim == dim for the NewGQA convention; explicit-HeadDim configs
	// (Qwen3: 16·128 = 2048 ≠ 1024) feed Wo's larger input dim here.
	concat := g.Permute(attnOut, []int{1, 0, 2}).Reshape(seqLen, innerDim)

	return loraForward(gqa.LoRAO, gqa.Wo, concat)
}

// ProjectQKV projects x to per-head Q/K/V with the optional per-head
// QK-norm and RoPE applied (norm BEFORE rope, HF Qwen3 order). Returns
//
//	qH (numQ, seq, headDim), kH (numKV, seq, headDim), vH (numKV, seq, headDim)
//
// with RoPE at absolute positions startPos..startPos+seq-1 on Q and K
// (never V). Shared by Forward and ForwardCached; exported so golden
// parity tests can compare the post-qknorm+rope Q/K stages against HF
// reference dumps without duplicating the projection math.
func (gqa *GQA) ProjectQKV(x *g.Tensor, startPos int) (qH, kH, vH *g.Tensor) {
	seqLen := x.Shape()[0]
	headDim := gqa.HeadDim
	numQ := gqa.NumQueryHeads
	numKV := gqa.NumKVHeads

	q := loraForward(gqa.LoRAQ, gqa.Wq, x) // (seq, numQ*headDim)
	k := loraForward(gqa.LoRAK, gqa.Wk, x) // (seq, numKV*headDim)
	v := loraForward(gqa.LoRAV, gqa.Wv, x)

	// Per-head RMSNorm on the (seq·heads, headDim) view — row r is
	// (token r/heads, head r%heads), and RMSNorm is per-row, so the
	// row order is immaterial; what matters is normalising each
	// (token, head) headDim-vector independently.
	if gqa.QNorm != nil {
		q = gqa.QNorm.Forward(q.Reshape(seqLen*numQ, headDim))
	}
	if gqa.KNorm != nil {
		k = gqa.KNorm.Forward(k.Reshape(seqLen*numKV, headDim))
	}

	// (seq, heads, headDim) → (heads, seq, headDim).
	qH = g.Permute(q.Reshape(seqLen, numQ, headDim), []int{1, 0, 2})
	kH = g.Permute(k.Reshape(seqLen, numKV, headDim), []int{1, 0, 2})
	vH = g.Permute(v.Reshape(seqLen, numKV, headDim), []int{1, 0, 2})

	if gqa.RoPE != nil {
		qH = gqa.RoPE.Apply(qH, startPos)
		kH = gqa.RoPE.Apply(kH, startPos)
	}
	return qH, kH, vH
}

// ForwardCached computes GQA for new tokens x (newSeq, dim) against a
// KV cache — the autoregressive decode path. The cache must be created
// with dim = NumKVHeads*HeadDim; posOffset is the absolute position of
// x's first token and must equal cache.Len() before this call (K/V are
// RoPE-rotated at absolute cache positions before being appended, so
// cached entries are position-final).
//
// Masking: a query at absolute position p attends keys 0..p — no mask
// work for single-token steps, staircase mask for multi-token prefill
// chunks. Inference-only (runs under NoGrad); mirrors
// MultiHeadAttention.ForwardCached structurally but routes the
// score/value matmuls through Accelerate per KV head (with the
// group's query heads batched into one GEMM).
func (gqa *GQA) ForwardCached(x *g.Tensor, cache *KVCache, layerIdx, posOffset int) *g.Tensor {
	var out *g.Tensor
	g.NoGrad(func() {
		out = gqa.forwardCached(x, cache, layerIdx, posOffset)
	})
	return out
}

func (gqa *GQA) forwardCached(x *g.Tensor, cache *KVCache, layerIdx, posOffset int) *g.Tensor {
	newSeq := x.Shape()[0]
	headDim := gqa.HeadDim
	numQ := gqa.NumQueryHeads
	numKV := gqa.NumKVHeads
	groupSize := numQ / numKV
	kvDim := numKV * headDim
	if cache.Dim != kvDim {
		panic("gorch/nn: GQA.ForwardCached cache dim must be numKVHeads*headDim")
	}
	// cache.Len() reads layer 0, which a multi-layer model has already
	// grown by the time layers ≥ 1 run — check this layer's own length.
	if have := len(cache.Keys[layerIdx]) / kvDim; posOffset != have {
		panic("gorch/nn: GQA.ForwardCached posOffset must equal the layer's cached length")
	}

	qH, kH, vH := gqa.ProjectQKV(x, posOffset)
	if qH.IsOnMetal() {
		qH.ToCPU()
	}
	if kH.IsOnMetal() {
		kH.ToCPU()
	}
	if vH.IsOnMetal() {
		vH.ToCPU()
	}

	// Cache rows are (token, kvDim): permute K/V back from
	// (numKV, newSeq, headDim) to (newSeq, numKV·headDim) row layout.
	kRows := g.Permute(kH, []int{1, 0, 2}).Reshape(newSeq, kvDim)
	vRows := g.Permute(vH, []int{1, 0, 2}).Reshape(newSeq, kvDim)
	cache.Append(layerIdx, kRows.Data(), vRows.Data())

	totalSeq := len(cache.Keys[layerIdx]) / kvDim
	cachedK := cache.Keys[layerIdx]   // flat (totalSeq, kvDim)
	cachedV := cache.Values[layerIdx] // flat (totalSeq, kvDim)
	qData := qH.Data()                // (numQ, newSeq, headDim)

	invScale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Per-KV-head attention. The cache's per-head K/V columns are
	// strided (stride kvDim); gather each head into a contiguous
	// scratch so the score and value matmuls run as Accelerate GEMMs
	// over the whole query group at once.
	kHead := make([]float32, totalSeq*headDim)
	vHead := make([]float32, totalSeq*headDim)
	scores := make([]float32, groupSize*newSeq*totalSeq)
	attnOut := make([]float32, numQ*newSeq*headDim) // (numQ, newSeq, headDim)

	for kv := 0; kv < numKV; kv++ {
		colOff := kv * headDim
		for j := 0; j < totalSeq; j++ {
			copy(kHead[j*headDim:(j+1)*headDim], cachedK[j*kvDim+colOff:j*kvDim+colOff+headDim])
			copy(vHead[j*headDim:(j+1)*headDim], cachedV[j*kvDim+colOff:j*kvDim+colOff+headDim])
		}

		// This KV head's query group is a contiguous (groupSize·newSeq,
		// headDim) slab of qData (heads are the leading dim).
		qBlock := qData[kv*groupSize*newSeq*headDim : (kv+1)*groupSize*newSeq*headDim]

		// scores = q · kᵀ · invScale: (groupSize·newSeq, totalSeq).
		accelerate.SgemmTransB(groupSize*newSeq, totalSeq, headDim, invScale, qBlock, kHead, 0.0, scores)

		// Causal staircase mask in absolute coordinates: row r is query
		// i = r % newSeq at absolute position posOffset+i; keys j >
		// posOffset+i are masked. For newSeq == 1 nothing is masked.
		if gqa.Causal && newSeq > 1 {
			for r := 0; r < groupSize*newSeq; r++ {
				absPos := posOffset + r%newSeq
				row := scores[r*totalSeq : (r+1)*totalSeq]
				for j := absPos + 1; j < totalSeq; j++ {
					row[j] = float32(math.Inf(-1))
				}
			}
		}

		softmaxRowsAccelerated(scores, groupSize*newSeq, totalSeq)

		// attn @ V: (groupSize·newSeq, totalSeq) × (totalSeq, headDim)
		// → this group's slab of the (numQ, newSeq, headDim) output.
		outBlock := attnOut[kv*groupSize*newSeq*headDim : (kv+1)*groupSize*newSeq*headDim]
		accelerate.Sgemm(groupSize*newSeq, headDim, totalSeq, 1.0, scores, vHead, 0.0, outBlock)
	}

	// (numQ, newSeq, headDim) → (newSeq, numQ·headDim).
	concat := make([]float32, newSeq*numQ*headDim)
	for h := 0; h < numQ; h++ {
		for i := 0; i < newSeq; i++ {
			copy(concat[(i*numQ+h)*headDim:(i*numQ+h+1)*headDim],
				attnOut[(h*newSeq+i)*headDim:(h*newSeq+i+1)*headDim])
		}
	}
	return loraForward(gqa.LoRAO, gqa.Wo, g.NewTensor(concat, newSeq, numQ*headDim))
}

// softmaxRowsAccelerated softmaxes each row of a (rows, cols) buffer in
// place: scalar per-row max subtraction, then vectorised exp / sum /
// scale via Accelerate. -Inf entries (causal mask) exp to exactly 0.
func softmaxRowsAccelerated(buf []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := buf[r*cols : (r+1)*cols]
		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		for j := range row {
			row[j] -= maxVal
		}
		accelerate.Exp(row, row)
		sum := accelerate.Sum(row)
		if sum > 0 {
			accelerate.VScale(row, 1.0/sum, row)
		}
	}
}

// Parameters returns the four projection matrices (and the optional
// QK-norm gammas when enabled).
func (gqa *GQA) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, gqa.Wq.Parameters()...)
	params = append(params, gqa.Wk.Parameters()...)
	params = append(params, gqa.Wv.Parameters()...)
	params = append(params, gqa.Wo.Parameters()...)
	if gqa.QNorm != nil {
		params = append(params, gqa.QNorm.Parameters()...)
	}
	if gqa.KNorm != nil {
		params = append(params, gqa.KNorm.Parameters()...)
	}
	return params
}
