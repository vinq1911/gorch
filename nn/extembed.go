//go:build darwin

package nn

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
	"github.com/vinq1911/gorch/metal"
)

// ExtendedEmbedding — vocab-extension surgery for a frozen pretrained
// embedding (plan 0008 §3.2).
//
// Design: SPLIT TENSORS, not row masking. Base holds the pretrained
// rows (frozen, ids 0..baseVocab-1); Ext holds the appended rows
// (trainable, ids baseVocab..baseVocab+numExt-1). The "only appended
// rows train" property is structural: Base is never an autograd input
// of either the lookup or the tied head, so no gradient can reach it
// and optimizer state is sized to Ext alone. A RowMaskedGrad fallback
// is explicitly rejected (plan §3.2) for optimizer-state bloat and
// masking-bug risk.
type ExtendedEmbedding struct {
	Base *g.Tensor // (baseVocab, dim) — frozen pretrained rows
	Ext  *g.Tensor // (numExt, dim) — trainable appended rows
	Dim  int
}

// ToMetal moves Base and Ext into unified memory (plan 0009 X4). The
// lookup and interleave loops stay on the CPU (Data() is the read
// barrier), but (1) the lookup output becomes Metal-resident, so the
// whole transformer chain downstream starts resident instead of being
// stripped of GPU dispatch at the very first op, and (2) the tied-head
// GEMMs in Logits dispatch MPS instead of Accelerate when they clear
// the matmul threshold. Both tensors stay f32: Base rows are read by
// the CPU lookup, Ext is a trainable param (optimizer contract).
func (e *ExtendedEmbedding) ToMetal(dev *metal.Device) {
	e.Base.ToMetal(dev)
	e.Ext.ToMetal(dev)
}

// NewExtendedEmbedding wraps the frozen base embedding matrix and
// appends numExt trainable rows initialised to the mean of the base
// rows plus N(0, 0.02) noise — the standard vocab-extension init that
// keeps new-token logits in-distribution at step 0.
func NewExtendedEmbedding(base *g.Tensor, numExt int) *ExtendedEmbedding {
	if base.Dim() != 2 {
		panic("gorch/nn: ExtendedEmbedding base must be 2-D (vocab, dim)")
	}
	if numExt <= 0 {
		panic("gorch/nn: ExtendedEmbedding numExt must be ≥ 1")
	}
	baseVocab, dim := base.Shape()[0], base.Shape()[1]

	// Mean of base rows.
	mean := make([]float64, dim)
	bd := base.Data()
	for r := 0; r < baseVocab; r++ {
		row := bd[r*dim : (r+1)*dim]
		for j, v := range row {
			mean[j] += float64(v)
		}
	}
	for j := range mean {
		mean[j] /= float64(baseVocab)
	}

	ext := g.RandN(numExt, dim)
	ed := ext.Data()
	for r := 0; r < numExt; r++ {
		for j := 0; j < dim; j++ {
			ed[r*dim+j] = ed[r*dim+j]*0.02 + float32(mean[j])
		}
	}
	ext.SetRequiresGrad(true)

	return &ExtendedEmbedding{Base: base, Ext: ext, Dim: dim}
}

// BaseVocab returns the number of frozen pretrained rows.
func (e *ExtendedEmbedding) BaseVocab() int { return e.Base.Shape()[0] }

// VocabSize returns the total extended vocabulary size.
func (e *ExtendedEmbedding) VocabSize() int { return e.Base.Shape()[0] + e.Ext.Shape()[0] }

// Forward looks up embeddings for ids over the extended vocabulary:
// ids < baseVocab read (frozen) Base rows, ids ≥ baseVocab read
// (trainable) Ext rows. Returns (len(ids), dim). Gradient flows ONLY
// into the Ext rows that were looked up — Base is not an autograd
// input.
func (e *ExtendedEmbedding) Forward(ids []int) *g.Tensor {
	baseVocab := e.BaseVocab()
	numExt := e.Ext.Shape()[0]
	vocab := baseVocab + numExt
	dim := e.Dim

	// Residency-inheriting output (plan 0009 X4): when the embedding
	// tables live on Metal the hidden-state chain starts Metal-resident,
	// which is what lets every block op downstream dispatch its GPU
	// path. The row copies below run on the CPU through unified memory.
	out := g.ZerosLike(e.Ext, len(ids), dim)
	od := out.Data()
	bd, ed := e.Base.Data(), e.Ext.Data()
	for i, id := range ids {
		switch {
		case id < 0 || id >= vocab:
			panic(fmt.Sprintf("gorch/nn: ExtendedEmbedding id %d out of range [0, %d)", id, vocab))
		case id < baseVocab:
			copy(od[i*dim:(i+1)*dim], bd[id*dim:(id+1)*dim])
		default:
			r := id - baseVocab
			copy(od[i*dim:(i+1)*dim], ed[r*dim:(r+1)*dim])
		}
	}

	if g.GradEnabled() && e.Ext.RequiresGrad() {
		out.SetRequiresGrad(true)
		idsCopy := append([]int{}, ids...)
		out.SetGradFn("ExtEmbedLookup", []*g.Tensor{e.Ext}, func(grad *g.Tensor) []*g.Tensor {
			dExt := g.ZerosLike(e.Ext, numExt, dim)
			dd := dExt.Data()
			gd := grad.Data()
			for i, id := range idsCopy {
				if id < baseVocab {
					continue
				}
				r := id - baseVocab
				for j := 0; j < dim; j++ {
					dd[r*dim+j] += gd[i*dim+j]
				}
			}
			return []*g.Tensor{dExt}
		})
	}
	return out
}

// Logits applies the tied LM head over the split vocabulary:
//
//	logits = concat(h @ Baseᵀ, h @ Extᵀ) along the vocab dim
//
// h is (M, dim), the result (M, baseVocab+numExt). The backward is a
// fused node: dL/dh receives contributions from BOTH column blocks
// (the softmax normaliser couples every vocab column to h), while the
// weight gradient is computed for Ext ONLY — Base is not an autograd
// input, so the dW GEMM over the 151,936 frozen columns never runs.
//
// This is semantically the plan's
// concat(MatMulTransB(h, Base), MatMul(h, Transpose2D(Ext))) with the
// dead base-side dW structurally removed.
func (e *ExtendedEmbedding) Logits(h *g.Tensor) *g.Tensor {
	if h.Dim() != 2 || h.Shape()[1] != e.Dim {
		panic("gorch/nn: ExtendedEmbedding.Logits requires h of shape (M, dim)")
	}
	M := h.Shape()[0]
	dim := e.Dim
	v0 := e.BaseVocab()
	v1 := e.Ext.Shape()[0]

	// Residency-inheriting output (plan 0009 X1 rule, applied in X2):
	// when the hidden chain is Metal-resident the concatenated logits
	// stay resident too, so CrossEntropyLoss can dispatch the K2 fused
	// Metal kernels on the gathered supervised path (qwen SupervisedLoss).
	out := g.ZerosLike(h, M, v0+v1)
	od := out.Data()
	// Two GEMMs into scratch blocks, then row-interleave into the
	// concatenated layout (a TransB GEMM writes a contiguous (M, N)
	// result, so it cannot target the strided column block directly).
	//
	// GPU path (plan 0009 X4): with h and the tables Metal-resident,
	// g.MatMulTransB (no autograd — this node's fused GradFn below is
	// the autograd) dispatches the head GEMMs on MPS above the matmul
	// threshold; the base-side GEMM at (M, 151936, 1024) is the single
	// biggest matmul in the training step after the blocks themselves.
	// Below threshold or CPU-resident it is the same Accelerate call as
	// before, through unified memory.
	if h.IsOnMetal() && e.Base.IsOnMetal() && e.Ext.IsOnMetal() {
		lb := g.MatMulTransB(h.Detach(), e.Base.Detach())
		le := g.MatMulTransB(h.Detach(), e.Ext.Detach())
		lbd, led := lb.Data(), le.Data()
		for i := 0; i < M; i++ {
			copy(od[i*(v0+v1):i*(v0+v1)+v0], lbd[i*v0:(i+1)*v0])
			copy(od[i*(v0+v1)+v0:(i+1)*(v0+v1)], led[i*v1:(i+1)*v1])
		}
	} else {
		lb := make([]float32, M*v0)
		le := make([]float32, M*v1)
		accelerate.SgemmTransB(M, v0, dim, 1.0, h.Data(), e.Base.Data(), 0.0, lb)
		accelerate.SgemmTransB(M, v1, dim, 1.0, h.Data(), e.Ext.Data(), 0.0, le)
		for i := 0; i < M; i++ {
			copy(od[i*(v0+v1):i*(v0+v1)+v0], lb[i*v0:(i+1)*v0])
			copy(od[i*(v0+v1)+v0:(i+1)*(v0+v1)], le[i*v1:(i+1)*v1])
		}
	}

	if g.GradEnabled() && (h.RequiresGrad() || e.Ext.RequiresGrad()) {
		out.SetRequiresGrad(true)
		base := e.Base
		ext := e.Ext
		out.SetGradFn("ExtEmbedLogits", []*g.Tensor{h, ext}, func(grad *g.Tensor) []*g.Tensor {
			// GPU path (plan 0009 X4): split the (M, v0+v1) grad into
			// resident column blocks on the CPU (one sync + memcpy),
			// then run the three grad GEMMs — dh's two contributions and
			// dExt — through the MPS-dispatching no-autograd matmuls.
			// The base-side dW GEMM stays structurally absent.
			if grad.IsOnMetal() && base.IsOnMetal() && ext.IsOnMetal() {
				gd := grad.Data()
				gb := g.ZerosLike(grad, M, v0)
				ge := g.ZerosLike(grad, M, v1)
				gbd, ged := gb.Data(), ge.Data()
				for i := 0; i < M; i++ {
					copy(gbd[i*v0:(i+1)*v0], gd[i*(v0+v1):i*(v0+v1)+v0])
					copy(ged[i*v1:(i+1)*v1], gd[i*(v0+v1)+v0:(i+1)*(v0+v1)])
				}
				var dh *g.Tensor
				if h.RequiresGrad() {
					dh = g.Add(g.MatMul(gb, base.Detach()), g.MatMul(ge, ext.Detach()))
				}
				var dExt *g.Tensor
				if ext.RequiresGrad() {
					dExt = g.MatMulTransA(ge, h.Detach())
				}
				return []*g.Tensor{dh, dExt}
			}

			gd := grad.Data()
			// Split the (M, v0+v1) grad into its column blocks.
			gb := make([]float32, M*v0)
			ge := make([]float32, M*v1)
			for i := 0; i < M; i++ {
				copy(gb[i*v0:(i+1)*v0], gd[i*(v0+v1):i*(v0+v1)+v0])
				copy(ge[i*v1:(i+1)*v1], gd[i*(v0+v1)+v0:(i+1)*(v0+v1)])
			}

			// dh = gb @ Base + ge @ Ext — both blocks contribute.
			var dh *g.Tensor
			if h.RequiresGrad() {
				dh = g.ZerosLike(h, M, dim)
				accelerate.Sgemm(M, dim, v0, 1.0, gb, base.Data(), 0.0, dh.Data())
				accelerate.Sgemm(M, dim, v1, 1.0, ge, ext.Data(), 1.0, dh.Data())
			}

			// dExt = geᵀ @ h — Ext columns only; the base-side dW GEMM
			// is structurally absent.
			var dExt *g.Tensor
			if ext.RequiresGrad() {
				dExt = g.Zeros(v1, dim)
				accelerate.SgemmTransA(v1, dim, M, 1.0, ge, h.Data(), 0.0, dExt.Data())
			}
			return []*g.Tensor{dh, dExt}
		})
	}
	return out
}

// Parameters returns the trainable appended rows {Ext} only.
func (e *ExtendedEmbedding) Parameters() []*g.Tensor {
	return []*g.Tensor{e.Ext}
}
