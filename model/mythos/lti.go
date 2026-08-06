//go:build darwin

package mythos

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// LTIInjection is the Linear-Time-Invariant stable mixing of the
// recurrent block's previous hidden state with the current block's
// output and the original embedding.
//
// The OpenMythos paper writes this as
//
//	h_{t+1} = A · h_t + B · e + Block(h_t, e)
//
// for learnable matrices A, B that satisfy a stability constraint
// (eigenvalues of A inside the unit disc, parameterized as the matrix
// exponential of -log eigenvalues). For the v1 port we simplify to a
// per-channel diagonal A and B — i.e., learnable scalar damping per
// hidden dim. This preserves the stability property by clamping each
// scalar to (0, 1) via sigmoid, and keeps the math autograd-ready
// without introducing a matrix-exp op.
//
// The full matrix-A parameterisation is captured as a follow-up in
// plan 0001 Phase 5 (scale-up). For mythos_tiny on TinyStories the
// scalar form should be enough to demonstrate the recurrent-depth
// claim — the architecture's value comes from the iterative refinement,
// not from the precise spectral structure of A.
//
// Per-channel damping signals: damp[i] ∈ (0, 1), with init = sigmoid(-1) ≈ 0.27,
// then h_{t+1} = damp ⊙ h_t + (1 - damp) ⊙ Block(h_t).
type LTIInjection struct {
	Dim      int
	DampLogit *g.Tensor // raw logits, shape (1, dim); pass through sigmoid for damp ∈ (0,1)
}

// NewLTIInjection initialises per-channel damping logits so sigmoid(logit)
// = cfg.LTIDampInit. Inverse sigmoid: log(p/(1-p)). For the default 0.5
// damping that works out to 0 — initialising at 0 gives equal weight to
// the previous hidden and the new block contribution, then the optimiser
// learns the right per-channel mixing.
func NewLTIInjection(dim int, dampInit float32) *LTIInjection {
	logit := g.Zeros(1, dim)
	if dampInit != 0.5 {
		// Inverse sigmoid to set the desired init: logit = log(p/(1-p)).
		p := float64(dampInit)
		if p <= 0 || p >= 1 {
			panic("mythos: LTIDampInit must be in (0, 1)")
		}
		v := float32(math.Log(p / (1 - p)))
		for i := range logit.Data() {
			logit.Data()[i] = v
		}
	}
	logit.SetRequiresGrad(true)
	return &LTIInjection{Dim: dim, DampLogit: logit}
}

// Apply mixes h_prev with h_block:
//
//	damp = sigmoid(DampLogit)            // per-channel ∈ (0, 1)
//	out  = damp ⊙ h_prev + (1 - damp) ⊙ h_block
//
// Both inputs are (M, dim). Output (M, dim). Autograd-aware end-to-end
// via gorch's broadcast ops (MulB / SubB) and Sigmoid; gradient flows
// back to DampLogit through Sigmoid.
func (l *LTIInjection) Apply(hPrev, hBlock *g.Tensor) *g.Tensor {
	if hPrev.Shape()[1] != l.Dim || hBlock.Shape()[1] != l.Dim {
		panic("mythos: LTI dim mismatch")
	}
	// (1, dim) gate; broadcasts across the M rows of hPrev/hBlock.
	gate := g.Sigmoid(l.DampLogit)
	one := g.Full(1.0, 1, l.Dim)
	complement := g.SubB(one, gate)
	hPrevScaled := g.MulB(hPrev, gate)
	hNewScaled := g.MulB(hBlock, complement)
	return g.Add(hPrevScaled, hNewScaled)
}

// Parameters returns the damping logits (the only learnable part).
func (l *LTIInjection) Parameters() []*g.Tensor {
	return []*g.Tensor{l.DampLogit}
}
