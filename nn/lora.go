//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
)

// LoRALinear — Low-Rank Adaptation of a frozen Linear layer
// (Hu et al. 2021; plan 0008 §3.1).
//
//	y = Base(x) + (x @ Aᵀ @ Bᵀ) · (Alpha/R)
//
// Base stays frozen (Weight/Bias RequiresGrad=false — see Freeze);
// only the low-rank factors A (r, in) and B (out, r) train. B is
// zero-initialised so the adapter starts as an exact identity: the
// wrapped layer is bit-identical to Base until the first optimizer
// step. The forward is a composition of existing autograd ops
// (MatMul, Transpose2D, Add) plus a scalar-scale node — no new
// autograd primitives.
type LoRALinear struct {
	Base  *Linear
	A     *g.Tensor // (r, in) — N(0, 0.02) init
	B     *g.Tensor // (out, r) — zero init ⇒ adapter starts as identity
	Alpha float32
	R     int

	merged bool // true after Merge(): delta folded into Base.Weight
}

// NewLoRALinear wraps base with a rank-r adapter. The base layer is
// frozen (Freeze) as a side effect — LoRA's contract is that only A/B
// train.
func NewLoRALinear(base *Linear, r int, alpha float32) *LoRALinear {
	if r <= 0 {
		panic("gorch/nn: LoRA rank must be ≥ 1")
	}
	in := base.Weight.Shape()[1]
	out := base.Weight.Shape()[0]

	a := g.RandN(r, in)
	for i := range a.Data() {
		a.Data()[i] *= 0.02
	}
	a.SetRequiresGrad(true)

	b := g.Zeros(out, r)
	b.SetRequiresGrad(true)

	l := &LoRALinear{Base: base, A: a, B: b, Alpha: alpha, R: r}
	l.Freeze()
	return l
}

// Freeze marks the base layer's Weight and Bias as non-trainable.
func (l *LoRALinear) Freeze() {
	l.Base.Weight.SetRequiresGrad(false)
	l.Base.Bias.SetRequiresGrad(false)
}

// Forward computes y = Base(x) + MatMul(MatMul(x, Aᵀ), Bᵀ)·(Alpha/R).
// After Merge() the delta lives inside Base.Weight and the adapter
// path is skipped (zero-overhead inference).
func (l *LoRALinear) Forward(x *g.Tensor) *g.Tensor {
	base := l.Base.Forward(x)
	if l.merged {
		return base
	}
	xa := g.MatMul(x, g.Transpose2D(l.A))   // (batch, r)
	xab := g.MatMul(xa, g.Transpose2D(l.B)) // (batch, out)
	delta := g.Scale(xab, l.Alpha/float32(l.R))
	return g.Add(base, delta)
}

// Merged reports whether the adapter is currently folded into
// Base.Weight.
func (l *LoRALinear) Merged() bool { return l.merged }

// Merge folds B·A·(Alpha/R) into Base.Weight for zero-overhead
// inference. A and B keep their unmerged values so training can
// resume after Unmerge. Idempotent.
func (l *LoRALinear) Merge() {
	if l.merged {
		return
	}
	l.applyDelta(+1)
	l.merged = true
}

// Unmerge subtracts the adapter delta from Base.Weight, restoring the
// unmerged state. Idempotent.
func (l *LoRALinear) Unmerge() {
	if !l.merged {
		return
	}
	l.applyDelta(-1)
	l.merged = false
}

// applyDelta does Base.Weight += sign · B·A·(Alpha/R).
func (l *LoRALinear) applyDelta(sign float32) {
	if l.Base.Weight.Dtype() != g.Float32 {
		panic("gorch/nn: LoRA Merge/Unmerge requires an f32 base weight — the bf16 frozen path (plan 0009 X4) trains unmerged; export by merging into an f32 copy of the base")
	}
	out := l.Base.Weight.Shape()[0]
	in := l.Base.Weight.Shape()[1]
	// delta = B (out, r) @ A (r, in) → (out, in)
	delta := make([]float32, out*in)
	accelerate.Sgemm(out, in, l.R, sign*l.Alpha/float32(l.R), l.B.Data(), l.A.Data(), 0.0, delta)
	w := l.Base.Weight.Data()
	accelerate.VAdd(w, delta, w)
}

// Parameters returns the trainable adapter factors {A, B} only —
// the frozen base is deliberately excluded.
func (l *LoRALinear) Parameters() []*g.Tensor {
	return []*g.Tensor{l.A, l.B}
}

// loraForward routes a projection through its optional LoRA adapter:
// adapters wrap the SAME base layer they shadow, so when l is nil the
// plain base forward runs unchanged.
func loraForward(l *LoRALinear, base *Linear, x *g.Tensor) *g.Tensor {
	if l == nil {
		return base.Forward(x)
	}
	if l.Base != base {
		panic("gorch/nn: LoRA adapter does not wrap this projection's base Linear")
	}
	return l.Forward(x)
}
