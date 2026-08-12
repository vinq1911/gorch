//go:build darwin

package gorch

import "testing"

// Regression tests for the 2026-08-12 autograd retention bug.
//
// Backward() used to run with grad tracking ON. Backward closures
// compute with ordinary graph-building ops (MatMul, Scale,
// Transpose2D...), so each gradient got its own GradFn referencing the
// forward tensors it consumed. Those gradients are stored in
// inp.grad — and a PARAMETER's .grad survives until the optimizer's
// ZeroGrad, i.e. across every gradient-accumulation micro-step. Each
// parameter therefore rooted a live copy of that micro-step's entire
// forward+backward graph: ~5 GB per micro-step, unbounded, and
// invisible to runtime.GC() because it was all genuinely reachable.
//
// The exact invariants, in preference order (cheapest first):
//   1. the backward pass runs with grad tracking OFF (mechanism)
//   2. no stored .grad ever owns a GradFn (invariant)
//   3. accumulation across micro-steps stays numerically correct

// TestBackwardRunsUntracked is the mechanism guard. Every op builds
// its GradFn only when GradEnabled(), so proving the backward pass
// runs with tracking OFF proves no backward-produced tensor can own a
// graph. A backward closure is the only place that can observe this,
// so install one and have it report.
func TestBackwardRunsUntracked(t *testing.T) {
	x := NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	x.SetRequiresGrad(true)

	out := Sum(MatMul(x, x))
	seen := 0
	sawTracking := false
	// Wrap the real node: our closure observes grad mode, then hands
	// off to the graph beneath it.
	inner := out.gradFn
	out.SetGradFn("probe", inner.inputs, func(g *Tensor) []*Tensor {
		seen++
		sawTracking = GradEnabled()
		return inner.backward(g)
	})

	out.Backward()

	if seen == 0 {
		t.Fatal("probe closure never ran")
	}
	if sawTracking {
		t.Error("backward closures run with grad tracking ON — every gradient they " +
			"compute will own a GradFn, and once stored in a parameter's .grad it " +
			"pins that micro-step's whole forward graph (2026-08-12 leak)")
	}
}

// TestStoredGradsOwnNoGraph asserts the invariant that makes the leak
// impossible: a tensor stored in .grad must never own a GradFn, so it
// can never root a forward graph.
func TestStoredGradsOwnNoGraph(t *testing.T) {
	x := NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	w := NewTensor([]float32{0.5, -0.5, 0.25, 1}, 2, 2)
	x.SetRequiresGrad(true)
	w.SetRequiresGrad(true)

	// Several micro-steps WITHOUT ZeroGrad — the accumulation pattern
	// that made the leak unbounded.
	for i := 0; i < 4; i++ {
		Sum(MatMul(x, w)).Backward()
		for name, p := range map[string]*Tensor{"x": x, "w": w} {
			if p.Grad() == nil {
				t.Fatalf("micro-step %d: %s has no grad", i, name)
			}
			if p.Grad().gradFn != nil {
				t.Fatalf("micro-step %d: %s.grad owns a GradFn (%q) — it is rooting "+
					"a forward graph and will pin its activations",
					i, name, p.Grad().gradFn.name)
			}
		}
	}
}

// TestAccumulationStillCorrect guards the numerics the fix could have
// broken: N micro-steps of the same input must accumulate to exactly N
// times the single-step gradient.
func TestAccumulationStillCorrect(t *testing.T) {
	single := func() []float32 {
		x := NewTensor([]float32{1, 2, 3, 4}, 2, 2)
		w := NewTensor([]float32{0.5, -0.5, 0.25, 1}, 2, 2)
		w.SetRequiresGrad(true)
		Sum(MatMul(x, w)).Backward()
		out := make([]float32, len(w.Grad().Data()))
		copy(out, w.Grad().Data())
		return out
	}
	one := single()

	x := NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	w := NewTensor([]float32{0.5, -0.5, 0.25, 1}, 2, 2)
	w.SetRequiresGrad(true)
	const micro = 3
	for i := 0; i < micro; i++ {
		Sum(MatMul(x, w)).Backward()
	}
	got := w.Grad().Data()
	for i := range one {
		want := one[i] * micro
		if diff := got[i] - want; diff > 1e-5 || diff < -1e-5 {
			t.Errorf("accumulated grad[%d] = %v, want %v (%d micro-steps)", i, got[i], want, micro)
		}
	}
}

// TestNoGradRestoredAfterBackward verifies Backward's NoGrad scope does
// not leak: ops after a Backward must still build a graph, or every
// subsequent forward would silently stop being differentiable.
func TestNoGradRestoredAfterBackward(t *testing.T) {
	x := NewTensor([]float32{1, 2}, 1, 2)
	w := NewTensor([]float32{1, 1}, 2, 1)
	w.SetRequiresGrad(true)
	Sum(MatMul(x, w)).Backward()

	if !GradEnabled() {
		t.Fatal("grad tracking still disabled after Backward returned")
	}
	if out := MatMul(x, w); out.gradFn == nil {
		t.Fatal("forward after Backward did not build a graph")
	}
}
