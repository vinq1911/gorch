//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// The tests here are the contract from checkpoint.go's package comment,
// one test per clause, plus the failure modes the pre-implementation
// design review named. A checkpointing bug does not crash — it produces
// subtly wrong or silently zero gradients — so every one of these is a
// gradient-value comparison against the uncheckpointed pass, not a
// smoke test.

func scalarT(v float32, req bool) *Tensor {
	t := NewTensor([]float32{v}, 1, 1)
	t.SetRequiresGrad(req)
	return t
}

func gradVal(t *testing.T, x *Tensor) float32 {
	t.Helper()
	if x.Grad() == nil {
		t.Fatalf("gradient is nil")
	}
	return x.Grad().Data()[0]
}

func closeEnough(a, b float32, tol float64) bool {
	return math.Abs(float64(a-b)) <= tol*(1+math.Abs(float64(b)))
}

// TestCheckpointScalarGradientsMatch is the smallest end-to-end proof:
// f(x) = (x·a + b)², with x an activation and a, b leaf "parameters"
// captured by the closure. Checkpointed and plain gradients must agree
// for all three, and the analytic values are known.
func TestCheckpointScalarGradientsMatch(t *testing.T) {
	build := func(ckpt bool) (x, a, b *Tensor) {
		x, a, b = scalarT(3, true), scalarT(2, true), scalarT(5, true)
		fn := func(in *Tensor) *Tensor {
			u := Add(Mul(in, a), b)
			return Mul(u, u)
		}
		var y *Tensor
		if ckpt {
			y = Checkpoint("t", x, fn)
		} else {
			y = fn(x)
		}
		Sum(y).Backward()
		return x, a, b
	}

	x0, a0, b0 := build(false)
	x1, a1, b1 := build(true)

	// u = 3·2+5 = 11, y = 121. dy/dx = 2u·a = 44, dy/da = 2u·x = 66,
	// dy/db = 2u = 22.
	for _, c := range []struct {
		name       string
		plain, ck  *Tensor
		analytical float32
	}{
		{"x", x0, x1, 44},
		{"a", a0, a1, 66},
		{"b", b0, b1, 22},
	} {
		p, k := gradVal(t, c.plain), gradVal(t, c.ck)
		if !closeEnough(p, c.analytical, 1e-6) {
			t.Fatalf("%s: plain grad %v != analytic %v", c.name, p, c.analytical)
		}
		if !closeEnough(k, p, 1e-6) {
			t.Fatalf("%s: checkpointed grad %v != plain grad %v", c.name, k, p)
		}
	}
}

// TestCheckpointBuildsNoGraphDuringForward is the memory claim itself:
// the checkpointed output must have exactly one autograd predecessor
// (the saved input), not the segment's internal ops. If this regresses,
// checkpointing still computes correct gradients while saving nothing —
// the silent failure that a memory measurement would catch only later.
func TestCheckpointBuildsNoGraphDuringForward(t *testing.T) {
	x := scalarT(3, true)
	a := scalarT(2, true)
	y := Checkpoint("t", x, func(in *Tensor) *Tensor {
		u := Mul(in, a)
		u = Add(u, a)
		return Mul(u, u)
	})

	if y.gradFn == nil {
		t.Fatal("checkpoint output has no gradFn")
	}
	if got := len(y.gradFn.inputs); got != 1 {
		t.Fatalf("checkpoint node has %d inputs, want exactly 1 (the saved input)", got)
	}
	if y.gradFn.inputs[0] != x {
		t.Fatal("checkpoint node's input is not the saved activation")
	}
	// Count every node reachable from y: the checkpoint node and x, and
	// nothing else. A retained segment graph would add ~4 more.
	seen := map[*Tensor]bool{}
	var walk func(*Tensor)
	walk = func(n *Tensor) {
		if seen[n] {
			return
		}
		seen[n] = true
		if n.gradFn != nil {
			for _, in := range n.gradFn.inputs {
				walk(in)
			}
		}
	}
	walk(y)
	if len(seen) != 2 {
		t.Fatalf("graph reachable from checkpoint output has %d nodes, want 2 (output + saved input)", len(seen))
	}
	// And the captured parameter must NOT be an input of the outer node:
	// the local backward accumulates into it, so listing it here would
	// double count.
	if seen[a] {
		t.Fatal("captured parameter is reachable from the outer graph — double accumulation")
	}
}

// TestCheckpointGradientAccumulatesAcrossMicroSteps — parameters keep
// .grad until the optimizer's ZeroGrad, and the recompute's local
// backward must ADD to it, not replace it. Two half-scaled micro-steps
// must equal one full-scale step.
func TestCheckpointGradientAccumulatesAcrossMicroSteps(t *testing.T) {
	a := scalarT(2, true)
	fn := func(in *Tensor) *Tensor { return Mul(Mul(in, a), in) } // a·x²

	// Reference: one pass at x=3 plus one at x=4, uncheckpointed.
	ref := scalarT(2, true)
	for _, xv := range []float32{3, 4} {
		x := scalarT(xv, true)
		Sum(Mul(Mul(x, ref), x)).Backward()
	}

	for _, xv := range []float32{3, 4} {
		x := scalarT(xv, true)
		Sum(Checkpoint("t", x, fn)).Backward()
	}

	// d(a·x²)/da = x² → 9 + 16 = 25.
	if got := gradVal(t, ref); !closeEnough(got, 25, 1e-6) {
		t.Fatalf("reference accumulated grad %v, want 25", got)
	}
	if got := gradVal(t, a); !closeEnough(got, gradVal(t, ref), 1e-6) {
		t.Fatalf("checkpointed accumulated grad %v != reference %v", got, gradVal(t, ref))
	}
}

// TestCheckpointSharedParameterAcrossSegments — the same leaf parameter
// used by two consecutive checkpoint segments must receive both
// contributions exactly once each (the review's "tied adapter" case).
func TestCheckpointSharedParameterAcrossSegments(t *testing.T) {
	run := func(ckpt bool) (*Tensor, *Tensor) {
		x, a := scalarT(3, true), scalarT(2, true)
		seg := func(in *Tensor) *Tensor { return Mul(in, a) }
		h := x
		for i := 0; i < 2; i++ {
			if ckpt {
				h = Checkpoint("seg", h, seg)
			} else {
				h = seg(h)
			}
		}
		Sum(h).Backward()
		return x, a
	}
	x0, a0 := run(false)
	x1, a1 := run(true)

	// y = a²x → dy/da = 2ax = 12, dy/dx = a² = 4.
	if got := gradVal(t, a0); !closeEnough(got, 12, 1e-6) {
		t.Fatalf("plain shared-param grad %v, want 12", got)
	}
	if got := gradVal(t, a1); !closeEnough(got, gradVal(t, a0), 1e-6) {
		t.Fatalf("checkpointed shared-param grad %v != plain %v", got, gradVal(t, a0))
	}
	if got := gradVal(t, x1); !closeEnough(got, gradVal(t, x0), 1e-6) {
		t.Fatalf("checkpointed dx %v != plain %v", got, gradVal(t, x0))
	}
}

// TestCheckpointBranchedInput — one activation consumed by two
// checkpoint segments. The outer engine sums both dx contributions;
// this catches an implementation that returned dx by writing x.grad
// itself instead of handing it back to the engine.
func TestCheckpointBranchedInput(t *testing.T) {
	run := func(ckpt bool) *Tensor {
		x := scalarT(3, true)
		sq := func(in *Tensor) *Tensor { return Mul(in, in) }
		cube := func(in *Tensor) *Tensor { return Mul(Mul(in, in), in) }
		var l, r *Tensor
		if ckpt {
			l, r = Checkpoint("l", x, sq), Checkpoint("r", x, cube)
		} else {
			l, r = sq(x), cube(x)
		}
		Sum(Add(l, r)).Backward()
		return x
	}
	// d(x² + x³)/dx at 3 = 6 + 27 = 33.
	if got := gradVal(t, run(false)); !closeEnough(got, 33, 1e-6) {
		t.Fatalf("plain branched grad %v, want 33", got)
	}
	if got := gradVal(t, run(true)); !closeEnough(got, 33, 1e-6) {
		t.Fatalf("checkpointed branched grad %v, want 33", got)
	}
}

// TestCheckpointNested — a checkpoint segment inside another segment's
// recompute. The inner Checkpoint is entered from inside enableGrad
// while the outer Backward's NoGrad scope is still on the stack, so
// this is the test that the grad-mode save/restore actually nests
// (an implementation using a bool instead of a saved depth returns to
// the WRONG mode here and the outer recompute silently stops building
// its graph).
func TestCheckpointNested(t *testing.T) {
	run := func(ckpt bool) (*Tensor, *Tensor) {
		x, a := scalarT(3, true), scalarT(2, true)
		inner := func(in *Tensor) *Tensor { return Mul(in, a) }
		outer := func(in *Tensor) *Tensor {
			if ckpt {
				in = Checkpoint("inner", in, inner)
			} else {
				in = inner(in)
			}
			return Mul(in, in)
		}
		var y *Tensor
		if ckpt {
			y = Checkpoint("outer", x, outer)
		} else {
			y = outer(x)
		}
		Sum(y).Backward()
		return x, a
	}
	x0, a0 := run(false)
	x1, a1 := run(true)
	// y = (ax)² → dy/dx = 2a²x = 24, dy/da = 2ax² = 36.
	if got := gradVal(t, x0); !closeEnough(got, 24, 1e-6) {
		t.Fatalf("plain nested dx %v, want 24", got)
	}
	if got := gradVal(t, x1); !closeEnough(got, gradVal(t, x0), 1e-6) {
		t.Fatalf("nested checkpoint dx %v != plain %v", got, gradVal(t, x0))
	}
	if got := gradVal(t, a1); !closeEnough(got, gradVal(t, a0), 1e-6) {
		t.Fatalf("nested checkpoint da %v != plain %v", got, gradVal(t, a0))
	}
}

// TestCheckpointStoredGradsCarryNoGraph — the invariant the whole
// NoGrad-wrapped Backward exists to protect (the 2026-08-12 unbounded-
// retention post-mortem). A recompute that leaked tracking into the
// stored gradients would pin every segment's graph through the
// parameters' .grad until ZeroGrad.
func TestCheckpointStoredGradsCarryNoGraph(t *testing.T) {
	x, a := scalarT(3, true), scalarT(2, true)
	Sum(Checkpoint("t", x, func(in *Tensor) *Tensor {
		u := Mul(in, a)
		return Mul(u, u)
	})).Backward()

	for name, p := range map[string]*Tensor{"x": x, "a": a} {
		if p.Grad() == nil {
			t.Fatalf("%s has no gradient", name)
		}
		if p.Grad().gradFn != nil {
			t.Fatalf("%s.grad owns a graph — the recompute leaked tracking", name)
		}
		if p.Grad().requiresGrad {
			t.Fatalf("%s.grad requires grad — the recompute leaked tracking", name)
		}
	}
}

// TestCheckpointRestoresGradMode — the recompute forces tracking on
// mid-backward; a panic must not leave the process-global counter
// corrupted, and normal exit must restore the enclosing NoGrad depth
// exactly.
func TestCheckpointRestoresGradMode(t *testing.T) {
	if !GradEnabled() {
		t.Fatal("test started with tracking already off")
	}
	x := scalarT(3, true)
	Sum(Checkpoint("t", x, func(in *Tensor) *Tensor { return Mul(in, in) })).Backward()
	if !GradEnabled() {
		t.Fatal("grad mode not restored after a checkpointed backward")
	}

	func() {
		defer func() { _ = recover() }()
		NoGrad(func() {
			enableGrad(func() { panic("boom") })
		})
	}()
	if !GradEnabled() {
		t.Fatal("grad mode not restored after a panic inside enableGrad")
	}
}

// TestCheckpointIsInertUnderNoGrad — inference must not pay for the
// GradFn or the retained input.
func TestCheckpointIsInertUnderNoGrad(t *testing.T) {
	x := scalarT(3, true)
	var y *Tensor
	NoGrad(func() {
		y = Checkpoint("t", x, func(in *Tensor) *Tensor { return Mul(in, in) })
	})
	if y.gradFn != nil {
		t.Fatal("Checkpoint built a node under NoGrad")
	}
	if got := y.Data()[0]; got != 9 {
		t.Fatalf("value %v, want 9", got)
	}
}

// TestCheckpointIdentitySegment — the degenerate closure that returns
// its argument. dx is the incoming gradient; a naive implementation
// that seeds y.grad and reads xl.grad afterwards returns nil here
// (y and xl are the same tensor), silently zeroing everything upstream.
func TestCheckpointIdentitySegment(t *testing.T) {
	x := scalarT(3, true)
	y := Checkpoint("id", x, func(in *Tensor) *Tensor { return in })
	Sum(Scale(y, 7)).Backward()
	if got := gradVal(t, x); !closeEnough(got, 7, 1e-6) {
		t.Fatalf("identity-segment dx %v, want 7", got)
	}
}

// TestCheckpointSegmentReturningLeaf — a closure that passes a captured
// parameter straight through. The recompute produces no graph to walk,
// so a naive save/restore of the root's gradient drops the segment's
// whole contribution to that parameter. Uncheckpointed, it gets it.
func TestCheckpointSegmentReturningLeaf(t *testing.T) {
	run := func(ckpt bool) *Tensor {
		x, p := scalarT(3, true), scalarT(2, true)
		fn := func(_ *Tensor) *Tensor { return p }
		var y *Tensor
		if ckpt {
			y = Checkpoint("leaf", x, fn)
		} else {
			y = fn(x)
		}
		Sum(Scale(y, 5)).Backward()
		return p
	}
	if got := gradVal(t, run(false)); !closeEnough(got, 5, 1e-6) {
		t.Fatalf("plain passthrough grad %v, want 5", got)
	}
	if got := gradVal(t, run(true)); !closeEnough(got, 5, 1e-6) {
		t.Fatalf("checkpointed passthrough grad %v, want 5", got)
	}

	// And when the leaf already carries an accumulated gradient, the
	// segment's contribution must ADD to it, not replace it.
	x, p := scalarT(3, true), scalarT(2, true)
	p.grad = NewTensor([]float32{100}, 1, 1)
	Sum(Scale(Checkpoint("leaf", x, func(_ *Tensor) *Tensor { return p }), 5)).Backward()
	if got := gradVal(t, p); !closeEnough(got, 105, 1e-6) {
		t.Fatalf("accumulated passthrough grad %v, want 105", got)
	}
}

// TestCheckpointMatrixSegment exercises the shapes a real block sees
// (matmul + nonlinearity, not just scalars) and checks every element.
func TestCheckpointMatrixSegment(t *testing.T) {
	build := func(ckpt bool) (x, w *Tensor) {
		x = NewTensor([]float32{0.1, -0.2, 0.3, 0.4, -0.5, 0.6}, 3, 2)
		x.SetRequiresGrad(true)
		w = NewTensor([]float32{0.7, -0.8, 0.9, 1.1}, 2, 2)
		w.SetRequiresGrad(true)
		fn := func(in *Tensor) *Tensor { return Tanh(MatMul(in, w)) }
		var y *Tensor
		if ckpt {
			y = Checkpoint("m", x, fn)
		} else {
			y = fn(x)
		}
		Sum(Mul(y, y)).Backward()
		return x, w
	}
	x0, w0 := build(false)
	x1, w1 := build(true)
	for name, pair := range map[string][2]*Tensor{"dx": {x0, x1}, "dw": {w0, w1}} {
		a, b := pair[0].Grad().Data(), pair[1].Grad().Data()
		for i := range a {
			if !closeEnough(b[i], a[i], 1e-6) {
				t.Fatalf("%s[%d]: checkpointed %v != plain %v", name, i, b[i], a[i])
			}
		}
	}
}

// TestCheckpointSegmentDoneHook — the trainer relies on this firing at
// BOTH points where a segment's intermediates die: after the no-grad
// forward and after the recompute. Missing the forward half is the
// regression that made checkpointing save 30% instead of an order of
// magnitude on the Metal path (see CheckpointSegmentDone).
func TestCheckpointSegmentDoneHook(t *testing.T) {
	forward, total := 0, 0
	CheckpointSegmentDone = func() { total++ }
	defer func() { CheckpointSegmentDone = nil }()

	x := scalarT(3, true)
	h := x
	for i := 0; i < 3; i++ {
		h = Checkpoint("seg", h, func(in *Tensor) *Tensor { return Mul(in, in) })
	}
	forward = total
	if forward != 3 {
		t.Fatalf("segment hook fired %d times during the forward, want 3", forward)
	}
	Sum(h).Backward()
	if total-forward != 3 {
		t.Fatalf("segment hook fired %d times during backward, want 3", total-forward)
	}
}
