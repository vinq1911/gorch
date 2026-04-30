//go:build darwin

package gorch

import "testing"

// TestDetachStripsAutograd: a detached tensor must have requires_grad
// false and gradFn nil even when the source has both set. The shape
// is copied so detached's reshape doesn't mutate the source's shape.
func TestDetachStripsAutograd(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, 2, 2).SetRequiresGrad(true)
	b := NewTensor([]float32{5, 6, 7, 8}, 2, 2).SetRequiresGrad(true)
	out := MatMul(a, b) // builds the graph

	if !out.RequiresGrad() || out.gradFn == nil {
		t.Fatal("source tensor should track grad before Detach")
	}
	d := out.Detach()
	if d.RequiresGrad() {
		t.Fatal("detached tensor still tracks grad")
	}
	if d.gradFn != nil {
		t.Fatal("detached tensor still has gradFn")
	}
}

// TestDetachSharesStorage: mutating the detached tensor's data writes
// through to the source's data. This is the contract — Detach is a
// non-tracking handle to the same memory, NOT a copy.
func TestDetachSharesStorage(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, 4)
	d := a.Detach()
	d.Data()[0] = 42
	if a.Data()[0] != 42 {
		t.Fatalf("Detach didn't share storage: source[0] = %g, want 42", a.Data()[0])
	}
}

// TestDetachIsGoroutineLocal: a goroutine that uses Detach to opt out
// of autograd should not affect a concurrent training step in another
// goroutine. With Detach, the global NoGrad counter is never touched,
// so this is automatically goroutine-local. We verify by setting up
// a graph in goroutine B *while* A is doing inference via Detach, and
// confirming B's graph builds correctly.
func TestDetachIsGoroutineLocal(t *testing.T) {
	a := NewTensor([]float32{1, 2}, 2).SetRequiresGrad(true)
	b := NewTensor([]float32{3, 4}, 2).SetRequiresGrad(true)

	// Goroutine "inference path" — uses Detach. We just ensure the
	// detached tensor doesn't track grad regardless of timing.
	doneInfer := make(chan struct{})
	go func() {
		defer close(doneInfer)
		for i := 0; i < 100; i++ {
			d := a.Detach()
			if d.RequiresGrad() {
				t.Errorf("detached requires grad in concurrent goroutine")
				return
			}
		}
	}()

	// Goroutine "training path" — keeps building the autograd graph
	// concurrently. Each MatMul should produce a tensor that tracks grad.
	doneTrain := make(chan struct{})
	go func() {
		defer close(doneTrain)
		for i := 0; i < 100; i++ {
			c := NewTensor([]float32{float32(i), float32(i + 1)}, 1, 2).SetRequiresGrad(true)
			out := Mul(c, NewTensor([]float32{1, 1}, 1, 2))
			_ = a
			_ = b
			if !out.RequiresGrad() {
				t.Errorf("training path lost grad mid-run")
				return
			}
		}
	}()

	<-doneInfer
	<-doneTrain
}
