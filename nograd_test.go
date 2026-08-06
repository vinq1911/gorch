//go:build darwin

package gorch

import (
	"testing"
)

// TestNoGradSkipsAutogradConstruction: under NoGrad, ops must not
// build the autograd graph — output tensors should have
// requiresGrad=false and gradFn=nil even when their inputs require
// grad. That's what frees activations to be GC'd between forwards.
func TestNoGradSkipsAutogradConstruction(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, 2, 2).SetRequiresGrad(true)
	b := NewTensor([]float32{5, 6, 7, 8}, 2, 2).SetRequiresGrad(true)

	// Outside NoGrad: graph IS built.
	out := MatMul(a, b)
	if !out.RequiresGrad() {
		t.Error("outside NoGrad: MatMul output should require grad")
	}
	if out.gradFn == nil {
		t.Error("outside NoGrad: MatMul output should have gradFn")
	}

	// Inside NoGrad: no graph.
	NoGrad(func() {
		out2 := MatMul(a, b)
		if out2.RequiresGrad() {
			t.Error("inside NoGrad: MatMul output should not require grad")
		}
		if out2.gradFn != nil {
			t.Error("inside NoGrad: MatMul output should have nil gradFn")
		}
	})

	// Outside again: graph re-builds.
	out3 := Add(a, b)
	if !out3.RequiresGrad() {
		t.Error("after NoGrad: Add output should require grad again")
	}
}

// TestNoGradPreservesNumericalResult: the output values must be the
// same with and without NoGrad — only the autograd metadata differs.
func TestNoGradPreservesNumericalResult(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, 2, 2).SetRequiresGrad(true)
	b := NewTensor([]float32{0.5, -0.5, 1, 2}, 2, 2).SetRequiresGrad(true)

	withGrad := MatMul(a, b)
	var withoutGrad *Tensor
	NoGrad(func() {
		withoutGrad = MatMul(a, b)
	})

	for i := range withGrad.Data() {
		if withGrad.Data()[i] != withoutGrad.Data()[i] {
			t.Fatalf("idx %d: with=%g without=%g", i, withGrad.Data()[i], withoutGrad.Data()[i])
		}
	}
}

// TestNoGradIsNested: nested NoGrad scopes don't re-enable autograd
// in the inner scope.
func TestNoGradIsNested(t *testing.T) {
	a := NewTensor([]float32{1, 2}, 2).SetRequiresGrad(true)

	NoGrad(func() {
		if GradEnabled() {
			t.Fatal("inside NoGrad, GradEnabled should be false")
		}
		NoGrad(func() {
			if GradEnabled() {
				t.Fatal("inside nested NoGrad, GradEnabled should be false")
			}
			out := Neg(a)
			if out.RequiresGrad() {
				t.Fatal("inside nested NoGrad, op output should not require grad")
			}
		})
		if GradEnabled() {
			t.Fatal("after inner NoGrad, outer scope should still be NoGrad")
		}
	})
	if !GradEnabled() {
		t.Fatal("after both NoGrad scopes exit, GradEnabled should be true")
	}
}
