//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestMHABatchedAndPerHeadAgree: with the same input, weights, and
// no autograd, the batched-CPU forward and the per-head-loop forward
// must produce identical attention output. They previously
// disagreed because the batched path divided scores by headDim
// instead of sqrt(headDim).
//
// useBatched is selected based on x.RequiresGrad() in MHA.Forward;
// to exercise both paths, we run once outside NoGrad with x
// requires_grad=true (per-head loop) and once inside NoGrad with
// the same x (batched path).
func TestMHABatchedAndPerHeadAgree(t *testing.T) {
	const dim, numHeads, seqLen = 16, 4, 5
	mha := NewMultiHeadAttention(dim, numHeads)

	x := g.RandN(seqLen, dim)

	// Per-head loop path: triggered when x.RequiresGrad() is true.
	xGrad := g.NewTensor(x.Data(), seqLen, dim).SetRequiresGrad(true)
	yPerHead := mha.Forward(xGrad, seqLen)

	// Batched path: triggered when no autograd is built.
	var yBatched *g.Tensor
	g.NoGrad(func() {
		xBatched := g.NewTensor(x.Data(), seqLen, dim)
		yBatched = mha.Forward(xBatched, seqLen)
	})

	if !shapesEqualHeadScale(yPerHead.Shape(), yBatched.Shape()) {
		t.Fatalf("shape mismatch: per-head=%v batched=%v",
			yPerHead.Shape(), yBatched.Shape())
	}
	for i := range yPerHead.Data() {
		d := math.Abs(float64(yPerHead.Data()[i] - yBatched.Data()[i]))
		if d > 1e-3 {
			t.Fatalf("[%d] per-head=%g batched=%g diff=%g",
				i, yPerHead.Data()[i], yBatched.Data()[i], d)
		}
	}
}

func shapesEqualHeadScale(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
