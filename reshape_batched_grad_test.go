//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// TestReshapePropagatesAutograd: t.Reshape(...) on a tensor with
// requires_grad must produce a tensor with autograd hooked up so
// that downstream Backward reaches the source's gradient.
func TestReshapePropagatesAutograd(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3).SetRequiresGrad(true)
	r := a.Reshape(3, 2)
	if !r.RequiresGrad() {
		t.Fatal("reshape on a requires_grad tensor lost autograd")
	}
	loss := Sum(r)
	loss.Backward()

	// Sum's grad on r is all-ones; reshape backward is identity-shape.
	// So a.grad should be all 1s.
	for i, v := range a.Grad().Data() {
		if v != 1 {
			t.Fatalf("[%d] a.grad = %g, want 1", i, v)
		}
	}
}

// TestReshapeBackwardMatchesNumerical: numerical-vs-analytical for
// the reshape op specifically.
func TestReshapeBackwardMatchesNumerical(t *testing.T) {
	a := RandN(2, 6).SetRequiresGrad(true)
	loss := Sum(Mul(a.Reshape(4, 3), Ones(4, 3)))
	loss.Backward()
	dxAnalytic := append([]float32{}, a.Grad().Data()...)

	const h = 1e-3
	for i := range a.Data() {
		orig := a.Data()[i]
		a.Data()[i] = orig + h
		yPlus := Sum(Mul(a.Reshape(4, 3), Ones(4, 3))).Data()[0]
		a.Data()[i] = orig - h
		yMinus := Sum(Mul(a.Reshape(4, 3), Ones(4, 3))).Data()[0]
		a.Data()[i] = orig
		num := (yPlus - yMinus) / (2 * h)
		if math.Abs(float64(dxAnalytic[i]-num)) > 1e-2 {
			t.Fatalf("[%d] analytic=%g numeric=%g", i, dxAnalytic[i], num)
		}
	}
}

// TestBatchedMatMulBackwardMatchesNumerical: the new autograd hook
// must agree with central-difference numerical gradient.
func TestBatchedMatMulBackwardMatchesNumerical(t *testing.T) {
	const B, M, K, N = 2, 3, 4, 5
	a := RandN(B, M, K).SetRequiresGrad(true)
	b := RandN(B, K, N).SetRequiresGrad(true)

	loss := Sum(BatchedMatMul(a, b, B, M, N, K))
	loss.Backward()
	daA := append([]float32{}, a.Grad().Data()...)
	dbA := append([]float32{}, b.Grad().Data()...)

	const h = 1e-3
	checkSlice := func(name string, x *Tensor, ana []float32) {
		t.Helper()
		for i := range x.Data() {
			orig := x.Data()[i]
			x.Data()[i] = orig + h
			yPlus := Sum(BatchedMatMul(a, b, B, M, N, K)).Data()[0]
			x.Data()[i] = orig - h
			yMinus := Sum(BatchedMatMul(a, b, B, M, N, K)).Data()[0]
			x.Data()[i] = orig
			num := (yPlus - yMinus) / (2 * h)
			if math.Abs(float64(ana[i]-num)) > 5e-2 {
				t.Fatalf("%s[%d] analytic=%g numeric=%g", name, i, ana[i], num)
			}
		}
	}
	checkSlice("dA", a, daA)
	checkSlice("dB", b, dbA)
}

// TestBatchedMatMulTransBBackwardMatchesNumerical: same for the
// transpose-B variant.
func TestBatchedMatMulTransBBackwardMatchesNumerical(t *testing.T) {
	const B, M, N, K = 2, 3, 5, 4
	// Forward: a (B, M, K), b (B, N, K), out (B, M, N)  with C = A @ B^T.
	a := RandN(B, M, K).SetRequiresGrad(true)
	b := RandN(B, N, K).SetRequiresGrad(true)

	loss := Sum(BatchedMatMulTransB(a, b, B, M, N, K))
	loss.Backward()
	daA := append([]float32{}, a.Grad().Data()...)
	dbA := append([]float32{}, b.Grad().Data()...)

	const h = 1e-3
	checkSlice := func(name string, x *Tensor, ana []float32) {
		t.Helper()
		for i := range x.Data() {
			orig := x.Data()[i]
			x.Data()[i] = orig + h
			yPlus := Sum(BatchedMatMulTransB(a, b, B, M, N, K)).Data()[0]
			x.Data()[i] = orig - h
			yMinus := Sum(BatchedMatMulTransB(a, b, B, M, N, K)).Data()[0]
			x.Data()[i] = orig
			num := (yPlus - yMinus) / (2 * h)
			if math.Abs(float64(ana[i]-num)) > 5e-2 {
				t.Fatalf("%s[%d] analytic=%g numeric=%g", name, i, ana[i], num)
			}
		}
	}
	checkSlice("dA", a, daA)
	checkSlice("dB", b, dbA)
}
