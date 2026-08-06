//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// TestMeanAxisValues: hand-computed means for every axis of a 3-D tensor.
func TestMeanAxisValues(t *testing.T) {
	// shape (2, 2, 3)
	a := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,

		7, 8, 9,
		10, 11, 12,
	}, 2, 2, 3)

	m0 := MeanAxis(a, 0)
	if !sameShape(m0.Shape(), []int{2, 3}) {
		t.Fatalf("axis 0 shape %v", m0.Shape())
	}
	want0 := []float32{4, 5, 6, 7, 8, 9}
	for i, w := range want0 {
		if m0.Data()[i] != w {
			t.Fatalf("axis 0 [%d] = %g, want %g", i, m0.Data()[i], w)
		}
	}

	m1 := MeanAxis(a, 1)
	want1 := []float32{2.5, 3.5, 4.5, 8.5, 9.5, 10.5}
	for i, w := range want1 {
		if m1.Data()[i] != w {
			t.Fatalf("axis 1 [%d] = %g, want %g", i, m1.Data()[i], w)
		}
	}

	m2 := MeanAxis(a, 2)
	if !sameShape(m2.Shape(), []int{2, 2}) {
		t.Fatalf("axis 2 shape %v", m2.Shape())
	}
	want2 := []float32{2, 5, 8, 11}
	for i, w := range want2 {
		if m2.Data()[i] != w {
			t.Fatalf("axis 2 [%d] = %g, want %g", i, m2.Data()[i], w)
		}
	}
}

// TestVarAxisValues: biased and unbiased variance vs hand computation.
func TestVarAxisValues(t *testing.T) {
	a := NewTensor([]float32{
		1, 2,
		3, 4,
		5, 12,
	}, 3, 2) // columns: {1,3,5} and {2,4,12}

	// Population (unbiased=false), axis 0: mean {3,6};
	// col0: ((1-3)^2+(3-3)^2+(5-3)^2)/3 = 8/3; col1: (16+4+36)/3 = 56/3.
	vp := VarAxis(a, 0, false)
	if !sameShape(vp.Shape(), []int{2}) {
		t.Fatalf("shape %v", vp.Shape())
	}
	wantP := []float64{8.0 / 3, 56.0 / 3}
	for i, w := range wantP {
		if math.Abs(float64(vp.Data()[i])-w) > 1e-5 {
			t.Fatalf("population var[%d] = %g, want %g", i, vp.Data()[i], w)
		}
	}

	// Sample (unbiased=true): divide by n-1=2 → {4, 28}.
	vs := VarAxis(a, 0, true)
	wantS := []float64{4, 28}
	for i, w := range wantS {
		if math.Abs(float64(vs.Data()[i])-w) > 1e-5 {
			t.Fatalf("sample var[%d] = %g, want %g", i, vs.Data()[i], w)
		}
	}

	// axis 1 (rows), population: row {1,2} → 0.25, {3,4} → 0.25, {5,12} → 12.25.
	v1 := VarAxis(a, 1, false)
	want1 := []float64{0.25, 0.25, 12.25}
	for i, w := range want1 {
		if math.Abs(float64(v1.Data()[i])-w) > 1e-5 {
			t.Fatalf("axis1 var[%d] = %g, want %g", i, v1.Data()[i], w)
		}
	}
}

// TestMaxAxisValues: max over each axis, including a 1-D reduction to
// scalar shape (1,).
func TestMaxAxisValues(t *testing.T) {
	a := NewTensor([]float32{
		1, 9, 3,
		4, 5, -6,
	}, 2, 3)

	m0 := MaxAxis(a, 0)
	want0 := []float32{4, 9, 3}
	for i, w := range want0 {
		if m0.Data()[i] != w {
			t.Fatalf("axis 0 [%d] = %g, want %g", i, m0.Data()[i], w)
		}
	}

	m1 := MaxAxis(a, 1)
	want1 := []float32{9, 5}
	for i, w := range want1 {
		if m1.Data()[i] != w {
			t.Fatalf("axis 1 [%d] = %g, want %g", i, m1.Data()[i], w)
		}
	}

	s := MaxAxis(NewTensor([]float32{-3, -1, -7}, 3), 0)
	if !sameShape(s.Shape(), []int{1}) || s.Data()[0] != -1 {
		t.Fatalf("1-D max = %v %v, want shape [1] value -1", s.Shape(), s.Data())
	}
}

// TestMaxAxisGradRoutesToArgmax: gradient must land only on the argmax
// positions.
func TestMaxAxisGradRoutesToArgmax(t *testing.T) {
	a := NewTensor([]float32{
		1, 9, 3,
		4, 5, -6,
	}, 2, 3).SetRequiresGrad(true)
	Sum(MaxAxis(a, 1)).Backward()
	want := []float32{
		0, 1, 0, // row {1,9,3}: max 9 at index 1
		0, 1, 0, // row {4,5,-6}: max 5 at index 1
	}
	for i, w := range want {
		if a.Grad().Data()[i] != w {
			t.Fatalf("grad[%d] = %g, want %g", i, a.Grad().Data()[i], w)
		}
	}
}

// TestMeanAxisBackwardMatchesNumerical: analytic vs central-difference
// gradients over every axis (repo convention).
func TestMeanAxisBackwardMatchesNumerical(t *testing.T) {
	for axis := 0; axis < 3; axis++ {
		a := RandN(2, 3, 4).SetRequiresGrad(true)
		mask := RandN(MeanAxis(a.Detach(), axis).Shape()...)
		loss := func() *Tensor {
			return Sum(Mul(MeanAxis(a, axis), mask))
		}
		loss().Backward()
		ana := append([]float32{}, a.Grad().Data()...)

		const h = 1e-3
		for i := range a.Data() {
			orig := a.Data()[i]
			a.Data()[i] = orig + h
			yPlus := loss().Data()[0]
			a.Data()[i] = orig - h
			yMinus := loss().Data()[0]
			a.Data()[i] = orig
			num := (yPlus - yMinus) / (2 * h)
			if math.Abs(float64(ana[i]-num)) > 1e-2 {
				t.Fatalf("axis %d [%d]: analytic=%g numeric=%g", axis, i, ana[i], num)
			}
		}
	}
}

// TestVarAxisBackwardMatchesNumerical: same for VarAxis, both biased
// and unbiased.
func TestVarAxisBackwardMatchesNumerical(t *testing.T) {
	for _, unbiased := range []bool{false, true} {
		for axis := 0; axis < 3; axis++ {
			a := RandN(2, 3, 4).SetRequiresGrad(true)
			mask := RandN(VarAxis(a.Detach(), axis, unbiased).Shape()...)
			loss := func() *Tensor {
				return Sum(Mul(VarAxis(a, axis, unbiased), mask))
			}
			loss().Backward()
			ana := append([]float32{}, a.Grad().Data()...)

			const h = 1e-3
			for i := range a.Data() {
				orig := a.Data()[i]
				a.Data()[i] = orig + h
				yPlus := loss().Data()[0]
				a.Data()[i] = orig - h
				yMinus := loss().Data()[0]
				a.Data()[i] = orig
				num := (yPlus - yMinus) / (2 * h)
				if math.Abs(float64(ana[i]-num)) > 2e-2 {
					t.Fatalf("unbiased=%v axis %d [%d]: analytic=%g numeric=%g", unbiased, axis, i, ana[i], num)
				}
			}
		}
	}
}
