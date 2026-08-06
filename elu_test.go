//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// TestELUClosedForm: spot-check values against the alpha=1 definition.
func TestELUClosedForm(t *testing.T) {
	in := NewTensor([]float32{-2, -1, 0, 0.5, 3}, 5)
	out := ELU(in)
	want := []float64{
		math.Exp(-2) - 1, // -0.864664...
		math.Exp(-1) - 1, // -0.632120...
		0,
		0.5,
		3,
	}
	for i, w := range want {
		if math.Abs(float64(out.Data()[i])-w) > 1e-6 {
			t.Fatalf("[%d] ELU = %g, want %g", i, out.Data()[i], w)
		}
	}
}

// TestGELUErfClosedForm: spot-check values against 0.5*x*(1+erf(x/√2)).
func TestGELUErfClosedForm(t *testing.T) {
	xs := []float32{-3, -1, 0, 1, 2.5}
	in := NewTensor(xs, len(xs))
	out := GELUErf(in)
	for i, x := range xs {
		w := 0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))
		if math.Abs(float64(out.Data()[i])-w) > 1e-6 {
			t.Fatalf("[%d] GELUErf(%g) = %g, want %g", i, x, out.Data()[i], w)
		}
	}
	// Known reference point: GELU(1) = Phi(1) ≈ 0.8413447.
	one := GELUErf(NewTensor([]float32{1}, 1)).Data()[0]
	if math.Abs(float64(one)-0.8413447) > 1e-6 {
		t.Fatalf("GELUErf(1) = %g, want 0.8413447", one)
	}
}

// TestGELUErfDiffersFromTanhGELU guards plan 0006 risk #3: the exact
// erf GELU must differ measurably from the tanh approximation (an
// accidental substitution would make the diff exactly zero), while
// still tracking it closely.
func TestGELUErfDiffersFromTanhGELU(t *testing.T) {
	n := 2001
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = -5 + 10*float32(i)/float32(n-1)
	}
	in := NewTensor(xs, n)
	exact := GELUErf(in)
	approx := GELU(in)

	var maxDiff float64
	for i := range xs {
		d := math.Abs(float64(exact.Data()[i] - approx.Data()[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	// The tanh approximation peaks around |x|≈2 with ~3e-4 error.
	if maxDiff < 1e-4 {
		t.Fatalf("max |GELUErf - GELU(tanh)| = %g; too small — did GELUErf fall back to the tanh approximation?", maxDiff)
	}
	if maxDiff > 1e-2 {
		t.Fatalf("max |GELUErf - GELU(tanh)| = %g; too large — one of the implementations is wrong", maxDiff)
	}
}

// TestELUBackwardMatchesNumerical: analytic vs central-difference
// gradients (repo convention, see reshape_batched_grad_test.go).
func TestELUBackwardMatchesNumerical(t *testing.T) {
	a := RandN(3, 7).SetRequiresGrad(true)
	mask := RandN(3, 7)
	loss := Sum(Mul(ELU(a), mask))
	loss.Backward()
	ana := append([]float32{}, a.Grad().Data()...)

	const h = 1e-3
	for i := range a.Data() {
		orig := a.Data()[i]
		a.Data()[i] = orig + h
		yPlus := Sum(Mul(ELU(a), mask)).Data()[0]
		a.Data()[i] = orig - h
		yMinus := Sum(Mul(ELU(a), mask)).Data()[0]
		a.Data()[i] = orig
		num := (yPlus - yMinus) / (2 * h)
		if math.Abs(float64(ana[i]-num)) > 1e-2 {
			t.Fatalf("[%d] analytic=%g numeric=%g", i, ana[i], num)
		}
	}
}

// TestGELUErfBackwardMatchesNumerical: same for the exact GELU.
func TestGELUErfBackwardMatchesNumerical(t *testing.T) {
	a := RandN(3, 7).SetRequiresGrad(true)
	mask := RandN(3, 7)
	loss := Sum(Mul(GELUErf(a), mask))
	loss.Backward()
	ana := append([]float32{}, a.Grad().Data()...)

	const h = 1e-3
	for i := range a.Data() {
		orig := a.Data()[i]
		a.Data()[i] = orig + h
		yPlus := Sum(Mul(GELUErf(a), mask)).Data()[0]
		a.Data()[i] = orig - h
		yMinus := Sum(Mul(GELUErf(a), mask)).Data()[0]
		a.Data()[i] = orig
		num := (yPlus - yMinus) / (2 * h)
		if math.Abs(float64(ana[i]-num)) > 1e-2 {
			t.Fatalf("[%d] analytic=%g numeric=%g", i, ana[i], num)
		}
	}
}
