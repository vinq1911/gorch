//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

func TestRMSNormForwardSpec(t *testing.T) {
	// Hand-computed reference for a 2x4 input with gamma=ones.
	// x = [[1, 2, 3, 4], [-1, 0, 1, 2]]
	// row 0: ms = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5) ≈ 2.7386
	// row 1: ms = (1+0+1+4)/4 = 1.5, rms = sqrt(1.5) ≈ 1.2247
	rn := NewRMSNorm(4)
	rn.Eps = 0 // disable epsilon for clean numerical match
	x := g.NewTensor([]float32{1, 2, 3, 4, -1, 0, 1, 2}, 2, 4)

	y := rn.Forward(x)
	want := []float32{
		1 / 2.7386128, 2 / 2.7386128, 3 / 2.7386128, 4 / 2.7386128,
		-1 / 1.2247449, 0 / 1.2247449, 1 / 1.2247449, 2 / 1.2247449,
	}
	for i, w := range want {
		if math.Abs(float64(y.Data()[i]-w)) > 1e-5 {
			t.Fatalf("[%d]: got %g, want %g", i, y.Data()[i], w)
		}
	}
}

func TestRMSNormGammaScaling(t *testing.T) {
	// With gamma = 2*ones, output should be exactly 2x the gamma=ones output.
	rn1 := NewRMSNorm(8)
	rn2 := NewRMSNorm(8)
	for i := range rn2.Weight.Data() {
		rn2.Weight.Data()[i] = 2.0
	}
	x := g.RandN(3, 8)
	y1 := rn1.Forward(x)
	y2 := rn2.Forward(x)
	for i := range y1.Data() {
		want := y1.Data()[i] * 2
		if math.Abs(float64(y2.Data()[i]-want)) > 1e-5 {
			t.Fatalf("[%d]: gamma=2 got %g, want %g (= 2 * gamma=1)", i, y2.Data()[i], want)
		}
	}
}

// TestRMSNormBackwardMatchesNumerical compares the analytical gradient
// against a numerical (finite-difference) approximation. This is the
// gold-standard correctness check for any new autograd op — if it
// passes, the closed-form backward is right.
func TestRMSNormBackwardMatchesNumerical(t *testing.T) {
	rn := NewRMSNorm(5)
	for i := range rn.Weight.Data() {
		rn.Weight.Data()[i] = 1 + float32(i)*0.3 // non-trivial gamma
	}
	M, N := 3, 5
	x := g.RandN(M, N).SetRequiresGrad(true)

	// Analytical gradient via Backward on a scalar loss = sum(y).
	y := rn.Forward(x)
	loss := g.Sum(y)
	loss.Backward()
	dxAnalytic := append([]float32{}, x.Grad().Data()...)
	dwAnalytic := append([]float32{}, rn.Weight.Grad().Data()...)

	// Numerical gradient: dL/dx[i] ≈ (L(x + h*e_i) - L(x - h*e_i)) / (2h)
	const h = 1e-3
	xData := x.Data()

	// Build a fresh RMSNorm with same gamma for the numerical loop so
	// that x.Grad accumulation in analytical phase doesn't interfere.
	rn2 := NewRMSNorm(N)
	copy(rn2.Weight.Data(), rn.Weight.Data())

	dxNum := make([]float32, M*N)
	for i := range dxNum {
		orig := xData[i]
		xData[i] = orig + h
		yPlus := g.Sum(rn2.Forward(x)).Data()[0]
		xData[i] = orig - h
		yMinus := g.Sum(rn2.Forward(x)).Data()[0]
		xData[i] = orig
		dxNum[i] = (yPlus - yMinus) / (2 * h)
	}

	// Numerical wrt gamma.
	dwNum := make([]float32, N)
	wData := rn2.Weight.Data()
	for j := range dwNum {
		orig := wData[j]
		wData[j] = orig + h
		yPlus := g.Sum(rn2.Forward(x)).Data()[0]
		wData[j] = orig - h
		yMinus := g.Sum(rn2.Forward(x)).Data()[0]
		wData[j] = orig
		dwNum[j] = (yPlus - yMinus) / (2 * h)
	}

	check := func(label string, analytic, numeric []float32, tol float64) {
		t.Helper()
		for i := range analytic {
			d := math.Abs(float64(analytic[i] - numeric[i]))
			rel := d / (math.Abs(float64(numeric[i])) + 1e-6)
			if d > tol && rel > tol {
				t.Fatalf("%s[%d]: analytic=%g numeric=%g abs=%g rel=%g",
					label, i, analytic[i], numeric[i], d, rel)
			}
		}
	}
	check("dx", dxAnalytic, dxNum, 1e-2)
	check("dW", dwAnalytic, dwNum, 1e-2)
}

// TestRMSNormNoGradSkipsBackwardCache: in NoGrad, the per-row invRMS
// allocation should be skipped. Hard to assert directly; instead just
// check that Forward inside NoGrad produces a tensor with no graph.
func TestRMSNormNoGradSkipsBackwardCache(t *testing.T) {
	rn := NewRMSNorm(8)
	x := g.RandN(2, 8).SetRequiresGrad(true)
	var y *g.Tensor
	g.NoGrad(func() {
		y = rn.Forward(x)
	})
	if y.RequiresGrad() {
		t.Fatal("RMSNorm output inside NoGrad still requires grad")
	}
}

// TestRMSNormParameters: make sure Parameters() returns exactly gamma.
// Important for optimiser composition with the rest of a transformer.
func TestRMSNormParameters(t *testing.T) {
	rn := NewRMSNorm(16)
	params := rn.Parameters()
	if len(params) != 1 {
		t.Fatalf("got %d parameters, want 1 (just gamma)", len(params))
	}
	if params[0] != rn.Weight {
		t.Fatal("first parameter is not gamma")
	}
}
