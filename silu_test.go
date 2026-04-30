//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// TestSiLUForwardSpec: a few hand-computed values.
//
//	silu(0)   = 0
//	silu(1)   ≈ 1 * 0.7311 = 0.7311
//	silu(-1)  ≈ -1 * 0.2689 = -0.2689
//	silu(10)  ≈ 10 * 0.99995 ≈ 9.9995 (saturated)
//	silu(-10) ≈ -10 * 4.5e-5 ≈ -0.00045
func TestSiLUForwardSpec(t *testing.T) {
	a := NewTensor([]float32{0, 1, -1, 10, -10}, 5)
	y := SiLU(a)
	want := []float32{0, 0.7310586, -0.26894143, 9.9995, -0.00045417}
	for i, w := range want {
		if math.Abs(float64(y.Data()[i]-w)) > 1e-3 {
			t.Fatalf("[%d] silu(%g) = %g, want %g", i, a.Data()[i], y.Data()[i], w)
		}
	}
}

func TestSiLUBackwardMatchesNumerical(t *testing.T) {
	a := RandN(20).SetRequiresGrad(true)

	y := SiLU(a)
	loss := Sum(y)
	loss.Backward()
	dxAnalytic := append([]float32{}, a.Grad().Data()...)

	const h = 1e-3
	for i := range a.data {
		orig := a.data[i]
		a.data[i] = orig + h
		yPlus := Sum(SiLU(a)).Data()[0]
		a.data[i] = orig - h
		yMinus := Sum(SiLU(a)).Data()[0]
		a.data[i] = orig
		dxNum := (yPlus - yMinus) / (2 * h)
		d := math.Abs(float64(dxAnalytic[i] - dxNum))
		rel := d / (math.Abs(float64(dxNum)) + 1e-6)
		if d > 1e-2 && rel > 1e-2 {
			t.Fatalf("[%d] analytic=%g numeric=%g abs=%g rel=%g",
				i, dxAnalytic[i], dxNum, d, rel)
		}
	}
}

// TestSwiGLUMatchesUnfused: the fused op should produce the same
// result as SiLU(gate) * value within fp32 noise.
func TestSwiGLUMatchesUnfused(t *testing.T) {
	gate := RandN(3, 8)
	value := RandN(3, 8)

	fused := SwiGLU(gate, value)
	unfused := Mul(SiLU(gate), value)

	for i := range fused.Data() {
		d := math.Abs(float64(fused.Data()[i] - unfused.Data()[i]))
		if d > 1e-5 {
			t.Fatalf("[%d] fused=%g unfused=%g diff=%g",
				i, fused.Data()[i], unfused.Data()[i], d)
		}
	}
}

// TestSwiGLUBackwardMatchesNumerical: numerical-vs-analytical for
// both gate and value.
func TestSwiGLUBackwardMatchesNumerical(t *testing.T) {
	gate := RandN(2, 5).SetRequiresGrad(true)
	value := RandN(2, 5).SetRequiresGrad(true)

	y := SwiGLU(gate, value)
	loss := Sum(y)
	loss.Backward()
	dGateAnalytic := append([]float32{}, gate.Grad().Data()...)
	dValueAnalytic := append([]float32{}, value.Grad().Data()...)

	const h = 1e-3
	check := func(label string, x *Tensor, analytic []float32) {
		t.Helper()
		for i := range x.data {
			orig := x.data[i]
			x.data[i] = orig + h
			yPlus := Sum(SwiGLU(gate, value)).Data()[0]
			x.data[i] = orig - h
			yMinus := Sum(SwiGLU(gate, value)).Data()[0]
			x.data[i] = orig
			num := (yPlus - yMinus) / (2 * h)
			d := math.Abs(float64(analytic[i] - num))
			rel := d / (math.Abs(float64(num)) + 1e-6)
			if d > 1e-2 && rel > 1e-2 {
				t.Fatalf("%s[%d] analytic=%g numeric=%g abs=%g rel=%g",
					label, i, analytic[i], num, d, rel)
			}
		}
	}
	check("dGate", gate, dGateAnalytic)
	check("dValue", value, dValueAnalytic)
}

func TestSwiGLUShapeMismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	gate := RandN(3, 8)
	value := RandN(3, 7) // mismatch
	SwiGLU(gate, value)
}

// TestSiLUNoGradSkipsBackward: NoGrad-wrapped SiLU should produce a
// tensor without graph metadata.
func TestSiLUNoGradSkipsBackward(t *testing.T) {
	a := RandN(8).SetRequiresGrad(true)
	var y *Tensor
	NoGrad(func() {
		y = SiLU(a)
	})
	if y.RequiresGrad() {
		t.Fatal("SiLU output inside NoGrad still requires grad")
	}
}

// BenchmarkSiLU vs BenchmarkSwiGLUFused-vs-Unfused: shows whether the
// fused op is worth keeping. Needs to be measurably faster than
// SiLU(gate) * value to justify the extra surface area.
func BenchmarkSiLU(b *testing.B) {
	x := RandN(1024, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SiLU(x)
	}
}

func BenchmarkSwiGLUFused(b *testing.B) {
	gate := RandN(1024, 768)
	value := RandN(1024, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SwiGLU(gate, value)
	}
}

func BenchmarkSwiGLUUnfused(b *testing.B) {
	gate := RandN(1024, 768)
	value := RandN(1024, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Mul(SiLU(gate), value)
	}
}
