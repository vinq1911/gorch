//go:build darwin

package gorch

import (
	"math"
	"math/rand"
	"testing"
)

// Golden tests for the K4 SiLU/SwiGLU kernels (plan 0009 X2) and their
// vectorized Accelerate fallbacks. The scalar reference is the exact
// pre-K4 silu.go math:
//
//	silu(x)   = x·σ(x)          silu'(x)  = σ(x)(1 + x(1−σ(x)))
//	swiglu    = gate·σ(gate)·v  d/dgate   = v·σ(g)(1+g(1−σ(g)))
//	                            d/dvalue  = g·σ(g)
//
// Tolerances per the plan table: fwd 1e-4 abs (elementwise, no
// reductions), analytic-vs-numerical grad 1e-2 rel, GPU-vs-CPU parity
// with the csRetry min-over-attempts discipline.

func siluScalarRef(x []float32) (y []float32) {
	y = make([]float32, len(x))
	for i, v := range x {
		s := float32(1.0 / (1.0 + math.Exp(float64(-v))))
		y[i] = v * s
	}
	return y
}

func siluScalarBwd(x, g []float32) (dx []float32) {
	dx = make([]float32, len(x))
	for i, v := range x {
		s := float32(1.0 / (1.0 + math.Exp(float64(-v))))
		dx[i] = g[i] * s * (1 + v*(1-s))
	}
	return dx
}

func swigluScalarRef(gate, val []float32) (y []float32) {
	y = make([]float32, len(gate))
	for i := range gate {
		s := float32(1.0 / (1.0 + math.Exp(float64(-gate[i]))))
		y[i] = gate[i] * s * val[i]
	}
	return y
}

func swigluScalarBwd(gate, val, g []float32) (dGate, dVal []float32) {
	dGate = make([]float32, len(gate))
	dVal = make([]float32, len(gate))
	for i := range gate {
		s := float32(1.0 / (1.0 + math.Exp(float64(-gate[i]))))
		dGate[i] = g[i] * val[i] * s * (1 + gate[i]*(1-s))
		dVal[i] = g[i] * gate[i] * s
	}
	return dGate, dVal
}

// TestSiLUSwiGLUCPUMatchesScalarReference: the vectorized Accelerate
// path against the scalar math.
func TestSiLUSwiGLUCPUMatchesScalarReference(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	n := 3072*4 + 17 // odd size: vector body + tail
	x := csRandSlice(rng, n, 2.0)
	v := csRandSlice(rng, n, 1.5)
	w := csRandSlice(rng, n, 1.0)

	// SiLU fwd+bwd through the public op (CPU path).
	xt := NewTensor(x, n).SetRequiresGrad(true)
	y := SiLU(xt)
	Sum(Mul(y, NewTensor(w, n))).Backward()
	if d := csMaxAbsDiff(siluScalarRef(x), y.Data()); d > 1e-4 {
		t.Fatalf("SiLU fwd diff %.3g > 1e-4", d)
	}
	if d := csMaxAbsDiff(siluScalarBwd(x, w), xt.Grad().Data()); d > 1e-4 {
		t.Fatalf("SiLU bwd diff %.3g > 1e-4", d)
	}

	// SwiGLU fwd+bwd.
	gt := NewTensor(x, n).SetRequiresGrad(true)
	vt := NewTensor(v, n).SetRequiresGrad(true)
	y2 := SwiGLU(gt, vt)
	Sum(Mul(y2, NewTensor(w, n))).Backward()
	refDG, refDV := swigluScalarBwd(x, v, w)
	if d := csMaxAbsDiff(swigluScalarRef(x, v), y2.Data()); d > 1e-4 {
		t.Fatalf("SwiGLU fwd diff %.3g > 1e-4", d)
	}
	if d := csMaxAbsDiff(refDG, gt.Grad().Data()); d > 1e-4 {
		t.Fatalf("SwiGLU dGate diff %.3g > 1e-4", d)
	}
	if d := csMaxAbsDiff(refDV, vt.Grad().Data()); d > 1e-4 {
		t.Fatalf("SwiGLU dValue diff %.3g > 1e-4", d)
	}
}

// TestSwiGLUNumericalGrad: analytic backward vs central differences at
// 1e-2 relative (float64 reference forward).
func TestSwiGLUNumericalGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	n := 41
	gate := csRandSlice(rng, n, 1.5)
	val := csRandSlice(rng, n, 1.5)
	w := csRandSlice(rng, n, 1.0)

	lossAt := func(g64, v64 []float32) float64 {
		var l float64
		for i := range g64 {
			s := 1.0 / (1.0 + math.Exp(-float64(g64[i])))
			l += float64(g64[i]) * s * float64(v64[i]) * float64(w[i])
		}
		return l
	}

	gt := NewTensor(gate, n).SetRequiresGrad(true)
	vt := NewTensor(val, n).SetRequiresGrad(true)
	Sum(Mul(SwiGLU(gt, vt), NewTensor(w, n))).Backward()

	const h = 1e-3
	check := func(name string, base []float32, other func(p []float32) float64, analytic []float32) {
		for _, idx := range []int{0, 7, n / 2, n - 1} {
			plus := append([]float32(nil), base...)
			minus := append([]float32(nil), base...)
			plus[idx] += h
			minus[idx] -= h
			num := (other(plus) - other(minus)) / (2 * h)
			got := float64(analytic[idx])
			denom := math.Max(math.Abs(num), 1e-3)
			if rel := math.Abs(got-num) / denom; rel > 1e-2 {
				t.Errorf("%s numerical grad mismatch at %d: analytic %.6g vs numerical %.6g (rel %.3g)",
					name, idx, got, num, rel)
			}
		}
	}
	check("dGate", gate, func(p []float32) float64 { return lossAt(p, val) }, gt.Grad().Data())
	check("dValue", val, func(p []float32) float64 { return lossAt(gate, p) }, vt.Grad().Data())
}

// TestSiLUSwiGLUMetalMatchesCPU is the K4 kernel gate: Metal fwd+bwd
// vs the scalar reference at 1e-3 abs (plan tolerance, retry
// discipline), including autograd end-to-end on Metal-resident inputs.
func TestSiLUSwiGLUMetalMatchesCPU(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !siluPipelinesReady() {
		t.Fatal("silu/swiglu pipelines not compiled by InitMetal")
	}
	rng := rand.New(rand.NewSource(23))
	n := 1024*3 + 7
	x := csRandSlice(rng, n, 2.0)
	v := csRandSlice(rng, n, 1.5)
	w := csRandSlice(rng, n, 1.0)

	refY := siluScalarRef(x)
	refDX := siluScalarBwd(x, w)
	refY2 := swigluScalarRef(x, v)
	refDG, refDV := swigluScalarBwd(x, v, w)

	csRetry(t, "silu_metal_autograd", func() (float64, bool) {
		xt := NewTensorOnMetal(gpuHandle.Dev, x, n).SetRequiresGrad(true)
		y := SiLU(xt)
		if !y.IsOnMetal() {
			return math.Inf(1), false
		}
		Sum(Mul(y, NewTensorOnMetal(gpuHandle.Dev, w, n))).Backward()
		d := csMaxAbsDiff(refY, y.Data())
		if d2 := csMaxAbsDiff(refDX, xt.Grad().Data()); d2 > d {
			d = d2
		}
		return d, d <= 1e-3
	})

	csRetry(t, "swiglu_metal_autograd", func() (float64, bool) {
		gt := NewTensorOnMetal(gpuHandle.Dev, x, n).SetRequiresGrad(true)
		vt := NewTensorOnMetal(gpuHandle.Dev, v, n).SetRequiresGrad(true)
		y := SwiGLU(gt, vt)
		if !y.IsOnMetal() {
			return math.Inf(1), false
		}
		Sum(Mul(y, NewTensorOnMetal(gpuHandle.Dev, w, n))).Backward()
		d := csMaxAbsDiff(refY2, y.Data())
		if d2 := csMaxAbsDiff(refDG, gt.Grad().Data()); d2 > d {
			d = d2
		}
		if d2 := csMaxAbsDiff(refDV, vt.Grad().Data()); d2 > d {
			d = d2
		}
		return d, d <= 1e-3
	})
}
