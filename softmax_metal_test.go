//go:build darwin

package gorch

import (
	"math"
	"math/rand"
	"testing"
)

// Golden tests for the fused causal softmax (plan 0009 K1), written
// BEFORE the Metal kernels per the §4.2 Azure-codex protocol. The CPU
// reference is the composed op chain from ops.go/attention_ops.go
// (Full + Mul + MaskFill + Softmax) — the exact chain the fusion
// replaces in nn/gqa.go. Tolerances per plan: fwd/bwd parity 1e-3 abs
// (GPU vs CPU), analytic-vs-numerical grad 1e-2, with the
// min-over-attempts retry discipline from audio/mimi (a transient
// load-induced BLAS blip doesn't repeat; a real regression fails
// twice).

// csRetry runs check(); on failure it recomputes once and only fails
// if the failure reproduces.
func csRetry(t *testing.T, name string, check func() (float64, bool)) {
	t.Helper()
	if d, ok := check(); ok {
		t.Logf("%s: max diff %.3g (attempt 1)", name, d)
		return
	}
	t.Logf("%s: attempt 1 failed — recomputing once to rule out load-induced nondeterminism", name)
	d, ok := check()
	if !ok {
		t.Fatalf("%s: max diff %.3g exceeds tolerance on both attempts", name, d)
	}
	t.Logf("%s: max diff %.3g (attempt 2)", name, d)
}

func csMaxAbsDiff(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if math.IsNaN(float64(a[i])) || math.IsNaN(float64(b[i])) {
			return math.Inf(1)
		}
		if d > m {
			m = d
		}
	}
	return m
}

// csComposedReference builds softmax(maskfill(scale·x)) from the
// existing CPU ops — the pre-K1 nn/gqa.go chain, op for op — and
// returns (y, dx) for the given upstream grad w.
func csComposedReference(xData []float32, heads, qSeq, kSeq int, scale float32, w []float32) (y, dx []float32) {
	x := NewTensor(xData, heads, qSeq, kSeq).SetRequiresGrad(true)
	scaleVec := Full(scale, heads, qSeq, kSeq)
	scaled := Mul(x, scaleVec)
	flat := scaled.Reshape(heads*qSeq, kSeq)
	baseMask := CausalMask(kSeq)
	// Staircase for kSeq > qSeq: allowed j <= i + (kSeq - qSeq). For the
	// square case this is CausalMask exactly.
	offset := kSeq - qSeq
	fullMask := make([]bool, heads*qSeq*kSeq)
	for h := 0; h < heads; h++ {
		for i := 0; i < qSeq; i++ {
			for j := 0; j < kSeq; j++ {
				masked := j > i+offset
				if kSeq == qSeq {
					masked = baseMask[i*kSeq+j]
				}
				fullMask[(h*qSeq+i)*kSeq+j] = masked
			}
		}
	}
	masked := MaskFill(flat, fullMask, -1e9)
	soft := Softmax(masked)
	loss := Sum(Mul(soft, NewTensor(w, heads*qSeq, kSeq)))
	loss.Backward()
	return soft.Data(), x.Grad().Data()
}

func csRandSlice(rng *rand.Rand, n int, scale float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64()) * scale
	}
	return s
}

// TestCausalSoftmaxCPUMatchesComposedReference verifies the fused CPU
// path (the kernel's oracle) against the composed op chain, forward
// and backward.
func TestCausalSoftmaxCPUMatchesComposedReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, tc := range []struct{ heads, qSeq, kSeq int }{
		{4, 13, 13},   // small odd square
		{2, 7, 12},    // staircase (kSeq > qSeq, prefill window shape)
		{3, 300, 300}, // cols > threadgroup size, forces strided loops on GPU later
	} {
		scale := float32(1.0 / math.Sqrt(128))
		n := tc.heads * tc.qSeq * tc.kSeq
		xData := csRandSlice(rng, n, 2.0)
		w := csRandSlice(rng, n, 1.0)

		refY, refDX := csComposedReference(xData, tc.heads, tc.qSeq, tc.kSeq, scale, w)

		x := NewTensor(xData, tc.heads, tc.qSeq, tc.kSeq).SetRequiresGrad(true)
		y := CausalSoftmax(x, tc.heads, tc.qSeq, scale)
		loss := Sum(Mul(y, NewTensor(w, tc.heads, tc.qSeq, tc.kSeq)))
		loss.Backward()

		if d := csMaxAbsDiff(refY, y.Data()); d > 1e-6 {
			t.Fatalf("heads=%d qSeq=%d kSeq=%d: fused CPU forward diff %.3g > 1e-6", tc.heads, tc.qSeq, tc.kSeq, d)
		}
		if d := csMaxAbsDiff(refDX, x.Grad().Data()); d > 1e-5 {
			t.Fatalf("heads=%d qSeq=%d kSeq=%d: fused CPU backward diff %.3g > 1e-5", tc.heads, tc.qSeq, tc.kSeq, d)
		}
	}
}

// TestCausalSoftmaxNumericalGrad checks the fused CPU backward (the
// kernel's backward oracle) against central-difference numerical
// gradients at 1e-2 relative (plan tolerance).
func TestCausalSoftmaxNumericalGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	heads, qSeq, kSeq := 2, 5, 5
	scale := float32(0.25)
	n := heads * qSeq * kSeq
	xData := csRandSlice(rng, n, 1.5)
	w := csRandSlice(rng, n, 1.0)

	lossAt := func(data []float32) float64 {
		x := NewTensor(data, heads, qSeq, kSeq)
		y := CausalSoftmax(x, heads, qSeq, scale)
		var l float64
		for i, v := range y.Data() {
			l += float64(v) * float64(w[i])
		}
		return l
	}

	x := NewTensor(xData, heads, qSeq, kSeq).SetRequiresGrad(true)
	y := CausalSoftmax(x, heads, qSeq, scale)
	Sum(Mul(y, NewTensor(w, heads, qSeq, kSeq))).Backward()
	analytic := x.Grad().Data()

	const h = 1e-3
	for _, idx := range []int{0, 3, 6, 12, 24, 31, 40, n - 1} {
		plus := append([]float32(nil), xData...)
		minus := append([]float32(nil), xData...)
		plus[idx] += h
		minus[idx] -= h
		num := (lossAt(plus) - lossAt(minus)) / (2 * h)
		got := float64(analytic[idx])
		denom := math.Max(math.Abs(num), 1e-3)
		if rel := math.Abs(got-num) / denom; rel > 1e-2 {
			t.Errorf("numerical grad mismatch at %d: analytic %.6g vs numerical %.6g (rel %.3g)", idx, got, num, rel)
		}
	}
}

// TestCausalSoftmaxMetalMatchesCPU is the K1 kernel gate: the Metal
// forward and backward must match the fused CPU oracle within 1e-3 abs
// (plan tolerance, retry discipline).
func TestCausalSoftmaxMetalMatchesCPU(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !softmaxPipelinesReady() {
		t.Fatal("softmax pipelines not compiled by InitMetal")
	}
	rng := rand.New(rand.NewSource(23))

	for _, tc := range []struct{ heads, qSeq, kSeq int }{
		{4, 13, 13},   // limit < threadgroup size, most lanes idle
		{2, 7, 12},    // staircase offset
		{3, 300, 300}, // strided loops (cols > 256)
		{16, 64, 64},  // the GQA head-count at a small seq
	} {
		scale := float32(1.0 / math.Sqrt(128))
		n := tc.heads * tc.qSeq * tc.kSeq
		xData := csRandSlice(rng, n, 2.0)
		gData := csRandSlice(rng, n, 1.0)
		rows := tc.heads * tc.qSeq

		// CPU oracle (fused reference).
		xCPU := NewTensor(xData, tc.heads, tc.qSeq, tc.kSeq)
		yCPU := causalSoftmaxForwardCPU(xCPU, rows, tc.kSeq, tc.qSeq, scale)
		dxCPU := softmaxBackwardCPU(yCPU, NewTensor(gData, tc.heads, tc.qSeq, tc.kSeq), rows, tc.kSeq, scale)

		csRetry(t, "softmax_causal_forward", func() (float64, bool) {
			xGPU := NewTensorOnMetal(gpuHandle.Dev, xData, tc.heads, tc.qSeq, tc.kSeq)
			yGPU := softmaxCausalForwardMetal(xGPU, rows, tc.kSeq, tc.qSeq, scale)
			if !yGPU.IsOnMetal() {
				return math.Inf(1), false
			}
			d := csMaxAbsDiff(yCPU.Data(), yGPU.Data())
			return d, d <= 1e-3
		})

		csRetry(t, "softmax_backward", func() (float64, bool) {
			xGPU := NewTensorOnMetal(gpuHandle.Dev, xData, tc.heads, tc.qSeq, tc.kSeq)
			yGPU := softmaxCausalForwardMetal(xGPU, rows, tc.kSeq, tc.qSeq, scale)
			gGPU := NewTensorOnMetal(gpuHandle.Dev, gData, tc.heads, tc.qSeq, tc.kSeq)
			dxGPU := softmaxBackwardMetal(yGPU, gGPU, rows, tc.kSeq, scale)
			if !dxGPU.IsOnMetal() {
				return math.Inf(1), false
			}
			d := csMaxAbsDiff(dxCPU.Data(), dxGPU.Data())
			return d, d <= 1e-3
		})
	}
}

// TestCausalSoftmaxAutogradEndToEndMetal runs the full autograd op on
// Metal-resident input and checks grads against the composed CPU
// reference — the integration shape nn/gqa.go actually uses.
func TestCausalSoftmaxAutogradEndToEndMetal(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	rng := rand.New(rand.NewSource(31))
	heads, seq := 4, 96
	scale := float32(1.0 / math.Sqrt(64))
	n := heads * seq * seq
	xData := csRandSlice(rng, n, 1.0)
	w := csRandSlice(rng, n, 1.0)

	refY, refDX := csComposedReference(xData, heads, seq, seq, scale, w)

	csRetry(t, "causal_softmax_autograd_metal", func() (float64, bool) {
		x := NewTensorOnMetal(gpuHandle.Dev, xData, heads, seq, seq).SetRequiresGrad(true)
		x.ZeroGrad()
		y := CausalSoftmax(x, heads, seq, scale)
		if !y.IsOnMetal() {
			return math.Inf(1), false
		}
		Sum(Mul(y, NewTensorOnMetal(gpuHandle.Dev, w, heads, seq, seq))).Backward()
		d := csMaxAbsDiff(refY, y.Data())
		if d2 := csMaxAbsDiff(refDX, x.Grad().Data()); d2 > d {
			d = d2
		}
		return d, d <= 1e-3
	})
}
