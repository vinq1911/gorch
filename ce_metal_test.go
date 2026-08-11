//go:build darwin

package gorch

import (
	"math"
	"math/rand"
	"testing"
)

// Golden tests for the fused cross-entropy (plan 0009 K2), written
// BEFORE the Metal kernels per the §4.2 Azure-codex protocol. The
// oracle is the pre-K2 loss.go implementation, reproduced here as the
// composed reference: loss = -mean(LogSoftmax(logits)[i, tgt[i]]),
// backward = softmax(logits) − onehot, scaled by grad/batch. The K2
// paths (vectorized CPU with saved logsumexp; Metal kernels) must
// match it, and the analytic grad must match central differences at
// the plan's 1e-2 relative tolerance. Metal-vs-CPU parity uses the
// min-over-attempts retry discipline (csRetry, softmax_metal_test.go).

// ceComposedReference computes loss and dLogits exactly the way the
// pre-K2 loss.go did, via the still-existing LogSoftmax/Softmax ops.
func ceComposedReference(logitsData []float32, tgt []int, batch, classes int) (loss float32, dx []float32) {
	x := NewTensor(logitsData, batch, classes)
	ls := LogSoftmax(x)
	var total float32
	for i := 0; i < batch; i++ {
		total -= ls.Data()[i*classes+tgt[i]]
	}
	loss = total / float32(batch)

	sm := Softmax(x)
	dx = make([]float32, batch*classes)
	scale := 1.0 / float32(batch) // upstream grad = 1
	for i := 0; i < batch; i++ {
		for j := 0; j < classes; j++ {
			dx[i*classes+j] = sm.Data()[i*classes+j] * scale
		}
		dx[i*classes+tgt[i]] -= scale
	}
	return loss, dx
}

func ceRandCase(rng *rand.Rand, batch, classes int) (logits []float32, tgt []int) {
	logits = make([]float32, batch*classes)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64()) * 2.0
	}
	tgt = make([]int, batch)
	for i := range tgt {
		tgt[i] = rng.Intn(classes)
	}
	return logits, tgt
}

// runCE runs the public CrossEntropyLoss end to end and returns the
// scalar loss and dLogits.
func runCE(logits *Tensor, tgt []int, batch int) (float32, []float32) {
	tt := Zeros(batch, 1)
	for i, c := range tgt {
		tt.Data()[i] = float32(c)
	}
	logits.SetRequiresGrad(true)
	loss := CrossEntropyLoss(logits, tt)
	loss.Backward()
	return loss.Data()[0], logits.Grad().Data()
}

// TestCrossEntropyCPUMatchesComposedReference: the K2 vectorized CPU
// path (the kernel's oracle) against the composed LogSoftmax/Softmax
// chain.
func TestCrossEntropyCPUMatchesComposedReference(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, tc := range []struct{ batch, classes int }{
		{7, 300},   // classes > threadgroup size on GPU later
		{3, 1000},  // long rows
		{16, 64},   // short rows, most lanes idle later
		{4, 20000}, // deep strided loops (vocab-scale direction)
	} {
		logits, tgt := ceRandCase(rng, tc.batch, tc.classes)

		refLoss, refDX := ceComposedReference(logits, tgt, tc.batch, tc.classes)
		gotLoss, gotDX := runCE(NewTensor(logits, tc.batch, tc.classes), tgt, tc.batch)

		if d := math.Abs(float64(refLoss - gotLoss)); d > 1e-4 {
			t.Fatalf("batch=%d classes=%d: loss %g vs ref %g (|Δ| %.3g > 1e-4)",
				tc.batch, tc.classes, gotLoss, refLoss, d)
		}
		if d := csMaxAbsDiff(refDX, gotDX); d > 1e-6 {
			t.Fatalf("batch=%d classes=%d: dLogits max diff %.3g > 1e-6", tc.batch, tc.classes, d)
		}
	}
}

// TestCrossEntropyNumericalGrad checks the analytic dLogits against
// central differences at 1e-2 relative (plan tolerance).
func TestCrossEntropyNumericalGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	batch, classes := 4, 37
	logits, tgt := ceRandCase(rng, batch, classes)

	// Float64 reference loss for the central difference — evaluating
	// the f32 op at ±h would put f32 rounding noise (~1e-7) over the
	// 2h divisor and swamp small gradient entries.
	lossAt := func(data []float32) float64 {
		var total float64
		for i := 0; i < batch; i++ {
			row := data[i*classes : (i+1)*classes]
			rowMax := float64(row[0])
			for _, v := range row[1:] {
				if float64(v) > rowMax {
					rowMax = float64(v)
				}
			}
			var sumExp float64
			for _, v := range row {
				sumExp += math.Exp(float64(v) - rowMax)
			}
			total += rowMax + math.Log(sumExp) - float64(row[tgt[i]])
		}
		return total / float64(batch)
	}

	_, analytic := runCE(NewTensor(logits, batch, classes), tgt, batch)

	const h = 1e-3
	n := batch * classes
	for _, idx := range []int{0, 5, 36, 37, 40, tgt[0], batch * classes / 2, n - 1} {
		plus := append([]float32(nil), logits...)
		minus := append([]float32(nil), logits...)
		plus[idx] += h
		minus[idx] -= h
		num := (lossAt(plus) - lossAt(minus)) / (2 * h)
		got := float64(analytic[idx])
		denom := math.Max(math.Abs(num), 1e-3)
		if rel := math.Abs(got-num) / denom; rel > 1e-2 {
			t.Errorf("numerical grad mismatch at %d: analytic %.6g vs numerical %.6g (rel %.3g)",
				idx, got, num, rel)
		}
	}
}

// TestCrossEntropyMetalMatchesCPU is the K2 kernel gate: the Metal
// forward (loss + per-row logsumexp) and backward must match the
// vectorized CPU oracle. Loss at 1e-3 abs (magnitude ~log C ≈ 10),
// dLogits at 1e-5 abs (values bounded by grad/batch), retry
// discipline per plan R5.
func TestCrossEntropyMetalMatchesCPU(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !cePipelinesReady() {
		t.Fatal("cross-entropy pipelines not compiled by InitMetal")
	}
	rng := rand.New(rand.NewSource(43))

	for _, tc := range []struct{ batch, classes int }{
		{7, 300},   // strided loops (classes > 256)
		{16, 64},   // short rows
		{3, 20000}, // vocab-scale strided depth
		{257, 129}, // rows > threadgroup count sanity, odd sizes
	} {
		logits, tgt := ceRandCase(rng, tc.batch, tc.classes)

		// CPU oracle (fused reference).
		xCPU := NewTensor(logits, tc.batch, tc.classes)
		refTotal, refLse := ceForwardCPU(xCPU, tgt, tc.batch, tc.classes)
		refDX := ceBackwardCPU(xCPU, refLse, tgt, tc.batch, tc.classes, 1.0/float32(tc.batch))

		csRetry(t, "cross_entropy_forward", func() (float64, bool) {
			xGPU := NewTensorOnMetal(gpuHandle.Dev, logits, tc.batch, tc.classes)
			total, lse := ceForwardMetal(xGPU, tgt, tc.batch, tc.classes)
			if !lse.IsOnMetal() {
				return math.Inf(1), false
			}
			d := math.Abs(float64(total-refTotal)) / float64(tc.batch)
			if d2 := csMaxAbsDiff(refLse, lse.Data()); d2 > d {
				d = d2
			}
			return d, d <= 1e-3
		})

		csRetry(t, "cross_entropy_backward", func() (float64, bool) {
			xGPU := NewTensorOnMetal(gpuHandle.Dev, logits, tc.batch, tc.classes)
			_, lse := ceForwardMetal(xGPU, tgt, tc.batch, tc.classes)
			dxGPU := ceBackwardMetal(xGPU, lse, tgt, tc.batch, tc.classes, 1.0/float32(tc.batch))
			if !dxGPU.IsOnMetal() {
				return math.Inf(1), false
			}
			d := csMaxAbsDiff(refDX.Data(), dxGPU.Data())
			return d, d <= 1e-5
		})
	}
}

// TestCrossEntropyAutogradEndToEndMetal exercises the full public op
// with Metal-resident logits (threshold lowered so the small test
// shape dispatches) — the integration shape the lm_head chain uses —
// and checks against the composed CPU reference.
func TestCrossEntropyAutogradEndToEndMetal(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	prev := CEMetalMinElements
	CEMetalMinElements = 1
	defer func() { CEMetalMinElements = prev }()

	rng := rand.New(rand.NewSource(59))
	batch, classes := 33, 4111
	logits, tgt := ceRandCase(rng, batch, classes)
	refLoss, refDX := ceComposedReference(logits, tgt, batch, classes)

	csRetry(t, "cross_entropy_autograd_metal", func() (float64, bool) {
		x := NewTensorOnMetal(gpuHandle.Dev, logits, batch, classes)
		gotLoss, gotDX := runCE(x, tgt, batch)
		d := math.Abs(float64(refLoss-gotLoss)) / 1e2 // loss scale headroom
		if d2 := csMaxAbsDiff(refDX, gotDX); d2 > d {
			d = d2
		}
		return d, d <= 1e-5
	})
}
