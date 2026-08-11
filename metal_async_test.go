//go:build darwin

package gorch

import (
	"math"
	"math/rand"
	"testing"
)

// R6 async-mode parity (plan 0009 X2): a training-shaped chain of
// GPU-dispatched ops (MPS matmuls, K1 causal softmax, K4 SwiGLU,
// elementwise kernels) interleaved with CPU-computed ops (Permute —
// a Go loop over unified memory) must produce IDENTICAL results with
// commit-without-wait dispatch, because every CPU read is fenced by
// syncForCPU/Data(). Identical means bit-exact: the same GPU commands
// run in the same order; only the host wait points move.

func asyncChainRun(t *testing.T, xData, w1Data, w2Data, wData []float32, heads, seq, dim int) (loss float32, dx, dw1 []float32) {
	t.Helper()
	dev := MetalDev()
	x := NewTensorOnMetal(dev, xData, seq, dim).SetRequiresGrad(true)
	w1 := NewTensorOnMetal(dev, w1Data, dim, heads*seq).SetRequiresGrad(true)
	w2 := NewTensorOnMetal(dev, w2Data, seq, dim).SetRequiresGrad(true)

	h := MatMul(x, w1)                                        // MPS (threshold lowered)
	hp := Permute(h.Reshape(seq, heads, seq), []int{1, 0, 2}) // CPU loop → sync point
	sm := CausalSoftmax(hp, heads, seq, 0.25)                 // K1 kernel
	g1 := sm.Reshape(heads*seq, seq)
	g2 := MatMul(g1, w2)                                               // MPS
	act := SwiGLU(g2, Scale(g2, 0.5))                                  // K4 kernel + vDSP scale
	weighted := Mul(act, NewTensorOnMetal(dev, wData, heads*seq, dim)) // vec_mul kernel
	l := Sum(weighted)
	l.Backward()

	SyncMetal()
	return l.Data()[0], append([]float32(nil), x.Grad().Data()...),
		append([]float32(nil), w1.Grad().Data()...)
}

func TestMetalAsyncParity(t *testing.T) {
	if _, err := InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	prevThresh := MatMulMetalThreshold
	MatMulMetalThreshold = 0 // force MPS dispatch at test shapes
	defer func() { MatMulMetalThreshold = prevThresh }()

	rng := rand.New(rand.NewSource(101))
	heads, seq, dim := 4, 32, 64
	xData := csRandSlice(rng, seq*dim, 1.0)
	w1Data := csRandSlice(rng, dim*heads*seq, 0.1)
	w2Data := csRandSlice(rng, seq*dim, 0.1)
	wData := csRandSlice(rng, heads*seq*dim, 1.0)

	if MetalAsyncEnabled() {
		t.Fatal("async mode unexpectedly on at test start")
	}
	lossSync, dxSync, dw1Sync := asyncChainRun(t, xData, w1Data, w2Data, wData, heads, seq, dim)

	SetMetalAsync(true)
	defer SetMetalAsync(false)
	lossAsync, dxAsync, dw1Async := asyncChainRun(t, xData, w1Data, w2Data, wData, heads, seq, dim)

	if lossSync != lossAsync {
		t.Errorf("async loss %g != sync loss %g", lossAsync, lossSync)
	}
	if d := csMaxAbsDiff(dxSync, dxAsync); d != 0 {
		t.Errorf("async dx differs from sync dx: max abs diff %.3g", d)
	}
	if d := csMaxAbsDiff(dw1Sync, dw1Async); d != 0 {
		t.Errorf("async dW1 differs from sync dW1: max abs diff %.3g", d)
	}

	// Sanity: the numbers are finite and non-trivial.
	if math.IsNaN(float64(lossSync)) || lossSync == 0 {
		t.Errorf("degenerate loss %g", lossSync)
	}
}

// TestMetalAsyncCEParity runs the K2 fused CE end-to-end in async mode
// (the loss read is a host sync point mid-graph) and checks bit parity
// with sync mode.
func TestMetalAsyncCEParity(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	prev := CEMetalMinElements
	CEMetalMinElements = 1
	defer func() { CEMetalMinElements = prev }()

	rng := rand.New(rand.NewSource(103))
	batch, classes := 17, 3001
	logits, tgt := ceRandCase(rng, batch, classes)

	run := func() (float32, []float32) {
		x := NewTensorOnMetal(gpuHandle.Dev, logits, batch, classes)
		l, dx := runCE(x, tgt, batch)
		SyncMetal()
		return l, append([]float32(nil), dx...)
	}

	lossSync, dxSync := run()
	SetMetalAsync(true)
	defer SetMetalAsync(false)
	lossAsync, dxAsync := run()

	if lossSync != lossAsync {
		t.Errorf("async CE loss %g != sync %g", lossAsync, lossSync)
	}
	if d := csMaxAbsDiff(dxSync, dxAsync); d != 0 {
		t.Errorf("async CE dx differs: max abs diff %.3g", d)
	}
}
