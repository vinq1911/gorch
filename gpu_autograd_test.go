//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// TestGPUMatMulBackwardMatchesCPU verifies that the new GPU MatMul
// backward path produces gradients that match the CPU Accelerate
// path within fp32 noise. Uses identical input data on both paths.
func TestGPUMatMulBackwardMatchesCPU(t *testing.T) {
	gpu, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	// Deterministic small inputs.
	M, K, N := 4, 6, 5
	aData := make([]float32, M*K)
	bData := make([]float32, K*N)
	gradData := make([]float32, M*N)
	for i := range aData {
		aData[i] = float32(i)*0.1 - 1
	}
	for i := range bData {
		bData[i] = float32(i)*0.05 + 0.5
	}
	for i := range gradData {
		gradData[i] = float32(i)*0.03 - 0.2
	}

	// CPU path.
	aCPU := NewTensor(aData, M, K).SetRequiresGrad(true)
	bCPU := NewTensor(bData, K, N).SetRequiresGrad(true)
	outCPU := MatMul(aCPU, bCPU)
	gradCPU := NewTensor(gradData, M, N)
	gradsCPU := outCPU.gradFn.backward(gradCPU)

	// GPU path: same data on Metal.
	aGPU := NewTensorOnMetal(gpu.Dev, aData, M, K).SetRequiresGrad(true)
	bGPU := NewTensorOnMetal(gpu.Dev, bData, K, N).SetRequiresGrad(true)
	outGPU := MatMul(aGPU, bGPU)
	gradGPU := NewTensorOnMetal(gpu.Dev, gradData, M, N)
	gradsGPU := outGPU.gradFn.backward(gradGPU)

	if !gradsGPU[0].IsOnMetal() {
		t.Error("expected dA on Metal")
	}
	if !gradsGPU[1].IsOnMetal() {
		t.Error("expected dB on Metal")
	}

	checkClose(t, "dA", gradsCPU[0].Data(), gradsGPU[0].Data(), 1e-3)
	checkClose(t, "dB", gradsCPU[1].Data(), gradsGPU[1].Data(), 1e-3)
}

// TestGPUMatMulFallsBackWhenGradOnCPU confirms that if the gradient
// arrives on CPU even with weights on GPU, we fall back to CPU
// rather than panicking. This matters because upstream losses
// (CrossEntropy, MSE) currently produce CPU grads.
func TestGPUMatMulFallsBackWhenGradOnCPU(t *testing.T) {
	gpu, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	M, K, N := 3, 4, 2
	aData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	bData := []float32{1, 2, 3, 4, 5, 6, 7, 8}

	aGPU := NewTensorOnMetal(gpu.Dev, aData, M, K).SetRequiresGrad(true)
	bGPU := NewTensorOnMetal(gpu.Dev, bData, K, N).SetRequiresGrad(true)
	out := MatMul(aGPU, bGPU)

	// CPU grad, mixed device path.
	gradCPU := Ones(M, N)
	grads := out.gradFn.backward(gradCPU)

	// Output devices should follow the CPU path semantics — no panic.
	if grads[0].Size() != M*K || grads[1].Size() != K*N {
		t.Fatalf("wrong shapes: %v %v", grads[0].Shape(), grads[1].Shape())
	}
}

// TestMatMulTransAPublicOp verifies the helper used by Linear's GPU
// backward agrees with the CPU SgemmTransA path.
func TestMatMulTransAPublicOp(t *testing.T) {
	gpu, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	K, M, N := 3, 4, 5
	aData := make([]float32, K*M)
	bData := make([]float32, K*N)
	for i := range aData {
		aData[i] = float32(i) - 5
	}
	for i := range bData {
		bData[i] = float32(i)*0.2 + 1
	}

	aCPU := NewTensor(aData, K, M)
	bCPU := NewTensor(bData, K, N)
	cpuOut := MatMulTransA(aCPU, bCPU)

	aGPU := NewTensorOnMetal(gpu.Dev, aData, K, M)
	bGPU := NewTensorOnMetal(gpu.Dev, bData, K, N)
	gpuOut := MatMulTransA(aGPU, bGPU)

	if !gpuOut.IsOnMetal() {
		t.Error("expected GPU result on Metal")
	}
	checkClose(t, "MatMulTransA", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

func checkClose(t *testing.T, label string, a, b []float32, tol float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: length mismatch %d vs %d", label, len(a), len(b))
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol || math.IsNaN(float64(a[i])) || math.IsNaN(float64(b[i])) {
			t.Fatalf("%s[%d]: cpu=%g gpu=%g (diff=%g)", label, i, a[i], b[i], d)
		}
	}
}
