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
	// Force GPU dispatch regardless of shape (the threshold otherwise
	// keeps small matmuls on CPU; this test wants the GPU code path
	// for correctness verification).
	defer setMatMulMetalThresholdForTest(t, 0)()

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
	defer setMatMulMetalThresholdForTest(t, 0)()
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

// TestGPUBatchedMatMulBackwardMatchesCPU verifies the plan-0009-X1
// batched MPS backward (dA via batched transB, dB via the new batched
// transA) against the CPU Accelerate loop at 1e-3 abs.
func TestGPUBatchedMatMulBackwardMatchesCPU(t *testing.T) {
	gpu, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer setMatMulMetalThresholdForTest(t, 0)()

	batch, M, N, K := 3, 5, 4, 6
	aData := make([]float32, batch*M*K)
	bData := make([]float32, batch*K*N)
	gradData := make([]float32, batch*M*N)
	for i := range aData {
		aData[i] = float32(i)*0.07 - 1.1
	}
	for i := range bData {
		bData[i] = float32(i)*0.03 + 0.2
	}
	for i := range gradData {
		gradData[i] = float32(i)*0.05 - 0.6
	}

	// CPU path.
	aCPU := NewTensor(aData, batch, M, K).SetRequiresGrad(true)
	bCPU := NewTensor(bData, batch, K, N).SetRequiresGrad(true)
	outCPU := BatchedMatMul(aCPU, bCPU, batch, M, N, K)
	gradsCPU := outCPU.gradFn.backward(NewTensor(gradData, batch, M, N))

	// GPU path.
	aGPU := NewTensorOnMetal(gpu.Dev, aData, batch, M, K).SetRequiresGrad(true)
	bGPU := NewTensorOnMetal(gpu.Dev, bData, batch, K, N).SetRequiresGrad(true)
	outGPU := BatchedMatMul(aGPU, bGPU, batch, M, N, K)
	gradsGPU := outGPU.gradFn.backward(NewTensorOnMetal(gpu.Dev, gradData, batch, M, N))

	if !gradsGPU[0].IsOnMetal() || !gradsGPU[1].IsOnMetal() {
		t.Error("expected batched dA/dB on Metal")
	}
	checkClose(t, "batched dA", gradsCPU[0].Data(), gradsGPU[0].Data(), 1e-3)
	checkClose(t, "batched dB", gradsCPU[1].Data(), gradsGPU[1].Data(), 1e-3)
}

// TestGPUBatchedMatMulTransBBackwardMatchesCPU is the same check for
// BatchedMatMulTransB (dA via batched plain, dB via batched transA) —
// the attention-scores backward.
func TestGPUBatchedMatMulTransBBackwardMatchesCPU(t *testing.T) {
	gpu, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	defer setMatMulMetalThresholdForTest(t, 0)()

	batch, M, N, K := 4, 6, 5, 3
	aData := make([]float32, batch*M*K)
	bData := make([]float32, batch*N*K)
	gradData := make([]float32, batch*M*N)
	for i := range aData {
		aData[i] = float32(i)*0.04 - 0.9
	}
	for i := range bData {
		bData[i] = float32(i)*0.06 + 0.1
	}
	for i := range gradData {
		gradData[i] = float32(i)*0.02 - 0.3
	}

	aCPU := NewTensor(aData, batch, M, K).SetRequiresGrad(true)
	bCPU := NewTensor(bData, batch, N, K).SetRequiresGrad(true)
	outCPU := BatchedMatMulTransB(aCPU, bCPU, batch, M, N, K)
	gradsCPU := outCPU.gradFn.backward(NewTensor(gradData, batch, M, N))

	aGPU := NewTensorOnMetal(gpu.Dev, aData, batch, M, K).SetRequiresGrad(true)
	bGPU := NewTensorOnMetal(gpu.Dev, bData, batch, N, K).SetRequiresGrad(true)
	outGPU := BatchedMatMulTransB(aGPU, bGPU, batch, M, N, K)
	gradsGPU := outGPU.gradFn.backward(NewTensorOnMetal(gpu.Dev, gradData, batch, M, N))

	if !gradsGPU[0].IsOnMetal() || !gradsGPU[1].IsOnMetal() {
		t.Error("expected batched transB dA/dB on Metal")
	}
	checkClose(t, "transB dA", gradsCPU[0].Data(), gradsGPU[0].Data(), 1e-3)
	checkClose(t, "transB dB", gradsCPU[1].Data(), gradsGPU[1].Data(), 1e-3)
}

// TestResidencyPropagation asserts the plan-0009-X1 rule: ops with a
// Metal-resident input produce Metal-resident outputs even when the
// compute runs on CPU (below-threshold matmul, Go-loop softmax, …),
// and the Sum-loss backward seeds a Metal-resident grad.
func TestResidencyPropagation(t *testing.T) {
	gpu, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	data := make([]float32, 4*6)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	x := NewTensorOnMetal(gpu.Dev, data, 4, 6).SetRequiresGrad(true)

	// Below-threshold matmul: CPU sgemm, output must stay resident.
	w := NewTensorOnMetal(gpu.Dev, data, 6, 4)
	mm := MatMul(x, w)
	if !mm.IsOnMetal() {
		t.Error("below-threshold MatMul output lost residency")
	}
	if !Softmax(mm).IsOnMetal() {
		t.Error("Softmax output lost residency")
	}
	if !Permute(mm.Reshape(4, 2, 2), []int{1, 0, 2}).IsOnMetal() {
		t.Error("Permute output lost residency")
	}
	if !Scale(mm, 0.5).IsOnMetal() {
		t.Error("Scale output lost residency")
	}
	if !MaskFill(mm, make([]bool, mm.Size()), -1e9).IsOnMetal() {
		t.Error("MaskFill output lost residency")
	}

	// Loss-side seeding: Sum backward must produce a Metal grad.
	x.ZeroGrad()
	Sum(mm).Backward()
	if x.Grad() == nil || !x.Grad().IsOnMetal() {
		t.Error("Sum-loss backward did not seed a Metal-resident grad chain")
	}
}

// setMatMulMetalThresholdForTest temporarily lowers the threshold
// so tests can exercise the GPU dispatch path on small shapes.
// Returns a restore func to defer.
func setMatMulMetalThresholdForTest(t *testing.T, v int) func() {
	t.Helper()
	prev := MatMulMetalThreshold
	MatMulMetalThreshold = v
	return func() { MatMulMetalThreshold = prev }
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
