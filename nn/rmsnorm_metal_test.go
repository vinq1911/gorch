//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestRMSNormMetalForwardMatchesCPU pins the custom rmsnorm_forward
// Metal kernel against the existing CPU implementation. Plan 0004
// part A: forward parity at 1e-4 (looser than the 1e-5 the all-CPU
// numerical test uses, because GPU sums are non-deterministic in
// reduction order).
func TestRMSNormMetalForwardMatchesCPU(t *testing.T) {
	gpu, err := g.InitMetal()
	if err != nil {
		t.Skipf("InitMetal failed: %v", err)
	}
	_ = gpu

	const M, N = 8, 64
	xData := make([]float32, M*N)
	wData := make([]float32, N)
	// deterministic, non-trivial input
	for i := range xData {
		xData[i] = float32(i%17)*0.1 - 0.5
	}
	for i := range wData {
		wData[i] = 1 + float32(i%5)*0.2
	}

	// CPU reference.
	rnCPU := NewRMSNorm(N)
	copy(rnCPU.Weight.Data(), wData)
	xCPU := g.NewTensor(xData, M, N)
	yCPU := rnCPU.Forward(xCPU)

	// Metal path.
	rnGPU := NewRMSNorm(N)
	copy(rnGPU.Weight.Data(), wData)
	rnGPU.Weight.ToMetal(gpu.Dev)
	xGPU := g.NewTensorOnMetal(gpu.Dev, xData, M, N)
	yGPU := rnGPU.Forward(xGPU)

	if !yGPU.IsOnMetal() {
		t.Fatal("RMSNorm.Forward on Metal inputs returned a CPU tensor")
	}

	for i := range yCPU.Data() {
		d := math.Abs(float64(yCPU.Data()[i] - yGPU.Data()[i]))
		if d > 1e-4 {
			t.Fatalf("[%d]: CPU=%g GPU=%g abs=%g", i, yCPU.Data()[i], yGPU.Data()[i], d)
		}
	}
}

// TestRMSNormMetalBackwardMatchesCPU verifies the rmsnorm_dx Metal
// kernel produces gradients within 1e-3 of the CPU backward, plus
// dW (computed on the host) within the same tolerance. Goal-backward
// check that "GPU autograd actually trains."
func TestRMSNormMetalBackwardMatchesCPU(t *testing.T) {
	gpu, err := g.InitMetal()
	if err != nil {
		t.Skipf("InitMetal failed: %v", err)
	}

	const M, N = 4, 32
	xData := make([]float32, M*N)
	wData := make([]float32, N)
	for i := range xData {
		xData[i] = float32(i%13)*0.05 - 0.3
	}
	for i := range wData {
		wData[i] = 1 + float32(i%4)*0.25
	}

	// CPU.
	rnCPU := NewRMSNorm(N)
	copy(rnCPU.Weight.Data(), wData)
	xCPU := g.NewTensor(xData, M, N).SetRequiresGrad(true)
	yCPU := rnCPU.Forward(xCPU)
	lossCPU := g.Sum(yCPU)
	lossCPU.Backward()
	dxCPU := append([]float32{}, xCPU.Grad().Data()...)
	dwCPU := append([]float32{}, rnCPU.Weight.Grad().Data()...)

	// GPU.
	rnGPU := NewRMSNorm(N)
	copy(rnGPU.Weight.Data(), wData)
	rnGPU.Weight.ToMetal(gpu.Dev)
	rnGPU.Weight.SetRequiresGrad(true)
	xGPU := g.NewTensorOnMetal(gpu.Dev, xData, M, N).SetRequiresGrad(true)
	yGPU := rnGPU.Forward(xGPU)
	lossGPU := g.Sum(yGPU)
	lossGPU.Backward()

	if xGPU.Grad() == nil {
		t.Fatal("xGPU.Grad() is nil after Metal RMSNorm backward")
	}
	if rnGPU.Weight.Grad() == nil {
		t.Fatal("Weight.Grad() is nil after Metal RMSNorm backward")
	}

	dxGPU := xGPU.Grad().Data()
	dwGPU := rnGPU.Weight.Grad().Data()

	for i := range dxCPU {
		d := math.Abs(float64(dxCPU[i] - dxGPU[i]))
		if d > 1e-3 {
			t.Fatalf("dx[%d]: CPU=%g GPU=%g abs=%g", i, dxCPU[i], dxGPU[i], d)
		}
	}
	for i := range dwCPU {
		d := math.Abs(float64(dwCPU[i] - dwGPU[i]))
		if d > 1e-3 {
			t.Fatalf("dW[%d]: CPU=%g GPU=%g abs=%g", i, dwCPU[i], dwGPU[i], d)
		}
	}
}

// TestRMSNormMetalBackwardMatchesNumerical: gold-standard correctness
// — the GPU analytical gradient agrees with central finite-difference
// over the GPU forward path. Pins both the kernel math and the
// dispatch wiring.
func TestRMSNormMetalBackwardMatchesNumerical(t *testing.T) {
	gpu, err := g.InitMetal()
	if err != nil {
		t.Skipf("InitMetal failed: %v", err)
	}

	const M, N = 3, 8
	xData := make([]float32, M*N)
	for i := range xData {
		xData[i] = float32((i*7)%11)*0.1 - 0.4
	}

	rnGPU := NewRMSNorm(N)
	for j := range rnGPU.Weight.Data() {
		rnGPU.Weight.Data()[j] = 1 + float32(j)*0.3
	}
	rnGPU.Weight.ToMetal(gpu.Dev)
	rnGPU.Weight.SetRequiresGrad(true)
	x := g.NewTensorOnMetal(gpu.Dev, xData, M, N).SetRequiresGrad(true)

	y := rnGPU.Forward(x)
	loss := g.Sum(y)
	loss.Backward()
	dxAnalytic := append([]float32{}, x.Grad().Data()...)
	dwAnalytic := append([]float32{}, rnGPU.Weight.Grad().Data()...)

	// Numerical: perturb the unified-memory slice in place, re-run
	// forward, recompute loss. Each iteration spawns a fresh forward
	// graph so the Backward state from the analytic phase doesn't
	// interfere.
	const h = 1e-2
	xData2 := x.Data()

	rnNum := NewRMSNorm(N)
	copy(rnNum.Weight.Data(), rnGPU.Weight.Data())
	rnNum.Weight.ToMetal(gpu.Dev)

	dxNum := make([]float32, M*N)
	for i := range dxNum {
		orig := xData2[i]
		xData2[i] = orig + h
		yPlus := g.Sum(rnNum.Forward(x)).Data()[0]
		xData2[i] = orig - h
		yMinus := g.Sum(rnNum.Forward(x)).Data()[0]
		xData2[i] = orig
		dxNum[i] = (yPlus - yMinus) / (2 * h)
	}

	dwNum := make([]float32, N)
	wData := rnNum.Weight.Data()
	for j := range dwNum {
		orig := wData[j]
		wData[j] = orig + h
		yPlus := g.Sum(rnNum.Forward(x)).Data()[0]
		wData[j] = orig - h
		yMinus := g.Sum(rnNum.Forward(x)).Data()[0]
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
	check("dx", dxAnalytic, dxNum, 2e-2)
	check("dW", dwAnalytic, dwNum, 2e-2)
}
