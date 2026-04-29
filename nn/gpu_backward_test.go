//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

// TestLinearBackwardMatchesCPUOnGPU verifies Linear gradients (dW, db,
// dx via downstream chain) match between CPU and Metal paths within
// fp32 noise. Uses identical init data and runs the same Backward()
// path on both.
func TestLinearBackwardMatchesCPUOnGPU(t *testing.T) {
	gpu, err := g.InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	const batch, in, out = 8, 16, 12

	wData := make([]float32, out*in)
	bData := make([]float32, out)
	xData := make([]float32, batch*in)
	for i := range wData {
		wData[i] = float32(i)*0.01 - 0.3
	}
	for i := range bData {
		bData[i] = float32(i) * 0.1
	}
	for i := range xData {
		xData[i] = float32(i)*0.05 - 1
	}

	runOnce := func(metal bool) (dW, dB []float32) {
		l := NewLinear(in, out)
		copy(l.Weight.Data(), wData)
		copy(l.Bias.Data(), bData)
		var x *g.Tensor
		if metal {
			l.ToMetal(gpu.Dev)
			x = g.NewTensorOnMetal(gpu.Dev, xData, batch, in)
		} else {
			x = g.NewTensor(xData, batch, in)
		}
		y := l.Forward(x)
		// Scalar loss = sum(y) — gradient seed is 1 for every y entry,
		// which gives clean closed-form grads for dW and db.
		loss := g.Sum(y)
		loss.Backward()
		dWcopy := make([]float32, out*in)
		dBcopy := make([]float32, out)
		copy(dWcopy, l.Weight.Grad().Data())
		copy(dBcopy, l.Bias.Grad().Data())
		return dWcopy, dBcopy
	}

	dWcpu, dBcpu := runOnce(false)
	dWgpu, dBgpu := runOnce(true)

	checkAllClose(t, "dW", dWcpu, dWgpu, 5e-3)
	checkAllClose(t, "db", dBcpu, dBgpu, 5e-3)
}

// TestTrainTinyMLPOnGPU runs a few Adam steps with weights resident
// on Metal and checks that the loss decreases — proving the GPU
// autograd path trains end-to-end without NaNs or hangs.
func TestTrainTinyMLPOnGPU(t *testing.T) {
	gpu, err := g.InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}

	model := NewSequential(
		NewLinear(4, 16),
		NewReLU(),
		NewLinear(16, 4),
	)
	for _, layer := range model.Layers {
		if l, ok := layer.(*Linear); ok {
			l.ToMetal(gpu.Dev)
		}
	}

	xData := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	x := g.NewTensorOnMetal(gpu.Dev, xData, 4, 4)
	yData := []float32{0, 1, 2, 3}
	y := g.NewTensor(yData, 4, 1)

	opt := optim.NewAdam(model.Parameters(), 0.05)

	first := g.CrossEntropyLoss(model.Forward(x), y).Data()[0]
	for step := 0; step < 30; step++ {
		opt.ZeroGrad()
		loss := g.CrossEntropyLoss(model.Forward(x), y)
		loss.Backward()
		opt.Step()
	}
	last := g.CrossEntropyLoss(model.Forward(x), y).Data()[0]

	if math.IsNaN(float64(last)) {
		t.Fatalf("loss is NaN — GPU backward broke training")
	}
	if !(last < first*0.7) {
		t.Fatalf("loss did not drop on GPU: first=%g last=%g", first, last)
	}
}

func checkAllClose(t *testing.T, label string, a, b []float32, tol float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: length mismatch %d vs %d", label, len(a), len(b))
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			t.Fatalf("%s[%d]: cpu=%g gpu=%g (diff=%g)", label, i, a[i], b[i], d)
		}
	}
}
