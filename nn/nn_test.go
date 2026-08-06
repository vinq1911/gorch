//go:build darwin

package nn

import (
	"fmt"
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

func TestLinearForward(t *testing.T) {
	layer := NewLinear(3, 2)
	x := g.NewTensor([]float32{1, 2, 3}, 1, 3)
	out := layer.Forward(x)

	if out.Shape()[0] != 1 || out.Shape()[1] != 2 {
		t.Fatalf("shape = %v, want [1, 2]", out.Shape())
	}
}

func TestLinearParameters(t *testing.T) {
	layer := NewLinear(4, 3)
	params := layer.Parameters()
	if len(params) != 2 {
		t.Fatalf("got %d params, want 2", len(params))
	}
	// Weight: (3, 4), Bias: (1, 3)
	if params[0].Size() != 12 {
		t.Fatalf("weight size = %d, want 12", params[0].Size())
	}
	if params[1].Size() != 3 {
		t.Fatalf("bias size = %d, want 3", params[1].Size())
	}
}

func TestSequentialForward(t *testing.T) {
	model := NewSequential(
		NewLinear(2, 4),
		NewReLU(),
		NewLinear(4, 1),
	)
	x := g.NewTensor([]float32{1, 2}, 1, 2)
	out := model.Forward(x)

	if out.Shape()[0] != 1 || out.Shape()[1] != 1 {
		t.Fatalf("shape = %v, want [1, 1]", out.Shape())
	}
}

func TestSequentialParameters(t *testing.T) {
	model := NewSequential(
		NewLinear(2, 4),
		NewReLU(),
		NewLinear(4, 1),
	)
	params := model.Parameters()
	// Linear(2,4): weight(4,2) + bias(1,4) = 2 params
	// ReLU: 0 params
	// Linear(4,1): weight(1,4) + bias(1,1) = 2 params
	if len(params) != 4 {
		t.Fatalf("got %d params, want 4", len(params))
	}
}

// TestTrainXOR trains a small network to learn XOR — the "hello world" of neural nets.
func TestTrainXOR(t *testing.T) {
	model := NewSequential(
		NewLinear(2, 8),
		NewReLU(),
		NewLinear(8, 1),
	)

	opt := optim.NewAdam(model.Parameters(), 0.01)

	// XOR dataset: 4 samples
	inputs := g.NewTensor([]float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}, 4, 2)

	targets := g.NewTensor([]float32{0, 1, 1, 0}, 4, 1)

	var finalLoss float32
	for epoch := 0; epoch < 500; epoch++ {
		opt.ZeroGrad()

		pred := model.Forward(inputs)
		loss := g.MSELoss(pred, targets)
		loss.Backward()
		opt.Step()

		finalLoss = loss.Data()[0]
	}

	if finalLoss > 0.05 {
		t.Fatalf("XOR training failed: final loss = %f (want < 0.05)", finalLoss)
	}

	// Verify predictions
	pred := model.Forward(inputs)
	pData := pred.Data()
	xorTargets := []float32{0, 1, 1, 0}
	for i, target := range xorTargets {
		if math.Abs(float64(pData[i]-target)) > 0.15 {
			t.Errorf("XOR[%d] = %.3f, want ~%.0f", i, pData[i], target)
		}
	}
	fmt.Printf("  XOR final loss: %.6f, predictions: [%.3f, %.3f, %.3f, %.3f]\n",
		finalLoss, pData[0], pData[1], pData[2], pData[3])
}

// TestTrainSGD verifies SGD with momentum on a simple regression.
func TestTrainSGD(t *testing.T) {
	// Learn y = 2*x + 1
	layer := NewLinear(1, 1)
	opt := optim.NewSGD(layer.Parameters(), 0.01, 0.9)

	var finalLoss float32
	for epoch := 0; epoch < 200; epoch++ {
		opt.ZeroGrad()

		x := g.NewTensor([]float32{1, 2, 3, 4}, 4, 1)
		target := g.NewTensor([]float32{3, 5, 7, 9}, 4, 1)

		pred := layer.Forward(x)
		loss := g.MSELoss(pred, target)
		loss.Backward()
		opt.Step()

		finalLoss = loss.Data()[0]
	}

	if finalLoss > 0.1 {
		t.Fatalf("regression failed: final loss = %f (want < 0.1)", finalLoss)
	}
	fmt.Printf("  Regression (y=2x+1) final loss: %.6f\n", finalLoss)
}

func TestMultiHeadAttention_Causal_DefaultsTrue(t *testing.T) {
	mha := NewMultiHeadAttention(8, 2)
	if !mha.Causal {
		t.Fatal("default-constructed MultiHeadAttention should have Causal=true")
	}
}

func TestMultiHeadAttention_Bi_HasCausalFalse(t *testing.T) {
	mha := NewMultiHeadAttentionBi(8, 2)
	if mha.Causal {
		t.Fatal("NewMultiHeadAttentionBi should have Causal=false")
	}
}

func TestMultiHeadAttention_Bi_DiffersFromCausal(t *testing.T) {
	// On the same input and the same random Q/K/V projections,
	// causal and bidirectional attention must produce different
	// outputs whenever seq > 1 — bidirectional sees future tokens,
	// causal does not.
	const dim = 8
	const heads = 2
	const seq = 4

	mhaC := NewMultiHeadAttention(dim, heads)
	mhaB := NewMultiHeadAttentionBi(dim, heads)

	// Force identical weights so output difference comes purely from masking.
	for _, pair := range []struct {
		c, b *Linear
	}{
		{mhaC.Wq, mhaB.Wq},
		{mhaC.Wk, mhaB.Wk},
		{mhaC.Wv, mhaB.Wv},
		{mhaC.Wo, mhaB.Wo},
	} {
		copy(pair.b.Weight.Data(), pair.c.Weight.Data())
		copy(pair.b.Bias.Data(), pair.c.Bias.Data())
	}

	xData := make([]float32, seq*dim)
	for i := range xData {
		xData[i] = float32(i+1) * 0.1
	}
	xC := g.NewTensor(xData, seq, dim)
	xB := g.NewTensor(append([]float32(nil), xData...), seq, dim)

	outC := mhaC.Forward(xC, seq).Data()
	outB := mhaB.Forward(xB, seq).Data()
	if len(outC) != len(outB) {
		t.Fatalf("output shape mismatch: causal=%d bi=%d", len(outC), len(outB))
	}

	var maxDiff float32
	for i := range outC {
		d := outC[i] - outB[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff < 1e-4 {
		t.Errorf("causal and bidirectional outputs are identical (max diff %v); mask is not having an effect", maxDiff)
	}
}
