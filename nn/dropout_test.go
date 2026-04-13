//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

func TestDropoutTraining(t *testing.T) {
	dropout := NewDropout(0.5)
	x := g.Ones(1000)

	out := dropout.Forward(x)
	data := out.Data()

	// Count zeros — should be roughly 50%
	zeros := 0
	for _, v := range data {
		if v == 0 {
			zeros++
		}
	}

	// With p=0.5 and 1000 elements, expect ~500 zeros (tolerance: 350-650)
	if zeros < 350 || zeros > 650 {
		t.Fatalf("dropout zeros = %d, expected ~500 (350-650 range)", zeros)
	}

	// Non-zero values should be scaled by 1/(1-0.5) = 2.0
	for _, v := range data {
		if v != 0 && math.Abs(float64(v)-2.0) > 0.01 {
			t.Fatalf("non-zero dropout value = %f, want 2.0 (scaled)", v)
		}
	}
}

func TestDropoutEval(t *testing.T) {
	dropout := NewDropout(0.5)
	dropout.Eval() // disable dropout

	x := g.Ones(100)
	out := dropout.Forward(x)

	// All values should be unchanged
	for i, v := range out.Data() {
		if v != 1.0 {
			t.Fatalf("eval dropout[%d] = %f, want 1.0", i, v)
		}
	}
}

func TestDropoutBackward(t *testing.T) {
	dropout := NewDropout(0.3)
	x := g.Ones(100)
	x.SetRequiresGrad(true)

	out := dropout.Forward(x)
	loss := g.Sum(out)
	loss.Backward()

	if x.Grad() == nil {
		t.Fatal("x.grad is nil")
	}

	// Gradient should match the forward mask
	for i, v := range x.Grad().Data() {
		if out.Data()[i] == 0 {
			if v != 0 {
				t.Fatalf("grad[%d] = %f for zeroed position, want 0", i, v)
			}
		}
	}
}
