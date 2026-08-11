//go:build darwin

package nn

import (
	"math"
	"math/rand"
	"testing"

	g "github.com/vinq1911/gorch"
)

// loraToy builds the plan's 8×16 toy: Linear in=8 → out=16, rank 4.
func loraToy(t *testing.T, randomize bool) (*Linear, *LoRALinear) {
	t.Helper()
	base := NewLinear(8, 16)
	l := NewLoRALinear(base, 4, 8)
	if randomize {
		rng := rand.New(rand.NewSource(7))
		for i := range l.A.Data() {
			l.A.Data()[i] = float32(rng.NormFloat64()) * 0.3
		}
		for i := range l.B.Data() {
			l.B.Data()[i] = float32(rng.NormFloat64()) * 0.3
		}
	}
	return base, l
}

func randInput(seed int64, rows, cols int) *g.Tensor {
	rng := rand.New(rand.NewSource(seed))
	x := g.Zeros(rows, cols)
	for i := range x.Data() {
		x.Data()[i] = float32(rng.NormFloat64())
	}
	return x
}

// TestLoRAZeroInitParity — zero-init B ⇒ the wrapped layer is
// BIT-identical to the base forward (plan §3.1 gate).
func TestLoRAZeroInitParity(t *testing.T) {
	base, l := loraToy(t, false)
	x := randInput(1, 5, 8)

	want := base.Forward(x)
	got := l.Forward(x)
	for i := range want.Data() {
		if got.Data()[i] != want.Data()[i] {
			t.Fatalf("element %d: %v != %v (must be bit-identical at zero init)", i, got.Data()[i], want.Data()[i])
		}
	}
}

// TestLoRAFreezeAndParameters — Freeze marks the base non-trainable;
// Parameters exposes {A, B} only.
func TestLoRAFreezeAndParameters(t *testing.T) {
	base, l := loraToy(t, false)
	if base.Weight.RequiresGrad() || base.Bias.RequiresGrad() {
		t.Fatal("NewLoRALinear must freeze the base layer")
	}
	ps := l.Parameters()
	if len(ps) != 2 || ps[0] != l.A || ps[1] != l.B {
		t.Fatalf("Parameters() = %d tensors, want exactly {A, B}", len(ps))
	}
}

// lossOf runs the scalar test loss L = Σ (y ⊙ w) for a fixed weight
// tensor w — a smooth loss with a dense, non-trivial gradient.
func loraLoss(l *LoRALinear, x, w *g.Tensor) *g.Tensor {
	y := l.Forward(x)
	return g.Sum(g.Mul(y, w))
}

// TestLoRAGradCheck — finite-difference gradient checks on every
// element of A and B against the autograd backward (plan §3.1 gate).
func TestLoRAGradCheck(t *testing.T) {
	_, l := loraToy(t, true)
	x := randInput(2, 3, 8)
	w := randInput(3, 3, 16)

	loss := loraLoss(l, x, w)
	loss.Backward()
	gradA := l.A.Grad()
	gradB := l.B.Grad()
	if gradA == nil || gradB == nil {
		t.Fatal("A/B received no gradient")
	}

	fd := func(p *g.Tensor, i int) float64 {
		const eps = 1e-2
		orig := p.Data()[i]
		var plus, minus float32
		g.NoGrad(func() {
			p.Data()[i] = orig + eps
			plus = loraLoss(l, x, w).Data()[0]
			p.Data()[i] = orig - eps
			minus = loraLoss(l, x, w).Data()[0]
		})
		p.Data()[i] = orig
		return (float64(plus) - float64(minus)) / (2 * eps)
	}

	check := func(name string, p, grad *g.Tensor) {
		var worst float64
		for i := range p.Data() {
			want := fd(p, i)
			got := float64(grad.Data()[i])
			d := math.Abs(got-want) / (1e-3 + math.Abs(want))
			if d > worst {
				worst = d
			}
			if d > 5e-2 {
				t.Errorf("%s[%d]: autograd %.6g vs finite-diff %.6g (rel %.3g)", name, i, got, want, d)
			}
		}
		t.Logf("%s: worst |auto-fd|/(1e-3+|fd|) = %.3g over %d elements", name, worst, p.Size())
	}
	check("A", l.A, gradA)
	check("B", l.B, gradB)
}

// TestLoRAMergeParity — merged-vs-unmerged forward parity within 1e-6
// (plan §3.1 gate), and Unmerge restores the unmerged output.
func TestLoRAMergeParity(t *testing.T) {
	_, l := loraToy(t, true)
	x := randInput(4, 5, 8)

	unmerged := l.Forward(x)
	l.Merge()
	if !l.Merged() {
		t.Fatal("Merged() false after Merge")
	}
	merged := l.Forward(x)
	for i := range unmerged.Data() {
		d := math.Abs(float64(unmerged.Data()[i]) - float64(merged.Data()[i]))
		if d > 1e-6 {
			t.Fatalf("element %d: merged %v vs unmerged %v (|Δ| %.3g > 1e-6)", i, merged.Data()[i], unmerged.Data()[i], d)
		}
	}
	l.Unmerge()
	restored := l.Forward(x)
	for i := range unmerged.Data() {
		d := math.Abs(float64(unmerged.Data()[i]) - float64(restored.Data()[i]))
		if d > 1e-6 {
			t.Fatalf("element %d not restored after Unmerge: |Δ| %.3g", i, d)
		}
	}
}

// TestLinearFrozenDWSkip — with a frozen Weight the backward must
// still produce a correct dX (bit-identical to the always-compute
// path) while accumulating no gradient into Weight or Bias.
func TestLinearFrozenDWSkip(t *testing.T) {
	lin := NewLinear(8, 16)
	x1 := randInput(5, 3, 8).SetRequiresGrad(true)
	x2 := g.NewTensor(x1.Data(), 3, 8).SetRequiresGrad(true)
	w := randInput(6, 3, 16)

	// Reference: frozen weight but skip disabled.
	lin.Weight.SetRequiresGrad(false)
	lin.Bias.SetRequiresGrad(false)
	AlwaysComputeLinearDW = true
	g.Sum(g.Mul(lin.Forward(x1), w)).Backward()
	AlwaysComputeLinearDW = false

	// Skip path.
	g.Sum(g.Mul(lin.Forward(x2), w)).Backward()

	if lin.Weight.Grad() != nil || lin.Bias.Grad() != nil {
		t.Fatal("frozen Weight/Bias accumulated a gradient")
	}
	if x2.Grad() == nil {
		t.Fatal("input received no gradient through the frozen Linear")
	}
	for i := range x1.Grad().Data() {
		if x1.Grad().Data()[i] != x2.Grad().Data()[i] {
			t.Fatalf("dX[%d] differs between skip and always-compute paths: %v vs %v",
				i, x2.Grad().Data()[i], x1.Grad().Data()[i])
		}
	}

	// Unfrozen control: dW must still be produced.
	lin2 := NewLinear(8, 16)
	x3 := randInput(5, 3, 8).SetRequiresGrad(true)
	g.Sum(g.Mul(lin2.Forward(x3), w)).Backward()
	if lin2.Weight.Grad() == nil || lin2.Bias.Grad() == nil {
		t.Fatal("unfrozen Linear lost its weight/bias gradients")
	}
}
