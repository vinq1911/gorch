//go:build darwin

package optim

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestAdamWZeroDecayMatchesAdam: with weightDecay=0, AdamW must be
// numerically identical to Adam (modulo floating-point noise). This
// pins down the implementation — if it fails, AdamW is doing more
// than just the decoupled-decay extension.
func TestAdamWZeroDecayMatchesAdam(t *testing.T) {
	const N = 8

	makeParam := func() *g.Tensor {
		w := g.NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, N)
		w.SetRequiresGrad(true)
		return w
	}

	pAdam := makeParam()
	pAdamW := makeParam()

	adam := NewAdam([]*g.Tensor{pAdam}, 0.01)
	adamw := NewAdamW([]*g.Tensor{pAdamW}, 0.01, 0)

	// Same gradient sequence for both.
	gradData := []float32{0.1, -0.2, 0.05, 0.3, -0.1, 0.4, 0.0, -0.05}

	for step := 0; step < 20; step++ {
		// Stamp the same gradient on both params.
		gT := g.NewTensor(gradData, N)
		// The optimisers read p.Grad(); set it directly via the
		// internal field would require package-private access, so
		// we use a pseudo-step: zero, set, step.
		pAdam.ZeroGrad()
		pAdamW.ZeroGrad()
		// Hack: copy gradData into a fresh grad tensor by triggering an
		// op that propagates grad. Simplest: do x = p * 1, sum, backward.
		// But we want the EXACT gradient. So we accept that gorch's
		// optim reads p.Grad().Data(), which we can't write directly.
		// Workaround: build a loss whose gradient is gradData at p.
		// loss = sum(p * gradData) → dloss/dp = gradData.
		multA := g.NewTensor(gradData, N)
		multB := g.NewTensor(gradData, N)
		lossA := g.Sum(g.Mul(pAdam, multA))
		lossB := g.Sum(g.Mul(pAdamW, multB))
		lossA.Backward()
		lossB.Backward()

		adam.Step()
		adamw.Step()
		_ = gT
	}

	for i := 0; i < N; i++ {
		d := math.Abs(float64(pAdam.Data()[i] - pAdamW.Data()[i]))
		if d > 1e-5 {
			t.Fatalf("[%d] Adam=%g AdamW(wd=0)=%g diff=%g", i, pAdam.Data()[i], pAdamW.Data()[i], d)
		}
	}
}

// TestAdamWWeightDecayShrinks: with positive weight decay and zero
// gradient, parameters must decay toward zero. The Adam optimiser does
// nothing in this case (no gradient → no update); AdamW does.
func TestAdamWWeightDecayShrinks(t *testing.T) {
	p := g.NewTensor([]float32{10, -20, 30}, 3)
	p.SetRequiresGrad(true)

	const lr = 0.1
	const wd = 0.01
	opt := NewAdamW([]*g.Tensor{p}, lr, wd)

	before := append([]float32{}, p.Data()...)

	// Build a graph that produces zero gradient (loss = sum(p)*0).
	// Just call Backward with no graph — AdamW reads p.Grad which is nil
	// and skips the gradient term, but the decoupled-decay term still
	// fires only if grad is non-nil.
	// So we need a real graph that produces a zero gradient: loss = p·0.
	for step := 0; step < 50; step++ {
		opt.ZeroGrad()
		zero := g.Zeros(3)
		loss := g.Sum(g.Mul(p, zero))
		loss.Backward()
		opt.Step()
	}

	// Each step should multiply by (1 - lr*wd) = 0.999. After 50 steps:
	// expected ≈ before * 0.999^50 ≈ before * 0.9512.
	expectedFactor := float32(math.Pow(1-float64(lr)*float64(wd), 50))
	for i, b := range before {
		want := b * expectedFactor
		got := p.Data()[i]
		if math.Abs(float64(got-want)) > math.Abs(float64(b))*0.01 {
			t.Fatalf("[%d] before=%g got=%g want≈%g (factor=%g)", i, b, got, want, expectedFactor)
		}
	}
}

func TestClipGradNormBelowThreshold(t *testing.T) {
	// All gradients well below 1.0 — clipping should be a no-op.
	p1 := g.NewTensor([]float32{1, 2}, 2)
	p2 := g.NewTensor([]float32{3, 4}, 2)
	p1.SetRequiresGrad(true)
	p2.SetRequiresGrad(true)

	loss := g.Sum(g.Add(g.Mul(p1, g.NewTensor([]float32{0.1, 0.1}, 2)),
		g.Mul(p2, g.NewTensor([]float32{0.1, 0.1}, 2))))
	loss.Backward()

	// Total grad-norm: sqrt(0.1²+0.1²+0.1²+0.1²) = sqrt(0.04) = 0.2
	totalNorm := ClipGradNorm([]*g.Tensor{p1, p2}, 1.0)
	if math.Abs(float64(totalNorm-0.2)) > 1e-5 {
		t.Fatalf("totalNorm = %g, want 0.2", totalNorm)
	}
	for _, v := range p1.Grad().Data() {
		if math.Abs(float64(v-0.1)) > 1e-6 {
			t.Fatalf("clipping below threshold modified grad: %g", v)
		}
	}
}

func TestClipGradNormAboveThreshold(t *testing.T) {
	// Gradient much larger than maxNorm — should be scaled.
	p := g.NewTensor([]float32{1, 1}, 2)
	p.SetRequiresGrad(true)

	loss := g.Sum(g.Mul(p, g.NewTensor([]float32{10, 10}, 2)))
	loss.Backward()

	// Total norm: sqrt(10² + 10²) = sqrt(200) ≈ 14.14
	totalNorm := ClipGradNorm([]*g.Tensor{p}, 1.0)
	if math.Abs(float64(totalNorm-14.142136)) > 1e-3 {
		t.Fatalf("totalNorm = %g, want ≈14.14", totalNorm)
	}
	// After clipping, each grad element should be 10 * 1.0 / 14.14 ≈ 0.707
	wantClipped := float32(10) / float32(math.Sqrt(200))
	for i, v := range p.Grad().Data() {
		if math.Abs(float64(v-wantClipped)) > 1e-3 {
			t.Fatalf("[%d] grad=%g, want %g", i, v, wantClipped)
		}
	}
	// Final norm should be ≤ 1.0.
	var sumSq float64
	for _, v := range p.Grad().Data() {
		sumSq += float64(v) * float64(v)
	}
	finalNorm := math.Sqrt(sumSq)
	if finalNorm > 1.0+1e-3 {
		t.Fatalf("after clip totalNorm = %g, want ≤ 1.0", finalNorm)
	}
}

func TestClipGradNormHandlesNilGrad(t *testing.T) {
	// One param has no grad (e.g. wasn't used in the graph). Must not panic.
	p1 := g.NewTensor([]float32{1, 2}, 2)
	p2 := g.NewTensor([]float32{3, 4}, 2) // no grad set
	p1.SetRequiresGrad(true)

	loss := g.Sum(g.Mul(p1, g.NewTensor([]float32{1, 1}, 2)))
	loss.Backward()

	// p2.Grad() is nil; ClipGradNorm must skip it.
	_ = ClipGradNorm([]*g.Tensor{p1, p2}, 1.0)
}
