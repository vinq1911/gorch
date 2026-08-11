//go:build darwin

package optim

import (
	"math"
	"math/rand"
	"testing"

	g "github.com/vinq1911/gorch"
)

// Plan 0009 K7 gate: the vectorized Accelerate AdamW step must produce
// a trajectory bit-comparable to the scalar Go reference — the loss
// curve over 20 synthetic steps must match within 1e-6. The scalar
// loop stays in adamw.go behind UseScalarAdamW as the oracle.

// advSetup builds a deterministic two-group parameter set (mirroring
// the M1 NewAdamWGroups structure: base group + a group at 10× LR).
func advSetup(seed int64) ([]*g.Tensor, *AdamW) {
	rng := rand.New(rand.NewSource(seed))
	mk := func(shape ...int) *g.Tensor {
		t := g.Zeros(shape...)
		d := t.Data()
		for i := range d {
			d[i] = float32(rng.NormFloat64()) * 0.5
		}
		t.SetRequiresGrad(true)
		return t
	}
	// Odd sizes on purpose: exercise the vector body + scalar tail.
	p1 := mk(37, 11) // group 0 (base LR)
	p2 := mk(129)    // group 0
	p3 := mk(53, 7)  // group 1 (10× LR — the M1 per-group LR structure)
	opt := NewAdamWGroups([]ParamGroup{
		{Params: []*g.Tensor{p1, p2}, LR: 1e-3},
		{Params: []*g.Tensor{p3}, LR: 1e-2},
	}, 0.01)
	return []*g.Tensor{p1, p2, p3}, opt
}

// advGrads seeds deterministic per-step gradients: loss = Sum(p ⊙
// noise) has dL/dp = noise, so both trajectories see identical grads
// from the same rng stream as long as the shapes agree.
func advGrads(rng *rand.Rand, params []*g.Tensor) {
	for _, p := range params {
		p.ZeroGrad()
		noise := g.Zeros(p.Shape()...)
		d := noise.Data()
		for i := range d {
			d[i] = float32(rng.NormFloat64())
		}
		g.Sum(g.Mul(p, noise)).Backward()
	}
}

// advLoss is the synthetic loss the curves are compared on: mean(p²)
// over every param.
func advLoss(params []*g.Tensor) float64 {
	var sum float64
	var n int
	for _, p := range params {
		for _, v := range p.Data() {
			sum += float64(v) * float64(v)
		}
		n += p.Size()
	}
	return sum / float64(n)
}

func TestAdamWVectorizedMatchesScalarTrajectory(t *testing.T) {
	const steps = 20

	run := func(scalar bool) []float64 {
		prev := UseScalarAdamW
		UseScalarAdamW = scalar
		defer func() { UseScalarAdamW = prev }()

		params, opt := advSetup(1234)
		gradRng := rand.New(rand.NewSource(999))
		curve := make([]float64, 0, steps)
		for s := 0; s < steps; s++ {
			advGrads(gradRng, params)
			opt.Step()
			curve = append(curve, advLoss(params))
		}
		return curve
	}

	scalarCurve := run(true)
	vectorCurve := run(false)

	for s := 0; s < steps; s++ {
		diff := math.Abs(scalarCurve[s] - vectorCurve[s])
		if diff > 1e-6 {
			t.Fatalf("step %d: loss curve diverged: scalar %.9f vs vector %.9f (|Δ| %.3g > 1e-6)",
				s, scalarCurve[s], vectorCurve[s], diff)
		}
	}
	t.Logf("20-step trajectory parity: final loss scalar %.9f vs vector %.9f (|Δ| %.3g)",
		scalarCurve[steps-1], vectorCurve[steps-1],
		math.Abs(scalarCurve[steps-1]-vectorCurve[steps-1]))
}

// TestAdamWVectorizedParamParity compares the raw parameter values (not
// just the loss curve) after 20 steps — a tighter check that the fused
// C loop implements the exact update rule, per-group LRs included.
func TestAdamWVectorizedParamParity(t *testing.T) {
	const steps = 20

	run := func(scalar bool) []*g.Tensor {
		prev := UseScalarAdamW
		UseScalarAdamW = scalar
		defer func() { UseScalarAdamW = prev }()
		params, opt := advSetup(4321)
		gradRng := rand.New(rand.NewSource(777))
		for s := 0; s < steps; s++ {
			advGrads(gradRng, params)
			opt.Step()
		}
		return params
	}

	sp := run(true)
	vp := run(false)
	for i := range sp {
		sd, vd := sp[i].Data(), vp[i].Data()
		var maxDiff float64
		for j := range sd {
			d := math.Abs(float64(sd[j]) - float64(vd[j]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-6 {
			t.Errorf("param %d: max abs diff %.3g > 1e-6 after %d steps", i, maxDiff, steps)
		} else {
			t.Logf("param %d: max abs diff %.3g after %d steps", i, maxDiff, steps)
		}
	}
}
