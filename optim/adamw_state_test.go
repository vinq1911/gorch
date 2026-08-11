//go:build darwin

package optim

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

func setGrad(p *g.Tensor, v float32) {
	grad := g.Zeros(p.Shape()...)
	for i := range grad.Data() {
		grad.Data()[i] = v
	}
	p.ZeroGrad()
	// Accumulate via a backward-free path: assign directly.
	pGrad := grad
	// gorch has no SetGrad; drive through a trivial graph instead.
	_ = pGrad
	y := g.Mul(p, grad) // dy/dp = grad values
	s := g.Sum(y)
	s.Backward()
}

// TestAdamWGroupsLR — two param groups train at their own LRs, and a
// scheduler SetLR rescales both proportionally.
func TestAdamWGroupsLR(t *testing.T) {
	a := g.NewTensor([]float32{1, 1}, 2).SetRequiresGrad(true)
	b := g.NewTensor([]float32{1, 1}, 2).SetRequiresGrad(true)
	opt := NewAdamWGroups([]ParamGroup{
		{Params: []*g.Tensor{a}, LR: 0.1},
		{Params: []*g.Tensor{b}, LR: 0.2},
	}, 0)

	setGrad(a, 1)
	setGrad(b, 1)
	opt.Step()

	da := 1 - a.Data()[0]
	db := 1 - b.Data()[0]
	if da <= 0 || db <= 0 {
		t.Fatalf("params did not move: da=%v db=%v", da, db)
	}
	if r := db / da; math.Abs(float64(r)-2) > 1e-3 {
		t.Fatalf("group-B update / group-A update = %v, want 2 (LR ratio)", r)
	}

	// SetLR halves the base; ratio must persist.
	a.Data()[0], a.Data()[1] = 1, 1
	b.Data()[0], b.Data()[1] = 1, 1
	opt2 := NewAdamWGroups([]ParamGroup{
		{Params: []*g.Tensor{a}, LR: 0.1},
		{Params: []*g.Tensor{b}, LR: 0.2},
	}, 0)
	opt2.SetLR(0.05)
	setGrad(a, 1)
	setGrad(b, 1)
	opt2.Step()
	da2 := 1 - a.Data()[0]
	db2 := 1 - b.Data()[0]
	if r := db2 / da2; math.Abs(float64(r)-2) > 1e-3 {
		t.Fatalf("after SetLR, update ratio = %v, want 2", r)
	}
	if math.Abs(float64(da2/da)-0.5) > 1e-3 {
		t.Fatalf("SetLR(0.05) update / LR-0.1 update = %v, want 0.5", da2/da)
	}
}

// TestAdamWStateRoundtrip — StateTensors/LoadState restores the exact
// optimizer trajectory: a fresh optimizer with restored state must
// produce bit-identical updates to the uninterrupted one.
func TestAdamWStateRoundtrip(t *testing.T) {
	mk := func() *g.Tensor {
		return g.NewTensor([]float32{0.5, -0.25, 1.5}, 3).SetRequiresGrad(true)
	}
	p1 := mk()
	opt1 := NewAdamW([]*g.Tensor{p1}, 0.01, 0.01)
	for i := 0; i < 3; i++ {
		setGrad(p1, float32(i+1))
		opt1.Step()
	}

	// Snapshot state + params (copies).
	step, m, v := opt1.StateTensors()
	mCopy := [][]float32{append([]float32{}, m[0]...)}
	vCopy := [][]float32{append([]float32{}, v[0]...)}
	paramCopy := append([]float32{}, p1.Data()...)

	// Continue the original two more steps.
	for i := 3; i < 5; i++ {
		setGrad(p1, float32(i+1))
		opt1.Step()
	}

	// Resume a fresh optimizer from the snapshot.
	p2 := mk()
	copy(p2.Data(), paramCopy)
	opt2 := NewAdamW([]*g.Tensor{p2}, 0.01, 0.01)
	if err := opt2.LoadState(step, mCopy, vCopy); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	for i := 3; i < 5; i++ {
		setGrad(p2, float32(i+1))
		opt2.Step()
	}

	for i := range p1.Data() {
		if p1.Data()[i] != p2.Data()[i] {
			t.Fatalf("resumed trajectory diverged at element %d: %v vs %v", i, p2.Data()[i], p1.Data()[i])
		}
	}

	// Size validation.
	if err := opt2.LoadState(1, [][]float32{{1}}, [][]float32{{1}}); err == nil {
		t.Fatal("LoadState accepted mis-sized moments")
	}
}
