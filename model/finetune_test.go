//go:build darwin

package model

import (
	"testing"

	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// TestCausalLMLossOverfit confirms the public CausalLMLoss helper
// trains a tiny model end-to-end. We don't load real GPT-2 weights
// here — that's the e2e test. This is the unit-level smoke test.
func TestCausalLMLossOverfit(t *testing.T) {
	model := nn.NewGPT(16, 16, 2, 1, 16)
	tokens := []int{1, 2, 3, 4, 5, 6}

	opt := optim.NewAdam(model.Parameters(), 0.05)

	first := CausalLMLoss(model, tokens).Data()[0]
	for step := 0; step < 50; step++ {
		opt.ZeroGrad()
		loss := CausalLMLoss(model, tokens)
		loss.Backward()
		opt.Step()
	}
	last := CausalLMLoss(model, tokens).Data()[0]

	if !(last < first*0.5) {
		t.Fatalf("loss did not drop: first=%g last=%g", first, last)
	}
}

func TestCausalLMLossPanicsOnShortSequence(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for sequence length < 2")
		}
	}()
	model := nn.NewGPT(8, 4, 2, 1, 8)
	CausalLMLoss(model, []int{1})
}
