//go:build darwin

package nn

import (
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

// TestGPTAutogradReachesAllParameters confirms one training step on a
// tiny GPT updates every parameter group: token embedding, position
// embedding, attention Q/K/V/Wo, FFN1/FFN2, both layer norms, final
// layer norm, and the LM head. If any group is unchanged, the
// autograd graph isn't reaching it.
func TestGPTAutogradReachesAllParameters(t *testing.T) {
	g.InitMetal()
	model := NewGPT(16, 8, 2, 1, 16)

	// Snapshot every parameter before the step.
	before := make([][]float32, 0, len(model.Parameters()))
	for _, p := range model.Parameters() {
		snap := make([]float32, p.Size())
		copy(snap, p.Data())
		before = append(before, snap)
	}

	// LM training: predict next token. inputs t[0..n-2], targets t[1..n-1].
	tokens := []int{1, 2, 3, 4}
	inputs := tokens[:len(tokens)-1]
	tgtIDs := tokens[1:]

	logits := model.Forward(inputs) // (seq, vocab)
	targets := make([]float32, len(tgtIDs))
	for i, id := range tgtIDs {
		targets[i] = float32(id)
	}
	tgt := g.NewTensor(targets, len(tgtIDs), 1)

	loss := g.CrossEntropyLoss(logits, tgt)
	if !loss.RequiresGrad() {
		t.Fatal("loss does not require grad — autograd graph broken")
	}
	loss.Backward()

	opt := optim.NewSGD(model.Parameters(), 0.1, 0)
	opt.Step()

	names := []string{
		"TokenEmbed.Weight", "PosEmbed.Weight",
		"Attn.Wq.W", "Attn.Wq.b",
		"Attn.Wk.W", "Attn.Wk.b",
		"Attn.Wv.W", "Attn.Wv.b",
		"Attn.Wo.W", "Attn.Wo.b",
		"FFN1.W", "FFN1.b",
		"FFN2.W", "FFN2.b",
		"Norm1.W", "Norm1.b",
		"Norm2.W", "Norm2.b",
		"FinalNorm.W", "FinalNorm.b",
		"LMHead.W", "LMHead.b",
	}
	params := model.Parameters()
	if len(names) != len(params) {
		t.Logf("warning: name list (%d) and param list (%d) length differ", len(names), len(params))
	}
	for i, p := range params {
		moved := false
		for j := range p.Data() {
			if p.Data()[j] != before[i][j] {
				moved = true
				break
			}
		}
		var label string
		if i < len(names) {
			label = names[i]
		} else {
			label = "(unnamed)"
		}
		if !moved {
			t.Errorf("parameter %d (%s) did not change after training step — autograd not reaching it", i, label)
		}
	}
}

// TestGPTLossDecreasesOverfitOneSequence trains a tiny GPT to memorise
// a single token sequence. After enough steps, the loss should drop
// significantly — not necessarily to zero (the LMHead's tied-with-
// embedding init makes that hard) but well below the starting value.
func TestGPTLossDecreasesOverfitOneSequence(t *testing.T) {
	g.InitMetal()
	model := NewGPT(16, 16, 2, 1, 16)

	tokens := []int{1, 2, 3, 4, 5}
	inputs := tokens[:len(tokens)-1]
	tgtIDs := tokens[1:]
	targets := make([]float32, len(tgtIDs))
	for i, id := range tgtIDs {
		targets[i] = float32(id)
	}
	tgt := g.NewTensor(targets, len(tgtIDs), 1)

	opt := optim.NewAdam(model.Parameters(), 0.05)

	var firstLoss, lastLoss float32
	for step := 0; step < 50; step++ {
		opt.ZeroGrad()
		logits := model.Forward(inputs)
		loss := g.CrossEntropyLoss(logits, tgt)
		if step == 0 {
			firstLoss = loss.Data()[0]
		}
		lastLoss = loss.Data()[0]
		loss.Backward()
		opt.Step()
	}

	if !(lastLoss < firstLoss*0.5) {
		t.Fatalf("loss did not drop substantially: first=%g last=%g", firstLoss, lastLoss)
	}
}
