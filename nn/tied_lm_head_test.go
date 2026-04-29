//go:build darwin

package nn

import (
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

// TestTieLMHeadAliasesEmbedding: after tying, the LMHead weight
// pointer must be the same Go object as the embedding weight, and
// any in-place modification to one is visible through the other.
func TestTieLMHeadAliasesEmbedding(t *testing.T) {
	gpt := NewGPT(16, 8, 2, 1, 16)
	if gpt.TiedLMHead {
		t.Fatal("untied GPT reports TiedLMHead=true")
	}
	gpt.TieLMHeadToEmbedding()
	if !gpt.TiedLMHead {
		t.Fatal("after Tie, TiedLMHead is false")
	}
	if gpt.TokenEmbed.Weight != gpt.LMHead.Weight {
		t.Fatal("after tie, LMHead.Weight is not the same pointer as TokenEmbed.Weight")
	}
	// Mutate one, check the other reflects it.
	gpt.TokenEmbed.Weight.Data()[0] = 42
	if gpt.LMHead.Weight.Data()[0] != 42 {
		t.Fatal("tied weights do not share storage")
	}
}

// TestTiedParametersDeduplicates: Parameters() must list the shared
// embedding tensor only once. Optimizers iterate this list, so a
// duplicate would step the same buffer twice with two gradient
// signals and corrupt training.
func TestTiedParametersDeduplicates(t *testing.T) {
	gpt := NewGPT(16, 8, 2, 1, 16)
	beforeCount := len(gpt.Parameters())
	gpt.TieLMHeadToEmbedding()
	afterCount := len(gpt.Parameters())
	if afterCount != beforeCount-1 {
		t.Fatalf("after tie, parameter count = %d (was %d); expected to drop by 1", afterCount, beforeCount)
	}

	// Make sure the shared tensor appears exactly once.
	seen := 0
	for _, p := range gpt.Parameters() {
		if p == gpt.TokenEmbed.Weight {
			seen++
		}
	}
	if seen != 1 {
		t.Fatalf("shared tensor appears %d times in Parameters() — must be 1", seen)
	}
}

// TestTiedGradAccumulatesFromBothPaths: with tying, a backward pass
// through the LM head and the embedding should accumulate gradients
// into the same tensor buffer. The optimizer step should then see
// the combined gradient.
func TestTiedGradAccumulatesFromBothPaths(t *testing.T) {
	gpt := NewGPT(8, 4, 2, 1, 8)
	gpt.TieLMHeadToEmbedding()

	// Snapshot before.
	beforeWeight := append([]float32{}, gpt.TokenEmbed.Weight.Data()...)

	tokens := []int{1, 2, 3}
	targetIDs := []int{2, 3, 4}
	tgtData := make([]float32, len(targetIDs))
	for i, id := range targetIDs {
		tgtData[i] = float32(id)
	}
	tgt := g.NewTensor(tgtData, len(targetIDs), 1)

	// Reset grads, run forward + backward.
	for _, p := range gpt.Parameters() {
		p.ZeroGrad()
	}
	logits := gpt.Forward(tokens)
	loss := g.CrossEntropyLoss(logits, tgt)
	loss.Backward()

	// Both the embedding lookup and the LM-head Linear should have
	// accumulated grads into the shared tensor — meaning the grad is
	// non-zero in many positions.
	tokenGrad := gpt.TokenEmbed.Weight.Grad()
	if tokenGrad == nil {
		t.Fatal("embed weight has no grad")
	}
	nonZero := 0
	for _, v := range tokenGrad.Data() {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Fatal("tied weight grad is all zero — gradient did not flow")
	}

	// Step the optimizer. The shared tensor must change.
	opt := optim.NewSGD(gpt.Parameters(), 0.1, 0)
	opt.Step()
	moved := false
	for i, before := range beforeWeight {
		if gpt.TokenEmbed.Weight.Data()[i] != before {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatal("after step, tied weight did not change")
	}
}

// TestTieIdempotent: calling TieLMHeadToEmbedding twice is a no-op.
func TestTieIdempotent(t *testing.T) {
	gpt := NewGPT(8, 4, 2, 1, 8)
	gpt.TieLMHeadToEmbedding()
	count := len(gpt.Parameters())
	gpt.TieLMHeadToEmbedding()
	if got := len(gpt.Parameters()); got != count {
		t.Fatalf("idempotency violated: 1st call %d params, 2nd call %d", count, got)
	}
}
