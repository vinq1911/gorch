//go:build darwin

package model

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// CausalLMLoss runs the model forward on tokens[:n-1] and returns
// the next-token cross-entropy loss against tokens[1:]. The returned
// tensor is a scalar with autograd attached, ready for Backward()
// followed by an optimiser step.
//
// This is a thin convenience for the standard LM training pattern
// where the input and target sequences differ by a one-position
// shift. It does not handle batching across sequences (the rest of
// gorch operates one sequence at a time), so for multi-sequence
// training, sum or average losses across sequences in the caller.
//
// All parameters created by nn.NewGPT already have requires_grad=true,
// and LoadGPT2 preserves that, so fine-tuning a pretrained model
// requires nothing more than calling this loss + optim.Step in a loop.
func CausalLMLoss(model *nn.GPT, tokens []int) *g.Tensor {
	if len(tokens) < 2 {
		panic("model: CausalLMLoss needs at least 2 tokens")
	}
	inputs := tokens[:len(tokens)-1]
	targetIDs := tokens[1:]

	logits := model.Forward(inputs) // (n-1, vocab)

	tgt := make([]float32, len(targetIDs))
	for i, id := range targetIDs {
		tgt[i] = float32(id)
	}
	tgtTensor := g.NewTensor(tgt, len(targetIDs), 1)

	return g.CrossEntropyLoss(logits, tgtTensor)
}
