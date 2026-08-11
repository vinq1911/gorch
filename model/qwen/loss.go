//go:build darwin

package qwen

import (
	"fmt"

	g "github.com/vinq1911/gorch"
)

// SupervisedLoss — gathered supervised-position LM loss
// (plan 0008 §3.5).
//
// Instead of materialising full-sequence logits (seq × 168,336 ≈
// 690 MB f32 at seq 1024), the final-norm hidden states are gathered
// at the supervised positions only, the split tied head runs on those
// rows, and CrossEntropyLoss grades each position's next-token
// prediction. The gather IS the loss mask: unsupervised positions
// contribute nothing and cost nothing.
//
// supervised lists sequence positions i (0-based, each ≤ len(tokens)-2)
// whose next-token target is tokens[i+1].
//
// A CE-chunking hook (splitting the supervised rows into column- or
// row-chunks before the head matmul) is deliberately NOT implemented:
// at the overfit gate's ≤256-token sequences the gathered logits stay
// under ~70 MB per copy. If M2's 1024-token sequences push peak RSS,
// chunk here — the call site is the only place that needs to change.
func (vm *VoiceModel) SupervisedLoss(tokens []int, supervised []int) *g.Tensor {
	if len(supervised) == 0 {
		panic("qwen: SupervisedLoss requires ≥1 supervised position")
	}
	for _, pos := range supervised {
		if pos < 0 || pos >= len(tokens)-1 {
			panic(fmt.Sprintf("qwen: supervised position %d needs a next-token target in a %d-token sequence", pos, len(tokens)))
		}
	}
	h := vm.ForwardHidden(tokens) // (seq, hidden)
	hs := g.Gather(h, supervised) // (M, hidden) — autograd scatter-add backward
	logits := vm.Embed.Logits(hs) // (M, ExtVocab), grad → h and Ext only
	targets := make([]float32, len(supervised))
	for i, pos := range supervised {
		targets[i] = float32(tokens[pos+1])
	}
	return g.CrossEntropyLoss(logits, g.NewTensor(targets, len(targets), 1))
}
