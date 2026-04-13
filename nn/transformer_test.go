//go:build darwin

package nn

import (
	"fmt"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

func TestEmbeddingForward(t *testing.T) {
	emb := NewEmbedding(100, 16)
	out := emb.Forward([]int{5, 10, 15})
	if out.Shape()[0] != 3 || out.Shape()[1] != 16 {
		t.Fatalf("shape = %v, want [3, 16]", out.Shape())
	}
}

func TestLayerNormForward(t *testing.T) {
	ln := NewLayerNorm(4)
	x := g.NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	out := ln.Forward(x)

	if out.Shape()[0] != 2 || out.Shape()[1] != 4 {
		t.Fatalf("shape = %v, want [2, 4]", out.Shape())
	}

	// Each row should have approximately zero mean (since gamma=1, beta=0)
	for row := 0; row < 2; row++ {
		var sum float32
		for j := 0; j < 4; j++ {
			sum += out.Data()[row*4+j]
		}
		mean := sum / 4.0
		if mean > 0.01 || mean < -0.01 {
			t.Fatalf("row %d mean = %f, want ~0", row, mean)
		}
	}
}

func TestLayerNormBackward(t *testing.T) {
	ln := NewLayerNorm(4)
	x := g.NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	x.SetRequiresGrad(true)

	out := ln.Forward(x)
	loss := g.Sum(out)
	loss.Backward()

	if x.Grad() == nil {
		t.Fatal("x.grad is nil")
	}
	if ln.Weight.Grad() == nil {
		t.Fatal("weight.grad is nil")
	}
	if ln.Bias.Grad() == nil {
		t.Fatal("bias.grad is nil")
	}
}

func TestMultiHeadAttentionForward(t *testing.T) {
	dim := 16
	numHeads := 4
	seqLen := 8

	mha := NewMultiHeadAttention(dim, numHeads)
	x := g.RandN(seqLen, dim)
	out := mha.Forward(x, seqLen)

	if out.Shape()[0] != seqLen || out.Shape()[1] != dim {
		t.Fatalf("shape = %v, want [%d, %d]", out.Shape(), seqLen, dim)
	}
}

func TestMultiHeadAttentionBackward(t *testing.T) {
	dim := 8
	numHeads := 2
	seqLen := 4

	mha := NewMultiHeadAttention(dim, numHeads)
	x := g.RandN(seqLen, dim)
	x.SetRequiresGrad(true)

	out := mha.Forward(x, seqLen)
	loss := g.Sum(out)
	loss.Backward()

	if x.Grad() == nil {
		t.Fatal("x.grad is nil after MHA backward")
	}
	// Check that all weight gradients exist
	for i, p := range mha.Parameters() {
		if p.Grad() == nil {
			t.Fatalf("param %d grad is nil", i)
		}
	}
}

func TestTransformerBlockForward(t *testing.T) {
	dim := 16
	numHeads := 4
	seqLen := 8

	block := NewTransformerBlock(dim, numHeads)
	x := g.RandN(seqLen, dim)
	out := block.Forward(x, seqLen)

	if out.Shape()[0] != seqLen || out.Shape()[1] != dim {
		t.Fatalf("shape = %v, want [%d, %d]", out.Shape(), seqLen, dim)
	}
}

func TestTransformerBlockBackward(t *testing.T) {
	dim := 8
	numHeads := 2
	seqLen := 4

	block := NewTransformerBlock(dim, numHeads)
	x := g.RandN(seqLen, dim)
	x.SetRequiresGrad(true)

	out := block.Forward(x, seqLen)
	loss := g.Sum(out)
	loss.Backward()

	if x.Grad() == nil {
		t.Fatal("x.grad is nil after TransformerBlock backward")
	}
}

func TestGPTForward(t *testing.T) {
	vocabSize := 50
	dim := 16
	numHeads := 4
	numLayers := 2
	maxSeq := 32

	model := NewGPT(vocabSize, dim, numHeads, numLayers, maxSeq)

	tokens := []int{5, 10, 15, 20, 25}
	logits := model.Forward(tokens)

	if logits.Shape()[0] != 5 || logits.Shape()[1] != vocabSize {
		t.Fatalf("logits shape = %v, want [5, %d]", logits.Shape(), vocabSize)
	}

	fmt.Printf("  GPT params: %d\n", model.CountParameters())
}

func TestGPTBackward(t *testing.T) {
	model := NewGPT(30, 8, 2, 1, 16)

	tokens := []int{1, 5, 10}
	logits := model.Forward(tokens)

	// Use the last token's logits as loss
	loss := g.Sum(logits)
	loss.Backward()

	// Check all parameters have gradients
	for i, p := range model.Parameters() {
		if p.Grad() == nil {
			t.Fatalf("param %d (size=%d) has nil grad", i, p.Size())
		}
	}
}

func TestGPTTrainStep(t *testing.T) {
	// Test that a single training step works: forward → loss → backward → step
	model := NewGPT(20, 8, 2, 1, 16)

	tokens := []int{1, 5, 10, 3}
	// Target: predict next token (shift by 1)
	targets := g.NewTensor([]float32{5, 10, 3, 0}, 4, 1)

	logits := model.Forward(tokens)
	loss := g.CrossEntropyLoss(logits, targets)

	initialLoss := loss.Data()[0]
	loss.Backward()

	// Multiple Adam steps to ensure loss decreases (single SGD step can be flaky
	// with random init on small models)
	opt := optim.NewAdam(model.Parameters(), 0.01)
	for step := 0; step < 5; step++ {
		opt.ZeroGrad()
		l := g.CrossEntropyLoss(model.Forward(tokens), targets)
		l.Backward()
		opt.Step()
	}

	logits2 := model.Forward(tokens)
	loss2 := g.CrossEntropyLoss(logits2, targets)

	if loss2.Data()[0] >= initialLoss {
		t.Fatalf("loss did not decrease after 5 Adam steps: %.4f → %.4f", initialLoss, loss2.Data()[0])
	}
	fmt.Printf("  GPT train step: loss %.4f → %.4f (5 steps)\n", initialLoss, loss2.Data()[0])
}
