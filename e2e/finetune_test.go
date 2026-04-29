//go:build darwin && e2e

package e2e

import (
	"testing"
	"time"

	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/optim"
)

// TestFinetunePretrainedGPT2 downloads GPT-2 small, takes a single
// sentence, and runs a few fine-tuning steps. The point is to prove
// the training loop works on real pretrained weights — we expect
// loss to drop monotonically as the model overfits to the target.
//
// This test is heavy: 131 MB download + 124M param training steps on
// CPU (~ 10s per step). Keep step count low.
func TestFinetunePretrainedGPT2(t *testing.T) {
	cacheDir := t.TempDir()

	t.Log("Downloading GPT-2 small...")
	if err := model.DownloadGPT2("openai-community/gpt2", cacheDir); err != nil {
		t.Fatalf("download: %v", err)
	}

	tok, err := model.LoadTokenizer(cacheDir+"/vocab.json", cacheDir+"/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	cfg := model.GPT2Small()
	t.Log("Loading model...")
	loadStart := time.Now()
	gpt, err := model.LoadGPT2(cacheDir, cfg)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	t.Logf("Loaded %d params in %v", gpt.CountParameters(), time.Since(loadStart).Round(time.Millisecond))

	// Tiny training corpus — one short sentence. Fine-tuning to memorise
	// it is enough to demonstrate the loop works.
	tokens := tok.Encode("Gorch is a deep learning framework written in Go.")
	t.Logf("Training on %d tokens", len(tokens))

	const steps = 5
	const lr = 1e-5 // small lr — full pretrained weights, don't blow them up
	opt := optim.NewAdam(gpt.Parameters(), lr)

	losses := make([]float32, 0, steps+1)
	losses = append(losses, model.CausalLMLoss(gpt, tokens).Data()[0])
	t.Logf("step 0 loss: %g", losses[0])

	for i := 0; i < steps; i++ {
		stepStart := time.Now()
		opt.ZeroGrad()
		loss := model.CausalLMLoss(gpt, tokens)
		loss.Backward()
		opt.Step()
		losses = append(losses, loss.Data()[0])
		t.Logf("step %d loss: %g (%v)", i+1, losses[i+1], time.Since(stepStart).Round(time.Millisecond))
	}

	if !(losses[len(losses)-1] < losses[0]) {
		t.Fatalf("loss did not decrease: start=%g end=%g", losses[0], losses[len(losses)-1])
	}
}
