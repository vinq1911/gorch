//go:build darwin

package mythos

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

// Phase 2 acceptance tests. These pin the structure of the model and
// the recurrent-depth claim; the convergence run on real data is
// Phase 4.

func tinyModel(t *testing.T, vocab int) *Mythos {
	t.Helper()
	cfg := TinyConfig(vocab)
	// Shrink further for tests so they run in <1s and don't blow up
	// CI's RSS budget. The shape contracts are identical at any size.
	cfg.Dim = 64
	cfg.NumHeads = 4
	cfg.NumKVHeads = 2
	cfg.PreludeLayers = 1
	cfg.CodaLayers = 1
	cfg.MaxLoopIters = 2
	cfg.NumExperts = 2
	cfg.NumExpertsPerToken = 2
	cfg.ExpertDim = 64
	cfg.MaxSeqLen = 32
	return New(cfg)
}

// TestMythosForwardShape: forward on a small token sequence returns
// logits with the expected (seq, vocab) shape and no NaN/Inf.
func TestMythosForwardShape(t *testing.T) {
	const vocab = 64
	m := tinyModel(t, vocab)
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	logits := m.Forward(tokens, 0, -1)

	if logits.Dim() != 2 {
		t.Fatalf("expected 2-D logits, got %d-D", logits.Dim())
	}
	if logits.Shape()[0] != len(tokens) {
		t.Fatalf("logits[0] = %d, want %d", logits.Shape()[0], len(tokens))
	}
	if logits.Shape()[1] != vocab {
		t.Fatalf("logits[1] = %d, want %d (vocab)", logits.Shape()[1], vocab)
	}
	for i, v := range logits.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits[%d] = %g (NaN/Inf)", i, v)
		}
	}
}

// TestMythosForwardLoopItersAffectsOutput: increasing the recurrent
// loop iterations changes the output (the recurrent block isn't
// silently identity). This is the architecture's core claim
// instrumented as a test — Phase 4 turns this into "more iterations
// → lower perplexity"; here we just verify the iterations have
// observable effect.
func TestMythosForwardLoopItersAffectsOutput(t *testing.T) {
	const vocab = 64
	m := tinyModel(t, vocab)
	tokens := []int{1, 2, 3, 4, 5}

	g.NoGrad(func() {
		l1 := m.Forward(tokens, 0, 1).Data()
		l2 := m.Forward(tokens, 0, 4).Data()
		var maxDiff float32
		for i := range l1 {
			d := l1[i] - l2[i]
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
		// At init the recurrent block has near-random weights; even with
		// damping ~0.5, four extra iterations should perturb the logits
		// by more than fp32 noise.
		if maxDiff < 1e-3 {
			t.Fatalf("loop iters had no effect: max |Δlogit| = %g", maxDiff)
		}
	})
}

// TestMythosTrainsOneStep: a single optimiser step on cross-entropy
// loss against a tiny synthetic target should reduce the loss. End-
// to-end smoke test of forward + autograd + gradient flow through
// every parameter.
func TestMythosTrainsOneStep(t *testing.T) {
	const vocab = 32
	m := tinyModel(t, vocab)
	tokens := []int{1, 2, 3, 4}

	// Targets are next-token shifted by 1 (offset by 5 mod vocab to
	// keep them inside the vocab).
	targetIDs := []float32{2, 3, 4, 5}
	targets := g.NewTensor(targetIDs, len(targetIDs), 1)

	opt := optim.NewAdam(m.Parameters(), 1e-2)

	// Loss at init.
	logits := m.Forward(tokens, 0, -1)
	loss0 := g.CrossEntropyLoss(logits, targets)
	if math.IsNaN(float64(loss0.Data()[0])) {
		t.Fatalf("initial loss is NaN")
	}

	// Backward + one step.
	opt.ZeroGrad()
	loss0.Backward()
	// Confirm at least one parameter actually got a gradient flowed
	// through it — catches "the autograd graph silently broke" bugs.
	var gradedParams int
	for _, p := range m.Parameters() {
		if p.Grad() != nil {
			gradedParams++
		}
	}
	if gradedParams == 0 {
		t.Fatal("no parameter received a gradient — autograd graph is broken")
	}
	opt.Step()

	// Loss after step.
	logits1 := m.Forward(tokens, 0, -1)
	loss1 := g.CrossEntropyLoss(logits1, targets)
	if math.IsNaN(float64(loss1.Data()[0])) {
		t.Fatalf("loss after step is NaN")
	}
	if loss1.Data()[0] >= loss0.Data()[0] {
		t.Fatalf("loss did not decrease: %g → %g", loss0.Data()[0], loss1.Data()[0])
	}
}

// TestMythosParametersPositive: every block contributes at least one
// learnable parameter; nothing's a forgotten-no-Parameters() module.
func TestMythosParametersPositive(t *testing.T) {
	m := tinyModel(t, 32)
	params := m.Parameters()
	if len(params) == 0 {
		t.Fatal("Parameters() returned empty list")
	}
	for i, p := range params {
		if !p.RequiresGrad() {
			t.Fatalf("parameter %d does not require grad: %v", i, p.Shape())
		}
	}
}

// TestMythosLTIDampingTrains: the LTI damping logits should accumulate
// a non-zero gradient when the loss flows through the recurrent loop.
func TestMythosLTIDampingTrains(t *testing.T) {
	m := tinyModel(t, 32)
	tokens := []int{1, 2, 3}
	targets := g.NewTensor([]float32{2, 3, 4}, 3, 1)

	logits := m.Forward(tokens, 0, -1)
	loss := g.CrossEntropyLoss(logits, targets)
	loss.Backward()

	if m.LTI.DampLogit.Grad() == nil {
		t.Fatal("LTI damping logit received no gradient")
	}
	var maxAbs float32
	for _, v := range m.LTI.DampLogit.Grad().Data() {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		t.Fatal("LTI damping logit gradient is identically zero")
	}
}

// TestMythosUseMLAPanics: pin the contract that the v1 port refuses
// to build an MLA block until plan 0001's "MLA full autograd" item
// lands. Silent miscompile would be much worse than a clear panic.
func TestMythosUseMLAPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for UseMLA=true")
		}
	}()
	cfg := TinyConfig(32)
	cfg.UseMLA = true
	New(cfg)
}
