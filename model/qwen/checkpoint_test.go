//go:build darwin

package qwen

import (
	"fmt"
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// Gradient equivalence is THE deliverable of activation checkpointing:
// a checkpointing bug does not crash, it silently produces wrong or
// zero LoRA gradients. These tests run the real VoiceModel forward
// (extended embedding → GQA/QK-norm/RoPE/fused causal softmax/SwiGLU
// blocks → tied split head → gathered CE loss) with checkpointing on
// and off from identical weights and compare every trainable tensor's
// gradient elementwise.
//
// The CPU f32 path is the semantic gate and is what runs here: it is
// deterministic, so "equal" means equal to f32 round-off, and any
// structural error (double accumulation, a missing edge, a leaked
// graph) shows up as a gross mismatch rather than something a loose
// tolerance could hide. The GPU bf16 path's equivalence is measured
// separately by the trainer's loss-trajectory comparison, where the
// pre-existing kernel nondeterminism sets the floor.

// checkpointFixture builds two VoiceModels with bit-identical weights,
// one plain and one checkpointed at segment length seg.
func checkpointFixture(t *testing.T, cfg Config, seg int) (plain, ckpt *VoiceModel) {
	t.Helper()
	vc := VoiceConfig{LoRARank: 2, LoRAAlpha: 4, NumExt: 16}
	plain = NewVoiceModel(New(cfg), vc)

	vcCk := vc
	vcCk.CheckpointEvery = seg
	ckpt = NewVoiceModel(New(cfg), vcCk)

	// Copy every tensor across, not just the trainable ones: the random
	// init differs per model.
	copyAll := func(dst, src *g.Tensor) {
		t.Helper()
		if dst.Size() != src.Size() {
			t.Fatalf("shape mismatch copying weights: %v vs %v", dst.Shape(), src.Shape())
		}
		copy(dst.Data(), src.Data())
	}
	srcParams := plain.Base.Parameters()
	dstParams := ckpt.Base.Parameters()
	if len(srcParams) != len(dstParams) {
		t.Fatalf("parameter count mismatch: %d vs %d", len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		copyAll(dstParams[i], srcParams[i])
	}
	_, sp := plain.TrainableParams()
	_, dp := ckpt.TrainableParams()
	for i := range sp {
		copyAll(dp[i], sp[i])
	}
	return plain, ckpt
}

// compareGrads asserts every trainable gradient matches elementwise and
// returns the worst relative error seen.
func compareGrads(t *testing.T, plain, ckpt *VoiceModel, tol float64) float64 {
	t.Helper()
	names, pp := plain.TrainableParams()
	_, cp := ckpt.TrainableParams()
	worst := 0.0
	nonzero := 0
	for i, name := range names {
		pg, cg := pp[i].Grad(), cp[i].Grad()
		if pg == nil && cg == nil {
			continue
		}
		if pg == nil || cg == nil {
			t.Fatalf("%s: gradient present on only one side (plain=%v checkpointed=%v)", name, pg != nil, cg != nil)
		}
		a, b := pg.Data(), cg.Data()
		if len(a) != len(b) {
			t.Fatalf("%s: gradient length %d vs %d", name, len(a), len(b))
		}
		for j := range a {
			if a[j] != 0 {
				nonzero++
			}
			d := math.Abs(float64(a[j] - b[j]))
			rel := d / (1e-8 + math.Abs(float64(a[j])))
			if d > 1e-9 && rel > worst {
				worst = rel
			}
			if d > tol*(1e-3+math.Abs(float64(a[j]))) {
				t.Fatalf("%s[%d]: checkpointed grad %v != plain %v (|Δ| %.3g)", name, j, b[j], a[j], d)
			}
		}
	}
	// Guard against the test passing because BOTH sides are all zeros —
	// which is exactly what a broken checkpoint would produce.
	if nonzero == 0 {
		t.Fatal("all reference gradients are zero; the test proves nothing")
	}
	return worst
}

// TestCheckpointGradientEquivalence is the gate: LoRA A/B and the
// extended-embedding rows must receive identical gradients with
// checkpointing on and off, at every segment length.
func TestCheckpointGradientEquivalence(t *testing.T) {
	cfg := tinyCfg()
	cfg.NumLayers = 4
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2, 11, 72}
	supervised := []int{2, 3, 5, 6, 8}

	for _, seg := range []int{1, 2, 3, 4} {
		t.Run(fmt.Sprintf("every=%d", seg), func(t *testing.T) {
			plain, ckpt := checkpointFixture(t, cfg, seg)

			lp := plain.SupervisedLoss(tokens, supervised)
			lc := ckpt.SupervisedLoss(tokens, supervised)
			if d := math.Abs(float64(lp.Data()[0] - lc.Data()[0])); d > 1e-5 {
				t.Fatalf("loss differs before backward: plain %v checkpointed %v (|Δ| %.3g)",
					lp.Data()[0], lc.Data()[0], d)
			}
			lp.Backward()
			lc.Backward()

			worst := compareGrads(t, plain, ckpt, 1e-4)
			t.Logf("segment %d: worst relative gradient error %.3g", seg, worst)
		})
	}
}

// TestCheckpointGradientEquivalenceUnderAccumulation — the trainer runs
// several micro-steps before ZeroGrad, so the recompute's accumulation
// into a parameter's live .grad has to compose with the outer engine's.
func TestCheckpointGradientEquivalenceUnderAccumulation(t *testing.T) {
	cfg := tinyCfg()
	cfg.NumLayers = 4
	plain, ckpt := checkpointFixture(t, cfg, 1)

	batches := []struct {
		tokens []int
		sup    []int
	}{
		{[]int{1, 5, 70, 71, 3, 9, 64, 2}, []int{2, 3, 5}},
		{[]int{7, 66, 12, 4, 73, 8, 2, 40}, []int{0, 4, 6}},
		{[]int{3, 3, 68, 69, 1, 2, 5, 70}, []int{1, 2, 3, 5}},
	}
	scale := func(s float32, x *g.Tensor) *g.Tensor {
		return g.Mul(x, g.NewTensor([]float32{s}, 1))
	}
	for _, b := range batches {
		scale(1.0/3, plain.SupervisedLoss(b.tokens, b.sup)).Backward()
		scale(1.0/3, ckpt.SupervisedLoss(b.tokens, b.sup)).Backward()
	}
	t.Logf("worst relative gradient error after 3 accumulated micro-steps: %.3g",
		compareGrads(t, plain, ckpt, 1e-4))
}

// TestCheckpointGradientEquivalenceFullDepth runs the real 28-layer
// geometry (random weights, short sequence) so the equivalence claim is
// not only tested at truncated depth — a per-layer error that cancels
// at 4 layers would not at 28.
func TestCheckpointGradientEquivalenceFullDepth(t *testing.T) {
	if testing.Short() {
		t.Skip("28-layer random-weight forward is slow")
	}
	cfg := tinyCfg()
	cfg.NumLayers = 28
	plain, ckpt := checkpointFixture(t, cfg, 1)
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2}
	supervised := []int{2, 3, 5, 6}
	plain.SupervisedLoss(tokens, supervised).Backward()
	ckpt.SupervisedLoss(tokens, supervised).Backward()
	t.Logf("28 layers: worst relative gradient error %.3g", compareGrads(t, plain, ckpt, 1e-4))
}

// TestCheckpointForwardValueUnchanged — checkpointing must not perturb
// the forward at all, in training or inference mode. (Under NoGrad the
// checkpoint wrapper is inert by construction; this pins that.)
func TestCheckpointForwardValueUnchanged(t *testing.T) {
	cfg := tinyCfg()
	cfg.NumLayers = 4
	plain, ckpt := checkpointFixture(t, cfg, 2)
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2}

	for _, noGrad := range []bool{false, true} {
		var a, b *g.Tensor
		run := func() {
			a = plain.ForwardHidden(tokens)
			b = ckpt.ForwardHidden(tokens)
		}
		if noGrad {
			g.NoGrad(run)
		} else {
			run()
		}
		ad, bd := a.Data(), b.Data()
		for i := range ad {
			if math.Abs(float64(ad[i]-bd[i])) > 1e-5 {
				t.Fatalf("noGrad=%v hidden[%d]: %v vs %v", noGrad, i, ad[i], bd[i])
			}
		}
	}
}

// TestCheckpointNoBlockGraphRetained — the memory claim at model level:
// with checkpointing on, the graph reachable from the hidden states
// must contain one node per SEGMENT, not per op. A regression here
// keeps the gradients correct and silently gives back the footprint.
func TestCheckpointNoBlockGraphRetained(t *testing.T) {
	cfg := tinyCfg()
	cfg.NumLayers = 8
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2}

	count := func(vm *VoiceModel) int {
		h := vm.ForwardHidden(tokens)
		return g.GraphSize(h)
	}
	plain, ckpt := checkpointFixture(t, cfg, 1)
	nPlain, nCkpt := count(plain), count(ckpt)
	if nCkpt >= nPlain/4 {
		t.Fatalf("checkpointed graph has %d nodes vs %d plain — the segment graphs are still retained", nCkpt, nPlain)
	}
	// 8 checkpoint nodes + the embedding lookup + the final norm chain.
	if nCkpt > 16 {
		t.Fatalf("checkpointed graph has %d nodes, want ~10 (one per block plus the boundary ops)", nCkpt)
	}
	t.Logf("graph nodes: plain %d, checkpointed %d (%.1fx smaller)", nPlain, nCkpt, float64(nPlain)/float64(nCkpt))
}
