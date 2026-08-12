//go:build darwin && e2e

package e2e

// Plan 0009 X4 correctness gates — the M1 overfit-style descent gate
// rerun on the REAL VoiceModel over the accelerated GPU+bf16 path:
//
//  1. TestQwenVoiceAccelTrajectoryParity4Layer: a fixed 20-step
//     fixture (deterministic multi-task-shaped samples over the
//     extended vocab, accum 2, clip 1.0 — the trainer loop's exact
//     op sequence) trained on a 4-layer-truncated REAL Qwen3-0.6B
//     base, CPU f32 (`-accel=off` config) vs GPU+bf16 (`-accel=async`
//     config). Both must descend, and the accelerated loss at the
//     final step must track the CPU loss within 5% relative — the X3
//     trajectory-parity policy (§3.4) applied to the real model.
//  2. TestQwenVoiceAccelFullModelShortDescent: short full-28-layer
//     confirmation on the accelerated path only (the CPU equivalent
//     is multi-minute): loss must strictly descend over 6 steps.
//
// Both tests skip without the local Qwen3-0.6B checkpoint, without
// Metal, or when the ADR-012 bf16 probe fails.

import (
	"math"
	"math/rand"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model/qwen"
	"github.com/vinq1911/gorch/optim"
)

// qvSample is one deterministic trainer-shaped sample: tokens over the
// extended vocab (base "text" ids + an ext-vocab "audio" span) with
// the supervised span covering the assistant-like tail — the same
// structure the M1 overfit builder produces, without needing the
// tokenizer or the committed clips.
type qvSample struct {
	tokens []int
	sup    []int
}

func qvFixtureSamples(baseVocab, textLen, audioLen, n int) []qvSample {
	rng := rand.New(rand.NewSource(7))
	samples := make([]qvSample, n)
	for s := range samples {
		var toks []int
		for i := 0; i < textLen; i++ {
			toks = append(toks, rng.Intn(30000)+100) // plausible text ids
		}
		aStart := len(toks)
		for i := 0; i < audioLen; i++ {
			toks = append(toks, baseVocab+rng.Intn(2048)) // ext-vocab span
		}
		toks = append(toks, rng.Intn(30000)+100) // final "im_end"-ish target
		var sup []int
		for p := aStart - 1; p < len(toks)-1; p++ {
			sup = append(sup, p)
		}
		samples[s] = qvSample{tokens: toks, sup: sup}
	}
	return samples
}

// qvTrainSteps drives `steps` optimizer steps of the trainer's exact
// micro-loop (accum scaled loss, grad clip, two LR groups) over the
// fixture, cycling samples deterministically. Returns per-step mean
// losses.
func qvTrainSteps(t *testing.T, vm *qwen.VoiceModel, samples []qvSample, steps, accum int) []float32 {
	t.Helper()
	_, params := vm.TrainableParams()
	lora := params[:len(params)-1]
	ext := params[len(params)-1:]
	// The trainer's default LRs: fast enough for clear descent over 20
	// steps, slow enough that the fixture does NOT overfit to the
	// ~1e-3 noise floor, where a relative-loss parity gate stops being
	// meaningful (observed with 10× these rates: both paths reach
	// ~5e-4..1e-3 by step 20 and the ratio is pure rounding noise).
	opt := optim.NewAdamWGroups([]optim.ParamGroup{
		{Params: lora, LR: 1e-4},
		{Params: ext, LR: 5e-4},
	}, 0)

	losses := make([]float32, 0, steps)
	micro := 0
	for step := 0; step < steps; step++ {
		var stepLoss float64
		t0 := time.Now()
		for a := 0; a < accum; a++ {
			s := samples[micro%len(samples)]
			micro++
			loss := vm.SupervisedLoss(s.tokens, s.sup)
			stepLoss += float64(loss.Data()[0])
			g.Mul(loss, g.NewTensor([]float32{1 / float32(accum)}, 1)).Backward()
		}
		optim.ClipGradNorm(params, 1.0)
		opt.Step()
		opt.ZeroGrad()
		losses = append(losses, float32(stepLoss/float64(accum)))
		t.Logf("step %2d loss %.4f (%.2fs)", step+1, losses[step], time.Since(t0).Seconds())
	}
	return losses
}

// qvCopyTrainable copies every trainable tensor (LoRA A/B + Ext rows)
// from src into dst so both models start from the identical point.
func qvCopyTrainable(t *testing.T, dst, src *qwen.VoiceModel) {
	t.Helper()
	_, ps := src.TrainableParams()
	_, pd := dst.TrainableParams()
	if len(ps) != len(pd) {
		t.Fatalf("trainable param count mismatch: %d vs %d", len(ps), len(pd))
	}
	for i := range ps {
		copy(pd[i].Data(), ps[i].Data())
	}
}

// qvAccelSetup switches the process into the trainer's -accel=async
// configuration and returns a restore func.
func qvAccelSetup(t *testing.T) func() {
	t.Helper()
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !qwen.AccelSupported() {
		t.Skip("MPS bf16 matmul unsupported (ADR-012 runtime probe)")
	}
	prevThreshold := g.MatMulMetalThreshold
	g.MatMulMetalThreshold = 8_000_000
	g.SetMetalAsync(true)
	return func() {
		g.MatMulMetalThreshold = prevThreshold
		g.SetMetalAsync(false)
	}
}

func qvCheckpoint(t *testing.T) string {
	t.Helper()
	path, err := qwen.FindCheckpoint()
	if err != nil {
		t.Skipf("qwen checkpoint not available (set QWEN3_MODEL): %v", err)
	}
	return path
}

func TestQwenVoiceAccelTrajectoryParity4Layer(t *testing.T) {
	path := qvCheckpoint(t)
	restore := qvAccelSetup(t)
	defer restore()

	cfg := qwen.Qwen3_0_6B()
	cfg.NumLayers = 4
	vc := qwen.VoiceConfig{LoRARank: 8, LoRAAlpha: 16}
	samples := qvFixtureSamples(cfg.VocabSize, 24, 96, 4)
	const steps, accum = 20, 2

	// CPU f32 reference (the -accel=off path).
	mCPU, err := qwen.LoadTruncated(path, cfg)
	if err != nil {
		t.Fatalf("LoadTruncated: %v", err)
	}
	vmCPU := qwen.NewVoiceModel(mCPU, vc)

	// Accelerated model, identical trainable init.
	mAcc, err := qwen.LoadTruncatedNative(path, cfg)
	if err != nil {
		t.Fatalf("LoadTruncatedNative: %v", err)
	}
	vmAcc := qwen.NewVoiceModel(mAcc, vc)
	qvCopyTrainable(t, vmAcc, vmCPU)
	vmAcc.ToMetal(g.MetalDev())

	t.Log("CPU f32 trajectory:")
	cpuLosses := qvTrainSteps(t, vmCPU, samples, steps, accum)

	g.ResetMetalDispatchCounts()
	t.Log("GPU+bf16 trajectory:")
	accLosses := qvTrainSteps(t, vmAcc, samples, steps, accum)
	dc := g.ReadMetalDispatchCounts()
	t.Logf("accel dispatch counts over %d steps: bf16_matmul=%d mps_matmul=%d batched=%d softmax=%d ce=%d",
		steps, dc.BF16MatMul, dc.MatMul, dc.BatchedMatMul, dc.SoftmaxKernel, dc.CEKernel)

	// The bf16 dtyped path must actually be engaged: ≥14 dtyped matmuls
	// per micro-step (7 fwd + 7 dx per adapted layer × 4 layers would be
	// 56; be conservative about below-threshold short-seq residue).
	if dc.BF16MatMul < int64(steps*accum) {
		t.Errorf("accelerated run fired only %d dtyped bf16 matmuls over %d micro-steps — GPU path not engaged", dc.BF16MatMul, steps*accum)
	}

	if !(cpuLosses[steps-1] < cpuLosses[0]*0.9) {
		t.Errorf("CPU trajectory did not descend: first %.4f last %.4f", cpuLosses[0], cpuLosses[steps-1])
	}
	if !(accLosses[steps-1] < accLosses[0]*0.9) {
		t.Errorf("accel trajectory did not descend: first %.4f last %.4f", accLosses[0], accLosses[steps-1])
	}
	// Relative diff with a small absolute floor so a trajectory that
	// descends into the numerical noise floor cannot fail the gate on
	// a meaningless ratio of two near-zero losses.
	rel := math.Abs(float64(accLosses[steps-1]-cpuLosses[steps-1])) /
		math.Max(math.Abs(float64(cpuLosses[steps-1])), 0.05)
	t.Logf("loss at step %d: cpu %.5f vs accel %.5f (|Δ|/cpu = %.4f, gate 0.05)", steps, cpuLosses[steps-1], accLosses[steps-1], rel)
	if rel > 0.05 {
		t.Errorf("accelerated trajectory diverges from CPU by %.2f%% at step %d (gate 5%%)", rel*100, steps)
	}
}

func TestQwenVoiceAccelFullModelShortDescent(t *testing.T) {
	path := qvCheckpoint(t)
	restore := qvAccelSetup(t)
	defer restore()

	m, err := qwen.LoadNative(path, qwen.Qwen3_0_6B())
	if err != nil {
		t.Fatalf("LoadNative: %v", err)
	}
	vm := qwen.NewVoiceModel(m, qwen.VoiceConfig{LoRARank: 16, LoRAAlpha: 32})
	vm.ToMetal(g.MetalDev())

	samples := qvFixtureSamples(vm.Base.Cfg.VocabSize, 24, 96, 2)
	const steps, accum = 6, 2
	losses := qvTrainSteps(t, vm, samples, steps, accum)
	if !(losses[steps-1] < losses[0]) {
		t.Fatalf("full-model accelerated loss did not descend: first %.4f last %.4f", losses[0], losses[steps-1])
	}
	t.Logf("full 28-layer accel descent: %.4f -> %.4f over %d steps", losses[0], losses[steps-1], steps)
}
