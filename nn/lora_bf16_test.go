//go:build darwin

package nn

// Plan 0009 X3 loss-trajectory golden: a 50-step synthetic LoRA run
// with the frozen base weight in bf16 on Metal (the MPS dtyped matmul
// path, threshold lowered to 0) must track the all-f32 CPU trajectory
// within 5% relative loss at step 50. min-over-2-attempts per the
// established R5 retry discipline (BLAS threading nondeterminism —
// retry once, real regressions fail twice).

import (
	"math"
	"math/rand"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

const (
	loraTrajIn    = 128
	loraTrajOut   = 128
	loraTrajBatch = 64
	loraTrajSteps = 50
)

// loraTrajFill deterministically fills a slice from its own generator.
func loraTrajFill(dst []float32, seed int64, scale float32) {
	rng := rand.New(rand.NewSource(seed))
	for i := range dst {
		dst[i] = float32(rng.NormFloat64()) * scale
	}
}

// loraTrajRun trains a rank-8 LoRA adapter over a frozen 128×128 base
// for 50 AdamW steps on a fixed synthetic regression target and
// returns the loss trajectory. bf16Frozen=true stores the base weight
// in bf16 on Metal (input Metal-resident too) so every base matmul —
// forward and the dx backward — takes the B4 dtyped GPU path.
func loraTrajRun(t *testing.T, bf16Frozen bool) []float32 {
	t.Helper()
	base := NewLinear(loraTrajIn, loraTrajOut)
	loraTrajFill(base.Weight.Data(), 101, 0.05)
	lora := NewLoRALinear(base, 8, 16)
	loraTrajFill(lora.A.Data(), 102, 0.02)

	xd := make([]float32, loraTrajBatch*loraTrajIn)
	yd := make([]float32, loraTrajBatch*loraTrajOut)
	loraTrajFill(xd, 103, 1.0)
	loraTrajFill(yd, 104, 1.0)
	x := g.NewTensor(xd, loraTrajBatch, loraTrajIn)
	// x requires grad like a mid-network activation would (the real
	// workload's frozen Linears must still produce dx for the layers
	// below them — the dtyped bf16 backward path).
	x.SetRequiresGrad(true)
	y := g.NewTensor(yd, loraTrajBatch, loraTrajOut)

	if bf16Frozen {
		dev := g.MetalDev()
		base.Weight = base.Weight.ToBF16() // fresh bf16 copy, requiresGrad=false
		base.ToMetal(dev)                  // moves bf16 Weight + f32 Bias, sets the GPU forward path
		x = x.ToMetal(dev)
	}

	opt := optim.NewAdamW(lora.Parameters(), 1e-2, 0.0)
	losses := make([]float32, 0, loraTrajSteps)
	for i := 0; i < loraTrajSteps; i++ {
		opt.ZeroGrad()
		pred := lora.Forward(x)
		d := g.Sub(pred, y)
		loss := g.Mean(g.Mul(d, d))
		loss.Backward()
		opt.Step()
		losses = append(losses, loss.Data()[0])
	}
	return losses
}

func TestLoRABF16FrozenLossTrajectory(t *testing.T) {
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !g.MetalBF16MatMulSupported() {
		t.Skip("MPS bf16 matmul unsupported (ADR-012 runtime probe)")
	}
	prev := g.MatMulMetalThreshold
	g.MatMulMetalThreshold = 0 // the 64×128×128 base matmul must hit the dtyped GPU path
	defer func() { g.MatMulMetalThreshold = prev }()

	attempt := func() (relDiff float64, f50, b50 float32, bfDispatches int64) {
		fRun := loraTrajRun(t, false)
		c0 := g.ReadMetalDispatchCounts()
		bRun := loraTrajRun(t, true)
		c1 := g.ReadMetalDispatchCounts()
		f50, b50 = fRun[loraTrajSteps-1], bRun[loraTrajSteps-1]
		relDiff = math.Abs(float64(b50-f50)) / math.Abs(float64(f50))
		return relDiff, f50, b50, c1.BF16MatMul - c0.BF16MatMul
	}

	// min-over-2-attempts (plan §3.4 parity policy / risk R5).
	rel, f50, b50, disp := attempt()
	if disp < int64(loraTrajSteps)*2 {
		t.Fatalf("bf16 run fired only %d dtyped MPS dispatches over %d steps (want >= %d: fwd+dx per step) — the GPU bf16 path did not engage",
			disp, loraTrajSteps, loraTrajSteps*2)
	}
	if rel > 0.05 {
		rel2, f50b, b50b, _ := attempt()
		t.Logf("attempt 1: |Δ|/f32 = %.4f (f32 %.6f, bf16 %.6f); attempt 2: %.4f (f32 %.6f, bf16 %.6f)",
			rel, f50, b50, rel2, f50b, b50b)
		if rel2 > 0.05 {
			t.Fatalf("bf16-frozen LoRA loss at step 50 diverges from f32 by %.2f%% / %.2f%% (gate 5%%)",
				rel*100, rel2*100)
		}
	} else {
		t.Logf("loss at step 50: f32 %.6f, bf16-frozen %.6f (|Δ|/f32 = %.4f, gate 0.05; %d dtyped dispatches)",
			f50, b50, rel, disp)
	}

	// Sanity: the run actually learned something (B starts at zero, so
	// step-1 loss is the frozen-base residual).
	f := loraTrajRun(t, false)
	if !(f[loraTrajSteps-1] < f[0]*0.9) {
		t.Fatalf("f32 trajectory did not descend: first %.6f last %.6f", f[0], f[loraTrajSteps-1])
	}
}
