//go:build darwin

package nn

import (
	"math"
	"math/rand"
	"testing"

	g "github.com/vinq1911/gorch"
)

// Golden tests for the K6 rope_apply Metal kernel (plan 0009 X2b).
// CPU reference: the nn/rope.go rotate loops (pinned by rope_test.go
// incl. TestRoPEBackwardMatchesNumerical). Tolerances per the plan
// table: fwd 1e-4 abs; the backward is the same rotation with sin
// negated, so it shares the tolerance.

func ropeRandSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

func ropeMaxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// TestRoPEMetalMatchesCPU: Metal fwd + autograd bwd vs the CPU path
// for both pair conventions, at startPos 0 and a nonzero offset, on
// the workload's (heads, seq, headDim) layout.
func TestRoPEMetalMatchesCPU(t *testing.T) {
	gpu, err := g.InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !g.RoPEMetalReady() {
		t.Fatal("rope_apply pipeline not compiled by InitMetal")
	}
	rng := rand.New(rand.NewSource(29))

	const heads, seq, headDim, maxSeq = 4, 19, 32, 64
	n := heads * seq * headDim

	for _, tc := range []struct {
		name     string
		style    RopeStyle
		startPos int
	}{
		{"llama_pos0", RopeLlama, 0},
		{"llama_pos17", RopeLlama, 17},
		{"neox_pos0", RopeGPTNeoX, 0},
		{"neox_pos5", RopeGPTNeoX, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := ropeRandSlice(rng, n)
			w := ropeRandSlice(rng, n)
			rope := NewRoPE(headDim, maxSeq, 10000, tc.style)

			// CPU reference fwd + bwd (weighted loss → non-uniform grad).
			xc := g.NewTensor(src, heads, seq, headDim).SetRequiresGrad(true)
			yc := rope.Apply(xc, tc.startPos)
			g.Sum(g.Mul(yc, g.NewTensor(w, heads, seq, headDim))).Backward()
			refY := append([]float32(nil), yc.Data()...)
			refDX := append([]float32(nil), xc.Grad().Data()...)

			// Metal path (fresh RoPE module so the table upload path runs).
			ropeM := NewRoPE(headDim, maxSeq, 10000, tc.style)
			xm := g.NewTensorOnMetal(gpu.Dev, src, heads, seq, headDim).SetRequiresGrad(true)
			ym := ropeM.Apply(xm, tc.startPos)
			if !ym.IsOnMetal() {
				t.Fatal("RoPE.Apply on Metal input returned a CPU tensor")
			}
			g.Sum(g.Mul(ym, g.NewTensorOnMetal(gpu.Dev, w, heads, seq, headDim))).Backward()

			if d := ropeMaxAbsDiff(refY, ym.Data()); d > 1e-4 {
				t.Errorf("fwd diff %.3g > 1e-4", d)
			}
			if d := ropeMaxAbsDiff(refDX, xm.Grad().Data()); d > 1e-4 {
				t.Errorf("bwd diff %.3g > 1e-4", d)
			}
		})
	}
}

// TestLinearMetalBiasAddAndDb pins the X2b vec_bias_add forward path
// (nonzero bias — the pre-X2b gpu_backward test only checked grads,
// which are bias-independent) and the col_sum db kernel under a
// non-uniform upstream grad.
func TestLinearMetalBiasAddAndDb(t *testing.T) {
	gpu, err := g.InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	rng := rand.New(rand.NewSource(31))

	const batch, in, out = 37, 24, 18
	wData := ropeRandSlice(rng, out*in)
	bData := ropeRandSlice(rng, out)
	xData := ropeRandSlice(rng, batch*in)
	lossW := ropeRandSlice(rng, batch*out)

	run := func(onMetal bool) (y, db []float32) {
		l := NewLinear(in, out)
		copy(l.Weight.Data(), wData)
		copy(l.Bias.Data(), bData)
		var x *g.Tensor
		var lw *g.Tensor
		if onMetal {
			l.ToMetal(gpu.Dev)
			x = g.NewTensorOnMetal(gpu.Dev, xData, batch, in)
			lw = g.NewTensorOnMetal(gpu.Dev, lossW, batch, out)
		} else {
			x = g.NewTensor(xData, batch, in)
			lw = g.NewTensor(lossW, batch, out)
		}
		yT := l.Forward(x)
		g.Sum(g.Mul(yT, lw)).Backward()
		y = append([]float32(nil), yT.Data()...)
		db = append([]float32(nil), l.Bias.Grad().Data()...)
		return y, db
	}

	yCPU, dbCPU := run(false)
	yGPU, dbGPU := run(true)

	if d := ropeMaxAbsDiff(yCPU, yGPU); d > 1e-3 {
		t.Errorf("Linear Metal forward (bias add) diff %.3g > 1e-3", d)
	}
	if d := ropeMaxAbsDiff(dbCPU, dbGPU); d > 1e-3 {
		t.Errorf("Linear Metal db (col_sum) diff %.3g > 1e-3", d)
	}
}
