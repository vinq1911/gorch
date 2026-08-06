//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestMLAOutputShape: forward produces (seqLen, dim) regardless of
// the unusual internal head splits.
func TestMLAOutputShape(t *testing.T) {
	const dim = 32
	const numHeads = 4
	const nopeDim, ropeDim, valDim = 4, 4, 6
	const kvLora = 8
	const seqLen = 7

	mla := NewMLA(dim, numHeads, nopeDim, ropeDim, valDim, kvLora)
	x := g.RandN(seqLen, dim)
	g.NoGrad(func() {
		y := mla.Forward(x, 0)
		if y.Shape()[0] != seqLen || y.Shape()[1] != dim {
			t.Fatalf("output shape = %v, want [%d %d]", y.Shape(), seqLen, dim)
		}
		// All values finite.
		for i, v := range y.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("[%d] non-finite: %g", i, v)
			}
		}
	})
}

// TestMLAOddRopeDimPanics: rope head dim must be even (RoPE pairs).
func TestMLAOddRopeDimPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on odd ropeHeadDim")
		}
	}()
	NewMLA(16, 4, 4, 3, 4, 8) // ropeDim=3 is odd
}

// TestMLAKVLatentSizeMatchesPaper: the WkvDown projection produces
// kvLoraRank + ropeHeadDim outputs. WkvUp consumes kvLoraRank inputs
// and produces numHeads * (nopeDim + valueDim) outputs. These shapes
// are what makes MLA's KV-cache savings work — the cache stores the
// kvLoraRank latent + ropeHeadDim shared rope key (per token),
// instead of full per-head K/V.
func TestMLAKVLatentSizeMatchesPaper(t *testing.T) {
	const dim, numHeads = 32, 4
	const nopeDim, ropeDim, valDim = 4, 4, 6
	const kvLora = 8
	mla := NewMLA(dim, numHeads, nopeDim, ropeDim, valDim, kvLora)

	if got := mla.WkvDown.Weight.Shape(); got[0] != kvLora+ropeDim {
		t.Fatalf("WkvDown out = %d, want %d (kvLoraRank + ropeDim)", got[0], kvLora+ropeDim)
	}
	if got := mla.WkvUp.Weight.Shape(); got[0] != numHeads*(nopeDim+valDim) || got[1] != kvLora {
		t.Fatalf("WkvUp shape = %v, want [%d %d]", got, numHeads*(nopeDim+valDim), kvLora)
	}

	// Per-token KV-cache footprint (latent + shared rope key) vs
	// full per-head K/V. For DeepSeek-V2 (kvLora=512, ropeDim=64,
	// numHeads=128, headDim=192): MLA = 576 floats, full = 24576
	// floats. ~42× reduction.
	mlaCache := kvLora + ropeDim                    // floats per token
	fullCache := 2 * numHeads * (nopeDim + ropeDim) // K + V at full per-head dim
	if mlaCache >= fullCache {
		t.Fatalf("MLA cache %d not smaller than full %d — defeats the point", mlaCache, fullCache)
	}
}

// TestMLACausalMask: pos 0's output must not depend on later tokens.
func TestMLACausalMask(t *testing.T) {
	const dim, numHeads = 16, 2
	const nopeDim, ropeDim, valDim = 4, 4, 4
	const kvLora = 6
	mla := NewMLA(dim, numHeads, nopeDim, ropeDim, valDim, kvLora)

	x := g.RandN(5, dim)
	g.NoGrad(func() {
		y0 := mla.Forward(x, 0)
		// Perturb position 3 of x.
		for j := 0; j < dim; j++ {
			x.Data()[3*dim+j] += 99
		}
		y1 := mla.Forward(x, 0)
		// Position 0..2 outputs should be unchanged.
		for s := 0; s < 3; s++ {
			for j := 0; j < dim; j++ {
				idx := s*dim + j
				if math.Abs(float64(y0.Data()[idx]-y1.Data()[idx])) > 1e-3 {
					t.Fatalf("causal violated at pos %d dim %d: %g → %g",
						s, j, y0.Data()[idx], y1.Data()[idx])
				}
			}
		}
	})
}

// TestMLAWithRoPE: with RoPE attached, output differs from no-RoPE
// baseline on the same input.
func TestMLAWithRoPE(t *testing.T) {
	const dim, numHeads = 16, 2
	const nopeDim, ropeDim, valDim = 4, 4, 4
	const kvLora = 6
	mla := NewMLA(dim, numHeads, nopeDim, ropeDim, valDim, kvLora)

	mlaR := NewMLA(dim, numHeads, nopeDim, ropeDim, valDim, kvLora)
	// Stamp identical weights.
	for i, w := range []*Linear{mla.Wq, mla.WkvDown, mla.WkvUp, mla.Wo} {
		other := []*Linear{mlaR.Wq, mlaR.WkvDown, mlaR.WkvUp, mlaR.Wo}[i]
		copy(other.Weight.Data(), w.Weight.Data())
		copy(other.Bias.Data(), w.Bias.Data())
	}
	mlaR.RoPE = NewRoPE(ropeDim, 32, 10000, RopeLlama)

	x := g.RandN(4, dim)
	g.NoGrad(func() {
		yPlain := mla.Forward(x, 0)
		yRope := mlaR.Forward(x, 0)
		var maxAbs, maxDiff float32
		for i := range yPlain.Data() {
			if av := float32(math.Abs(float64(yPlain.Data()[i]))); av > maxAbs {
				maxAbs = av
			}
			d := float32(math.Abs(float64(yPlain.Data()[i] - yRope.Data()[i])))
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff < 0.01*maxAbs {
			t.Fatalf("RoPE didn't change output: maxDiff=%g maxAbs=%g", maxDiff, maxAbs)
		}
	})
}

func TestMLAParametersIncludesAllProjections(t *testing.T) {
	mla := NewMLA(16, 2, 4, 4, 4, 8)
	// 4 Linears × 2 (W, b) = 8 tensors.
	if got := len(mla.Parameters()); got != 8 {
		t.Fatalf("got %d parameters, want 8", got)
	}
}
