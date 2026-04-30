//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestGQAEqualHeadsMatchesMHANormScale: when numKVHeads == numQueryHeads,
// GQA reduces to standard MHA. Compare to a manually-rolled batched
// reference using sqrt(headDim) — the standard scaled-dot-product
// attention scale.
//
// We do NOT compare against the existing nn.MultiHeadAttention's
// batched path because that path has a long-standing bug — it
// divides by headDim instead of sqrt(headDim). The per-head loop
// path (used for autograd) correctly uses sqrt. Filed separately;
// not fixing in this PR to keep the diff focused.
func TestGQAEqualHeadsMatchesMHANormScale(t *testing.T) {
	const dim, numHeads, seqLen = 16, 4, 6
	headDim := dim / numHeads

	g.NoGrad(func() {
		gqa := NewGQA(dim, numHeads, numHeads) // groupSize=1
		x := g.RandN(seqLen, dim)
		yGQA := gqa.Forward(x, 0)

		// Reference: project, reshape, permute, scale by sqrt(headDim),
		// causal-mask softmax, attend, project back.
		q := gqa.Wq.Forward(x)
		k := gqa.Wk.Forward(x)
		v := gqa.Wv.Forward(x)
		qH := g.Permute(q.Reshape(seqLen, numHeads, headDim), []int{1, 0, 2})
		kH := g.Permute(k.Reshape(seqLen, numHeads, headDim), []int{1, 0, 2})
		vH := g.Permute(v.Reshape(seqLen, numHeads, headDim), []int{1, 0, 2})
		scores := g.BatchedMatMulTransB(qH, kH, numHeads, seqLen, seqLen, headDim)
		invScale := float32(1.0 / math.Sqrt(float64(headDim)))
		for i := range scores.Data() {
			scores.Data()[i] *= invScale
		}
		for h := 0; h < numHeads; h++ {
			block := scores.Data()[h*seqLen*seqLen : (h+1)*seqLen*seqLen]
			for i := 0; i < seqLen; i++ {
				for j := i + 1; j < seqLen; j++ {
					block[i*seqLen+j] = -1e9
				}
			}
			softmaxInPlace(block, seqLen)
		}
		attnOut := g.BatchedMatMul(scores, vH, numHeads, seqLen, headDim, seqLen)
		concat := g.Permute(attnOut, []int{1, 0, 2}).Reshape(seqLen, dim)
		yRef := gqa.Wo.Forward(concat)

		for i := range yGQA.Data() {
			d := math.Abs(float64(yGQA.Data()[i] - yRef.Data()[i]))
			if d > 1e-3 {
				t.Fatalf("[%d] GQA=%g ref=%g diff=%g", i, yGQA.Data()[i], yRef.Data()[i], d)
			}
		}
	})
}

// TestGQAGroupSizeReducesKV: with numKVHeads < numQueryHeads, the
// projection matrices for K/V should be smaller (kvDim < dim) and
// the forward should still produce the right output shape.
func TestGQAGroupSizeReducesKV(t *testing.T) {
	const dim, numQ, numKV, seqLen = 16, 4, 2, 6
	gqa := NewGQA(dim, numQ, numKV)

	// kvDim = numKV * headDim = 2 * 4 = 8. Wk/Wv have shape (8, 16).
	if got := gqa.Wk.Weight.Shape(); got[0] != 8 || got[1] != 16 {
		t.Fatalf("Wk shape = %v, want [8 16]", got)
	}
	if got := gqa.Wv.Weight.Shape(); got[0] != 8 || got[1] != 16 {
		t.Fatalf("Wv shape = %v, want [8 16]", got)
	}
	// Wq/Wo are full (dim, dim).
	if got := gqa.Wq.Weight.Shape(); got[0] != 16 || got[1] != 16 {
		t.Fatalf("Wq shape = %v, want [16 16]", got)
	}

	x := g.RandN(seqLen, dim)
	g.NoGrad(func() {
		y := gqa.Forward(x, 0)
		if y.Shape()[0] != seqLen || y.Shape()[1] != dim {
			t.Fatalf("output shape = %v, want [%d %d]", y.Shape(), seqLen, dim)
		}
	})
}

// TestGQAWithRoPE: attaching RoPE must change the output relative to
// no-RoPE baseline. We do NOT test "shift startPos changes output" —
// RoPE is specifically designed to make attention depend only on
// relative position (t-s), so shifting all positions by a constant
// is invariant by construction.
func TestGQAWithRoPE(t *testing.T) {
	const dim, numQ, numKV, seqLen = 16, 4, 2, 4
	headDim := dim / numQ

	x := g.RandN(seqLen, dim)
	g.NoGrad(func() {
		// Two GQA instances with identical projections, one with RoPE.
		gqaPlain := NewGQA(dim, numQ, numKV)
		gqaRope := NewGQA(dim, numQ, numKV)
		copy(gqaRope.Wq.Weight.Data(), gqaPlain.Wq.Weight.Data())
		copy(gqaRope.Wq.Bias.Data(), gqaPlain.Wq.Bias.Data())
		copy(gqaRope.Wk.Weight.Data(), gqaPlain.Wk.Weight.Data())
		copy(gqaRope.Wk.Bias.Data(), gqaPlain.Wk.Bias.Data())
		copy(gqaRope.Wv.Weight.Data(), gqaPlain.Wv.Weight.Data())
		copy(gqaRope.Wv.Bias.Data(), gqaPlain.Wv.Bias.Data())
		copy(gqaRope.Wo.Weight.Data(), gqaPlain.Wo.Weight.Data())
		copy(gqaRope.Wo.Bias.Data(), gqaPlain.Wo.Bias.Data())
		gqaRope.RoPE = NewRoPE(headDim, 32, 10000, RopeLlama)

		yPlain := gqaPlain.Forward(x, 0)
		yRope := gqaRope.Forward(x, 0)

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

// TestGQARoPERelativePositionInvariance: shifting all positions by
// a constant offset (startPos) must NOT change the output. This is
// the defining property of RoPE — attention depends only on
// relative position (t-s), so a uniform shift is invariant.
func TestGQARoPERelativePositionInvariance(t *testing.T) {
	const dim, numQ, numKV, seqLen = 16, 4, 2, 4
	headDim := dim / numQ
	gqa := NewGQA(dim, numQ, numKV)
	gqa.RoPE = NewRoPE(headDim, 32, 10000, RopeLlama)

	x := g.RandN(seqLen, dim)
	g.NoGrad(func() {
		y0 := gqa.Forward(x, 0)
		y4 := gqa.Forward(x, 4)
		for i := range y0.Data() {
			if math.Abs(float64(y0.Data()[i]-y4.Data()[i])) > 1e-4 {
				t.Fatalf("[%d] startPos=0 (%g) vs startPos=4 (%g) differ — relative-position invariance violated",
					i, y0.Data()[i], y4.Data()[i])
			}
		}
	})
}

// TestGQACausalMaskApplied: when Causal=true, position 0's output
// should not depend on tokens at positions ≥ 1. We verify by
// changing token at position 1 and checking position-0 output is
// unchanged.
func TestGQACausalMaskApplied(t *testing.T) {
	const dim, numQ, numKV, seqLen = 16, 4, 2, 4
	gqa := NewGQA(dim, numQ, numKV)
	gqa.Causal = true

	x := g.RandN(seqLen, dim)
	g.NoGrad(func() {
		y0 := gqa.Forward(x, 0)

		// Modify position 1 of x (in place — gorch tensors are mutable).
		for j := 0; j < dim; j++ {
			x.Data()[1*dim+j] += 99.0
		}
		y1 := gqa.Forward(x, 0)

		// Position 0's output should be unchanged (causal: pos 0 can't
		// see pos 1 onward).
		for j := 0; j < dim; j++ {
			if math.Abs(float64(y0.Data()[j]-y1.Data()[j])) > 1e-3 {
				t.Fatalf("causal violated: pos 0 output changed when pos 1 changed (delta=%g)",
					y0.Data()[j]-y1.Data()[j])
			}
		}
	})
}

// TestGQAValidatesShapes: panic on misconfigured dim/heads.
func TestGQAValidatesShapes(t *testing.T) {
	cases := []struct {
		name        string
		dim, q, kv  int
	}{
		{"dim not div by numQ", 17, 4, 2},
		{"numQ not div by numKV", 16, 6, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("expected panic for dim=%d, q=%d, kv=%d", tc.dim, tc.q, tc.kv)
				}
			}()
			NewGQA(tc.dim, tc.q, tc.kv)
		})
	}
}

func shapesEqual4(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
