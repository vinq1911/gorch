//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestExpertSwiGLUFFNShape: an expert produces (M, dim) output from
// (M, dim) input via the SwiGLU sandwich.
func TestExpertSwiGLUFFNShape(t *testing.T) {
	const dim, expDim, M = 16, 32, 5
	e := NewExpert(dim, expDim)
	x := g.RandN(M, dim)
	g.NoGrad(func() {
		y := e.Forward(x)
		if y.Shape()[0] != M || y.Shape()[1] != dim {
			t.Fatalf("expert output shape = %v, want [%d %d]", y.Shape(), M, dim)
		}
	})
}

// TestMoESingleExpertEqualsExpert: with numExperts=1 and K=1, the
// MoE block reduces to a single expert (router output is irrelevant
// after softmax-over-1 normalises to 1.0 weight). Output must match
// running that one expert directly.
func TestMoESingleExpertEqualsExpert(t *testing.T) {
	const dim, expDim, M = 8, 16, 4
	moe := NewMoE(dim, expDim, 1, 1)

	x := g.RandN(M, dim)
	g.NoGrad(func() {
		yMoE := moe.Forward(x)
		yExpert := moe.Experts[0].Forward(x)
		for i := range yMoE.Data() {
			d := math.Abs(float64(yMoE.Data()[i] - yExpert.Data()[i]))
			if d > 1e-4 {
				t.Fatalf("[%d] MoE=%g expert=%g diff=%g", i, yMoE.Data()[i], yExpert.Data()[i], d)
			}
		}
	})
}

// TestMoEPartitionsTokensAcrossExperts: with deterministic router
// outputs (we stamp the router weights), every token routes to a
// specific expert. Verify the output equals each expert's response
// on its assigned tokens.
func TestMoEPartitionsTokensAcrossExperts(t *testing.T) {
	const dim, expDim, M, N, K = 4, 8, 4, 2, 1
	moe := NewMoE(dim, expDim, N, K)

	// Force token 0,1 → expert 0, token 2,3 → expert 1 by stamping
	// router weights to lean strongly on input feature 0 vs feature 1.
	// Router weight shape: (numExperts, dim) = (2, 4).
	for i := range moe.Router.Weight.Data() {
		moe.Router.Weight.Data()[i] = 0
	}
	moe.Router.Weight.Data()[0*4+0] = 10  // expert 0 fires on x[0]
	moe.Router.Weight.Data()[1*4+1] = 10  // expert 1 fires on x[1]
	for i := range moe.Router.Bias.Data() {
		moe.Router.Bias.Data()[i] = 0
	}

	x := g.NewTensor([]float32{
		1, 0, 0, 0,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 1, 0, 0,
	}, M, dim)

	g.NoGrad(func() {
		y := moe.Forward(x)

		// Expert 0 should have processed rows 0,1; expert 1 rows 2,3.
		// At K=1, weights are all 1.0 (single softmax-over-1).
		want01 := moe.Experts[0].Forward(g.Gather(x, []int{0, 1}))
		want23 := moe.Experts[1].Forward(g.Gather(x, []int{2, 3}))

		check := func(label string, gotRow, wantRow []float32) {
			t.Helper()
			for i := range gotRow {
				if math.Abs(float64(gotRow[i]-wantRow[i])) > 1e-3 {
					t.Fatalf("%s [%d]: got %g want %g", label, i, gotRow[i], wantRow[i])
				}
			}
		}
		check("token 0", y.Data()[0:dim], want01.Data()[0:dim])
		check("token 1", y.Data()[dim:2*dim], want01.Data()[dim:2*dim])
		check("token 2", y.Data()[2*dim:3*dim], want23.Data()[0:dim])
		check("token 3", y.Data()[3*dim:4*dim], want23.Data()[dim:2*dim])
	})
}

// TestMoETopKWeightingSums: with K=2, the routing weights for each
// token must sum to exactly 1.0 (softmax over the top-K logits).
// Verifies the weighted-sum scatter doesn't drift.
func TestMoETopKWeightingSums(t *testing.T) {
	const dim, expDim, M, N, K = 8, 16, 6, 4, 2
	moe := NewMoE(dim, expDim, N, K)

	x := g.RandN(M, dim)
	logits := moe.Router.Forward(x)
	_, topIdx := g.TopK(logits, K)

	topVals, _ := g.TopK(logits, K)
	weights := softmaxRows(topVals.Data(), M, K)

	for tok := 0; tok < M; tok++ {
		var sum float32
		for slot := 0; slot < K; slot++ {
			sum += weights[tok*K+slot]
		}
		if math.Abs(float64(sum-1)) > 1e-5 {
			t.Fatalf("token %d weights sum = %g, want 1", tok, sum)
		}
	}
	// And topIdx values are valid expert indices.
	for _, e := range topIdx {
		if e < 0 || e >= N {
			t.Fatalf("topIdx contains invalid expert %d", e)
		}
	}
}

// TestMoELoadBalanceLossMinimisedAtUniform: when router prob is
// uniform across experts, load-balance loss should be ~1.0 (the
// minimum). Far from uniform, it's higher.
//
// Stamp the router to produce identical logits → uniform softmax
// probabilities. fraction[e] = 1/N (uniform top-K assignment by
// chance can deviate, so use small tolerance).
func TestMoELoadBalanceLossMinimisedAtUniform(t *testing.T) {
	const dim, expDim, M, N, K = 8, 16, 64, 4, 2
	moe := NewMoE(dim, expDim, N, K)

	// Zero out router → all logits equal → uniform softmax.
	for i := range moe.Router.Weight.Data() {
		moe.Router.Weight.Data()[i] = 0
	}
	for i := range moe.Router.Bias.Data() {
		moe.Router.Bias.Data()[i] = 0
	}

	x := g.RandN(M, dim)
	loss := moe.LoadBalanceLoss(x)

	// At uniform: pProb[e] = 1/N, fraction[e] = 1/N (all experts
	// are tied with random tie-break, so tieflows go uniformly).
	// loss = N * sum_e (1/N)*(1/N) = N * N * (1/N²) = 1.
	if math.Abs(float64(loss-1.0)) > 0.05 {
		t.Fatalf("load-balance loss at uniform = %g, want ≈1.0", loss)
	}
}

// TestMoELoadBalanceLossHighWhenSkewed: stamping router to favour
// one expert pushes loss above 1.
func TestMoELoadBalanceLossHighWhenSkewed(t *testing.T) {
	const dim, expDim, M, N, K = 8, 16, 64, 4, 2

	// Two MoEs sharing the same input. One has a uniform router, the
	// other has a router heavily biased toward expert 0. Compare the
	// *relative* losses — the absolute number depends on randomness
	// in TopK's second pick across tied logits.
	uniformMoE := NewMoE(dim, expDim, N, K)
	for i := range uniformMoE.Router.Weight.Data() {
		uniformMoE.Router.Weight.Data()[i] = 0
	}
	for i := range uniformMoE.Router.Bias.Data() {
		uniformMoE.Router.Bias.Data()[i] = 0
	}

	skewedMoE := NewMoE(dim, expDim, N, K)
	for i := range skewedMoE.Router.Weight.Data() {
		skewedMoE.Router.Weight.Data()[i] = 0
	}
	for d := 0; d < dim; d++ {
		skewedMoE.Router.Weight.Data()[0*dim+d] = 5
	}
	for i := range skewedMoE.Router.Bias.Data() {
		skewedMoE.Router.Bias.Data()[i] = 0
	}
	skewedMoE.Router.Bias.Data()[0] = 10

	x := g.RandN(M, dim)
	uniformLoss := uniformMoE.LoadBalanceLoss(x)
	skewedLoss := skewedMoE.LoadBalanceLoss(x)

	// Skewed must be measurably higher than uniform. Relative
	// comparison guards against the absolute-number flakiness we
	// were getting (1.14 sometimes, 1.4 other times) due to TopK's
	// tied-logit tiebreak choosing different "second pick" experts.
	if skewedLoss <= uniformLoss*1.05 {
		t.Fatalf("skewed=%g not >5%% above uniform=%g", skewedLoss, uniformLoss)
	}
}

// TestMoEExpertWeightsReceiveGradient: with the autograd-aware path
// (Gather + Mul + ScatterAdd + Add), every expert that processed at
// least one token must receive a non-zero gradient on its Linear
// weights. Pre-fix MoE.Forward used a manual-scatter loop that broke
// the chain; expert weights stayed at init.
func TestMoEExpertWeightsReceiveGradient(t *testing.T) {
	const dim, expDim, M, N, K = 8, 16, 6, 4, 2
	moe := NewMoE(dim, expDim, N, K)

	x := g.RandN(M, dim)
	y := moe.Forward(x)
	loss := g.Sum(y)
	loss.Backward()

	// Find which experts processed tokens this batch.
	logits := moe.Router.Forward(x)
	_, topIdx := g.TopK(logits, K)
	used := make(map[int]bool)
	for _, e := range topIdx {
		used[e] = true
	}
	if len(used) == 0 {
		t.Fatal("no experts used — test setup error")
	}
	for e := range used {
		params := moe.Experts[e].Parameters()
		any := false
		for _, p := range params {
			if grad := p.Grad(); grad != nil {
				for _, v := range grad.Data() {
					if v != 0 {
						any = true
						break
					}
				}
			}
			if any {
				break
			}
		}
		if !any {
			t.Errorf("expert %d processed tokens but received no gradient", e)
		}
	}
}

func TestMoEValidatesK(t *testing.T) {
	cases := []struct {
		k        int
		nExperts int
	}{
		{0, 4},
		{-1, 4},
		{5, 4},
	}
	for _, tc := range cases {
		t.Run("invalid_k", func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("expected panic for k=%d/N=%d", tc.k, tc.nExperts)
				}
			}()
			NewMoE(8, 16, tc.nExperts, tc.k)
		})
	}
}

// TestMoEParametersIncludesAllExperts: the parameter list must
// include the router + every expert's three Linear layers (so the
// optimiser updates everything).
func TestMoEParametersIncludesAllExperts(t *testing.T) {
	const dim, expDim, N, K = 8, 16, 4, 2
	moe := NewMoE(dim, expDim, N, K)
	// Router: 2 (W, b). Each expert: 3 Linears × 2 = 6. N experts.
	// Total: 2 + 4*6 = 26.
	want := 2 + N*6
	if got := len(moe.Parameters()); got != want {
		t.Fatalf("got %d parameters, want %d", got, want)
	}
}
