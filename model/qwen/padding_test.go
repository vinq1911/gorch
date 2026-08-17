//go:build darwin

package qwen

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// Padding invariance for the supervised training path.
//
// The trainer may pad a training sequence up to a bucket boundary so
// that the GPU sees a small fixed set of matmul shapes instead of one
// per distinct sample length (cmd/qwenvoice-train padToBucket). Padding
// is appended at the END and the supervised position list is not
// touched, which is only sound because:
//
//   - attention is CAUSAL, so a real query row never attends a pad
//     column (and CausalSoftmax leaves masked columns at exactly 0.0);
//   - every other op is per-position (RMSNorm over the hidden dim,
//     elementwise SwiGLU, RoPE by absolute position);
//   - the loss GATHERS supervised positions rather than averaging over
//     the sequence, so pads enter neither numerator nor denominator.
//
// "Sound" is an argument; these tests are the evidence. They are the
// gate on any future change to masking, normalisation or the loss
// reduction — all three would break padding silently, producing wrong
// gradients rather than an error.

// padWith appends n copies of pad to a copy of tokens.
func padWith(tokens []int, n, pad int) []int {
	out := make([]int, 0, len(tokens)+n)
	out = append(out, tokens...)
	for i := 0; i < n; i++ {
		out = append(out, pad)
	}
	return out
}

// gradSnapshot copies every trainable gradient, keyed by param index.
func gradSnapshot(t *testing.T, vm *VoiceModel) [][]float32 {
	t.Helper()
	_, params := vm.TrainableParams()
	out := make([][]float32, len(params))
	for i, p := range params {
		gr := p.Grad()
		if gr == nil {
			t.Fatalf("trainable param %d has no gradient to snapshot", i)
		}
		cp := make([]float32, len(gr.Data()))
		copy(cp, gr.Data())
		out[i] = cp
	}
	return out
}

func zeroGrads(vm *VoiceModel) {
	_, params := vm.TrainableParams()
	for _, p := range params {
		p.ZeroGrad()
	}
}

// TestSupervisedLossPaddingInvariance is the core proof: appending pad
// tokens must change neither the loss nor ANY trainable gradient.
//
// Tolerance, not exact equality. Padding grows the M dimension of every
// per-position GEMM and the K dimension of the attn@V GEMM, so the
// backend is free to tile and to order its accumulations differently —
// the mathematics is identical, the floating-point rounding need not
// be. The bound below is a relative one against the unpadded gradient's
// own magnitude; anything structurally wrong (a leaking pad row, a
// length-normalised loss, a mask applied after the softmax rather than
// inside it) moves gradients by far more than this.
func TestSupervisedLossPaddingInvariance(t *testing.T) {
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2, 17, 33}
	supervised := []int{2, 3, 5, 6, 8}

	// Both pad choices the design considered. If causal isolation is
	// real, the pad VALUE cannot matter; if it is not, a distinctive id
	// and a repeated real token will disagree, which is the point.
	for _, tc := range []struct {
		name string
		pad  int
	}{
		{"constant-base-vocab-id", 0},
		{"repeat-last-real-token", tokens[len(tokens)-1]},
		{"trainable-ext-row", 70},
	} {
		t.Run(tc.name, func(t *testing.T) {
			vm := tinyVoice(t)

			loss := vm.SupervisedLoss(tokens, supervised)
			base := loss.Data()[0]
			loss.Backward()
			want := gradSnapshot(t, vm)
			zeroGrads(vm)

			padded := padWith(tokens, 22, tc.pad)
			ploss := vm.SupervisedLoss(padded, supervised)
			got := ploss.Data()[0]
			ploss.Backward()
			have := gradSnapshot(t, vm)

			if d := math.Abs(float64(base - got)); d > 1e-5*math.Abs(float64(base))+1e-6 {
				t.Fatalf("loss changed under padding: unpadded %v padded %v (|Δ| %.3g) — "+
					"pad positions are entering the loss", base, got, d)
			}

			for i := range want {
				var maxAbs, maxDiff float64
				for j := range want[i] {
					a, b := float64(want[i][j]), float64(have[i][j])
					if math.Abs(a) > maxAbs {
						maxAbs = math.Abs(a)
					}
					if d := math.Abs(a - b); d > maxDiff {
						maxDiff = d
					}
				}
				if maxDiff > 1e-5*maxAbs+1e-7 {
					t.Fatalf("param %d gradient changed under padding: max|Δ| %.3g against "+
						"max|g| %.3g — padding is reaching the gradients", i, maxDiff, maxAbs)
				}
			}
		})
	}
}

// TestPadTokenRowGainsNothingFromBeingPadding is the sharpest single
// probe for a masking leak: pad with a trainable extended-vocabulary
// row that occurs NOWHERE in the real sequence and nowhere among the
// targets, then check that row's gradient against the unpadded run.
//
// The naive form of this test — "the pad row's gradient must be zero" —
// is WRONG here, and the first draft of it failed for that reason. The
// extended rows are the tied OUTPUT HEAD as well as the input
// embedding, so every vocabulary row takes a gradient from the
// cross-entropy at every supervised position (the -p_j*h term for the
// classes that are not the target), whether or not that row's token
// appears anywhere. That contribution exists identically in the
// unpadded run.
//
// So the invariant is not "zero" but "zero EXTRA": the only new thing
// padding introduces for this row is len(pad) embedding lookups, whose
// gradient must vanish because pad positions are causally unreachable
// and ungathered. Comparing the row against its unpadded self isolates
// exactly that.
func TestPadTokenRowGainsNothingFromBeingPadding(t *testing.T) {
	vm := tinyVoice(t)
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2}
	supervised := []int{1, 2, 3, 4, 5, 6}

	const baseVocab = 64
	const padID = 78 // extended row 14
	for _, tok := range tokens {
		if tok == padID {
			t.Fatalf("test setup: pad id %d occurs in the real sequence", padID)
		}
	}
	for _, p := range supervised {
		if tokens[p+1] == padID {
			t.Fatalf("test setup: pad id %d is a target", padID)
		}
	}

	ext := vm.Embed.Ext
	dim := ext.Shape()[1]
	row := padID - baseVocab

	loss := vm.SupervisedLoss(tokens, supervised)
	loss.Backward()
	want := make([]float32, dim)
	copy(want, ext.Grad().Data()[row*dim:(row+1)*dim])
	zeroGrads(vm)

	ploss := vm.SupervisedLoss(padWith(tokens, 24, padID), supervised)
	ploss.Backward()
	got := ext.Grad().Data()[row*dim : (row+1)*dim]

	// The head contribution must be present (otherwise the test is
	// vacuous — a row that never gets a gradient cannot detect a leak).
	var mag float64
	for _, v := range want {
		mag += math.Abs(float64(v))
	}
	if mag == 0 {
		t.Fatal("the pad row has no gradient even unpadded, so this test cannot detect a leak")
	}

	for j := range want {
		a, b := float64(want[j]), float64(got[j])
		if math.Abs(a-b) > 1e-5*math.Abs(a)+1e-7 {
			t.Fatalf("extended row %d element %d: gradient %v unpadded, %v padded — appearing "+
				"24 times as padding added gradient to a row it must not touch", row, j, a, b)
		}
	}
}

// TestPaddingLeavesNoNonFiniteState guards the one way end-padding can
// poison a real position despite the mask: masked columns still take
// part in the attn@V GEMM as 0.0*V[j], and 0*NaN is NaN. So the
// invariant is not "pads are masked" but "pads carry finite
// activations". A pad token whose embedding row is ordinary satisfies
// it; this pins that the whole padded forward/backward stays finite.
func TestPaddingLeavesNoNonFiniteState(t *testing.T) {
	vm := tinyVoice(t)
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2}
	supervised := []int{1, 2, 3, 4, 5, 6}

	padded := padWith(tokens, 24, 0)
	loss := vm.SupervisedLoss(padded, supervised)
	if v := float64(loss.Data()[0]); math.IsNaN(v) || math.IsInf(v, 0) {
		t.Fatalf("padded loss is not finite: %v", v)
	}
	loss.Backward()
	_, params := vm.TrainableParams()
	for i, p := range params {
		for j, v := range p.Grad().Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("param %d gradient element %d is not finite (%v) after a padded backward", i, j, v)
			}
		}
	}
}

// TestPaddingInvarianceUnderCheckpointing repeats the gradient proof
// with activation checkpointing on, because the backward pass then
// REBUILDS each segment's forward from its saved input. If the
// recompute saw a different sequence length, a different mask or a
// different RoPE offset from the original forward, the gradients would
// be wrong in a way the uncheckpointed test cannot see.
func TestPaddingInvarianceUnderCheckpointing(t *testing.T) {
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2, 17, 33}
	supervised := []int{2, 3, 5, 6, 8}

	vm := tinyVoice(t)
	vm.CheckpointEvery = 1

	loss := vm.SupervisedLoss(tokens, supervised)
	base := loss.Data()[0]
	loss.Backward()
	want := gradSnapshot(t, vm)
	zeroGrads(vm)

	ploss := vm.SupervisedLoss(padWith(tokens, 22, 0), supervised)
	got := ploss.Data()[0]
	ploss.Backward()
	have := gradSnapshot(t, vm)

	if d := math.Abs(float64(base - got)); d > 1e-5*math.Abs(float64(base))+1e-6 {
		t.Fatalf("checkpointed loss changed under padding: %v vs %v (|Δ| %.3g)", base, got, d)
	}
	for i := range want {
		var maxAbs, maxDiff float64
		for j := range want[i] {
			a, b := float64(want[i][j]), float64(have[i][j])
			if math.Abs(a) > maxAbs {
				maxAbs = math.Abs(a)
			}
			if d := math.Abs(a - b); d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-5*maxAbs+1e-7 {
			t.Fatalf("checkpointed param %d gradient changed under padding: max|Δ| %.3g "+
				"against max|g| %.3g", i, maxDiff, maxAbs)
		}
	}
}

// TestCausalSoftmaxMasksBeforeNormalising is the load-bearing property
// underneath every padding claim, isolated. Masking AFTER the softmax
// (normalise over all columns, then zero the disallowed ones) would
// also leave pad columns at 0, and would still look "masked" — but it
// would change the probabilities of the REAL columns, because the pads
// would have taken a share of the denominator. Then padding would
// silently alter every real position's attention.
//
// The probe: give the pad columns enormously larger scores than the
// real ones. Under masked-then-normalised softmax the real row is
// untouched; under normalised-then-masked it collapses.
func TestCausalSoftmaxMasksBeforeNormalising(t *testing.T) {
	const heads, qSeq = 1, 4
	// Row i may attend columns 0..i; columns beyond are "pad".
	scores := make([]float32, heads*qSeq*qSeq)
	for i := 0; i < qSeq; i++ {
		for j := 0; j < qSeq; j++ {
			if j <= i {
				scores[i*qSeq+j] = float32(j) * 0.25
			} else {
				scores[i*qSeq+j] = 50 // would dominate any softmax it entered
			}
		}
	}
	y := g.CausalSoftmax(g.NewTensor(scores, heads, qSeq, qSeq), heads, qSeq, 1.0)
	d := y.Data()

	for i := 0; i < qSeq; i++ {
		var sum float64
		for j := 0; j < qSeq; j++ {
			v := float64(d[i*qSeq+j])
			if j > i {
				if v != 0 {
					t.Fatalf("row %d column %d is %v, want exactly 0 — a masked column "+
						"with a nonzero probability lets padding into real positions", i, j, v)
				}
				continue
			}
			sum += v
		}
		if math.Abs(sum-1) > 1e-6 {
			t.Fatalf("row %d allowed columns sum to %v, want 1 — the softmax normalised "+
				"over the masked columns too, so appending padding would change every "+
				"real position's attention", i, sum)
		}
	}
}
