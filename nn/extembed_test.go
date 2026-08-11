//go:build darwin

package nn

import (
	"math"
	"math/rand"
	"testing"

	g "github.com/vinq1911/gorch"
)

const (
	teBaseVocab = 32
	teNumExt    = 6
	teDim       = 8
)

// testExtEmbed builds a small ExtendedEmbedding plus the equivalent
// naive concatenated full matrix (teBaseVocab+teNumExt, teDim).
func testExtEmbed(t *testing.T) (*ExtendedEmbedding, *g.Tensor) {
	t.Helper()
	rng := rand.New(rand.NewSource(11))
	base := g.Zeros(teBaseVocab, teDim)
	for i := range base.Data() {
		base.Data()[i] = float32(rng.NormFloat64())
	}
	e := NewExtendedEmbedding(base, teNumExt)

	full := g.Zeros(teBaseVocab+teNumExt, teDim)
	copy(full.Data()[:teBaseVocab*teDim], base.Data())
	copy(full.Data()[teBaseVocab*teDim:], e.Ext.Data())
	return e, full
}

// TestExtEmbedInit — Ext rows start at mean(Base rows) + N(0, 0.02).
func TestExtEmbedInit(t *testing.T) {
	e, _ := testExtEmbed(t)
	mean := make([]float64, teDim)
	bd := e.Base.Data()
	for r := 0; r < teBaseVocab; r++ {
		for j := 0; j < teDim; j++ {
			mean[j] += float64(bd[r*teDim+j])
		}
	}
	for j := range mean {
		mean[j] /= teBaseVocab
	}
	for r := 0; r < teNumExt; r++ {
		for j := 0; j < teDim; j++ {
			d := math.Abs(float64(e.Ext.Data()[r*teDim+j]) - mean[j])
			if d > 0.2 { // 10σ of the 0.02 noise
				t.Fatalf("Ext[%d,%d] deviates %.3g from the base-row mean (noise σ=0.02)", r, j, d)
			}
		}
	}
	if !e.Ext.RequiresGrad() {
		t.Fatal("Ext must require grad")
	}
	if e.VocabSize() != teBaseVocab+teNumExt {
		t.Fatalf("VocabSize %d", e.VocabSize())
	}
}

// TestExtEmbedLookupParity — partition lookup matches a naive
// EmbeddingLookup over the concatenated matrix on mixed base/ext ids
// (plan §3.2 gate).
func TestExtEmbedLookupParity(t *testing.T) {
	e, full := testExtEmbed(t)
	ids := []int{0, 31, 32, 5, 37, 32, 0, 33}

	got := e.Forward(ids)
	want := g.EmbeddingLookup(full, ids)
	for i := range want.Data() {
		if got.Data()[i] != want.Data()[i] {
			t.Fatalf("element %d: %v != %v", i, got.Data()[i], want.Data()[i])
		}
	}
}

// TestExtEmbedLookupGradFlow — Base receives zero (structurally NO)
// gradient; Ext receives exactly the rows a full-matrix reference
// would place there (plan §3.2 gate).
func TestExtEmbedLookupGradFlow(t *testing.T) {
	e, full := testExtEmbed(t)
	full.SetRequiresGrad(true)
	ids := []int{2, 34, 32, 32, 7, 36}
	w := randInput(21, len(ids), teDim)

	g.Sum(g.Mul(e.Forward(ids), w)).Backward()
	g.Sum(g.Mul(g.EmbeddingLookup(full, ids), w)).Backward()

	if e.Base.Grad() != nil {
		t.Fatal("frozen Base received a gradient")
	}
	if e.Ext.Grad() == nil {
		t.Fatal("Ext received no gradient")
	}
	refExt := full.Grad().Data()[teBaseVocab*teDim:]
	for i := range refExt {
		if e.Ext.Grad().Data()[i] != refExt[i] {
			t.Fatalf("dExt[%d] = %v, full-matrix reference %v", i, e.Ext.Grad().Data()[i], refExt[i])
		}
	}
	// The looked-up base rows DID get gradient in the reference —
	// confirming the structural difference is exactly the frozen rows.
	baseRef := full.Grad().Data()[:teBaseVocab*teDim]
	var nonzero bool
	for _, v := range baseRef {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("reference base-row grads all zero — test would not detect a leak")
	}
}

// TestExtEmbedLogitsParity — split tied head matches the full-matrix
// reference logits (plan §3.2 gate).
func TestExtEmbedLogitsParity(t *testing.T) {
	e, full := testExtEmbed(t)
	h := randInput(22, 5, teDim)

	got := e.Logits(h)
	want := g.MatMulTransB(h, full)
	if got.Shape()[1] != teBaseVocab+teNumExt {
		t.Fatalf("logits vocab dim %d", got.Shape()[1])
	}
	for i := range want.Data() {
		d := math.Abs(float64(got.Data()[i]) - float64(want.Data()[i]))
		if d > 1e-6 {
			t.Fatalf("logit %d: %v vs %v (|Δ| %.3g)", i, got.Data()[i], want.Data()[i], d)
		}
	}
}

// TestExtEmbedLogitsGradFlow — through a CE loss on the split head:
// dL/dh matches the full-matrix autograd reference (BOTH column
// blocks contribute), dExt matches the reference's ext rows, and Base
// receives nothing.
func TestExtEmbedLogitsGradFlow(t *testing.T) {
	e, full := testExtEmbed(t)
	full.SetRequiresGrad(true)

	h1 := randInput(23, 4, teDim).SetRequiresGrad(true)
	h2 := g.NewTensor(h1.Data(), 4, teDim).SetRequiresGrad(true)
	targets := g.NewTensor([]float32{3, 33, 35, 0}, 4, 1) // mix of base and ext classes

	loss1 := g.CrossEntropyLoss(e.Logits(h1), targets)
	loss1.Backward()
	loss2 := g.CrossEntropyLoss(g.MatMul(h2, g.Transpose2D(full)), targets)
	loss2.Backward()

	if d := math.Abs(float64(loss1.Data()[0]) - float64(loss2.Data()[0])); d > 1e-6 {
		t.Fatalf("loss %v vs reference %v", loss1.Data()[0], loss2.Data()[0])
	}
	if e.Base.Grad() != nil {
		t.Fatal("frozen Base received a gradient through the head")
	}
	if h1.Grad() == nil || e.Ext.Grad() == nil {
		t.Fatal("h or Ext received no gradient")
	}
	for i := range h1.Grad().Data() {
		d := math.Abs(float64(h1.Grad().Data()[i]) - float64(h2.Grad().Data()[i]))
		if d > 1e-6 {
			t.Fatalf("dh[%d]: split %v vs full %v — base-block contribution missing?",
				i, h1.Grad().Data()[i], h2.Grad().Data()[i])
		}
	}
	refExt := full.Grad().Data()[teBaseVocab*teDim:]
	for i := range refExt {
		d := math.Abs(float64(e.Ext.Grad().Data()[i]) - float64(refExt[i]))
		if d > 1e-6 {
			t.Fatalf("dExt[%d]: %v vs reference %v", i, e.Ext.Grad().Data()[i], refExt[i])
		}
	}
}
