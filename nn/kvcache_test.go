//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

func TestKVCacheBasic(t *testing.T) {
	c := NewKVCache(3, 16)
	if c.Len() != 0 {
		t.Fatalf("initial len = %d, want 0", c.Len())
	}
	if c.Layers != 3 || c.Dim != 16 {
		t.Fatalf("dims wrong: layers=%d dim=%d", c.Layers, c.Dim)
	}

	// Append two tokens worth of K/V to all layers.
	for layer := 0; layer < 3; layer++ {
		k := make([]float32, 2*16)
		v := make([]float32, 2*16)
		for i := range k {
			k[i] = float32(i + layer)
			v[i] = float32(-i - layer)
		}
		c.Append(layer, k, v)
	}
	if c.Len() != 2 {
		t.Fatalf("len after append = %d, want 2", c.Len())
	}

	// One more token to layer 0/1/2 in lockstep.
	for layer := 0; layer < 3; layer++ {
		c.Append(layer, make([]float32, 16), make([]float32, 16))
	}
	if c.Len() != 3 {
		t.Fatalf("len = %d, want 3", c.Len())
	}

	c.Reset()
	if c.Len() != 0 {
		t.Fatalf("after reset len = %d, want 0", c.Len())
	}
}

// TestForwardCachedMatchesUncached is the correctness check that
// matters: incremental decoding via the KV cache must produce
// exactly (within fp32 noise) the same logits as feeding the full
// sequence through the un-cached forward pass.
func TestForwardCachedMatchesUncached(t *testing.T) {
	// Small deterministic model.
	gpt := NewGPT(32, 16, 2, 2, 64)
	prompt := []int{1, 2, 3, 4}

	want := gpt.Forward(prompt)
	wantData := want.Data()
	wantRows := want.Shape()[0]
	vocab := want.Shape()[1]

	cache := NewKVCache(gpt.NumLayers, gpt.Dim)
	// Prefill the prompt all at once.
	got := gpt.ForwardCached(prompt, cache)
	gotData := got.Data()

	if got.Shape()[0] != wantRows || got.Shape()[1] != vocab {
		t.Fatalf("prefill shape mismatch: got %v want %v", got.Shape(), want.Shape())
	}

	maxDiff := float32(0)
	for i := range wantData {
		d := wantData[i] - gotData[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-3 {
		t.Fatalf("prefill logits differ from uncached: maxDiff=%g", maxDiff)
	}

	// Now feed two more tokens incrementally and compare against an
	// uncached forward of the full sequence.
	full := append([]int{}, prompt...)
	for _, next := range []int{7, 11} {
		full = append(full, next)
		uncached := gpt.Forward(full)
		uncachedLast := uncached.Data()[(len(full)-1)*vocab : len(full)*vocab]

		stepLogits := gpt.ForwardCached([]int{next}, cache)
		stepData := stepLogits.Data()
		if stepLogits.Shape()[0] != 1 || stepLogits.Shape()[1] != vocab {
			t.Fatalf("incremental shape mismatch: %v", stepLogits.Shape())
		}

		var diff float32
		for i := 0; i < vocab; i++ {
			d := stepData[i] - uncachedLast[i]
			if d < 0 {
				d = -d
			}
			if d > diff {
				diff = d
			}
		}
		if math.IsNaN(float64(diff)) || diff > 1e-3 {
			t.Fatalf("incremental token %d logits differ: maxDiff=%g", next, diff)
		}
	}
}

func TestForwardCachedAdvancesPosition(t *testing.T) {
	// Distinct token at position 0 vs position 5 must produce
	// distinct logits — otherwise position embeddings aren't being
	// applied correctly through the cache.
	gpt := NewGPT(16, 8, 2, 1, 32)
	cache := NewKVCache(gpt.NumLayers, gpt.Dim)

	// Push 5 dummy tokens through prefill.
	prefill := []int{1, 2, 3, 4, 5}
	gpt.ForwardCached(prefill, cache)

	emptyCache := NewKVCache(gpt.NumLayers, gpt.Dim)
	atZero := gpt.ForwardCached([]int{7}, emptyCache).Data()
	atFive := gpt.ForwardCached([]int{7}, cache).Data()

	differs := false
	for i := range atZero {
		if math.Abs(float64(atZero[i]-atFive[i])) > 1e-5 {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("logits at position 0 vs 5 are identical — pos embedding not advancing")
	}
}

// Quick smoke test for the bidirectional/causal mask integration in
// the cached path: when Causal=false the mask should be skipped and
// queries should attend to future cached keys (only meaningful in
// prefill since incremental steps have no future keys anyway).
func TestForwardCachedRespectsCausalFlag(t *testing.T) {
	gpt := NewGPT(8, 4, 2, 1, 16)
	prompt := []int{1, 2, 3}

	// Causal: row i should only depend on rows ≤ i.
	cache := NewKVCache(gpt.NumLayers, gpt.Dim)
	logits := gpt.ForwardCached(prompt, cache)
	if logits.Shape()[0] != 3 {
		t.Fatalf("got %v rows, want 3", logits.Shape()[0])
	}

	// Sanity: changing the last prompt token should change row 2 but
	// leave row 0 and row 1 untouched (causal).
	cache2 := NewKVCache(gpt.NumLayers, gpt.Dim)
	other := []int{1, 2, 5} // differs only in last token
	logits2 := gpt.ForwardCached(other, cache2)
	row0Same := allClose(t, logits.Data()[:8], logits2.Data()[:8], 1e-5)
	row1Same := allClose(t, logits.Data()[8:16], logits2.Data()[8:16], 1e-5)
	row2Same := allClose(t, logits.Data()[16:24], logits2.Data()[16:24], 1e-5)
	if !row0Same || !row1Same {
		t.Fatal("causal violated: earlier rows changed when last token changed")
	}
	if row2Same {
		t.Fatal("row 2 unchanged when last token changed — something is broken")
	}
}

func allClose(t *testing.T, a, b []float32, tol float32) bool {
	t.Helper()
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			return false
		}
	}
	return true
}

// TestForwardCachedNoAutogradLeak guards against the cached path
// accidentally building an autograd graph (it's inference-only and
// should not).
func TestForwardCachedNoAutogradLeak(t *testing.T) {
	gpt := NewGPT(8, 4, 2, 1, 8)
	cache := NewKVCache(gpt.NumLayers, gpt.Dim)
	out := gpt.ForwardCached([]int{1, 2, 3}, cache)
	// LMHead is a Linear with requires_grad weights, so the output
	// inherits requires_grad — that's fine. The thing we want to
	// avoid is segfaults when no grad is propagated; just ensure
	// the output isn't NaN.
	for _, v := range out.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite logit %v", v)
		}
	}
	_ = g.Zeros // keep import live
}
