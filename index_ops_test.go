//go:build darwin

package gorch

import (
	"math"
	"math/rand"
	"testing"
)

// ---------- Gather ----------

func TestGatherForwardSpec(t *testing.T) {
	src := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}, 4, 3)
	idx := []int{0, 2, 1, 2}
	out := Gather(src, idx)

	want := []float32{
		1, 2, 3,
		7, 8, 9,
		4, 5, 6,
		7, 8, 9,
	}
	if len(out.Data()) != len(want) {
		t.Fatalf("size mismatch: got %d want %d", len(out.Data()), len(want))
	}
	for i, w := range want {
		if out.Data()[i] != w {
			t.Fatalf("[%d] got %g want %g", i, out.Data()[i], w)
		}
	}
}

func TestGatherBackwardScatterAdds(t *testing.T) {
	src := RandN(5, 4).SetRequiresGrad(true)
	// Index 1 appears twice — its gradient should be summed.
	idx := []int{1, 1, 3}

	loss := Sum(Gather(src, idx))
	loss.Backward()

	// Each gathered row contributes 1 to the source row's grad
	// (because loss = sum of 12 elements, dgrad/dsrc[i,j] = 1 if
	// row i is gathered at least once, weighted by repetition).
	want := make([]float32, 5*4)
	for j := 0; j < 4; j++ {
		want[1*4+j] = 2 // index 1 appears twice
		want[3*4+j] = 1 // index 3 once
	}
	for i, w := range want {
		if math.Abs(float64(src.Grad().Data()[i]-w)) > 1e-5 {
			t.Fatalf("[%d] grad=%g want %g", i, src.Grad().Data()[i], w)
		}
	}
}

func TestGatherInvalidIndexPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on out-of-range index")
		}
	}()
	src := RandN(3, 2)
	Gather(src, []int{0, 10})
}

// ---------- ScatterAdd ----------

func TestScatterAddForwardSpec(t *testing.T) {
	// src 3 rows, scatter into N=5 with overlap (idx[0] = idx[2] = 1).
	src := NewTensor([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, 3, 2)
	idx := []int{1, 3, 1}
	out := ScatterAdd(src, idx, 5)
	want := []float32{
		0, 0, // row 0 untouched
		6, 8, // row 1 = src[0] + src[2] = (1+5, 2+6)
		0, 0,
		3, 4, // row 3 = src[1]
		0, 0,
	}
	for i, w := range want {
		if out.Data()[i] != w {
			t.Fatalf("[%d] got %g want %g", i, out.Data()[i], w)
		}
	}
}

// TestScatterAddIsGatherInverse: Scatter then Gather at the same
// indices must round-trip src exactly (up to overlap edge cases).
func TestScatterAddIsGatherInverse(t *testing.T) {
	src := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2)
	idx := []int{2, 0, 5, 7} // no overlaps
	scatterOut := ScatterAdd(src, idx, 8)
	gatherBack := Gather(scatterOut, idx)
	for i, want := range src.Data() {
		if gatherBack.Data()[i] != want {
			t.Fatalf("[%d] round-trip failed: got %g want %g", i, gatherBack.Data()[i], want)
		}
	}
}

// TestScatterAddBackwardMatchesNumerical: analytical-vs-numerical for
// both unique indices and overlapping ones.
func TestScatterAddBackwardMatchesNumerical(t *testing.T) {
	src := RandN(4, 3).SetRequiresGrad(true)
	idx := []int{2, 0, 2, 5} // index 2 appears twice
	loss := Sum(ScatterAdd(src, idx, 6))
	loss.Backward()
	dSrcAnalytic := append([]float32{}, src.Grad().Data()...)

	const h = 1e-3
	for i := range src.Data() {
		orig := src.Data()[i]
		src.Data()[i] = orig + h
		yPlus := Sum(ScatterAdd(src, idx, 6)).Data()[0]
		src.Data()[i] = orig - h
		yMinus := Sum(ScatterAdd(src, idx, 6)).Data()[0]
		src.Data()[i] = orig
		num := (yPlus - yMinus) / (2 * h)
		if math.Abs(float64(dSrcAnalytic[i]-num)) > 1e-2 {
			t.Fatalf("[%d] analytic=%g numeric=%g", i, dSrcAnalytic[i], num)
		}
	}
}

func TestScatterAddInvalidIndexPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on out-of-range index")
		}
	}()
	src := RandN(2, 3)
	ScatterAdd(src, []int{0, 99}, 5)
}

func TestScatterAddIdxLengthMismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on idx length mismatch")
		}
	}()
	src := RandN(3, 2)
	ScatterAdd(src, []int{0, 1}, 5) // idx length 2 != src rows 3
}

// ---------- TopK ----------

func TestTopKValuesAndIndices(t *testing.T) {
	x := NewTensor([]float32{
		1, 5, 3, 2, 4,
		9, 7, 8, 6, 10,
	}, 2, 5)
	values, idx := TopK(x, 3)
	if !shapesEqualIdx(values.Shape(), []int{2, 3}) {
		t.Fatalf("values shape = %v, want [2 3]", values.Shape())
	}
	wantValues := []float32{5, 4, 3, 10, 9, 8}
	wantIdx := []int{1, 4, 2, 4, 0, 2}
	for i, w := range wantValues {
		if values.Data()[i] != w {
			t.Fatalf("values[%d] = %g, want %g", i, values.Data()[i], w)
		}
	}
	for i, w := range wantIdx {
		if idx[i] != w {
			t.Fatalf("idx[%d] = %d, want %d", i, idx[i], w)
		}
	}
}

func TestTopKValidatesK(t *testing.T) {
	for _, k := range []int{0, -1, 100} {
		t.Run("k_invalid", func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("expected panic for k=%d", k)
				}
			}()
			TopK(RandN(2, 5), k)
		})
	}
}

// ---------- Multinomial ----------

func TestMultinomialDeterministicAtTemperatureZero(t *testing.T) {
	// All probability on one element — sample must always pick it.
	probs := NewTensor([]float32{
		1, 0, 0, 0,
		0, 0, 0, 1,
	}, 2, 4)
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 20; trial++ {
		out := Multinomial(probs, rng)
		if out[0] != 0 || out[1] != 3 {
			t.Fatalf("trial %d: got %v want [0 3]", trial, out)
		}
	}
}

func TestMultinomialApproxMatchesProbs(t *testing.T) {
	// Probability [0.7, 0.3] — over many samples, fraction of 0s
	// should be ~0.7 within Monte-Carlo noise.
	probs := NewTensor([]float32{0.7, 0.3}, 1, 2)
	rng := rand.New(rand.NewSource(123))
	const N = 5000
	count0 := 0
	for i := 0; i < N; i++ {
		out := Multinomial(probs, rng)
		if out[0] == 0 {
			count0++
		}
	}
	frac := float64(count0) / float64(N)
	if math.Abs(frac-0.7) > 0.03 {
		t.Fatalf("frac 0 = %g, expected ~0.7 (±0.03)", frac)
	}
}

// ---------- RepeatInterleave ----------

func TestRepeatInterleave2D(t *testing.T) {
	// (kvHeads=2, headDim=3) repeat n=2 → (4, 3) where kv[0] appears
	// twice in a row, then kv[1] twice.
	src := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)
	out := RepeatInterleave(src, 2)
	want := []float32{
		1, 2, 3,
		1, 2, 3,
		4, 5, 6,
		4, 5, 6,
	}
	if !shapesEqualIdx(out.Shape(), []int{4, 3}) {
		t.Fatalf("shape = %v want [4 3]", out.Shape())
	}
	for i, w := range want {
		if out.Data()[i] != w {
			t.Fatalf("[%d] got %g want %g", i, out.Data()[i], w)
		}
	}
}

func TestRepeatInterleave4DGQAShape(t *testing.T) {
	// (B=2, S=3, kvHeads=4, headDim=5) → repeat n=2 →
	// (2, 3, 8, 5). This is the canonical GQA expansion shape.
	const B, S, K, D = 2, 3, 4, 5
	src := RandN(B, S, K, D)
	out := RepeatInterleave(src, 2)
	want := []int{B, S, K * 2, D}
	if !shapesEqualIdx(out.Shape(), want) {
		t.Fatalf("shape = %v want %v", out.Shape(), want)
	}
	// Spot-check: out[1, 2, 5, 3] should equal src[1, 2, 5/2, 3] = src[1, 2, 2, 3].
	got := out.At(1, 2, 5, 3)
	expected := src.At(1, 2, 2, 3)
	if got != expected {
		t.Fatalf("out[1,2,5,3]=%g, want src[1,2,2,3]=%g", got, expected)
	}
}

func TestRepeatInterleaveBackwardSums(t *testing.T) {
	// For loss = sum(repeat_interleave(x, n)), each x[i] contributes
	// to n positions in the output, so dx[i] = n.
	src := NewTensor([]float32{1, 2, 3, 4}, 2, 2).SetRequiresGrad(true)
	loss := Sum(RepeatInterleave(src, 3))
	loss.Backward()
	for i, v := range src.Grad().Data() {
		if math.Abs(float64(v-3)) > 1e-5 {
			t.Fatalf("[%d] grad=%g want 3", i, v)
		}
	}
}

func TestRepeatInterleaveBackwardMatchesNumerical(t *testing.T) {
	src := RandN(2, 4, 3).SetRequiresGrad(true)
	const n = 2

	loss := Sum(RepeatInterleave(src, n))
	loss.Backward()
	dxAnalytic := append([]float32{}, src.Grad().Data()...)

	const h = 1e-3
	for i := range src.Data() {
		orig := src.Data()[i]
		src.Data()[i] = orig + h
		yPlus := Sum(RepeatInterleave(src, n)).Data()[0]
		src.Data()[i] = orig - h
		yMinus := Sum(RepeatInterleave(src, n)).Data()[0]
		src.Data()[i] = orig
		num := (yPlus - yMinus) / (2 * h)
		if math.Abs(float64(dxAnalytic[i]-num)) > 1e-2 {
			t.Fatalf("[%d] analytic=%g numeric=%g", i, dxAnalytic[i], num)
		}
	}
}

func shapesEqualIdx(a, b []int) bool {
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
