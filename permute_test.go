//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// TestPermute2DMatchesTranspose: with perm=[1,0] on a 2-D tensor, the
// result must equal the existing Transpose2D output element-wise.
func TestPermute2DMatchesTranspose(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	viaPermute := Permute(a, []int{1, 0})
	viaTranspose := Transpose2D(a)
	if !shapesEqual(viaPermute.Shape(), viaTranspose.Shape()) {
		t.Fatalf("shape mismatch: %v vs %v", viaPermute.Shape(), viaTranspose.Shape())
	}
	for i := range viaPermute.Data() {
		if viaPermute.Data()[i] != viaTranspose.Data()[i] {
			t.Fatalf("[%d] permute=%g transpose=%g", i, viaPermute.Data()[i], viaTranspose.Data()[i])
		}
	}
}

// TestPermute3DSwapMiddleAndLast: classic (B, S, D) → (B, D, S) check.
func TestPermute3DSwapMiddleAndLast(t *testing.T) {
	// Shape (2, 3, 4) — values are flat-index for easy verification.
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	a := NewTensor(data, 2, 3, 4)
	out := Permute(a, []int{0, 2, 1}) // (2, 4, 3)
	if !shapesEqual(out.Shape(), []int{2, 4, 3}) {
		t.Fatalf("shape = %v, want [2 4 3]", out.Shape())
	}
	// Spot-check: original a[1, 2, 3] (flat 1*12 + 2*4 + 3 = 23)
	// becomes out[1, 3, 2] (flat 1*12 + 3*3 + 2 = 23). Same flat
	// position only because the cyclic rearrangement happens to land
	// here. Better check: value at out[0,0,0] = a[0,0,0] = 0; out
	// [0,0,1] = a[0,1,0] = 4; out[0,1,0] = a[0,0,1] = 1.
	want := map[[3]int]float32{
		{0, 0, 0}: 0,
		{0, 0, 1}: 4,
		{0, 1, 0}: 1,
		{0, 3, 2}: 11,
		{1, 2, 1}: 18,
	}
	for idx, expected := range want {
		got := out.At(idx[0], idx[1], idx[2])
		if got != expected {
			t.Fatalf("out[%v] = %g, want %g", idx, got, expected)
		}
	}
}

// TestPermuteRoundTrip: applying perm and then its inverse should
// recover the original tensor element-wise.
func TestPermuteRoundTrip(t *testing.T) {
	a := RandN(3, 4, 5, 6)
	perm := []int{2, 0, 3, 1}
	inv := []int{1, 3, 0, 2} // such that inv[perm[i]] = i
	out := Permute(Permute(a, perm), inv)
	for i := range a.Data() {
		if a.Data()[i] != out.Data()[i] {
			t.Fatalf("[%d] round-trip failed: got %g, want %g", i, out.Data()[i], a.Data()[i])
		}
	}
}

// TestPermuteBackwardMatchesNumerical: classic numerical-vs-analytical
// check. Permute's backward is the inverse permutation of grad, so the
// analytical and numerical gradients should agree to fp32 precision.
func TestPermuteBackwardMatchesNumerical(t *testing.T) {
	a := RandN(2, 3, 4).SetRequiresGrad(true)
	perm := []int{2, 0, 1}

	loss := Sum(Permute(a, perm))
	loss.Backward()
	dxAnalytic := append([]float32{}, a.Grad().Data()...)

	const h = 1e-3
	for i := range a.Data() {
		orig := a.Data()[i]
		a.Data()[i] = orig + h
		yPlus := Sum(Permute(a, perm)).Data()[0]
		a.Data()[i] = orig - h
		yMinus := Sum(Permute(a, perm)).Data()[0]
		a.Data()[i] = orig
		num := (yPlus - yMinus) / (2 * h)
		if math.Abs(float64(dxAnalytic[i]-num)) > 1e-2 {
			t.Fatalf("[%d] analytic=%g numeric=%g", i, dxAnalytic[i], num)
		}
	}
}

func TestPermuteValidatesInput(t *testing.T) {
	a := RandN(2, 3, 4)
	cases := []struct {
		name string
		perm []int
	}{
		{"too short", []int{0, 1}},
		{"too long", []int{0, 1, 2, 3}},
		{"duplicate axis", []int{0, 0, 1}},
		{"out of range", []int{0, 1, 5}},
		{"negative axis", []int{0, -1, 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("expected panic for perm=%v", tc.perm)
				}
			}()
			Permute(a, tc.perm)
		})
	}
}

// TestPermuteHeadReshape: the shape-juggle MultiHeadAttention does
// today — (B, S, H*D) → (B, H, S, D) — should work via reshape +
// permute and round-trip cleanly. This is the load-bearing case for
// upcoming GQA/MLA modules.
func TestPermuteHeadReshape(t *testing.T) {
	const B, S, H, D = 2, 4, 3, 5
	a := RandN(B, S, H*D)

	// (B, S, H*D) → (B, S, H, D) reshape → (B, H, S, D) permute
	r := a.Reshape(B, S, H, D)
	out := Permute(r, []int{0, 2, 1, 3})
	if !shapesEqual(out.Shape(), []int{B, H, S, D}) {
		t.Fatalf("got shape %v, want [%d %d %d %d]", out.Shape(), B, H, S, D)
	}

	// Reverse: (B, H, S, D) permute → (B, S, H, D) → reshape (B, S, H*D)
	back := Permute(out, []int{0, 2, 1, 3}).Reshape(B, S, H*D)
	for i := range a.Data() {
		if a.Data()[i] != back.Data()[i] {
			t.Fatalf("head-reshape round-trip failed at [%d]: %g vs %g",
				i, a.Data()[i], back.Data()[i])
		}
	}
}

func shapesEqual(a, b []int) bool {
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
