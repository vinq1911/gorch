//go:build darwin

package gorch

import (
	"math"
	"math/rand"
	"testing"
)

// Golden tests for the X2b permute_copy Metal kernel (plan 0009).
// CPU reference: the permute.go destination walk (already pinned by
// permute_test.go, incl. a numerical-grad test). GPU-vs-CPU parity is
// exact-copy semantics, so the tolerance is 0 — but the csRetry
// discipline is kept for consistency with the other kernel gates.

// permuteScalarRef is the pre-X2b element walk, kept verbatim as the
// test oracle (the production CPU path now uses run copies).
func permuteScalarRef(src []float32, srcShape, perm []int) []float32 {
	nd := len(srcShape)
	dstShape := make([]int, nd)
	for i, p := range perm {
		dstShape[i] = srcShape[p]
	}
	srcStride := make([]int, nd)
	srcStride[nd-1] = 1
	for i := nd - 2; i >= 0; i-- {
		srcStride[i] = srcStride[i+1] * srcShape[i+1]
	}
	total := 1
	for _, d := range dstShape {
		total *= d
	}
	dst := make([]float32, total)
	idx := make([]int, nd)
	for k := 0; k < total; k++ {
		var srcOff int
		for i, p := range perm {
			srcOff += idx[i] * srcStride[p]
		}
		dst[k] = src[srcOff]
		for i := nd - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < dstShape[i] {
				break
			}
			idx[i] = 0
		}
	}
	return dst
}

// TestPermuteCPUFastPathMatchesScalarRef pins the new run-copy CPU
// fast path (perm keeps the innermost dim) and the element-walk
// fallback against the scalar oracle.
func TestPermuteCPUFastPathMatchesScalarRef(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	cases := []struct {
		shape []int
		perm  []int
	}{
		{[]int{5, 7, 3}, []int{1, 0, 2}},       // run-copy path (attention head split)
		{[]int{4, 6, 2, 3}, []int{2, 0, 1, 3}}, // run-copy path, rank 4
		{[]int{5, 7, 3}, []int{2, 1, 0}},       // element-walk fallback
		{[]int{6, 8}, []int{1, 0}},             // 2-D transpose (walk)
	}
	for _, c := range cases {
		n := 1
		for _, d := range c.shape {
			n *= d
		}
		src := csRandSlice(rng, n, 1.0)
		want := permuteScalarRef(src, c.shape, c.perm)
		got := Permute(NewTensor(src, c.shape...), c.perm)
		if d := csMaxAbsDiff(want, got.Data()); d != 0 {
			t.Errorf("shape %v perm %v: CPU permute diff %g != 0", c.shape, c.perm, d)
		}
	}
}

// TestPermuteMetalMatchesCPU is the X2b permute kernel gate: Metal
// fwd + autograd bwd parity vs the scalar oracle at the attention
// head-reshape shapes, plus a rank-4 general perm.
func TestPermuteMetalMatchesCPU(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	rng := rand.New(rand.NewSource(11))

	cases := []struct {
		name  string
		shape []int
		perm  []int
	}{
		{"head_split_102", []int{37, 16, 24}, []int{1, 0, 2}},
		{"concat_102", []int{16, 37, 24}, []int{1, 0, 2}},
		{"rank4_2031", []int{3, 5, 4, 6}, []int{2, 0, 3, 1}},
		{"transpose_2d", []int{33, 17}, []int{1, 0}},
	}
	for _, c := range cases {
		n := 1
		for _, d := range c.shape {
			n *= d
		}
		src := csRandSlice(rng, n, 1.0)
		w := csRandSlice(rng, n, 1.0)
		refY := permuteScalarRef(src, c.shape, c.perm)

		// CPU autograd reference for the backward.
		xc := NewTensor(src, c.shape...).SetRequiresGrad(true)
		yc := Permute(xc, c.perm)
		Sum(Mul(yc, NewTensor(w, yc.Shape()...))).Backward()
		refDX := append([]float32(nil), xc.Grad().Data()...)

		csRetry(t, "permute_metal_"+c.name, func() (float64, bool) {
			xt := NewTensorOnMetal(gpuHandle.Dev, src, c.shape...).SetRequiresGrad(true)
			y := Permute(xt, c.perm)
			if !y.IsOnMetal() {
				return math.Inf(1), false
			}
			Sum(Mul(y, NewTensorOnMetal(gpuHandle.Dev, w, y.Shape()...))).Backward()
			d := csMaxAbsDiff(refY, y.Data())
			if d2 := csMaxAbsDiff(refDX, xt.Grad().Data()); d2 > d {
				d = d2
			}
			return d, d == 0 // pure copy: bit-exact
		})
	}
}

// TestRepeatInterleaveMetalMatchesCPU is the X2b GQA-expansion kernel
// gate: fwd expand + bwd sum-back parity vs the CPU path at the
// nn/gqa.go shapes ((numKV, 1, seq*headDim) repeated groupSize) and a
// generic rank-2 case. The forward is a copy (bit-exact); the backward
// sums n values in a fixed order, so it is bit-exact too (same order
// as the CPU loop).
func TestRepeatInterleaveMetalMatchesCPU(t *testing.T) {
	gpuHandle, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	rng := rand.New(rand.NewSource(13))

	cases := []struct {
		name  string
		shape []int
		n     int
	}{
		{"gqa_kv_expand", []int{8, 1, 37 * 16}, 2},
		{"rank2", []int{5, 9}, 3},
		{"rank3_wide_repeat", []int{2, 4, 11}, 5},
	}
	for _, c := range cases {
		total := 1
		for _, d := range c.shape {
			total *= d
		}
		src := csRandSlice(rng, total, 1.0)

		// CPU reference fwd+bwd with a weighted loss.
		xc := NewTensor(src, c.shape...).SetRequiresGrad(true)
		yc := RepeatInterleave(xc, c.n)
		wLen := yc.Size()
		w := csRandSlice(rng, wLen, 1.0)
		Sum(Mul(yc, NewTensor(w, yc.Shape()...))).Backward()
		refY := append([]float32(nil), yc.Data()...)
		refDX := append([]float32(nil), xc.Grad().Data()...)

		csRetry(t, "repeat_metal_"+c.name, func() (float64, bool) {
			xt := NewTensorOnMetal(gpuHandle.Dev, src, c.shape...).SetRequiresGrad(true)
			y := RepeatInterleave(xt, c.n)
			if !y.IsOnMetal() {
				return math.Inf(1), false
			}
			Sum(Mul(y, NewTensorOnMetal(gpuHandle.Dev, w, y.Shape()...))).Backward()
			d := csMaxAbsDiff(refY, y.Data())
			if d2 := csMaxAbsDiff(refDX, xt.Grad().Data()); d2 > d {
				d = d2
			}
			return d, d <= 1e-6
		})
	}
}
