//go:build darwin

package gorch

import "fmt"

// Permute returns a view of t with its dimensions reordered according
// to perm. Like PyTorch's tensor.permute / NumPy's transpose with an
// axes argument. perm must be a permutation of (0..ndim-1).
//
// Example: a (B, S, D) tensor permuted to (1, 0, 2) becomes (S, B, D).
//
// Output is a fresh allocation with copied data, NOT a stride-only
// view — gorch tensors don't carry strides today, so a "permute" that
// preserved the underlying buffer would lie about element layout.
// Permute writes the data in the new logical order.
//
// Backward is the inverse permutation applied to the upstream grad.
//
// Plan 0001 Phase 1 item 3. Needed by every multi-head reshape — both
// MultiHeadAttention's existing per-head loop and the upcoming GQA/MLA
// modules use this shape over (B, S, H, D) ↔ (B, H, S, D).
func Permute(t *Tensor, perm []int) *Tensor {
	nd := t.Dim()
	if len(perm) != nd {
		panic(fmt.Sprintf("gorch: Permute perm length %d != tensor ndim %d", len(perm), nd))
	}
	// Validate perm is a permutation of (0..nd-1).
	seen := make([]bool, nd)
	for _, p := range perm {
		if p < 0 || p >= nd || seen[p] {
			panic(fmt.Sprintf("gorch: Permute invalid perm %v for ndim %d", perm, nd))
		}
		seen[p] = true
	}

	srcShape := t.shape
	dstShape := make([]int, nd)
	for i, p := range perm {
		dstShape[i] = srcShape[p]
	}

	// Strides for source (row-major, last dim contiguous).
	srcStride := make([]int, nd)
	srcStride[nd-1] = 1
	for i := nd - 2; i >= 0; i-- {
		srcStride[i] = srcStride[i+1] * srcShape[i+1]
	}

	out := Zeros(dstShape...)
	dstData := out.data
	srcData := t.data

	// Walk destination in row-major order; for each dst index, compute
	// src offset by mapping each dst dim back to its src dim via perm.
	idx := make([]int, nd)
	total := numElements(dstShape)
	for k := 0; k < total; k++ {
		// Compute src offset.
		var srcOff int
		for i, p := range perm {
			srcOff += idx[i] * srcStride[p]
		}
		dstData[k] = srcData[srcOff]

		// Increment dst index in row-major order.
		for i := nd - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < dstShape[i] {
				break
			}
			idx[i] = 0
		}
	}

	if GradEnabled() && t.requiresGrad {
		out.requiresGrad = true
		// Inverse permutation: if perm maps src→dst as dst[i] = src[perm[i]],
		// then the inverse maps dst→src as src[perm[i]] = dst[i],
		// i.e., invPerm[perm[i]] = i.
		invPerm := make([]int, nd)
		for i, p := range perm {
			invPerm[p] = i
		}
		out.gradFn = &GradFn{
			name:   "Permute",
			inputs: []*Tensor{t},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{Permute(grad, invPerm)}
			},
		}
	}
	return out
}
