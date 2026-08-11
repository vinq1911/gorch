//go:build darwin

package gorch

import (
	"fmt"

	"github.com/vinq1911/gorch/metal"
)

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
	if t.dtype == BFloat16 {
		return downcastToBF16(Permute(promoteToF32(t), perm))
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

	total := numElements(dstShape)
	var out *Tensor
	if t.buf != nil && nd <= 8 && permutePipelineReady() {
		// Metal path (plan 0009 X2b): the permute_copy kernel gathers
		// each destination element from src via the permuted strides —
		// no host sync, chain stays resident (Permute was 25% of the
		// residual CPU samples in the X2 profile).
		out = permuteMetal(t, dstShape, srcStride, perm, total)
	} else {
		out = ZerosLike(t, dstShape...)
		syncForCPU(t)
		permuteCPU(t.data, out.data, dstShape, srcStride, perm, total)
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

// permuteCPU walks the destination in row-major order, copying
// contiguous inner runs when the innermost dim is unpermuted
// (perm[nd-1] == nd-1 — every attention head-reshape hits this) and
// falling back to the element walk otherwise.
func permuteCPU(srcData, dstData []float32, dstShape, srcStride []int, perm []int, total int) {
	nd := len(dstShape)
	idx := make([]int, nd)
	if nd >= 2 && perm[nd-1] == nd-1 {
		inner := dstShape[nd-1]
		for k := 0; k < total; k += inner {
			var srcOff int
			for i := 0; i < nd-1; i++ {
				srcOff += idx[i] * srcStride[perm[i]]
			}
			copy(dstData[k:k+inner], srcData[srcOff:srcOff+inner])
			for i := nd - 2; i >= 0; i-- {
				idx[i]++
				if idx[i] < dstShape[i] {
					break
				}
				idx[i] = 0
			}
		}
		return
	}
	for k := 0; k < total; k++ {
		var srcOff int
		for i, p := range perm {
			srcOff += idx[i] * srcStride[p]
		}
		dstData[k] = srcData[srcOff]
		for i := nd - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < dstShape[i] {
				break
			}
			idx[i] = 0
		}
	}
}

// permutePipelineReady reports whether the permute_copy kernel was
// compiled by InitMetal.
func permutePipelineReady() bool {
	if gpu == nil {
		return false
	}
	_, ok := gpu.pipelines["permute_copy"]
	return ok
}

// permuteMetal dispatches permute_copy: one thread per destination
// element, src offset computed from the permuted strides. Returns a
// Metal-backed tensor of dstShape.
func permuteMetal(t *Tensor, dstShape, srcStride, perm []int, total int) *Tensor {
	dev := gpu.Dev
	out := ZerosOnMetal(dev, dstShape...)

	nd := len(dstShape)
	metaBuf := dev.NewBuffer((1 + 2*nd) * 4)
	meta := metaBuf.Uint32Slice()
	meta[0] = uint32(nd)
	for i := 0; i < nd; i++ {
		meta[1+i] = uint32(dstShape[i])
		meta[1+nd+i] = uint32(srcStride[perm[i]])
	}

	gpu.Queue.Dispatch1D(gpu.pipe("permute_copy"),
		[]*metal.Buffer{t.buf, metaBuf, out.buf}, total)
	metalPermuteDispatches.Add(1)
	metaBuf.Release()
	return out
}
