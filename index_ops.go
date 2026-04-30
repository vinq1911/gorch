//go:build darwin

package gorch

import (
	"fmt"
	"math/rand"
	"sort"
)

// Gather selects rows from a 2-D tensor by integer indices.
//
//	src: (N, D), idx: []int of length M  →  out: (M, D)
//	out[i, :] = src[idx[i], :]
//
// PyTorch equivalent: torch.index_select(src, dim=0, index=idx).
// MoE expert dispatch is the canonical use case: gather routed
// tokens by destination expert before the grouped matmul.
//
// Backward: scatter-add — for each i, dx[idx[i], :] += grad[i, :].
// Indices that don't appear get zero gradient; indices that appear
// multiple times have their gradients summed (correct for repeated
// gather followed by a sum-style downstream).
//
// Plan 0001 Phase 1 item 4.
func Gather(src *Tensor, idx []int) *Tensor {
	if src.Dim() != 2 {
		panic("gorch: Gather requires 2-D source")
	}
	N, D := src.shape[0], src.shape[1]
	M := len(idx)

	out := Zeros(M, D)
	for i, j := range idx {
		if j < 0 || j >= N {
			panic(fmt.Sprintf("gorch: Gather index %d out of range [0, %d)", j, N))
		}
		copy(out.data[i*D:(i+1)*D], src.data[j*D:(j+1)*D])
	}

	if GradEnabled() && src.requiresGrad {
		out.requiresGrad = true
		idxCopy := append([]int{}, idx...)
		out.gradFn = &GradFn{
			name:   "Gather",
			inputs: []*Tensor{src},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(N, D)
				for i, j := range idxCopy {
					for d := 0; d < D; d++ {
						dx.data[j*D+d] += grad.data[i*D+d]
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// ScatterAdd is the autograd-aware inverse of Gather.
//
//	src: (M, D), idx: []int of length M, dim N
//	out: (N, D) initially zeroed; for each i:
//	     out[idx[i], :] += src[i, :]
//
// Indices that appear multiple times in idx accumulate the rows of
// src at those positions (the "add" in scatter-add). Indices that
// don't appear keep their initial zero. PyTorch equivalent:
// torch.zeros(N, D).index_add_(0, idx, src).
//
// MoE's expert-dispatch scatter is the canonical use case:
// after each expert produces its output rows, scatter-add them
// (weighted by routing probabilities) back into the global output
// at the correct token positions.
//
// Backward (with respect to src): for each i,
//
//	d/dsrc[i, :] = grad[idx[i], :]
//
// — the same operation as Gather. Gather and ScatterAdd are duals.
//
// Plan 0001 Phase 1 follow-up; needed by MoE training.
func ScatterAdd(src *Tensor, idx []int, N int) *Tensor {
	if src.Dim() != 2 {
		panic("gorch: ScatterAdd requires 2-D source")
	}
	M, D := src.shape[0], src.shape[1]
	if len(idx) != M {
		panic(fmt.Sprintf("gorch: ScatterAdd idx length %d != src rows %d", len(idx), M))
	}

	out := Zeros(N, D)
	for i, j := range idx {
		if j < 0 || j >= N {
			panic(fmt.Sprintf("gorch: ScatterAdd index %d out of range [0, %d)", j, N))
		}
		for d := 0; d < D; d++ {
			out.data[j*D+d] += src.data[i*D+d]
		}
	}

	if GradEnabled() && src.requiresGrad {
		out.requiresGrad = true
		idxCopy := append([]int{}, idx...)
		out.gradFn = &GradFn{
			name:   "ScatterAdd",
			inputs: []*Tensor{src},
			backward: func(grad *Tensor) []*Tensor {
				// d/dsrc[i, :] = grad[idx[i], :] — that's just Gather(grad, idx).
				dSrc := Zeros(M, D)
				for i, j := range idxCopy {
					copy(dSrc.data[i*D:(i+1)*D], grad.data[j*D:(j+1)*D])
				}
				return []*Tensor{dSrc}
			},
		}
	}
	return out
}

// TopK returns the values and indices of the K largest elements
// along the last dimension of a 2-D tensor.
//
//	x: (B, V)  →  values: (B, K), indices: (B, K)
//
// Order is descending by value. Used by MoE's top-k router to pick
// the K most-likely experts per token, and as a building block for
// nucleus / top-K sampling.
//
// No autograd — TopK is a discrete selection used in inference and
// in MoE forward where the indices feed gather; its "gradient" with
// respect to inputs is undefined (subgradient is zero almost
// everywhere). Use the values directly if a differentiable path is
// needed; the indices are integer.
//
// Plan 0001 Phase 1 item 5.
func TopK(x *Tensor, k int) (values *Tensor, indices []int) {
	if x.Dim() != 2 {
		panic("gorch: TopK requires 2-D input")
	}
	B, V := x.shape[0], x.shape[1]
	if k <= 0 || k > V {
		panic(fmt.Sprintf("gorch: TopK k=%d out of range [1, %d]", k, V))
	}

	values = Zeros(B, k)
	indices = make([]int, B*k)

	// Per-row partial selection: O(V log K) using a min-heap would be
	// faster, but at vocab/expert sizes (≤ a few thousand) and B
	// usually small, a sort + take-K is simpler and fast enough.
	type idxVal struct {
		val float32
		idx int
	}
	for b := 0; b < B; b++ {
		row := x.data[b*V : (b+1)*V]
		pairs := make([]idxVal, V)
		for i, v := range row {
			pairs[i] = idxVal{val: v, idx: i}
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].val > pairs[j].val })
		for i := 0; i < k; i++ {
			values.data[b*k+i] = pairs[i].val
			indices[b*k+i] = pairs[i].idx
		}
	}
	return values, indices
}

// Multinomial samples one index from each row of a probability
// distribution. probs: (B, V) where each row sums to 1; returns
// indices of length B.
//
// Used for sampling-based generation — the existing model.sample()
// inlines this for vocab; the generic public op makes it reusable
// for MoE / future routing decisions.
//
// rng can be nil for the default global rand source.
//
// Plan 0001 Phase 1 item 6.
func Multinomial(probs *Tensor, rng *rand.Rand) []int {
	if probs.Dim() != 2 {
		panic("gorch: Multinomial requires 2-D input")
	}
	B, V := probs.shape[0], probs.shape[1]
	out := make([]int, B)
	rnd := func() float64 {
		if rng != nil {
			return rng.Float64()
		}
		return rand.Float64()
	}
	for b := 0; b < B; b++ {
		r := rnd()
		var cum float64
		row := probs.data[b*V : (b+1)*V]
		picked := V - 1
		for j, p := range row {
			cum += float64(p)
			if r < cum {
				picked = j
				break
			}
		}
		out[b] = picked
	}
	return out
}

// RepeatInterleave repeats each element of the second-to-last
// dimension `n` times along that dimension. Used by GQA to expand
// shared K/V heads across query groups:
//
//	src: (..., kvHeads, headDim)
//	out: (..., kvHeads*n, headDim)  where each kv head appears n
//	                                consecutive times.
//
// For example, with 4 query heads and 2 KV heads, n=2 maps:
//	kv[0] → q[0], q[1]
//	kv[1] → q[2], q[3]
//
// Implemented for any tensor rank; the dim being repeated is at
// position dim-2 (the second-to-last). For a 2-D (kvHeads, headDim)
// input that's just dim 0; for a 4-D (B, S, kvHeads, headDim) it's
// dim 2. This matches PyTorch's torch.repeat_interleave with axis=-2.
//
// Backward: sum every n consecutive rows of grad along dim-2 to
// recover the original kvHeads dim.
//
// Plan 0001 Phase 1 item 7.
func RepeatInterleave(src *Tensor, n int) *Tensor {
	nd := src.Dim()
	if nd < 2 {
		panic("gorch: RepeatInterleave requires rank ≥ 2")
	}
	if n <= 0 {
		panic("gorch: RepeatInterleave n must be ≥ 1")
	}
	dim := nd - 2
	srcShape := src.shape
	dstShape := append([]int{}, srcShape...)
	dstShape[dim] = srcShape[dim] * n

	// Flatten everything into (outer, kvHeads, inner) for a clean loop.
	outer := 1
	for i := 0; i < dim; i++ {
		outer *= srcShape[i]
	}
	K := srcShape[dim]
	inner := 1
	for i := dim + 1; i < nd; i++ {
		inner *= srcShape[i]
	}

	out := Zeros(dstShape...)
	for o := 0; o < outer; o++ {
		for k := 0; k < K; k++ {
			srcOff := (o*K + k) * inner
			for r := 0; r < n; r++ {
				dstOff := (o*K*n + k*n + r) * inner
				copy(out.data[dstOff:dstOff+inner], src.data[srcOff:srcOff+inner])
			}
		}
	}

	if GradEnabled() && src.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "RepeatInterleave",
			inputs: []*Tensor{src},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(srcShape...)
				for o := 0; o < outer; o++ {
					for k := 0; k < K; k++ {
						srcOff := (o*K + k) * inner
						for r := 0; r < n; r++ {
							dstOff := (o*K*n + k*n + r) * inner
							for i := 0; i < inner; i++ {
								dx.data[srcOff+i] += grad.data[dstOff+i]
							}
						}
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

