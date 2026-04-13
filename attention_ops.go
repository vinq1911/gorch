//go:build darwin

package gorch

import (
	"math"

	"github.com/vinq1911/gorch/accelerate"
)

// MaskFill returns a copy of a with positions where mask[i]==true set to val.
// Used to apply causal masks before softmax (fill with -inf).
func MaskFill(a *Tensor, mask []bool, val float32) *Tensor {
	if len(mask) != a.Size() {
		panic("gorch: MaskFill mask length mismatch")
	}
	out := Zeros(a.shape...)
	copy(out.data, a.data)
	for i, m := range mask {
		if m {
			out.data[i] = val
		}
	}
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "MaskFill",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(a.shape...)
				for i, m := range mask {
					if !m {
						dx.data[i] = grad.data[i]
					}
					// masked positions get zero gradient
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// CausalMask generates a causal (upper-triangular) mask for a seq x seq attention matrix.
// Returns a flat bool slice where mask[i*seq+j] = true if j > i (future positions).
func CausalMask(seq int) []bool {
	mask := make([]bool, seq*seq)
	for i := 0; i < seq; i++ {
		for j := i + 1; j < seq; j++ {
			mask[i*seq+j] = true
		}
	}
	return mask
}

// EmbeddingLookup indexes into a weight matrix by integer IDs.
// weight: (vocab, dim), ids: flat list of token IDs.
// Returns: (len(ids), dim).
func EmbeddingLookup(weight *Tensor, ids []int) *Tensor {
	if weight.Dim() != 2 {
		panic("gorch: EmbeddingLookup requires 2-D weight (vocab, dim)")
	}
	dim := weight.shape[1]
	n := len(ids)
	out := Zeros(n, dim)

	for i, id := range ids {
		copy(out.data[i*dim:(i+1)*dim], weight.data[id*dim:(id+1)*dim])
	}

	if weight.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "EmbeddingLookup",
			inputs: []*Tensor{weight},
			backward: func(grad *Tensor) []*Tensor {
				dw := Zeros(weight.shape...)
				for i, id := range ids {
					for j := 0; j < dim; j++ {
						dw.data[id*dim+j] += grad.data[i*dim+j]
					}
				}
				return []*Tensor{dw}
			},
		}
	}
	return out
}

// ScaledMatMul computes (a @ b) / sqrt(scale).
// a: (M, K), b: (K, N). Returns: (M, N).
func ScaledMatMul(a, b *Tensor, scale float32) *Tensor {
	out := MatMul(a, b)
	invScale := 1.0 / float32(math.Sqrt(float64(scale)))
	result := Zeros(out.shape...)
	accelerate.VScale(out.data, invScale, result.data)

	if out.requiresGrad {
		result.requiresGrad = true
		result.gradFn = &GradFn{
			name:   "ScaledMatMul",
			inputs: out.gradFn.inputs,
			backward: func(grad *Tensor) []*Tensor {
				// Scale the gradient, then delegate to MatMul backward
				scaledGrad := Zeros(grad.shape...)
				accelerate.VScale(grad.data, invScale, scaledGrad.data)
				return out.gradFn.backward(scaledGrad)
			},
		}
	}
	return result
}
