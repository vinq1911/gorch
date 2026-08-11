//go:build darwin

package gorch

import (
	"fmt"
	"math"

	"github.com/vinq1911/gorch/metal"
)

// Fused causal softmax — plan 0009 phase K1 (X2 kernel #1).
//
// The GQA/MLA attention chain used to build softmax(mask(scale·QKᵀ))
// out of four ops (Full + Mul + MaskFill + Softmax), materializing
// 4–5 seq² intermediates per layer (at seq 1500: ~470 MB/layer) and
// spending 42% of the measured X0 step wall clock in the softmax +
// mask/scale class. CausalSoftmax fuses the whole chain into one op
// with one output tensor:
//
//	y[r, j] = exp(scale·x[r, j] − m_r) / Σ_{k allowed} exp(scale·x[r, k] − m_r)   if j allowed
//	        = 0                                                                    otherwise
//
// where row r of the (heads·qSeq, kSeq) view corresponds to query
// position i = r mod qSeq, and column j is allowed iff j ≤ i + (kSeq −
// qSeq) (pure causal for qSeq == kSeq, staircase for prefill windows).
// m_r is the row max over allowed columns (numerical stability, same
// as the Softmax CPU reference in ops.go).
//
// Backward (chain rule through the scale):
//
//	dx = scale · y ⊙ (g − Σ_j (g ⊙ y))
//
// which is the ops.go Softmax backward identity times the fused scale
// factor. Masked lanes have y = 0 and therefore dx = 0 — masked score
// positions receive no gradient, matching MaskFill's backward.
//
// Dispatch: the Metal kernels (softmax_causal_forward / softmax_
// backward in metal/kernels.go, rmsnorm template: one threadgroup of
// 256 per row, strided loops, tree reduction, f32 accumulation) run
// when x is Metal-resident and InitMetal compiled the pipelines; the
// CPU path below is the test oracle and the fallback, and still fuses
// the chain (no mask/scale tensors are ever materialized).

const softmaxThreadgroupSize = 256

// softmaxPipelinesReady reports whether the fused-softmax pipelines
// were compiled by InitMetal.
func softmaxPipelinesReady() bool {
	if gpu == nil {
		return false
	}
	if _, ok := gpu.pipelines["softmax_causal_forward"]; !ok {
		return false
	}
	if _, ok := gpu.pipelines["softmax_backward"]; !ok {
		return false
	}
	return true
}

// CausalSoftmax computes the fused scale + causal-mask + softmax over
// the last dimension of x, which must be (heads, qSeq, kSeq) or
// (heads·qSeq, kSeq) with kSeq ≥ qSeq. scale is the attention scale
// (1/√headDim); the causal rule allows column j for query row i iff
// j ≤ i + (kSeq − qSeq). Output has x's shape, is autograd-aware, and
// inherits x's Metal residency.
func CausalSoftmax(x *Tensor, heads, qSeq int, scale float32) *Tensor {
	if x.dtype == BFloat16 {
		return downcastToBF16(CausalSoftmax(promoteToF32(x), heads, qSeq, scale))
	}
	nd := x.Dim()
	if nd != 2 && nd != 3 {
		panic("gorch: CausalSoftmax requires a 2-D or 3-D tensor")
	}
	cols := x.shape[nd-1]
	rows := x.Size() / cols
	if rows != heads*qSeq {
		panic(fmt.Sprintf("gorch: CausalSoftmax rows %d != heads %d * qSeq %d", rows, heads, qSeq))
	}
	if cols < qSeq {
		panic(fmt.Sprintf("gorch: CausalSoftmax kSeq %d < qSeq %d", cols, qSeq))
	}

	var out *Tensor
	if x.buf != nil && softmaxPipelinesReady() {
		out = softmaxCausalForwardMetal(x, rows, cols, qSeq, scale)
	} else {
		out = causalSoftmaxForwardCPU(x, rows, cols, qSeq, scale)
	}

	if GradEnabled() && x.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "CausalSoftmax",
			inputs: []*Tensor{x},
			backward: func(grad *Tensor) []*Tensor {
				if out.buf != nil && grad.buf != nil && softmaxPipelinesReady() {
					return []*Tensor{softmaxBackwardMetal(out, grad, rows, cols, scale)}
				}
				return []*Tensor{softmaxBackwardCPU(out, grad, rows, cols, scale)}
			},
		}
	}
	return out
}

// causalSoftmaxForwardCPU is the fused CPU reference — mathematically
// identical to Softmax(MaskFill(Scale(x))) from ops.go /
// attention_ops.go, without materializing any intermediate.
func causalSoftmaxForwardCPU(x *Tensor, rows, cols, qSeq int, scale float32) *Tensor {
	out := ZerosLike(x, x.shape...)
	offset := cols - qSeq
	for r := 0; r < rows; r++ {
		i := r % qSeq
		limit := i + offset // inclusive last allowed column
		row := x.data[r*cols : (r+1)*cols]
		orow := out.data[r*cols : (r+1)*cols]

		rowMax := float32(math.Inf(-1))
		for j := 0; j <= limit; j++ {
			v := row[j] * scale
			if v > rowMax {
				rowMax = v
			}
		}
		var sum float32
		for j := 0; j <= limit; j++ {
			e := float32(math.Exp(float64(row[j]*scale - rowMax)))
			orow[j] = e
			sum += e
		}
		inv := 1.0 / sum
		for j := 0; j <= limit; j++ {
			orow[j] *= inv
		}
		// Columns > limit stay exactly 0 (fresh zero allocation).
	}
	return out
}

// softmaxBackwardCPU: dx = scale · y ⊙ (g − Σ(g⊙y)) per row — the
// ops.go Softmax backward closure math times the fused scale.
func softmaxBackwardCPU(y, grad *Tensor, rows, cols int, scale float32) *Tensor {
	dx := zerosLikeEither(y.shape, grad, y)
	for r := 0; r < rows; r++ {
		yrow := y.data[r*cols : (r+1)*cols]
		grow := grad.data[r*cols : (r+1)*cols]
		drow := dx.data[r*cols : (r+1)*cols]
		var dot float32
		for j := 0; j < cols; j++ {
			dot += grow[j] * yrow[j]
		}
		for j := 0; j < cols; j++ {
			drow[j] = scale * yrow[j] * (grow[j] - dot)
		}
	}
	return dx
}

// softmaxCausalForwardMetal dispatches the softmax_causal_forward
// kernel: one threadgroup of 256 lanes per row. x must be Metal-
// resident. Returns a Metal-backed y of x's shape.
func softmaxCausalForwardMetal(x *Tensor, rows, cols, qSeq int, scale float32) *Tensor {
	dev := gpu.Dev
	out := ZerosOnMetal(dev, x.shape...)

	dimsBuf := dev.NewBuffer(3 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(rows)
	dims[1] = uint32(cols)
	dims[2] = uint32(qSeq)

	scaleBuf := dev.NewBuffer(4)
	scaleBuf.FloatSlice()[0] = scale

	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("softmax_causal_forward"),
		[]*metal.Buffer{x.buf, dimsBuf, scaleBuf, out.buf},
		rows,
		softmaxThreadgroupSize,
	)
	metalSoftmaxDispatches.Add(1)
	dimsBuf.Release()
	scaleBuf.Release()
	return out
}

// softmaxBackwardMetal dispatches the softmax_backward kernel:
// dx = scale · y ⊙ (g − Σ(g⊙y)). y and grad must be Metal-resident
// (callers copy a CPU grad up first if needed). Returns Metal-backed
// dx of y's shape.
func softmaxBackwardMetal(y, grad *Tensor, rows, cols int, scale float32) *Tensor {
	dev := gpu.Dev
	dx := ZerosOnMetal(dev, y.shape...)

	dimsBuf := dev.NewBuffer(2 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(rows)
	dims[1] = uint32(cols)

	scaleBuf := dev.NewBuffer(4)
	scaleBuf.FloatSlice()[0] = scale

	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("softmax_backward"),
		[]*metal.Buffer{y.buf, grad.buf, dimsBuf, scaleBuf, dx.buf},
		rows,
		softmaxThreadgroupSize,
	)
	metalSoftmaxDispatches.Add(1)
	dimsBuf.Release()
	scaleBuf.Release()
	return dx
}
