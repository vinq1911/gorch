//go:build darwin

package gorch

import (
	"fmt"

	"github.com/vinq1911/gorch/metal"
)

// RMSNorm Metal forward + backward dispatch helpers. Plan 0004 part A,
// first non-MatMul GPU autograd kernel.
//
// The kernels themselves live in metal/kernels.go ("rmsnorm_forward"
// and "rmsnorm_dx"); this file is the Go-side glue that allocates
// uniform buffers, binds tensor buffers, and submits the dispatch.
//
// Higher-level callers (nn/rmsnorm.go) treat this as the GPU branch
// of their forward; the autograd graph is wired up there, not here.

const rmsnormThreadgroupSize = 256

// rmsnormPipelinesReady reports whether the precompiled pipelines
// needed for the GPU path exist on the global GPU singleton. Returns
// false before InitMetal — callers must fall back to the CPU path.
func rmsnormPipelinesReady() bool {
	if gpu == nil {
		return false
	}
	if _, ok := gpu.pipelines["rmsnorm_forward"]; !ok {
		return false
	}
	if _, ok := gpu.pipelines["rmsnorm_dx"]; !ok {
		return false
	}
	return true
}

// RMSNormForwardMetal runs rmsnorm_forward on x (M, N) with the given
// per-feature weight (N,) and epsilon, returning a Metal-backed y of
// shape (M, N) and a Metal-backed invRMS of shape (M,). Both x and
// weight must already be on the global Metal device. invRMS is the
// per-row 1/√(mean(x²)+eps) used by the matching backward kernel.
//
// The autograd graph is NOT registered here — callers (nn/rmsnorm.go)
// own that. This is intentionally low-level: it returns invRMS as a
// separate tensor so backward can reuse it without recomputing.
//
// Panics if x is not 2-D, weight has the wrong shape, or InitMetal
// hasn't been called.
func RMSNormForwardMetal(x, weight *Tensor, eps float32) (y, invRMS *Tensor) {
	if !rmsnormPipelinesReady() {
		panic("gorch: RMSNormForwardMetal called before InitMetal compiled rmsnorm pipelines")
	}
	if x.Dim() != 2 {
		panic("gorch: RMSNormForwardMetal requires 2-D x (M, N)")
	}
	M, N := x.shape[0], x.shape[1]
	if weight.Dim() != 1 || weight.shape[0] != N {
		panic(fmt.Sprintf("gorch: RMSNormForwardMetal weight shape %v incompatible with x last dim %d", weight.shape, N))
	}
	if x.buf == nil || weight.buf == nil {
		panic("gorch: RMSNormForwardMetal requires x and weight to be on Metal")
	}

	dev := gpu.Dev
	// Output buffers.
	yBuf := dev.NewBuffer(M * N * 4)
	invBuf := dev.NewBuffer(M * 4)

	// Uniforms: dims = [M, N] as uint32, eps as a 1-element float buffer.
	dimsBuf := dev.NewBuffer(2 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(M)
	dims[1] = uint32(N)

	epsBuf := dev.NewBuffer(4)
	epsBuf.FloatSlice()[0] = eps

	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("rmsnorm_forward"),
		[]*metal.Buffer{x.buf, weight.buf, dimsBuf, epsBuf, yBuf, invBuf},
		M,                      // one threadgroup per row
		rmsnormThreadgroupSize, // 256 lanes
	)
	dimsBuf.Release()
	epsBuf.Release()

	y = &Tensor{
		dtype: Float32,
		data:  yBuf.FloatSlice(),
		shape: []int{M, N},
		buf:   yBuf,
	}
	invRMS = &Tensor{
		dtype: Float32,
		data:  invBuf.FloatSlice(),
		shape: []int{M},
		buf:   invBuf,
	}
	return y, invRMS
}

// RMSNormBackwardDXMetal runs rmsnorm_dx on (x, weight, grad, invRMS)
// → dx of shape (M, N), and rmsnorm_dgamma (plan 0009 K5, one
// threadgroup per column) → dW of shape (N,). The pre-K5 host dW loop
// forced a full GPU sync in every RMSNorm backward; both grads are now
// Metal-backed and the chain stays resident.
//
// Returns Metal-backed dx and dW tensors.
func RMSNormBackwardDXMetal(x, weight, grad, invRMS *Tensor) (dx, dw *Tensor) {
	if !rmsnormPipelinesReady() {
		panic("gorch: RMSNormBackwardDXMetal called before InitMetal compiled rmsnorm pipelines")
	}
	if x.Dim() != 2 || grad.Dim() != 2 {
		panic("gorch: RMSNormBackwardDXMetal requires 2-D x and grad")
	}
	M, N := x.shape[0], x.shape[1]
	if grad.shape[0] != M || grad.shape[1] != N {
		panic(fmt.Sprintf("gorch: RMSNormBackwardDXMetal shape mismatch: x=%v grad=%v", x.shape, grad.shape))
	}
	if invRMS.Dim() != 1 || invRMS.shape[0] != M {
		panic(fmt.Sprintf("gorch: RMSNormBackwardDXMetal invRMS shape %v incompatible with x rows %d", invRMS.shape, M))
	}
	if x.buf == nil || weight.buf == nil || grad.buf == nil || invRMS.buf == nil {
		panic("gorch: RMSNormBackwardDXMetal requires all inputs to be on Metal")
	}

	dev := gpu.Dev
	dxBuf := dev.NewBuffer(M * N * 4)

	dimsBuf := dev.NewBuffer(2 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(M)
	dims[1] = uint32(N)

	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("rmsnorm_dx"),
		[]*metal.Buffer{x.buf, weight.buf, grad.buf, invRMS.buf, dimsBuf, dxBuf},
		M,
		rmsnormThreadgroupSize,
	)
	dx = &Tensor{
		dtype: Float32,
		data:  dxBuf.FloatSlice(),
		shape: []int{M, N},
		buf:   dxBuf,
	}

	// dW[j] = Σ_i grad[i,j] * x[i,j] * invRMS[i] — the K5 per-column
	// kernel (one threadgroup of 256 per column, rows strided).
	dw = ZerosOnMetal(dev, N)
	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("rmsnorm_dgamma"),
		[]*metal.Buffer{x.buf, grad.buf, invRMS.buf, dimsBuf, dw.buf},
		N,
		rmsnormThreadgroupSize,
	)
	metalColReduceDispatches.Add(1)
	dimsBuf.Release()
	return dx, dw
}
