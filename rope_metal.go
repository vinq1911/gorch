//go:build darwin

package gorch

import "github.com/vinq1911/gorch/metal"

// RoPE Metal dispatch — plan 0009 K6 (X2b wave).
//
// The kernel (rope_apply in metal/kernels.go) performs the elementwise
// pair rotation with cos/sin tables precomputed on host and uploaded
// once per RoPE module (nn/rope.go owns the tables and the autograd
// wiring; this file is only the dispatch glue, mirroring the
// rmsnorm_metal.go split). Forward and backward are the same kernel:
// the inverse rotation is the forward rotation with sin negated, so
// sign = +1 dispatches forward and sign = −1 dispatches backward on
// the upstream grad.

// RoPEMetalReady reports whether the rope_apply pipeline was compiled
// by InitMetal.
func RoPEMetalReady() bool {
	if gpu == nil {
		return false
	}
	_, ok := gpu.pipelines["rope_apply"]
	return ok
}

// RoPEApplyMetal rotates src (shape (..., seqLen, headDim), Metal-
// resident) using Metal-resident cos/sin tables of shape (maxSeq *
// half). outer is the product of the dims before seqLen; half =
// headDim/2; neox selects the interleaved pair convention; sign is +1
// for forward, −1 for the inverse (backward) rotation. Returns a
// Metal-backed tensor of src's shape. No autograd — callers own the
// graph wiring.
func RoPEApplyMetal(src, cosT, sinT *Tensor, outer, seqLen, half, startPos int, neox bool, sign float32) *Tensor {
	if src.buf == nil || cosT.buf == nil || sinT.buf == nil {
		panic("gorch: RoPEApplyMetal requires Metal-resident src and tables")
	}
	dev := gpu.Dev
	out := ZerosOnMetal(dev, src.shape...)

	dimsBuf := dev.NewBuffer(5 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(outer)
	dims[1] = uint32(seqLen)
	dims[2] = uint32(half)
	dims[3] = uint32(startPos)
	if neox {
		dims[4] = 1
	}

	signBuf := dev.NewBuffer(4)
	signBuf.FloatSlice()[0] = sign

	gpu.Queue.Dispatch1D(gpu.pipe("rope_apply"),
		[]*metal.Buffer{src.buf, cosT.buf, sinT.buf, dimsBuf, signBuf, out.buf},
		outer*seqLen*half)
	metalRopeDispatches.Add(1)
	dimsBuf.Release()
	signBuf.Release()
	return out
}
