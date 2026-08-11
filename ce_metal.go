//go:build darwin

package gorch

import (
	"github.com/vinq1911/gorch/metal"
)

// Fused cross-entropy Metal dispatch — plan 0009 K2 (X2 kernel #2).
//
// Kernels: cross_entropy_forward / cross_entropy_backward in
// metal/kernels.go, following the rmsnorm template (one threadgroup of
// 256 per row, strided loops over N — N up to ~168k vocab — tree
// reduction in threadgroup memory, f32 accumulation). Golden tests:
// ce_metal_test.go (CPU reference: ceForwardCPU/ceBackwardCPU in
// loss.go, which match the pre-K2 LogSoftmax-based implementation).
//
// Forward writes per-row loss (logsumexp − x[target]) and per-row
// logsumexp; the host sums the ≤batch losses (batch ≤ ~1500 floats —
// negligible) and keeps the logsumexp tensor for backward. Backward
// writes dx = (exp(x − lse) − onehot)·scale in one pass without ever
// materializing a softmax tensor.

const ceThreadgroupSize = 256

// cePipelinesReady reports whether the K2 pipelines were compiled.
func cePipelinesReady() bool {
	if gpu == nil {
		return false
	}
	if _, ok := gpu.pipelines["cross_entropy_forward"]; !ok {
		return false
	}
	if _, ok := gpu.pipelines["cross_entropy_backward"]; !ok {
		return false
	}
	return true
}

// ceTargetsBuffer uploads target class indices as a uint32 buffer.
func ceTargetsBuffer(tgt []int) *metal.Buffer {
	buf := gpu.Dev.NewBuffer(len(tgt) * 4)
	s := buf.Uint32Slice()
	for i, t := range tgt {
		s[i] = uint32(t)
	}
	return buf
}

// ceForwardMetal dispatches cross_entropy_forward over the (batch,
// classes) logits and returns the summed loss plus the Metal-resident
// per-row logsumexp tensor (kept for backward). logits must be Metal-
// resident.
func ceForwardMetal(logits *Tensor, tgt []int, batch, classes int) (total float32, lse *Tensor) {
	dev := gpu.Dev
	lse = ZerosOnMetal(dev, batch)
	rowLoss := dev.NewBuffer(batch * 4)

	tgtBuf := ceTargetsBuffer(tgt)
	dimsBuf := dev.NewBuffer(2 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(batch)
	dims[1] = uint32(classes)

	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("cross_entropy_forward"),
		[]*metal.Buffer{logits.buf, tgtBuf, dimsBuf, rowLoss, lse.buf},
		batch,
		ceThreadgroupSize,
	)
	metalCEDispatches.Add(1)

	// The loss scalar is read by the host every step: wait for the
	// dispatch (async mode) and sum the ≤batch per-row losses.
	metal.SyncQueue()
	losses := rowLoss.FloatSlice()
	for i := 0; i < batch; i++ {
		total += losses[i]
	}
	rowLoss.Release()
	tgtBuf.Release()
	dimsBuf.Release()
	return total, lse
}

// ceBackwardMetal dispatches cross_entropy_backward: dx[i,j] =
// (exp(x[i,j] − lse[i]) − onehot[i,j]) · scale, Metal-resident.
func ceBackwardMetal(logits, lse *Tensor, tgt []int, batch, classes int, scale float32) *Tensor {
	dev := gpu.Dev
	dx := ZerosOnMetal(dev, batch, classes)

	tgtBuf := ceTargetsBuffer(tgt)
	dimsBuf := dev.NewBuffer(2 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(batch)
	dims[1] = uint32(classes)
	scaleBuf := dev.NewBuffer(4)
	scaleBuf.FloatSlice()[0] = scale

	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("cross_entropy_backward"),
		[]*metal.Buffer{logits.buf, tgtBuf, lse.buf, dimsBuf, scaleBuf, dx.buf},
		batch,
		ceThreadgroupSize,
	)
	metalCEDispatches.Add(1)
	tgtBuf.Release()
	dimsBuf.Release()
	scaleBuf.Release()
	return dx
}
