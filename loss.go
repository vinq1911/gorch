//go:build darwin

package gorch

import (
	"math"

	"github.com/vinq1911/gorch/accelerate"
)

// MSELoss computes mean squared error: mean((pred - target)^2).
func MSELoss(pred, target *Tensor) *Tensor {
	diff := Sub(pred, target)
	sq := Mul(diff, diff)
	return Mean(sq)
}

// CEMetalMinElements gates the K2 fused-CE Metal kernels: below this
// many logit elements the per-dispatch overhead (~0.2–1 ms round trip)
// beats the GPU win and the vectorized CPU path runs instead. The
// workload shape (1500, 168320) is ~252M elements — far above. A
// package-level var (like MatMulMetalThreshold) so tests and benches
// can lower it.
var CEMetalMinElements = 1 << 21 // ~2M

// CrossEntropyLoss computes cross-entropy loss for classification.
// logits: (batch, classes) raw scores (pre-softmax).
// targets: (batch, 1) integer class labels stored as float32.
// Returns a scalar loss = -mean(logsoftmax(logits)[i, target[i]]).
//
// Following the standard mixed-precision pattern, the loss is computed
// in fp32 over fp32 logits. If logits arrive bf16, they're upcast and
// the scalar loss returns f32 — losses don't need bf16 storage and
// keeping them f32 avoids extra rounding noise in the optimiser.
//
// Plan 0009 K2 (X2 kernel #2): the old implementation built a full
// LogSoftmax tensor forward and *recomputed* Softmax in backward —
// 4.1 s/step at (1500, 168k) in the X0 per-op table, the #2 pure-Go
// loop. Now:
//
//   - Metal path (logits Metal-resident, pipelines compiled, ≥
//     ceMetalMinElements): fused cross_entropy_forward kernel computes
//     per-row logsumexp + target pick in one pass (f32 accumulation,
//     rmsnorm threadgroup template); backward is one
//     cross_entropy_backward dispatch writing softmax(x)−onehot scaled,
//     never materializing a second softmax. The per-row logsumexp is
//     kept (batch floats) and shared with backward.
//   - CPU path: vectorized per row via Accelerate (Max, VSAdd, Exp,
//     Sum, Log — the acc_vexp path) with the same saved-logsumexp
//     backward. No LogSoftmax tensor, no double softmax.
//
// The loss scalar itself always lives on CPU (it is read by the host
// every step); the backward dLogits inherits logits' residency so the
// whole backward chain stays on GPU (plan 0009 X1 item 3).
func CrossEntropyLoss(logits, targets *Tensor) *Tensor {
	if logits.Dim() != 2 {
		panic("gorch: CrossEntropyLoss requires 2-D logits (batch, classes)")
	}
	if logits.dtype == BFloat16 {
		return CrossEntropyLoss(promoteToF32(logits), targets)
	}
	batch := logits.shape[0]
	classes := logits.shape[1]

	tgt := make([]int, batch)
	syncForCPU(targets)
	for i := 0; i < batch; i++ {
		tgt[i] = int(targets.data[i])
	}

	useMetal := logits.buf != nil && cePipelinesReady() &&
		batch*classes >= CEMetalMinElements

	var total float32
	var lse *Tensor // Metal path: per-row logsumexp, Metal-resident
	var lseCPU []float32

	if useMetal {
		total, lse = ceForwardMetal(logits, tgt, batch, classes)
	} else {
		total, lseCPU = ceForwardCPU(logits, tgt, batch, classes)
	}
	loss := NewTensor([]float32{total / float32(batch)}, 1)

	if GradEnabled() && (logits.requiresGrad) {
		loss.requiresGrad = true
		loss.gradFn = &GradFn{
			name:   "CrossEntropyLoss",
			inputs: []*Tensor{logits},
			backward: func(grad *Tensor) []*Tensor {
				// dL/dlogits = (softmax(logits) − one_hot(targets)) * grad/batch,
				// with softmax reconstructed from the saved logsumexp.
				syncForCPU(grad) // scalar; typically CPU already
				scale := grad.data[0] / float32(batch)
				if useMetal && lse != nil && cePipelinesReady() {
					return []*Tensor{ceBackwardMetal(logits, lse, tgt, batch, classes, scale)}
				}
				return []*Tensor{ceBackwardCPU(logits, lseCPU, tgt, batch, classes, scale)}
			},
		}
	}
	return loss
}

// ceForwardCPU computes the summed loss and per-row logsumexp with
// Accelerate-vectorized row passes (max → exp(x−max) → sum → log).
// This is the K2 kernel's CPU oracle and the below-threshold fallback.
func ceForwardCPU(logits *Tensor, tgt []int, batch, classes int) (total float32, lse []float32) {
	syncForCPU(logits)
	lse = make([]float32, batch)
	scratch := AcquireFloat32(classes)
	for i := 0; i < batch; i++ {
		row := logits.data[i*classes : (i+1)*classes]
		rowMax := accelerate.Max(row)
		accelerate.VSAdd(row, -rowMax, scratch)
		accelerate.Exp(scratch, scratch)
		sumExp := accelerate.Sum(scratch)
		l := rowMax + float32(math.Log(float64(sumExp)))
		lse[i] = l
		total += l - row[tgt[i]]
	}
	ReleaseFloat32(scratch)
	return total, lse
}

// ceBackwardCPU writes dx[i,j] = (exp(x[i,j]−lse[i]) − onehot) * scale
// using the forward's saved logsumexp — no softmax recomputation.
func ceBackwardCPU(logits *Tensor, lse []float32, tgt []int, batch, classes int, scale float32) *Tensor {
	syncForCPU(logits)
	dx := ZerosLike(logits, batch, classes)
	for i := 0; i < batch; i++ {
		row := logits.data[i*classes : (i+1)*classes]
		drow := dx.data[i*classes : (i+1)*classes]
		accelerate.VSAdd(row, -lse[i], drow)
		accelerate.Exp(drow, drow)
		accelerate.VScale(drow, scale, drow)
		drow[tgt[i]] -= scale
	}
	return dx
}
