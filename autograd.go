//go:build darwin

package gorch

import (
	"github.com/vinq1911/gorch/accelerate"
	"github.com/vinq1911/gorch/metal"
)

// Backward computes gradients for all tensors in the computation graph
// that require gradients, starting from this tensor (typically a scalar loss).
//
// This implements reverse-mode automatic differentiation by walking the
// computation graph backward through GradFn pointers.
//
// Mixed dtypes: when the loss is bf16, the seed grad is bf16 and grad
// accumulation upcasts to f32 for the addition, then rounds back to
// bf16 storage — same shape as PyTorch's bf16 mixed-precision: rounding
// noise on storage but not on the running sum.
func (t *Tensor) Backward() {
	if t.Size() != 1 {
		panic("gorch: Backward() only supported on scalar tensors (call Sum or Mean first)")
	}

	// THE WHOLE BACKWARD PASS RUNS UNTRACKED (2026-08-12 post-mortem).
	//
	// Backward closures compute with ordinary graph-building ops —
	// gpuLinearDx does g.MatMul(grad, W), the LoRA path uses Scale and
	// Transpose2D, and so on. With tracking left on, each of those
	// attaches a GradFn to the gradient it produces, and that GradFn
	// references the forward tensors it consumed (x, W, activations).
	// The gradient is then stored in inp.grad — and for a PARAMETER
	// that field lives until the optimizer's ZeroGrad, which runs once
	// per optimizer step, after all accumulation micro-steps.
	//
	// So every parameter's .grad became a root pinning that micro-
	// step's entire forward+backward graph. With accumulation, N
	// micro-step graphs were pinned at once: measured ~5 GB per
	// micro-step, 19.9 GB after four, unbounded thereafter. GC could
	// not help — the graphs were genuinely reachable. This reproduced
	// with the GPU path disabled entirely, which is what ruled out the
	// Metal command-buffer theory.
	//
	// PyTorch has the same rule: backward does not record unless you
	// ask for create_graph. Nothing here reads grad.gradFn, so first-
	// order semantics are the whole contract.
	NoGrad(func() {
		// Seed gradient: dL/dL = 1, in the loss's dtype.
		t.grad = onesLike(t)
		backpropFrom(t)
	})
}

// backpropFrom walks the graph reachable from root in reverse
// topological order, propagating root.grad into every input that
// requires gradients. The caller must seed root.grad and must call
// this inside a NoGrad scope — see the post-mortem on Backward for
// why the backward pass must not record.
//
// Factored out of Backward so gradient checkpointing (checkpoint.go)
// can reuse the exact same engine on a locally recomputed subgraph:
// two implementations of "walk the graph backward" would be two places
// for the retention and detach invariants to drift apart.
func backpropFrom(root *Tensor) {
	// Topological sort: collect all nodes in reverse order.
	visited := make(map[*Tensor]bool)
	var order []*Tensor
	var topo func(n *Tensor)
	topo = func(n *Tensor) {
		if visited[n] {
			return
		}
		visited[n] = true
		if n.gradFn != nil {
			for _, inp := range n.gradFn.inputs {
				topo(inp)
			}
		}
		order = append(order, n)
	}
	topo(root)

	// Walk in reverse topological order, propagating gradients.
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if n.gradFn == nil || n.grad == nil {
			continue
		}

		inputGrads := n.gradFn.backward(n.grad)
		for j, inp := range n.gradFn.inputs {
			if !inp.requiresGrad {
				continue
			}
			gr := inputGrads[j]
			if gr == nil {
				continue
			}
			// Belt and braces: even if some future backward closure
			// re-enables tracking internally, a stored .grad must
			// never own a graph. Detach shares storage, so this is
			// free. Invariant: p.grad == nil || p.grad.gradFn == nil.
			if gr.gradFn != nil {
				gr = gr.Detach()
			}
			if inp.grad == nil {
				inp.grad = gr
			} else {
				accumulateGrad(inp.grad, gr)
			}
		}
	}
}

// onesLike returns a 1-element tensor with value 1 in t's dtype.
func onesLike(t *Tensor) *Tensor {
	if t.dtype == BFloat16 {
		return NewTensorBF16([]uint16{f32ToBF16(1)}, 1)
	}
	return Ones(1)
}

// accumulateGrad does dst += add in-place, dispatching on dtype. For
// bf16 the addition runs in fp32 (upcast → add → downcast) which keeps
// the per-step rounding noise the same as a single bf16 store.
func accumulateGrad(dst, add *Tensor) {
	if dst.dtype == BFloat16 {
		for k := range dst.data16 {
			sum := bf16ToF32(dst.data16[k]) + bf16ToF32(add.data16[k])
			dst.data16[k] = f32ToBF16(sum)
		}
		return
	}
	// Both grads Metal-resident → in-place vec_add on GPU (plan 0009
	// X2b): grad accumulation was one of the residual host sync points
	// in async dispatch mode; each lane reads and writes only its own
	// element, so dst-aliasing is safe.
	if dst.buf != nil && add.buf != nil && gpu != nil {
		if p, ok := gpu.pipelines["vec_add"]; ok {
			gpu.Queue.Dispatch1D(p, []*metal.Buffer{dst.buf, add.buf, dst.buf}, dst.Size())
			return
		}
	}
	// CPU read-modify-write of possibly-GPU-written grads — wait for
	// pending async GPU work first (plan 0009 R6).
	syncForCPU(dst, add)
	accelerate.VAdd(dst.data, add.data, dst.data)
}

// noGradDepth tracks nested NoGrad scopes.
//
// IMPORTANT: this counter is process-global, not goroutine-local. A
// NoGrad scope opened in goroutine A turns off gradient tracking for
// every other goroutine until the scope closes — which is wrong if
// another goroutine is in the middle of a training step. There is no
// race on the counter itself (single-threaded reads dominate, and the
// observable failure mode is "wrong answer" not "data race"), but the
// semantic limitation is real.
//
// If you need to disable gradient tracking on specific tensors without
// affecting other goroutines, use Tensor.Detach() — it returns a new
// tensor handle sharing the same data but with requires_grad=false and
// no gradFn. That works at any scope and doesn't touch global state.
//
// PyTorch's torch.no_grad() has the same global-thread-local-ish
// limitation, and tensor.detach() is the same goroutine/thread-local
// escape hatch. Mirroring that pairing intentionally.
var noGradDepth int

// NoGrad executes fn with gradient tracking disabled.
//
// Process-global state — see the IMPORTANT note on noGradDepth above.
// For goroutine-local "don't track this" semantics, use Tensor.Detach()
// instead.
func NoGrad(fn func()) {
	noGradDepth++
	defer func() { noGradDepth-- }()
	fn()
}

// GradEnabled returns true if gradient tracking is currently active.
//
// Reads the process-global counter. See Tensor.Detach for a goroutine-
// local opt-out that doesn't touch this state.
func GradEnabled() bool {
	return noGradDepth == 0
}

// enableGrad executes fn with gradient tracking forced ON, restoring
// the enclosing NoGrad depth afterwards (including on panic).
//
// This is the inverse of NoGrad and exists for exactly one caller:
// Checkpoint's recompute, which must rebuild a local graph while the
// whole backward pass is running inside Backward's NoGrad scope. It is
// deliberately unexported — an exported "ignore every NoGrad scope my
// caller opened" primitive is a footgun, and it inherits every
// process-global caveat documented on noGradDepth (a concurrent
// goroutine's inference will build a graph for the duration).
func enableGrad(fn func()) {
	saved := noGradDepth
	noGradDepth = 0
	defer func() { noGradDepth = saved }()
	fn()
}
