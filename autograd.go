//go:build darwin

package gorch

// Backward computes gradients for all tensors in the computation graph
// that require gradients, starting from this tensor (typically a scalar loss).
//
// This implements reverse-mode automatic differentiation by walking the
// computation graph backward through GradFn pointers.
func (t *Tensor) Backward() {
	if t.Size() != 1 {
		panic("gorch: Backward() only supported on scalar tensors (call Sum or Mean first)")
	}

	// Seed gradient: dL/dL = 1
	t.grad = Ones(1)

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
	topo(t)

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
			if inp.grad == nil {
				inp.grad = inputGrads[j]
			} else {
				// Accumulate gradients (for tensors used multiple times).
				for k := range inp.grad.data {
					inp.grad.data[k] += inputGrads[j].data[k]
				}
			}
		}
	}
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
