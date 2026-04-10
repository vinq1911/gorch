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
var noGradDepth int

// NoGrad executes fn with gradient tracking disabled.
// Any tensors created inside fn will not track gradients.
func NoGrad(fn func()) {
	noGradDepth++
	defer func() { noGradDepth-- }()
	fn()
}

// GradEnabled returns true if gradient tracking is currently active.
func GradEnabled() bool {
	return noGradDepth == 0
}
