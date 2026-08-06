//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
)

// gpuLinearDx computes dL/dx = grad @ W on Metal.
// grad: (batch, out), W: (out, in) → dx: (batch, in).
// Returns a Metal-backed tensor so the chain stays on GPU.
func gpuLinearDx(grad, W *g.Tensor, batch, in, out int, needsGrad bool) *g.Tensor {
	if !needsGrad {
		// Match CPU semantics: when x doesn't need grad, return zeros
		// of the right shape. Allocating a small zero tensor is cheap
		// and avoids special-casing in the autograd accumulator.
		return g.Zeros(batch, in)
	}
	// Plain MatMul handles GPU dispatch internally when both inputs
	// are Metal-backed.
	return g.MatMul(grad, W)
}

// gpuLinearDw computes dL/dW = grad^T @ x on Metal.
// grad: (batch, out), x: (batch, in) → dW: (out, in).
func gpuLinearDw(grad, x *g.Tensor, batch, in, out int) *g.Tensor {
	return g.MatMulTransA(grad, x)
}

// linearDb sums grad along the batch dimension to give dL/db.
// Always runs on CPU — the bias has at most a few thousand elements,
// dispatching to GPU for that is pure overhead.
func linearDb(grad *g.Tensor, batch, out int) *g.Tensor {
	gData := grad.Data()
	dbData := make([]float32, out)
	for i := 0; i < batch; i++ {
		row := gData[i*out : (i+1)*out]
		for j, v := range row {
			dbData[j] += v
		}
	}
	return g.NewTensor(dbData, 1, out)
}
