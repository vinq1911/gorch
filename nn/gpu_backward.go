//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
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

// linearDb sums grad along the batch dimension to give dL/db. When
// grad is Metal-resident the col_sum kernel runs on GPU (plan 0009
// X2b: the CPU db loop forced a full GPU sync per Linear backward —
// the cost was never the few thousand output floats, it was the
// waitUntilCompleted round trip its Data() read implied). CPU fallback
// is a vectorized vDSP row accumulation.
func linearDb(grad *g.Tensor, batch, out int) *g.Tensor {
	if db := g.ColSumMetal(grad, batch, out); db != nil {
		return db
	}
	gData := grad.Data()
	dbData := make([]float32, out)
	for i := 0; i < batch; i++ {
		accelerate.VAdd(dbData, gData[i*out:(i+1)*out], dbData)
	}
	return g.NewTensor(dbData, 1, out)
}
