//go:build darwin

package gorch

import (
	"fmt"
	"math"
	"sync"

	"github.com/vinq1911/gorch/metal"
)

// GPU holds the shared Metal device, command queue, and compiled kernels.
// Initialize once with InitMetal().
type GPU struct {
	Dev   *metal.Device
	Queue *metal.CommandQueue

	mu        sync.Mutex
	pipelines map[string]*metal.Pipeline
}

var gpu *GPU

// InitMetal initializes the global Metal device and compiles kernels.
func InitMetal() (*GPU, error) {
	dev, err := metal.NewDevice()
	if err != nil {
		return nil, err
	}
	queue := dev.NewCommandQueue()
	g := &GPU{
		Dev:       dev,
		Queue:     queue,
		pipelines: make(map[string]*metal.Pipeline),
	}

	// Pre-compile all element-wise kernels.
	for _, name := range []string{"vec_add", "vec_sub", "vec_mul", "vec_div",
		"vec_relu", "vec_sigmoid", "vec_tanh_act", "vec_scale", "vec_sum"} {
		pipe, err := dev.CompileKernel(metal.KernelSource, name)
		if err != nil {
			return nil, fmt.Errorf("gorch: compile %s: %w", name, err)
		}
		g.pipelines[name] = pipe
	}

	gpu = g
	return g, nil
}

func (g *GPU) pipe(name string) *metal.Pipeline {
	return g.pipelines[name]
}

// ---------- Element-wise binary ops ----------

// Add returns a + b element-wise.
func Add(a, b *Tensor) *Tensor {
	assertSameShape(a, b)
	out := binaryOp(a, b, "vec_add", func(x, y float32) float32 { return x + y })
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Add",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{grad, grad}
			},
		}
	}
	return out
}

// Sub returns a - b element-wise.
func Sub(a, b *Tensor) *Tensor {
	assertSameShape(a, b)
	out := binaryOp(a, b, "vec_sub", func(x, y float32) float32 { return x - y })
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Sub",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{grad, Neg(grad)}
			},
		}
	}
	return out
}

// Mul returns a * b element-wise.
func Mul(a, b *Tensor) *Tensor {
	assertSameShape(a, b)
	out := binaryOp(a, b, "vec_mul", func(x, y float32) float32 { return x * y })
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Mul",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				// d(a*b)/da = b*grad, d(a*b)/db = a*grad
				return []*Tensor{
					binaryOp(grad, b, "vec_mul", func(x, y float32) float32 { return x * y }),
					binaryOp(grad, a, "vec_mul", func(x, y float32) float32 { return x * y }),
				}
			},
		}
	}
	return out
}

// Div returns a / b element-wise.
func Div(a, b *Tensor) *Tensor {
	assertSameShape(a, b)
	out := binaryOp(a, b, "vec_div", func(x, y float32) float32 { return x / y })
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Div",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				// d(a/b)/da = grad/b
				// d(a/b)/db = -a*grad/(b*b)
				ga := binaryOp(grad, b, "vec_div", func(x, y float32) float32 { return x / y })
				bb := binaryOp(b, b, "vec_mul", func(x, y float32) float32 { return x * y })
				ab := binaryOp(a, grad, "vec_mul", func(x, y float32) float32 { return x * y })
				gb := Neg(binaryOp(ab, bb, "vec_div", func(x, y float32) float32 { return x / y }))
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}

// ---------- Unary ops ----------

// Neg returns -a element-wise.
func Neg(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	for i, v := range a.data {
		out.data[i] = -v
	}
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Neg",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{Neg(grad)}
			},
		}
	}
	return out
}

// ReLU returns max(0, a) element-wise.
func ReLU(a *Tensor) *Tensor {
	out := unaryOp(a, "vec_relu", func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "ReLU",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				g := Zeros(a.shape...)
				for i, v := range a.data {
					if v > 0 {
						g.data[i] = grad.data[i]
					}
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Sigmoid returns 1/(1+exp(-a)) element-wise.
func Sigmoid(a *Tensor) *Tensor {
	out := unaryOp(a, "vec_sigmoid", func(x float32) float32 {
		return float32(1.0 / (1.0 + math.Exp(float64(-x))))
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Sigmoid",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
				g := Zeros(a.shape...)
				for i, v := range out.data {
					g.data[i] = grad.data[i] * v * (1 - v)
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Tanh returns tanh(a) element-wise.
func Tanh(a *Tensor) *Tensor {
	out := unaryOp(a, "vec_tanh_act", func(x float32) float32 {
		return float32(math.Tanh(float64(x)))
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Tanh",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// tanh'(x) = 1 - tanh(x)^2
				g := Zeros(a.shape...)
				for i, v := range out.data {
					g.data[i] = grad.data[i] * (1 - v*v)
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// ---------- Reduction ops ----------

// Sum returns the sum of all elements as a scalar tensor.
func Sum(a *Tensor) *Tensor {
	var s float32
	for _, v := range a.data {
		s += v
	}
	out := NewTensor([]float32{s}, 1)
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Sum",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// gradient of sum is all ones scaled by upstream grad
				return []*Tensor{Full(grad.data[0], a.shape...)}
			},
		}
	}
	return out
}

// Mean returns the mean of all elements as a scalar tensor.
func Mean(a *Tensor) *Tensor {
	s := Sum(a)
	n := float32(a.Size())
	out := NewTensor([]float32{s.data[0] / n}, 1)
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Mean",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{Full(grad.data[0]/n, a.shape...)}
			},
		}
	}
	return out
}

// ---------- MatMul ----------

// MatMul computes matrix multiplication: a @ b.
// a is (M, K), b is (K, N), result is (M, N).
func MatMul(a, b *Tensor) *Tensor {
	if a.Dim() != 2 || b.Dim() != 2 {
		panic("gorch: MatMul requires 2-D tensors")
	}
	M, K := a.shape[0], a.shape[1]
	K2, N := b.shape[0], b.shape[1]
	if K != K2 {
		panic(fmt.Sprintf("gorch: MatMul shape mismatch: (%d,%d) @ (%d,%d)", M, K, K2, N))
	}

	out := Zeros(M, N)

	if a.buf != nil && b.buf != nil && gpu != nil {
		// GPU path: MPS matmul
		outBuf := gpu.Dev.NewBuffer(M * N * 4)
		gpu.Queue.MatMul(a.buf, b.buf, outBuf, M, N, K)
		out.data = outBuf.FloatSlice()
		out.buf = outBuf
	} else {
		// CPU path: naive matmul
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				var s float32
				for k := 0; k < K; k++ {
					s += a.data[i*K+k] * b.data[k*N+j]
				}
				out.data[i*N+j] = s
			}
		}
	}

	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "MatMul",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				// dL/dA = grad @ B^T
				// dL/dB = A^T @ grad
				bT := transpose(b)
				aT := transpose(a)
				return []*Tensor{
					matMulCPU(grad, bT),
					matMulCPU(aT, grad),
				}
			},
		}
	}
	return out
}

// transpose returns the transpose of a 2-D tensor (CPU only, for autograd).
func transpose(a *Tensor) *Tensor {
	M, N := a.shape[0], a.shape[1]
	out := Zeros(N, M)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			out.data[j*M+i] = a.data[i*N+j]
		}
	}
	return out
}

// matMulCPU is a CPU-only matmul used in backward pass.
func matMulCPU(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]
	out := Zeros(M, N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float32
			for k := 0; k < K; k++ {
				s += a.data[i*K+k] * b.data[k*N+j]
			}
			out.data[i*N+j] = s
		}
	}
	return out
}

// ---------- dispatch helpers ----------

// binaryOp dispatches to Metal if both tensors are on GPU, else CPU.
func binaryOp(a, b *Tensor, kernelName string, cpuFn func(float32, float32) float32) *Tensor {
	out := Zeros(a.shape...)

	if a.buf != nil && b.buf != nil && gpu != nil {
		outBuf := gpu.Dev.NewBuffer(a.Size() * 4)
		gpu.Queue.Dispatch1D(gpu.pipe(kernelName), []*metal.Buffer{a.buf, b.buf, outBuf}, a.Size())
		out.data = outBuf.FloatSlice()
		out.buf = outBuf
	} else {
		for i := range a.data {
			out.data[i] = cpuFn(a.data[i], b.data[i])
		}
	}
	return out
}

// unaryOp dispatches to Metal if tensor is on GPU, else CPU.
func unaryOp(a *Tensor, kernelName string, cpuFn func(float32) float32) *Tensor {
	out := Zeros(a.shape...)

	if a.buf != nil && gpu != nil {
		outBuf := gpu.Dev.NewBuffer(a.Size() * 4)
		gpu.Queue.Dispatch1D(gpu.pipe(kernelName), []*metal.Buffer{a.buf, outBuf}, a.Size())
		out.data = outBuf.FloatSlice()
		out.buf = outBuf
	} else {
		for i, v := range a.data {
			out.data[i] = cpuFn(v)
		}
	}
	return out
}

func assertSameShape(a, b *Tensor) {
	if !sameShape(a.shape, b.shape) {
		panic(fmt.Sprintf("gorch: shape mismatch: %v vs %v", a.shape, b.shape))
	}
}
