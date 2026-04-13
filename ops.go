//go:build darwin

package gorch

import (
	"fmt"
	"math"
	"sync"

	"github.com/vinq1911/gorch/accelerate"
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

// ---------- Exp / Log ----------

// Exp returns e^a element-wise.
func Exp(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	accelerate.Exp(a.data, out.data)
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Exp",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// d(exp(x))/dx = exp(x)
				g := Zeros(a.shape...)
				for i, v := range out.data {
					g.data[i] = grad.data[i] * v
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Log returns ln(a) element-wise.
func Log(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	accelerate.Log(a.data, out.data)
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Log",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				g := Zeros(a.shape...)
				for i, v := range a.data {
					g.data[i] = grad.data[i] / v
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// ---------- Softmax / LogSoftmax ----------

// Softmax applies softmax along the last dimension of a 2-D tensor (batch, classes).
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
func Softmax(a *Tensor) *Tensor {
	if a.Dim() != 2 {
		panic("gorch: Softmax requires 2-D tensor (batch, classes)")
	}
	batch, classes := a.shape[0], a.shape[1]
	out := Zeros(batch, classes)

	for i := 0; i < batch; i++ {
		// Numerical stability: subtract max
		rowMax := a.data[i*classes]
		for j := 1; j < classes; j++ {
			if a.data[i*classes+j] > rowMax {
				rowMax = a.data[i*classes+j]
			}
		}
		var sum float32
		for j := 0; j < classes; j++ {
			out.data[i*classes+j] = float32(math.Exp(float64(a.data[i*classes+j] - rowMax)))
			sum += out.data[i*classes+j]
		}
		for j := 0; j < classes; j++ {
			out.data[i*classes+j] /= sum
		}
	}

	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Softmax",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(batch, classes)
				for i := 0; i < batch; i++ {
					// For each sample: dx = s * (grad - sum(grad * s))
					var dot float32
					for j := 0; j < classes; j++ {
						dot += grad.data[i*classes+j] * out.data[i*classes+j]
					}
					for j := 0; j < classes; j++ {
						dx.data[i*classes+j] = out.data[i*classes+j] * (grad.data[i*classes+j] - dot)
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// LogSoftmax applies log-softmax along the last dimension of a 2-D tensor.
// logsoftmax(x_i) = x_i - max(x) - log(sum(exp(x - max(x))))
func LogSoftmax(a *Tensor) *Tensor {
	if a.Dim() != 2 {
		panic("gorch: LogSoftmax requires 2-D tensor (batch, classes)")
	}
	batch, classes := a.shape[0], a.shape[1]
	out := Zeros(batch, classes)

	// Also store softmax for backward
	sm := make([]float32, batch*classes)

	for i := 0; i < batch; i++ {
		rowMax := a.data[i*classes]
		for j := 1; j < classes; j++ {
			if a.data[i*classes+j] > rowMax {
				rowMax = a.data[i*classes+j]
			}
		}
		var sumExp float32
		for j := 0; j < classes; j++ {
			sm[i*classes+j] = float32(math.Exp(float64(a.data[i*classes+j] - rowMax)))
			sumExp += sm[i*classes+j]
		}
		logSumExp := float32(math.Log(float64(sumExp)))
		for j := 0; j < classes; j++ {
			sm[i*classes+j] /= sumExp // now softmax
			out.data[i*classes+j] = a.data[i*classes+j] - rowMax - logSumExp
		}
	}

	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "LogSoftmax",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(batch, classes)
				for i := 0; i < batch; i++ {
					var sumGrad float32
					for j := 0; j < classes; j++ {
						sumGrad += grad.data[i*classes+j]
					}
					for j := 0; j < classes; j++ {
						dx.data[i*classes+j] = grad.data[i*classes+j] - sm[i*classes+j]*sumGrad
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// ---------- Reduction ops ----------

// Sum returns the sum of all elements as a scalar tensor.
func Sum(a *Tensor) *Tensor {
	s := accelerate.Sum(a.data)
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
		// CPU path: Accelerate BLAS sgemm
		accelerate.Sgemm(M, N, K, 1.0, a.data, b.data, 0.0, out.data)
	}

	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "MatMul",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				// dL/dA = grad @ B^T  (using BLAS transB)
				// dL/dB = A^T @ grad  (using BLAS transA)
				gM, gN := grad.shape[0], grad.shape[1]
				bK := b.shape[0] // B is KxN, we want grad @ B^T = (gM, gN) @ (gN, bK) but bK=K, gN=N
				dA := Zeros(gM, b.shape[0])
				accelerate.SgemmTransB(gM, b.shape[0], gN, 1.0, grad.data, b.data, 0.0, dA.data)

				dB := Zeros(a.shape[1], gN)
				_ = bK
				accelerate.SgemmTransA(a.shape[1], gN, a.shape[0], 1.0, a.data, grad.data, 0.0, dB.data)

				return []*Tensor{dA, dB}
			},
		}
	}
	return out
}

// ---------- dispatch helpers ----------

// Accelerate-backed CPU dispatch function type.
type accBinaryFn func(a, b, out []float32)
type accUnaryFn func(a, out []float32)

// binaryOp dispatches to Metal if both tensors are on GPU, Accelerate on CPU.
func binaryOp(a, b *Tensor, kernelName string, cpuFn func(float32, float32) float32) *Tensor {
	out := Zeros(a.shape...)

	if a.buf != nil && b.buf != nil && gpu != nil {
		outBuf := gpu.Dev.NewBuffer(a.Size() * 4)
		gpu.Queue.Dispatch1D(gpu.pipe(kernelName), []*metal.Buffer{a.buf, b.buf, outBuf}, a.Size())
		out.data = outBuf.FloatSlice()
		out.buf = outBuf
	} else if fn := accBinaryFor(kernelName); fn != nil {
		fn(a.data, b.data, out.data)
	} else {
		for i := range a.data {
			out.data[i] = cpuFn(a.data[i], b.data[i])
		}
	}
	return out
}

// unaryOp dispatches to Metal if tensor is on GPU, Accelerate on CPU.
func unaryOp(a *Tensor, kernelName string, cpuFn func(float32) float32) *Tensor {
	out := Zeros(a.shape...)

	if a.buf != nil && gpu != nil {
		outBuf := gpu.Dev.NewBuffer(a.Size() * 4)
		gpu.Queue.Dispatch1D(gpu.pipe(kernelName), []*metal.Buffer{a.buf, outBuf}, a.Size())
		out.data = outBuf.FloatSlice()
		out.buf = outBuf
	} else if fn := accUnaryFor(kernelName); fn != nil {
		fn(a.data, out.data)
	} else {
		for i, v := range a.data {
			out.data[i] = cpuFn(v)
		}
	}
	return out
}

// accBinaryFor returns the Accelerate function for a given binary kernel name.
func accBinaryFor(name string) accBinaryFn {
	switch name {
	case "vec_add":
		return accelerate.VAdd
	case "vec_sub":
		return accelerate.VSub
	case "vec_mul":
		return accelerate.VMul
	case "vec_div":
		return accelerate.VDiv
	default:
		return nil
	}
}

// accUnaryFor returns the Accelerate function for a given unary kernel name.
func accUnaryFor(name string) accUnaryFn {
	switch name {
	case "vec_relu":
		return accelerate.ReLU
	case "vec_sigmoid":
		return accelerate.Sigmoid
	case "vec_tanh_act":
		return accelerate.Tanh
	default:
		return nil
	}
}

func assertSameShape(a, b *Tensor) {
	if !sameShape(a.shape, b.shape) {
		panic(fmt.Sprintf("gorch: shape mismatch: %v vs %v", a.shape, b.shape))
	}
}
