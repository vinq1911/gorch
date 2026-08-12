//go:build darwin

package gorch

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/vinq1911/gorch/accelerate"
	"github.com/vinq1911/gorch/metal"
)

// ---------- Metal dispatch counters (plan 0009 X1 gate evidence) ----------

var (
	mpsMatMulDispatches        atomic.Int64
	mpsBatchedMatMulDispatches atomic.Int64
	metalSoftmaxDispatches     atomic.Int64
	metalSiluDispatches        atomic.Int64 // K4: vec_silu/vec_swiglu fwd+bwd
	metalCEDispatches          atomic.Int64 // K2: cross_entropy fwd+bwd
	metalPermuteDispatches     atomic.Int64 // X2b: permute_copy fwd+bwd
	metalRopeDispatches        atomic.Int64 // X2b/K6: rope_apply fwd+bwd
	metalRepeatDispatches      atomic.Int64 // X2b: repeat_interleave fwd+bwd
	metalColReduceDispatches   atomic.Int64 // X2b/K5: rmsnorm_dgamma + col_sum
	metalBiasAddDispatches     atomic.Int64 // X2b: vec_bias_add (Linear fwd bias)
	mpsBF16MatMulDispatches    atomic.Int64 // X3/B4: MPS dtyped (bf16-operand) matmuls, fwd+bwd
)

// MetalDispatchCounts is a snapshot of how many times each GPU dispatch
// class has fired since the last reset. Tests use it to assert that
// above-threshold matmuls (forward AND backward) actually hit MPS.
type MetalDispatchCounts struct {
	MatMul        int64 // MPS single matmuls (plain/transA/transB), fwd+bwd
	BatchedMatMul int64 // MPS batched matmuls (plain/transA/transB), fwd+bwd
	SoftmaxKernel int64 // fused causal-softmax fwd + softmax bwd kernel dispatches
	SiluKernel    int64 // K4 SiLU/SwiGLU fwd + bwd kernel dispatches
	CEKernel      int64 // K2 fused cross-entropy fwd + bwd kernel dispatches
	PermuteKernel int64 // X2b permute_copy dispatches (fwd + grad)
	RopeKernel    int64 // X2b/K6 rope_apply dispatches (fwd + grad)
	RepeatKernel  int64 // X2b repeat_interleave fwd + bwd dispatches
	ColReduce     int64 // X2b/K5 rmsnorm_dgamma + col_sum dispatches
	BiasAdd       int64 // X2b vec_bias_add dispatches (Linear GPU forward)
	BF16MatMul    int64 // X3/B4 MPS dtyped matmuls with a bf16 operand (single + batched), fwd+bwd
}

// ReadMetalDispatchCounts returns the current dispatch counters.
func ReadMetalDispatchCounts() MetalDispatchCounts {
	return MetalDispatchCounts{
		MatMul:        mpsMatMulDispatches.Load(),
		BatchedMatMul: mpsBatchedMatMulDispatches.Load(),
		SoftmaxKernel: metalSoftmaxDispatches.Load(),
		SiluKernel:    metalSiluDispatches.Load(),
		CEKernel:      metalCEDispatches.Load(),
		PermuteKernel: metalPermuteDispatches.Load(),
		RopeKernel:    metalRopeDispatches.Load(),
		RepeatKernel:  metalRepeatDispatches.Load(),
		ColReduce:     metalColReduceDispatches.Load(),
		BiasAdd:       metalBiasAddDispatches.Load(),
		BF16MatMul:    mpsBF16MatMulDispatches.Load(),
	}
}

// ResetMetalDispatchCounts zeroes the dispatch counters.
func ResetMetalDispatchCounts() {
	mpsMatMulDispatches.Store(0)
	mpsBatchedMatMulDispatches.Store(0)
	metalSoftmaxDispatches.Store(0)
	metalSiluDispatches.Store(0)
	metalCEDispatches.Store(0)
	metalPermuteDispatches.Store(0)
	metalRopeDispatches.Store(0)
	metalRepeatDispatches.Store(0)
	metalColReduceDispatches.Store(0)
	metalBiasAddDispatches.Store(0)
	mpsBF16MatMulDispatches.Store(0)
}

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
		"vec_relu", "vec_sigmoid", "vec_tanh_act", "vec_scale", "vec_sum",
		"vec_gelu", "vec_bias_add",
		// Plan 0004 part A — non-MatMul backward kernels.
		"rmsnorm_forward", "rmsnorm_dx",
		// Plan 0009 K1 — fused causal softmax fwd + softmax bwd.
		"softmax_causal_forward", "softmax_backward",
		// Plan 0009 K4 — SiLU/SwiGLU fwd + bwd.
		"vec_silu", "vec_silu_bwd", "vec_swiglu", "vec_swiglu_bwd",
		// Plan 0009 K2 — fused cross-entropy fwd + bwd.
		"cross_entropy_forward", "cross_entropy_backward",
		// Plan 0009 X2b — block-step CPU residue: permute, RoPE (K6),
		// GQA KV expansion, RMSNorm dgamma (K5), Linear db column sum.
		"permute_copy", "rope_apply",
		"repeat_interleave_fwd", "repeat_interleave_bwd",
		"rmsnorm_dgamma", "col_sum"} {
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

// MetalGPU returns the global Metal GPU singleton, or nil if
// InitMetal hasn't been called. Exposed so packages outside gorch
// (e.g. nn) can dispatch their own custom kernels via this GPU's
// queue, without each package wiring up its own Metal init path.
func MetalGPU() *GPU { return gpu }

// Pipe returns a pre-compiled pipeline by kernel name. Panics if
// the kernel wasn't registered at InitMetal time. Used by package-
// external dispatch helpers (e.g. RMSNormForwardMetal).
func (g *GPU) Pipe(name string) *metal.Pipeline {
	p, ok := g.pipelines[name]
	if !ok {
		panic("gorch: pipeline " + name + " not registered; add it to InitMetal's kernel list")
	}
	return p
}

// Queue returns the underlying Metal command queue. Same pattern as
// Pipe — exposed so external dispatch helpers can submit work.
func (g *GPU) MetalQueue() *metal.CommandQueue { return g.Queue }

// ---------- Element-wise binary ops ----------

// Add returns a + b element-wise.
func Add(a, b *Tensor) *Tensor {
	assertSameShape(a, b)
	requireSameDtype(a, b, "Add")
	if a.dtype == BFloat16 {
		return downcastToBF16(Add(promoteToF32(a), promoteToF32(b)))
	}
	out := binaryOp(a, b, "vec_add", func(x, y float32) float32 { return x + y })
	if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
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
	requireSameDtype(a, b, "Sub")
	if a.dtype == BFloat16 {
		return downcastToBF16(Sub(promoteToF32(a), promoteToF32(b)))
	}
	out := binaryOp(a, b, "vec_sub", func(x, y float32) float32 { return x - y })
	if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
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
	requireSameDtype(a, b, "Mul")
	if a.dtype == BFloat16 {
		return downcastToBF16(Mul(promoteToF32(a), promoteToF32(b)))
	}
	out := binaryOp(a, b, "vec_mul", func(x, y float32) float32 { return x * y })
	if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
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
	requireSameDtype(a, b, "Div")
	if a.dtype == BFloat16 {
		return downcastToBF16(Div(promoteToF32(a), promoteToF32(b)))
	}
	out := binaryOp(a, b, "vec_div", func(x, y float32) float32 { return x / y })
	if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
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
	if a.dtype == BFloat16 {
		return downcastToBF16(Neg(promoteToF32(a)))
	}
	out := ZerosLike(a, a.shape...)
	syncForCPU(a)
	for i, v := range a.data {
		out.data[i] = -v
	}
	if GradEnabled() && (a.requiresGrad) {
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

// Scale returns a * s element-wise for a scalar s, with autograd
// (backward: dx = grad * s). Replaces the Full(s, shape) + Mul pattern
// that materialized a full-size constant tensor just to scale by a
// scalar — at GQA seq-1500 shapes that was a 144 MB allocation per
// layer per forward (plan 0009 X1 item 5). CPU compute via vDSP
// through unified memory; the output inherits a's Metal residency.
func Scale(a *Tensor, s float32) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(Scale(promoteToF32(a), s))
	}
	out := ZerosLike(a, a.shape...)
	syncForCPU(a)
	accelerate.VScale(a.data, s, out.data)
	if GradEnabled() && a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Scale",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := zerosLikeEither(a.shape, grad, a)
				syncForCPU(grad)
				accelerate.VScale(grad.data, s, dx.data)
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// ReLU returns max(0, a) element-wise.
func ReLU(a *Tensor) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(ReLU(promoteToF32(a)))
	}
	out := unaryOp(a, "vec_relu", func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	})
	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "ReLU",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				g := zerosLikeEither(a.shape, grad, a)
				syncForCPU(a, grad)
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
	if a.dtype == BFloat16 {
		return downcastToBF16(Sigmoid(promoteToF32(a)))
	}
	out := unaryOp(a, "vec_sigmoid", func(x float32) float32 {
		return float32(1.0 / (1.0 + math.Exp(float64(-x))))
	})
	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Sigmoid",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
				g := zerosLikeEither(a.shape, grad, a)
				syncForCPU(out, grad)
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
	if a.dtype == BFloat16 {
		return downcastToBF16(Tanh(promoteToF32(a)))
	}
	out := unaryOp(a, "vec_tanh_act", func(x float32) float32 {
		return float32(math.Tanh(float64(x)))
	})
	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Tanh",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// tanh'(x) = 1 - tanh(x)^2
				g := zerosLikeEither(a.shape, grad, a)
				syncForCPU(out, grad)
				for i, v := range out.data {
					g.data[i] = grad.data[i] * (1 - v*v)
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// ---------- GELU ----------

// GELU returns the Gaussian Error Linear Unit activation: x * Phi(x).
// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Forward path is dispatched: Metal kernel (vec_gelu) on GPU, vForce
// vectorised tanh on CPU. The naive per-element math.Tanh loop was
// the dominant cost in transformer FFN forward (≈60% of EncodeBatch
// time on M5 before this).
func GELU(a *Tensor) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(GELU(promoteToF32(a)))
	}
	out := ZerosLike(a, a.shape...)

	if a.buf != nil && out.buf != nil && gpu != nil {
		// GPU: single dispatch of the precompiled vec_gelu kernel.
		gpu.Queue.Dispatch1D(gpu.pipe("vec_gelu"), []*metal.Buffer{a.buf, out.buf}, a.Size())
	} else {
		// CPU: compute inner = sqrt(2/pi) * (x + 0.044715*x^3) elementwise,
		// then one vForce vector-tanh, then 0.5 * x * (1 + tanh(inner)).
		// `inner` is a transient scratch buffer — pooled to drop
		// allocations across calls.
		syncForCPU(a)
		n := len(a.data)
		inner := AcquireFloat32(n)
		for i, x := range a.data {
			x3 := x * x * x
			inner[i] = 0.7978845608 * (x + 0.044715*x3) // sqrt(2/pi)
		}
		accelerate.Tanh(inner, inner)
		for i, x := range a.data {
			out.data[i] = 0.5 * x * (1 + inner[i])
		}
		ReleaseFloat32(inner)
	}

	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "GELU",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := zerosLikeEither(a.shape, grad, a)
				syncForCPU(a, grad)
				for i, x := range a.data {
					x3 := x * x * x
					inner := float32(0.7978845608) * (x + 0.044715*x3)
					tanhVal := float32(math.Tanh(float64(inner)))
					// d(GELU)/dx = 0.5*(1+tanh) + 0.5*x*(1-tanh^2)*0.7978845608*(1+3*0.044715*x^2)
					dtanh := 1 - tanhVal*tanhVal
					dx.data[i] = grad.data[i] * (0.5*(1+tanhVal) + 0.5*x*dtanh*0.7978845608*(1+3*0.044715*x*x))
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// ---------- Exp / Log ----------

// Exp returns e^a element-wise.
func Exp(a *Tensor) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(Exp(promoteToF32(a)))
	}
	out := ZerosLike(a, a.shape...)
	syncForCPU(a)
	accelerate.Exp(a.data, out.data)
	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Exp",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// d(exp(x))/dx = exp(x)
				g := zerosLikeEither(a.shape, grad, a)
				syncForCPU(out, grad)
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
	if a.dtype == BFloat16 {
		return downcastToBF16(Log(promoteToF32(a)))
	}
	out := ZerosLike(a, a.shape...)
	syncForCPU(a)
	accelerate.Log(a.data, out.data)
	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Log",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				g := zerosLikeEither(a.shape, grad, a)
				syncForCPU(a, grad)
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
	if a.dtype == BFloat16 {
		return downcastToBF16(Softmax(promoteToF32(a)))
	}
	batch, classes := a.shape[0], a.shape[1]
	out := ZerosLike(a, batch, classes)
	syncForCPU(a)

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

	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Softmax",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := zerosLikeEither([]int{batch, classes}, grad, out)
				syncForCPU(out, grad)
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
	if a.dtype == BFloat16 {
		return downcastToBF16(LogSoftmax(promoteToF32(a)))
	}
	batch, classes := a.shape[0], a.shape[1]
	out := ZerosLike(a, batch, classes)
	syncForCPU(a)

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

	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "LogSoftmax",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := zerosLikeEither([]int{batch, classes}, grad, out)
				syncForCPU(grad)
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
	if a.dtype == BFloat16 {
		return downcastToBF16(Sum(promoteToF32(a)))
	}
	syncForCPU(a)
	s := accelerate.Sum(a.data)
	out := NewTensor([]float32{s}, 1)
	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Sum",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				// Gradient of sum is all ones scaled by upstream grad.
				// The seed inherits a's Metal residency so the whole
				// backward chain of a GPU-resident graph starts on
				// Metal (plan 0009 X1 item 3: loss-side grad seeding).
				return []*Tensor{fullLike(a, grad.data[0], a.shape...)}
			},
		}
	}
	return out
}

// Mean returns the mean of all elements as a scalar tensor.
func Mean(a *Tensor) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(Mean(promoteToF32(a)))
	}
	s := Sum(a)
	n := float32(a.Size())
	out := NewTensor([]float32{s.data[0] / n}, 1)
	if GradEnabled() && (a.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Mean",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{fullLike(a, grad.data[0]/n, a.shape...)}
			},
		}
	}
	return out
}

// ---------- MatMul ----------

// MatMulMetalThreshold is the M*N*K product below which gorch uses
// Accelerate sgemm even when both operands are on Metal. Empirically
// (see doc/metal_crossover_results.json), MPS dispatch overhead on
// Apple M-series is ~1ms; below ~1G FMAs the dispatch dominates and
// CPU sgemm wins. Crossover lives between 768³ (GPU 0.45×) and
// 1024³ (GPU 1.22×). Threshold set conservatively at 512M FMAs.
//
// This is a package-level variable so callers benchmarking large-
// matmul GPU paths can lower it. Setting to 0 always uses GPU when
// possible.
var MatMulMetalThreshold = 512_000_000

// shouldUseMetalMatMul returns true when M*N*K is large enough that
// MPS dispatch is faster than Accelerate sgemm.
func shouldUseMetalMatMul(M, N, K int) bool {
	if gpu == nil {
		return false
	}
	return int64(M)*int64(N)*int64(K) >= int64(MatMulMetalThreshold)
}

// MatMul computes matrix multiplication: a @ b.
// a is (M, K), b is (K, N), result is (M, N).
func MatMul(a, b *Tensor) *Tensor {
	if a.Dim() != 2 || b.Dim() != 2 {
		panic("gorch: MatMul requires 2-D tensors")
	}
	if a.dtype == BFloat16 || b.dtype == BFloat16 {
		// Plan 0009 X3-B4: bf16 operands (frozen-path weights) dispatch
		// the MPS dtyped path with f32 accumulation + f32 output, or
		// widen-to-f32 below threshold. Mixed f32/bf16 is allowed here.
		return matMulBF16(a, b)
	}
	M, K := a.shape[0], a.shape[1]
	K2, N := b.shape[0], b.shape[1]
	if K != K2 {
		panic(fmt.Sprintf("gorch: MatMul shape mismatch: (%d,%d) @ (%d,%d)", M, K, K2, N))
	}

	out := zerosLikeEither([]int{M, N}, a, b)

	if a.buf != nil && b.buf != nil && shouldUseMetalMatMul(M, N, K) {
		// GPU path: MPS matmul. Only dispatched when M*N*K is large
		// enough for compute to pay off the dispatch overhead.
		gpu.Queue.MatMul(a.buf, b.buf, out.buf, M, N, K)
		mpsMatMulDispatches.Add(1)
	} else {
		// CPU path: Accelerate BLAS sgemm. Reads through unified-
		// memory slices when operands are Metal-backed.
		syncForCPU(a, b)
		accelerate.Sgemm(M, N, K, 1.0, a.data, b.data, 0.0, out.data)
	}

	if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "MatMul",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				// dL/dA = grad @ B^T  (M, N) @ (N, K) -> (M, K)
				// dL/dB = A^T @ grad  (K, M) @ (M, N) -> (K, N)
				gM, gN := grad.shape[0], grad.shape[1]
				bK := b.shape[0]

				// GPU path: every operand on Metal AND the dW shape
				// (gM*bK*gN ≈ M*N*K of the original forward) clears
				// the threshold. Otherwise CPU.
				if a.buf != nil && b.buf != nil && grad.buf != nil &&
					shouldUseMetalMatMul(gM, bK, gN) {
					dA := ZerosLike(a, gM, bK)
					gpu.Queue.MatMulTransB(grad.buf, b.buf, dA.buf, gM, bK, gN)
					mpsMatMulDispatches.Add(1)

					dB := ZerosLike(b, a.shape[1], gN)
					gpu.Queue.MatMulTransA(a.buf, grad.buf, dB.buf, a.shape[1], gN, a.shape[0])
					mpsMatMulDispatches.Add(1)
					return []*Tensor{dA, dB}
				}

				// CPU path: Accelerate BLAS. Grads inherit the residency
				// of the tensor they belong to so the backward chain
				// stays GPU-resident below the threshold too.
				syncForCPU(a, b, grad)
				dA := zerosLikeEither([]int{gM, bK}, a, grad)
				accelerate.SgemmTransB(gM, bK, gN, 1.0, grad.data, b.data, 0.0, dA.data)

				dB := zerosLikeEither([]int{a.shape[1], gN}, b, grad)
				accelerate.SgemmTransA(a.shape[1], gN, a.shape[0], 1.0, a.data, grad.data, 0.0, dB.data)

				return []*Tensor{dA, dB}
			},
		}
	}
	return out
}

// MatMulTransB computes a @ b^T.
// a is (M, K), b is (N, K), result is (M, N).
func MatMulTransB(a, b *Tensor) *Tensor {
	if a.Dim() != 2 || b.Dim() != 2 {
		panic("gorch: MatMulTransB requires 2-D tensors")
	}
	M, K := a.shape[0], a.shape[1]
	N, K2 := b.shape[0], b.shape[1]
	if K != K2 {
		panic(fmt.Sprintf("gorch: MatMulTransB shape mismatch: (%d,%d) @ (%d,%d)^T", M, K, N, K2))
	}
	if a.dtype == BFloat16 || b.dtype == BFloat16 {
		// Plan 0009 X3-B4. MatMulTransB carries no autograd (callers
		// like nn.Linear install their own GradFn), so the shared
		// no-grad dtyped helper covers both the MPS path and the
		// widen fallback — except the legacy CPU bf16-pair case,
		// which keeps its plan-0002 bf16-output semantics.
		if a.dtype == BFloat16 && b.dtype == BFloat16 && a.buf == nil && b.buf == nil {
			return downcastToBF16(MatMulTransB(promoteToF32(a), promoteToF32(b)))
		}
		return matMulDTGrad(a, b, M, N, K, false, true)
	}

	out := zerosLikeEither([]int{M, N}, a, b)

	if a.buf != nil && b.buf != nil && shouldUseMetalMatMul(M, N, K) {
		gpu.Queue.MatMulTransB(a.buf, b.buf, out.buf, M, N, K)
		mpsMatMulDispatches.Add(1)
	} else {
		syncForCPU(a, b)
		accelerate.SgemmTransB(M, N, K, 1.0, a.data, b.data, 0.0, out.data)
	}

	// No autograd for now — used in inference-only Linear forward
	return out
}

// MatMulTransA computes a^T @ b. a is (K, M), b is (K, N), result is (M, N).
// No autograd — used inside backward functions where the inputs are
// gradient tensors that should not extend the autograd graph.
func MatMulTransA(a, b *Tensor) *Tensor {
	if a.Dim() != 2 || b.Dim() != 2 {
		panic("gorch: MatMulTransA requires 2-D tensors")
	}
	K, M := a.shape[0], a.shape[1]
	K2, N := b.shape[0], b.shape[1]
	if K != K2 {
		panic(fmt.Sprintf("gorch: MatMulTransA shape mismatch: (%d,%d)^T @ (%d,%d)", K, M, K2, N))
	}
	if a.dtype == BFloat16 || b.dtype == BFloat16 {
		// Plan 0009 X3-B4 (no autograd, same contract as the f32 path).
		if a.dtype == BFloat16 && b.dtype == BFloat16 && a.buf == nil && b.buf == nil {
			return downcastToBF16(MatMulTransA(promoteToF32(a), promoteToF32(b)))
		}
		return matMulDTGrad(a, b, M, N, K, true, false)
	}

	out := zerosLikeEither([]int{M, N}, a, b)
	if a.buf != nil && b.buf != nil && shouldUseMetalMatMul(M, N, K) {
		gpu.Queue.MatMulTransA(a.buf, b.buf, out.buf, M, N, K)
		mpsMatMulDispatches.Add(1)
	} else {
		syncForCPU(a, b)
		accelerate.SgemmTransA(M, N, K, 1.0, a.data, b.data, 0.0, out.data)
	}
	return out
}

// BatchedMatMul computes out[i] = a[i] @ b[i] for i in 0..batchSize-1.
// a: (batchSize, M, K), b: (batchSize, K, N), out: (batchSize, M, N).
// Dispatches to MPS batched matmul on GPU, Accelerate BLAS loop on CPU.
//
// Backward (plan 0009 X1: residency-and-threshold-gated MPS path, CPU
// Accelerate loop otherwise):
//
//	dL/dA[i] = grad[i] @ B[i]^T  (batched MPS transB / per-batch SgemmTransB)
//	dL/dB[i] = A[i]^T @ grad[i]  (batched MPS transA / per-batch SgemmTransA)
func BatchedMatMul(a, b *Tensor, batchSize, M, N, K int) *Tensor {
	if a.dtype == BFloat16 || b.dtype == BFloat16 {
		// Plan 0009 X3-B4.
		return batchedMatMulBF16(a, b, batchSize, M, N, K, false)
	}
	out := zerosLikeEither([]int{batchSize, M, N}, a, b)

	if a.buf != nil && b.buf != nil && shouldUseMetalMatMul(batchSize*M, N, K) {
		gpu.Queue.BatchedMatMul(a.buf, b.buf, out.buf, M, N, K, batchSize)
		mpsBatchedMatMulDispatches.Add(1)
	} else {
		syncForCPU(a, b)
		for i := 0; i < batchSize; i++ {
			aOff := i * M * K
			bOff := i * K * N
			cOff := i * M * N
			accelerate.Sgemm(M, N, K, 1.0, a.data[aOff:aOff+M*K], b.data[bOff:bOff+K*N], 0.0, out.data[cOff:cOff+M*N])
		}
	}

	if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "BatchedMatMul",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				// GPU path: all operands Metal-resident and the grad
				// matmuls (same FMA count as forward) clear the
				// threshold. This moves attention backward onto MPS.
				if a.buf != nil && b.buf != nil && grad.buf != nil &&
					shouldUseMetalMatMul(batchSize*M, N, K) {
					// dA[i] = grad[i] @ B[i]^T: grad (M,N) times B stored
					// (K,N) read transposed → (M,K).
					dA := ZerosLike(a, batchSize, M, K)
					gpu.Queue.BatchedMatMulTransB(grad.buf, b.buf, dA.buf, M, K, N, batchSize)
					mpsBatchedMatMulDispatches.Add(1)
					// dB[i] = A[i]^T @ grad[i]: A stored (M,K) read
					// transposed, times grad (M,N) → (K,N).
					dB := ZerosLike(b, batchSize, K, N)
					gpu.Queue.BatchedMatMulTransA(a.buf, grad.buf, dB.buf, K, N, M, batchSize)
					mpsBatchedMatMulDispatches.Add(1)
					return []*Tensor{dA, dB}
				}

				syncForCPU(a, b, grad)
				dA := zerosLikeEither([]int{batchSize, M, K}, a, grad)
				dB := zerosLikeEither([]int{batchSize, K, N}, b, grad)
				for i := 0; i < batchSize; i++ {
					aOff := i * M * K
					bOff := i * K * N
					cOff := i * M * N
					// dA[i] = grad[i] @ B[i]^T
					accelerate.SgemmTransB(M, K, N, 1.0,
						grad.data[cOff:cOff+M*N], b.data[bOff:bOff+K*N],
						0.0, dA.data[aOff:aOff+M*K])
					// dB[i] = A[i]^T @ grad[i]
					accelerate.SgemmTransA(K, N, M, 1.0,
						a.data[aOff:aOff+M*K], grad.data[cOff:cOff+M*N],
						0.0, dB.data[bOff:bOff+K*N])
				}
				return []*Tensor{dA, dB}
			},
		}
	}
	return out
}

// BatchedMatMulTransB computes out[i] = a[i] @ b[i]^T for i in 0..batchSize-1.
// a: (batchSize, M, K), b: (batchSize, N, K), out: (batchSize, M, N).
//
// Backward (plan 0009 X1: residency-and-threshold-gated MPS path, CPU
// Accelerate loop otherwise):
//
//	dL/dA[i] = grad[i] @ B[i]      (batched MPS plain / per-batch Sgemm)
//	dL/dB[i] = grad[i]^T @ A[i]    (batched MPS transA / per-batch SgemmTransA)
func BatchedMatMulTransB(a, b *Tensor, batchSize, M, N, K int) *Tensor {
	if a.dtype == BFloat16 || b.dtype == BFloat16 {
		// Plan 0009 X3-B4.
		return batchedMatMulBF16(a, b, batchSize, M, N, K, true)
	}
	out := zerosLikeEither([]int{batchSize, M, N}, a, b)

	if a.buf != nil && b.buf != nil && shouldUseMetalMatMul(batchSize*M, N, K) {
		gpu.Queue.BatchedMatMulTransB(a.buf, b.buf, out.buf, M, N, K, batchSize)
		mpsBatchedMatMulDispatches.Add(1)
	} else {
		syncForCPU(a, b)
		for i := 0; i < batchSize; i++ {
			aOff := i * M * K
			bOff := i * N * K
			cOff := i * M * N
			accelerate.SgemmTransB(M, N, K, 1.0, a.data[aOff:aOff+M*K], b.data[bOff:bOff+N*K], 0.0, out.data[cOff:cOff+M*N])
		}
	}

	if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "BatchedMatMulTransB",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				if a.buf != nil && b.buf != nil && grad.buf != nil &&
					shouldUseMetalMatMul(batchSize*M, N, K) {
					// dA[i] = grad[i] @ B[i]: grad (M,N) times B (N,K)
					// plain → (M,K).
					dA := ZerosLike(a, batchSize, M, K)
					gpu.Queue.BatchedMatMul(grad.buf, b.buf, dA.buf, M, K, N, batchSize)
					mpsBatchedMatMulDispatches.Add(1)
					// dB[i] = grad[i]^T @ A[i]: grad stored (M,N) read
					// transposed, times A (M,K) → (N,K).
					dB := ZerosLike(b, batchSize, N, K)
					gpu.Queue.BatchedMatMulTransA(grad.buf, a.buf, dB.buf, N, K, M, batchSize)
					mpsBatchedMatMulDispatches.Add(1)
					return []*Tensor{dA, dB}
				}

				syncForCPU(a, b, grad)
				dA := zerosLikeEither([]int{batchSize, M, K}, a, grad)
				dB := zerosLikeEither([]int{batchSize, N, K}, b, grad)
				for i := 0; i < batchSize; i++ {
					aOff := i * M * K
					bOff := i * N * K
					cOff := i * M * N
					// dA[i] = grad[i] @ B[i]   (no transpose)
					accelerate.Sgemm(M, K, N, 1.0,
						grad.data[cOff:cOff+M*N], b.data[bOff:bOff+N*K],
						0.0, dA.data[aOff:aOff+M*K])
					// dB[i] = grad[i]^T @ A[i] — shape (N, M) @ (M, K) = (N, K)
					accelerate.SgemmTransA(N, K, M, 1.0,
						grad.data[cOff:cOff+M*N], a.data[aOff:aOff+M*K],
						0.0, dB.data[bOff:bOff+N*K])
				}
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

// binaryOp dispatches to Metal if both tensors are on GPU, Accelerate
// on CPU. The output inherits Metal residency when either input is
// Metal-backed (plan 0009 X1) — the CPU compute paths write through
// unified memory in that case.
func binaryOp(a, b *Tensor, kernelName string, cpuFn func(float32, float32) float32) *Tensor {
	out := zerosLikeEither(a.shape, a, b)

	if a.buf != nil && b.buf != nil && out.buf != nil && gpu != nil {
		gpu.Queue.Dispatch1D(gpu.pipe(kernelName), []*metal.Buffer{a.buf, b.buf, out.buf}, a.Size())
	} else if fn := accBinaryFor(kernelName); fn != nil {
		syncForCPU(a, b)
		fn(a.data, b.data, out.data)
	} else {
		syncForCPU(a, b)
		for i := range a.data {
			out.data[i] = cpuFn(a.data[i], b.data[i])
		}
	}
	return out
}

// unaryOp dispatches to Metal if tensor is on GPU, Accelerate on CPU.
// Output inherits the input's Metal residency.
func unaryOp(a *Tensor, kernelName string, cpuFn func(float32) float32) *Tensor {
	out := ZerosLike(a, a.shape...)

	if a.buf != nil && out.buf != nil && gpu != nil {
		gpu.Queue.Dispatch1D(gpu.pipe(kernelName), []*metal.Buffer{a.buf, out.buf}, a.Size())
	} else if fn := accUnaryFor(kernelName); fn != nil {
		syncForCPU(a)
		fn(a.data, out.data)
	} else {
		syncForCPU(a)
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
