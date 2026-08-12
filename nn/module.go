//go:build darwin

// Package nn provides neural network modules for gorch.
package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
	"github.com/vinq1911/gorch/metal"
)

// Module is a neural network layer or model.
type Module interface {
	// Forward computes the output given input.
	Forward(x *g.Tensor) *g.Tensor
	// Parameters returns all learnable parameters.
	Parameters() []*g.Tensor
}

// AlwaysComputeLinearDW disables the frozen-weight dW/db skip in
// Linear's backward (plan 0008 §3.6). Normally, when a Linear's
// Weight has RequiresGrad()==false the dL/dW GEMM (and the db sum for
// a frozen Bias) is skipped entirely — the autograd engine never
// accumulates gradients into frozen tensors, so computing them is
// pure waste; with a fully-frozen LoRA base model that skip removes
// one of the three big GEMMs from every Linear backward. This switch
// exists so trainers can measure the step-time delta.
var AlwaysComputeLinearDW = false

// ---------- Linear ----------

// Linear implements a fully connected layer: y = x @ W^T + b.
// Input shape: (batch, inFeatures), output shape: (batch, outFeatures).
type Linear struct {
	Weight  *g.Tensor // shape: (outFeatures, inFeatures)
	Bias    *g.Tensor // shape: (1, outFeatures)
	in      int
	out     int
	onMetal bool
}

// NewLinear creates a Linear layer with Kaiming uniform initialization.
func NewLinear(inFeatures, outFeatures int) *Linear {
	// Kaiming init: scale = sqrt(2 / inFeatures)
	scale := float32(math.Sqrt(2.0 / float64(inFeatures)))
	w := g.RandN(outFeatures, inFeatures)
	for i := range w.Data() {
		w.Data()[i] *= scale
	}
	w.SetRequiresGrad(true)

	b := g.Zeros(1, outFeatures)
	b.SetRequiresGrad(true)

	return &Linear{Weight: w, Bias: b, in: inFeatures, out: outFeatures}
}

// Forward computes y = x @ W^T + b using Accelerate BLAS.
func (l *Linear) Forward(x *g.Tensor) *g.Tensor {
	batch := x.Shape()[0]

	var out *g.Tensor

	if (l.onMetal && x.MetalBuffer() != nil && l.Weight.MetalBuffer() != nil) ||
		l.Weight.Dtype() != g.Float32 {
		// GPU path: x @ W^T via MPS MatMulTransB
		// x is (batch, in), W is (out, in), result is (batch, out)
		//
		// A bf16 Weight (plan 0009 X3: frozen-path base weights) always
		// takes this route — MatMulTransB dispatches the MPS dtyped
		// matmul (f32 accumulation, f32 output) when resident and above
		// threshold, and widens to f32 + Accelerate otherwise. The CPU
		// branch below reads Weight.Data(), which is nil for bf16.
		out = g.MatMulTransB(x, l.Weight)
		// Bias add via the vec_bias_add kernel (plan 0009 X2b): the
		// previous CPU loop read out.Data(), forcing a GPU sync per
		// Linear forward. Fallback: vectorized vDSP through unified
		// memory (Data() performs the required sync).
		if !g.BiasAddInPlaceMetal(out, l.Bias) {
			bData := l.Bias.Data()
			outData := out.Data()
			for i := 0; i < batch; i++ {
				row := outData[i*l.out : (i+1)*l.out]
				accelerate.VAdd(row, bData, row)
			}
		}
	} else {
		// CPU path: Accelerate BLAS. Allocate the output tensor once
		// and have sgemm write directly into its data slice. The
		// previous implementation made a fresh scratch buffer, ran
		// sgemm + bias-add into it, then created a tensor via
		// NewTensor — which copied the entire slice again. Single-
		// alloc cuts ~10% off forward time at GPT-2 small dims and
		// drops the GC churn from per-Linear allocations.
		//
		// The output inherits x's Metal residency (plan 0009 X1):
		// below-threshold or CPU-computed Linears in a GPU-resident
		// chain must not strip the chain of its residency — sgemm
		// writes through unified memory either way.
		out = g.ZerosLike(x, batch, l.out)
		outData := out.Data()
		accelerate.SgemmTransB(batch, l.out, l.in, 1.0, x.Data(), l.Weight.Data(), 0.0, outData)
		bData := l.Bias.Data()
		for i := 0; i < batch; i++ {
			row := outData[i*l.out : (i+1)*l.out]
			accelerate.VAdd(row, bData, row)
		}
	}

	// Autograd
	if x.RequiresGrad() || l.Weight.RequiresGrad() || l.Bias.RequiresGrad() {
		out.SetRequiresGrad(true)
		capturedX := x
		capturedW := l.Weight
		capturedIn := l.in
		capturedOut := l.out
		capturedBatch := batch

		out.SetGradFn("Linear", []*g.Tensor{capturedX, capturedW, l.Bias}, func(grad *g.Tensor) []*g.Tensor {
			// GPU path: when grad, x, and W are all on Metal, dispatch
			// dx and dW through MPS. The bias sum is always done on
			// CPU because it touches at most a few thousand floats.
			// Frozen-param fast path (plan 0008 §3.6): a gradient is
			// only accumulated into inputs with RequiresGrad()==true,
			// so for a frozen Weight/Bias the dW GEMM / db sum would
			// be computed and thrown away. Skip them (nil slots are
			// never read by the autograd engine for frozen inputs).
			needDW := capturedW.RequiresGrad() || AlwaysComputeLinearDW
			needDB := l.Bias.RequiresGrad() || AlwaysComputeLinearDW

			if (grad.IsOnMetal() && capturedX.IsOnMetal() && capturedW.IsOnMetal()) ||
				capturedW.Dtype() != g.Float32 {
				// The bf16-Weight case (plan 0009 X3) also takes this
				// branch regardless of residency: gpuLinearDx routes
				// through g.MatMul, whose dtyped dispatch handles a bf16
				// W on GPU or via the widen fallback — the CPU branch
				// below would read the nil Weight.Data().
				dx := gpuLinearDx(grad, capturedW, capturedBatch, capturedIn, capturedOut, capturedX.RequiresGrad())
				var dw, db *g.Tensor
				if needDW {
					dw = gpuLinearDw(grad, capturedX, capturedBatch, capturedIn, capturedOut)
				}
				if needDB {
					db = linearDb(grad, capturedBatch, capturedOut)
				}
				return []*g.Tensor{dx, dw, db}
			}

			gData := grad.Data()
			// dL/dx = grad @ W  (batch, out) @ (out, in) = (batch, in)
			// Grads inherit the residency of the tensor they belong to
			// so a partially-resident chain keeps its residency through
			// the CPU backward (plan 0009 X1).
			dx := g.ZerosLike(capturedX, capturedBatch, capturedIn)
			if capturedX.RequiresGrad() {
				accelerate.Sgemm(capturedBatch, capturedIn, capturedOut, 1.0, gData, capturedW.Data(), 0.0, dx.Data())
			}

			// dL/dW = grad^T @ x  (out, batch) @ (batch, in) = (out, in)
			// Frozen-dW skip (plan 0008 M1) composed with residency
			// inheritance (plan 0009 X1): only compute dW when the
			// weight actually trains, and allocate it with the
			// weight's residency when we do.
			var dw *g.Tensor
			if needDW {
				dw = g.ZerosLike(capturedW, capturedOut, capturedIn)
				accelerate.SgemmTransA(capturedOut, capturedIn, capturedBatch, 1.0, gData, capturedX.Data(), 0.0, dw.Data())
			}

			var db *g.Tensor
			if needDB {
				db = linearDb(grad, capturedBatch, capturedOut)
			}

			return []*g.Tensor{dx, dw, db}
		})
	}
	return out
}

func (l *Linear) Parameters() []*g.Tensor {
	return []*g.Tensor{l.Weight, l.Bias}
}

// ToMetal moves the Linear layer's weights to Metal GPU.
func (l *Linear) ToMetal(dev *metal.Device) {
	l.Weight.ToMetal(dev)
	l.Bias.ToMetal(dev)
	l.onMetal = true
}

// ToCPU moves the Linear layer back to CPU.
func (l *Linear) ToCPU() {
	l.Weight.ToCPU()
	l.Bias.ToCPU()
	l.onMetal = false
}

// ---------- Activations as modules ----------

// ReLUModule wraps the ReLU activation as a Module.
type ReLUModule struct{}

func NewReLU() *ReLUModule { return &ReLUModule{} }

func (r *ReLUModule) Forward(x *g.Tensor) *g.Tensor { return g.ReLU(x) }
func (r *ReLUModule) Parameters() []*g.Tensor       { return nil }

// SigmoidModule wraps sigmoid as a Module.
type SigmoidModule struct{}

func NewSigmoid() *SigmoidModule { return &SigmoidModule{} }

func (s *SigmoidModule) Forward(x *g.Tensor) *g.Tensor { return g.Sigmoid(x) }
func (s *SigmoidModule) Parameters() []*g.Tensor       { return nil }

// TanhModule wraps tanh as a Module.
type TanhModule struct{}

func NewTanh() *TanhModule { return &TanhModule{} }

func (t *TanhModule) Forward(x *g.Tensor) *g.Tensor { return g.Tanh(x) }
func (t *TanhModule) Parameters() []*g.Tensor       { return nil }

// ---------- Conv2d ----------

// Conv2d implements a 2D convolutional layer.
// Input shape: (batch, inChannels, H, W), output shape: (batch, outChannels, outH, outW).
type Conv2d struct {
	Weight  *g.Tensor // shape: (outChannels, inChannels, kernelSize, kernelSize)
	Bias    *g.Tensor // shape: (outChannels,)
	Stride  int
	Padding int
}

// NewConv2d creates a Conv2d layer with Kaiming initialization.
func NewConv2d(inChannels, outChannels, kernelSize, stride, padding int) *Conv2d {
	fanIn := float64(inChannels * kernelSize * kernelSize)
	scale := float32(math.Sqrt(2.0 / fanIn))

	w := g.RandN(outChannels, inChannels, kernelSize, kernelSize)
	for i := range w.Data() {
		w.Data()[i] *= scale
	}
	w.SetRequiresGrad(true)

	b := g.Zeros(outChannels)
	b.SetRequiresGrad(true)

	return &Conv2d{Weight: w, Bias: b, Stride: stride, Padding: padding}
}

func (c *Conv2d) Forward(x *g.Tensor) *g.Tensor {
	return g.Conv2dForward(x, c.Weight, c.Bias, c.Stride, c.Padding)
}

func (c *Conv2d) Parameters() []*g.Tensor {
	return []*g.Tensor{c.Weight, c.Bias}
}

// ---------- MaxPool2d ----------

// MaxPool2d implements 2D max pooling.
type MaxPool2d struct {
	KernelSize int
	Stride     int
}

// NewMaxPool2d creates a MaxPool2d layer.
func NewMaxPool2d(kernelSize, stride int) *MaxPool2d {
	return &MaxPool2d{KernelSize: kernelSize, Stride: stride}
}

func (m *MaxPool2d) Forward(x *g.Tensor) *g.Tensor {
	return g.MaxPool2dForward(x, m.KernelSize, m.Stride)
}

func (m *MaxPool2d) Parameters() []*g.Tensor { return nil }

// ---------- Flatten ----------

// Flatten reshapes (batch, C, H, W) to (batch, C*H*W) for transition from conv to linear.
type Flatten struct{}

func NewFlatten() *Flatten { return &Flatten{} }

func (f *Flatten) Forward(x *g.Tensor) *g.Tensor {
	return g.FlattenForward(x)
}

func (f *Flatten) Parameters() []*g.Tensor { return nil }

// ---------- Sequential ----------

// Sequential chains multiple modules in order.
type Sequential struct {
	Layers []Module
}

// NewSequential creates a Sequential model from the given layers.
func NewSequential(layers ...Module) *Sequential {
	return &Sequential{Layers: layers}
}

func (s *Sequential) Forward(x *g.Tensor) *g.Tensor {
	for _, layer := range s.Layers {
		x = layer.Forward(x)
	}
	return x
}

func (s *Sequential) Parameters() []*g.Tensor {
	var params []*g.Tensor
	for _, layer := range s.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
