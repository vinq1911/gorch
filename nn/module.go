//go:build darwin

// Package nn provides neural network modules for gorch.
package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
)

// Module is a neural network layer or model.
type Module interface {
	// Forward computes the output given input.
	Forward(x *g.Tensor) *g.Tensor
	// Parameters returns all learnable parameters.
	Parameters() []*g.Tensor
}

// ---------- Linear ----------

// Linear implements a fully connected layer: y = x @ W^T + b.
// Input shape: (batch, inFeatures), output shape: (batch, outFeatures).
type Linear struct {
	Weight *g.Tensor // shape: (outFeatures, inFeatures)
	Bias   *g.Tensor // shape: (1, outFeatures)
	in     int
	out    int
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

	// out = x @ W^T via BLAS: SgemmTransB
	// x is (batch, in), W is (out, in), W^T is (in, out), result is (batch, out)
	outData := make([]float32, batch*l.out)
	accelerate.SgemmTransB(batch, l.out, l.in, 1.0, x.Data(), l.Weight.Data(), 0.0, outData)

	// Add bias using Accelerate vDSP
	bData := l.Bias.Data()
	for i := 0; i < batch; i++ {
		row := outData[i*l.out : (i+1)*l.out]
		accelerate.VAdd(row, bData, row)
	}

	out := g.NewTensor(outData, batch, l.out)

	// Autograd
	if x.RequiresGrad() || l.Weight.RequiresGrad() || l.Bias.RequiresGrad() {
		out.SetRequiresGrad(true)
		capturedX := x
		capturedW := l.Weight
		capturedIn := l.in
		capturedOut := l.out
		capturedBatch := batch

		out.SetGradFn("Linear", []*g.Tensor{capturedX, capturedW, l.Bias}, func(grad *g.Tensor) []*g.Tensor {
			gData := grad.Data()

			// dL/dx = grad @ W  (batch, out) @ (out, in) = (batch, in)
			var dx *g.Tensor
			if capturedX.RequiresGrad() {
				dxData := make([]float32, capturedBatch*capturedIn)
				accelerate.Sgemm(capturedBatch, capturedIn, capturedOut, 1.0, gData, capturedW.Data(), 0.0, dxData)
				dx = g.NewTensor(dxData, capturedBatch, capturedIn)
			} else {
				dx = g.Zeros(capturedBatch, capturedIn)
			}

			// dL/dW = grad^T @ x  (out, batch) @ (batch, in) = (out, in)
			dwData := make([]float32, capturedOut*capturedIn)
			accelerate.SgemmTransA(capturedOut, capturedIn, capturedBatch, 1.0, gData, capturedX.Data(), 0.0, dwData)
			dw := g.NewTensor(dwData, capturedOut, capturedIn)

			// dL/db = sum of grad over batch (column sums)
			dbData := make([]float32, capturedOut)
			for i := 0; i < capturedBatch; i++ {
				row := gData[i*capturedOut : (i+1)*capturedOut]
				accelerate.VAdd(dbData, row, dbData)
			}
			db := g.NewTensor(dbData, 1, capturedOut)

			return []*g.Tensor{dx, dw, db}
		})
	}
	return out
}

func (l *Linear) Parameters() []*g.Tensor {
	return []*g.Tensor{l.Weight, l.Bias}
}

// ---------- Activations as modules ----------

// ReLUModule wraps the ReLU activation as a Module.
type ReLUModule struct{}

func NewReLU() *ReLUModule { return &ReLUModule{} }

func (r *ReLUModule) Forward(x *g.Tensor) *g.Tensor { return g.ReLU(x) }
func (r *ReLUModule) Parameters() []*g.Tensor        { return nil }

// SigmoidModule wraps sigmoid as a Module.
type SigmoidModule struct{}

func NewSigmoid() *SigmoidModule { return &SigmoidModule{} }

func (s *SigmoidModule) Forward(x *g.Tensor) *g.Tensor { return g.Sigmoid(x) }
func (s *SigmoidModule) Parameters() []*g.Tensor        { return nil }

// TanhModule wraps tanh as a Module.
type TanhModule struct{}

func NewTanh() *TanhModule { return &TanhModule{} }

func (t *TanhModule) Forward(x *g.Tensor) *g.Tensor { return g.Tanh(x) }
func (t *TanhModule) Parameters() []*g.Tensor        { return nil }

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
