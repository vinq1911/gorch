//go:build darwin

// Package nn provides neural network modules for gorch.
package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
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

// Forward computes y = x @ W^T + b with full autograd support.
func (l *Linear) Forward(x *g.Tensor) *g.Tensor {
	batch := x.Shape()[0]

	// Manual matmul: out = x @ W^T
	xData := x.Data()
	wData := l.Weight.Data()
	outData := make([]float32, batch*l.out)
	for i := 0; i < batch; i++ {
		for j := 0; j < l.out; j++ {
			var s float32
			for k := 0; k < l.in; k++ {
				s += xData[i*l.in+k] * wData[j*l.in+k] // W is (out, in), row j
			}
			outData[i*l.out+j] = s
		}
	}

	// Add bias
	bData := l.Bias.Data()
	for i := 0; i < batch; i++ {
		for j := 0; j < l.out; j++ {
			outData[i*l.out+j] += bData[j]
		}
	}

	out := g.NewTensor(outData, batch, l.out)

	// Set up autograd
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
			var dxData []float32
			if capturedX.RequiresGrad() {
				dxData = make([]float32, capturedBatch*capturedIn)
				for i := 0; i < capturedBatch; i++ {
					for k := 0; k < capturedIn; k++ {
						var s float32
						for j := 0; j < capturedOut; j++ {
							s += gData[i*capturedOut+j] * capturedW.Data()[j*capturedIn+k]
						}
						dxData[i*capturedIn+k] = s
					}
				}
			}

			// dL/dW = grad^T @ x  => accumulated: (out, in)
			dwData := make([]float32, capturedOut*capturedIn)
			for j := 0; j < capturedOut; j++ {
				for k := 0; k < capturedIn; k++ {
					var s float32
					for i := 0; i < capturedBatch; i++ {
						s += gData[i*capturedOut+j] * capturedX.Data()[i*capturedIn+k]
					}
					dwData[j*capturedIn+k] = s
				}
			}

			// dL/db = sum of grad over batch
			dbData := make([]float32, capturedOut)
			for i := 0; i < capturedBatch; i++ {
				for j := 0; j < capturedOut; j++ {
					dbData[j] += gData[i*capturedOut+j]
				}
			}

			var dx *g.Tensor
			if capturedX.RequiresGrad() {
				dx = g.NewTensor(dxData, capturedBatch, capturedIn)
			} else {
				dx = g.Zeros(capturedBatch, capturedIn)
			}
			dw := g.NewTensor(dwData, capturedOut, capturedIn)
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
