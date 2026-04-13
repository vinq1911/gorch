//go:build darwin

package nn

import (
	"math/rand"

	g "github.com/vinq1911/gorch"
)

// Dropout randomly zeroes elements with probability p during training.
// During inference (when training=false), it's a no-op.
type Dropout struct {
	P        float32 // probability of zeroing an element
	Training bool
}

// NewDropout creates a Dropout layer with the given drop probability.
func NewDropout(p float32) *Dropout {
	return &Dropout{P: p, Training: true}
}

// Forward applies dropout: zeroes random elements and scales remaining by 1/(1-p).
func (d *Dropout) Forward(x *g.Tensor) *g.Tensor {
	if !d.Training || d.P == 0 {
		return x
	}

	mask := make([]float32, x.Size())
	scale := 1.0 / (1.0 - d.P)
	for i := range mask {
		if rand.Float32() >= d.P {
			mask[i] = scale
		}
		// else mask[i] = 0 (zero value)
	}

	out := g.Zeros(x.Shape()...)
	xData := x.Data()
	outData := out.Data()
	for i := range outData {
		outData[i] = xData[i] * mask[i]
	}

	if x.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("Dropout", []*g.Tensor{x}, func(grad *g.Tensor) []*g.Tensor {
			dx := g.Zeros(x.Shape()...)
			dxData := dx.Data()
			gData := grad.Data()
			for i := range dxData {
				dxData[i] = gData[i] * mask[i]
			}
			return []*g.Tensor{dx}
		})
	}
	return out
}

func (d *Dropout) Parameters() []*g.Tensor { return nil }

// Train sets the module to training mode.
func (d *Dropout) Train() { d.Training = true }

// Eval sets the module to evaluation mode (dropout disabled).
func (d *Dropout) Eval() { d.Training = false }
