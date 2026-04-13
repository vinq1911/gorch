//go:build darwin

// Package optim provides optimizers for gorch.
package optim

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// Optimizer updates model parameters using computed gradients.
type Optimizer interface {
	Step()
	ZeroGrad()
}

// ---------- SGD ----------

// SGD implements stochastic gradient descent with optional momentum.
type SGD struct {
	params   []*g.Tensor
	lr       float32
	momentum float32
	velocity [][]float32 // momentum buffers
}

// NewSGD creates an SGD optimizer.
func NewSGD(params []*g.Tensor, lr float32, momentum float32) *SGD {
	vel := make([][]float32, len(params))
	for i, p := range params {
		vel[i] = make([]float32, p.Size())
	}
	return &SGD{params: params, lr: lr, momentum: momentum, velocity: vel}
}

func (o *SGD) Step() {
	for i, p := range o.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		data := p.Data()
		gData := grad.Data()
		for j := range data {
			if o.momentum > 0 {
				o.velocity[i][j] = o.momentum*o.velocity[i][j] + gData[j]
				data[j] -= o.lr * o.velocity[i][j]
			} else {
				data[j] -= o.lr * gData[j]
			}
		}
	}
}

func (o *SGD) ZeroGrad() {
	for _, p := range o.params {
		p.ZeroGrad()
	}
}

// SetLR updates the learning rate (used by LR schedulers).
func (o *SGD) SetLR(lr float32) { o.lr = lr }

// GetLR returns the current learning rate.
func (o *SGD) GetLR() float32 { return o.lr }

// ---------- Adam ----------

// Adam implements the Adam optimizer (Kingma & Ba, 2014).
type Adam struct {
	params []*g.Tensor
	lr     float32
	beta1  float32
	beta2  float32
	eps    float32
	m      [][]float32 // first moment
	v      [][]float32 // second moment
	t      int         // timestep
}

// NewAdam creates an Adam optimizer with default betas (0.9, 0.999) and eps (1e-8).
func NewAdam(params []*g.Tensor, lr float32) *Adam {
	m := make([][]float32, len(params))
	v := make([][]float32, len(params))
	for i, p := range params {
		m[i] = make([]float32, p.Size())
		v[i] = make([]float32, p.Size())
	}
	return &Adam{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.999,
		eps:    1e-8,
		m:      m,
		v:      v,
	}
}

func (o *Adam) Step() {
	o.t++
	bc1 := 1 - float32(math.Pow(float64(o.beta1), float64(o.t)))
	bc2 := 1 - float32(math.Pow(float64(o.beta2), float64(o.t)))

	for i, p := range o.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		data := p.Data()
		gData := grad.Data()
		for j := range data {
			g := gData[j]
			o.m[i][j] = o.beta1*o.m[i][j] + (1-o.beta1)*g
			o.v[i][j] = o.beta2*o.v[i][j] + (1-o.beta2)*g*g

			mHat := o.m[i][j] / bc1
			vHat := o.v[i][j] / bc2
			data[j] -= o.lr * mHat / (float32(math.Sqrt(float64(vHat))) + o.eps)
		}
	}
}

func (o *Adam) ZeroGrad() {
	for _, p := range o.params {
		p.ZeroGrad()
	}
}

// SetLR updates the learning rate (used by LR schedulers).
func (o *Adam) SetLR(lr float32) { o.lr = lr }

// GetLR returns the current learning rate.
func (o *Adam) GetLR() float32 { return o.lr }
