//go:build darwin

package optim

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// AdamW implements the AdamW optimizer (Loshchilov & Hutter 2019).
//
// AdamW differs from gorch's existing Adam by *decoupling* weight decay
// from the gradient: instead of `grad = grad + wd * param` (the L2-
// regularisation form Adam tries and fails to do correctly because the
// `m`/`v` moving averages distort the decay), AdamW applies the decay
// directly to the parameter:
//
//	param ← param - lr * wd * param
//	         - lr * mHat / (sqrt(vHat) + eps)
//
// This is the default optimiser for Llama, Mistral, OpenMythos, and
// every other modern transformer training run. Adam alone is wrong for
// these models — the weight-decay regularisation is doing nothing
// useful when fused into the gradient.
//
// Plan 0001 Phase 1 item 12; called out in `0003-gemini-review.md` as
// missed by the external advisory and blocking for any serious training.
type AdamW struct {
	params      []*g.Tensor
	lr          float32
	beta1       float32
	beta2       float32
	eps         float32
	weightDecay float32
	m           [][]float32 // first moment
	v           [][]float32 // second moment
	t           int         // timestep
}

// NewAdamW creates an AdamW optimizer with default betas (0.9, 0.999),
// eps 1e-8, and the supplied weight-decay coefficient. PyTorch's
// default is 0.01.
func NewAdamW(params []*g.Tensor, lr, weightDecay float32) *AdamW {
	m := make([][]float32, len(params))
	v := make([][]float32, len(params))
	for i, p := range params {
		m[i] = make([]float32, p.Size())
		v[i] = make([]float32, p.Size())
	}
	return &AdamW{
		params:      params,
		lr:          lr,
		beta1:       0.9,
		beta2:       0.999,
		eps:         1e-8,
		weightDecay: weightDecay,
		m:           m,
		v:           v,
	}
}

// Step applies one AdamW update.
func (o *AdamW) Step() {
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
			gj := gData[j]
			o.m[i][j] = o.beta1*o.m[i][j] + (1-o.beta1)*gj
			o.v[i][j] = o.beta2*o.v[i][j] + (1-o.beta2)*gj*gj

			mHat := o.m[i][j] / bc1
			vHat := o.v[i][j] / bc2

			// Decoupled weight decay: applied directly to the param,
			// NOT folded into the gradient. This is the AdamW core idea.
			data[j] -= o.lr * (mHat/(float32(math.Sqrt(float64(vHat)))+o.eps) + o.weightDecay*data[j])
		}
	}
}

// ZeroGrad clears gradients on all tracked parameters.
func (o *AdamW) ZeroGrad() {
	for _, p := range o.params {
		p.ZeroGrad()
	}
}

// SetLR updates the learning rate (used by LR schedulers).
func (o *AdamW) SetLR(lr float32) { o.lr = lr }

// GetLR returns the current learning rate.
func (o *AdamW) GetLR() float32 { return o.lr }

// SetWeightDecay updates the weight-decay coefficient mid-training (used
// by some warmup-then-decay schedules).
func (o *AdamW) SetWeightDecay(wd float32) { o.weightDecay = wd }
