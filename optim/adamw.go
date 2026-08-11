//go:build darwin

package optim

import (
	"fmt"
	"math"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
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
	lrMul       []float32 // per-param multiplier on lr (param groups)
	beta1       float32
	beta2       float32
	eps         float32
	weightDecay float32
	m           [][]float32 // first moment
	v           [][]float32 // second moment
	t           int         // timestep
}

// ParamGroup pairs a parameter set with its own learning rate
// (plan 0008 §4.3: LoRA factors and Ext embedding rows train at
// different LRs). The FIRST group's LR is the base rate: a scheduler
// driving SetLR rescales every group proportionally, keeping the
// groups' LR ratios fixed.
type ParamGroup struct {
	Params []*g.Tensor
	LR     float32
}

// NewAdamW creates an AdamW optimizer with default betas (0.9, 0.999),
// eps 1e-8, and the supplied weight-decay coefficient. PyTorch's
// default is 0.01.
func NewAdamW(params []*g.Tensor, lr, weightDecay float32) *AdamW {
	return NewAdamWGroups([]ParamGroup{{Params: params, LR: lr}}, weightDecay)
}

// NewAdamWGroups creates an AdamW over several parameter groups, each
// with its own learning rate. groups[0].LR is the base rate (SetLR /
// GetLR operate on it); every other group's effective rate scales
// proportionally when a scheduler updates the base.
func NewAdamWGroups(groups []ParamGroup, weightDecay float32) *AdamW {
	if len(groups) == 0 || groups[0].LR == 0 {
		panic("optim: NewAdamWGroups needs ≥1 group with a non-zero base LR")
	}
	base := groups[0].LR
	var params []*g.Tensor
	var lrMul []float32
	for _, grp := range groups {
		for _, p := range grp.Params {
			params = append(params, p)
			lrMul = append(lrMul, grp.LR/base)
		}
	}
	m := make([][]float32, len(params))
	v := make([][]float32, len(params))
	for i, p := range params {
		m[i] = make([]float32, p.Size())
		v[i] = make([]float32, p.Size())
	}
	return &AdamW{
		params:      params,
		lr:          base,
		lrMul:       lrMul,
		beta1:       0.9,
		beta2:       0.999,
		eps:         1e-8,
		weightDecay: weightDecay,
		m:           m,
		v:           v,
	}
}

// UseScalarAdamW forces the pre-plan-0009-K7 scalar Go update loop.
// The vectorized Accelerate step (acc_adamw_step) is the default; the
// scalar loop stays as the numerical reference for the K7 trajectory-
// parity gate (loss curve over 20 synthetic steps within 1e-6).
var UseScalarAdamW = false

// Step applies one AdamW update.
//
// Plan 0009 K7: the per-element update runs through a single fused
// Accelerate/clang-vectorized C loop (accelerate.AdamWStep) instead of
// the scalar Go loop — the scalar path was 2.4 s/step at the workload's
// 172M-param embedding table + 28×15.7M block params (X0 per-op table).
// Math is identical (bias-corrected moments, decoupled weight decay);
// the scalar loop below is kept as the parity oracle behind
// UseScalarAdamW.
func (o *AdamW) Step() {
	o.t++
	bc1 := 1 - float32(math.Pow(float64(o.beta1), float64(o.t)))
	bc2 := 1 - float32(math.Pow(float64(o.beta2), float64(o.t)))

	for i, p := range o.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		if p.Dtype() != g.Float32 {
			// bf16 params have nil Data(); the moment slices are keyed to
			// f32 storage. Fail loudly instead of silently skipping
			// (plan 0009 §3.4 item B3).
			panic("optim: AdamW.Step on a non-f32 parameter — bf16 params must stay out of the optimizer (plan 0009 X3-B3)")
		}
		lr := o.lr * o.lrMul[i]
		data := p.Data()
		gData := grad.Data()

		if !UseScalarAdamW {
			accelerate.AdamWStep(data, gData, o.m[i], o.v[i],
				lr, o.beta1, o.beta2, o.eps, o.weightDecay, bc1, bc2)
			continue
		}

		for j := range data {
			gj := gData[j]
			o.m[i][j] = o.beta1*o.m[i][j] + (1-o.beta1)*gj
			o.v[i][j] = o.beta2*o.v[i][j] + (1-o.beta2)*gj*gj

			mHat := o.m[i][j] / bc1
			vHat := o.v[i][j] / bc2

			// Decoupled weight decay: applied directly to the param,
			// NOT folded into the gradient. This is the AdamW core idea.
			data[j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+o.eps) + o.weightDecay*data[j])
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

// StateTensors exposes the optimizer state for checkpointing
// (plan 0008 §3.4): the timestep and the live m/v moment slices, in
// parameter order. The slices are NOT copies — checkpoint writers
// must serialise them before the next Step.
func (o *AdamW) StateTensors() (step int, m, v [][]float32) {
	return o.t, o.m, o.v
}

// LoadState restores optimizer state saved via StateTensors. Lengths
// must match the current parameter list exactly.
func (o *AdamW) LoadState(step int, m, v [][]float32) error {
	if len(m) != len(o.params) || len(v) != len(o.params) {
		return fmt.Errorf("optim: AdamW.LoadState got %d/%d moment slices for %d params",
			len(m), len(v), len(o.params))
	}
	for i, p := range o.params {
		if len(m[i]) != p.Size() || len(v[i]) != p.Size() {
			return fmt.Errorf("optim: AdamW.LoadState param %d: moment sizes %d/%d != param size %d",
				i, len(m[i]), len(v[i]), p.Size())
		}
	}
	o.t = step
	for i := range o.m {
		copy(o.m[i], m[i])
		copy(o.v[i], v[i])
	}
	return nil
}
