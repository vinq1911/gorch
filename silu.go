//go:build darwin

package gorch

import (
	"github.com/vinq1911/gorch/accelerate"
	"github.com/vinq1911/gorch/metal"
)

// SiLU / SwiGLU — plan 0009 K4 (X2 kernel set).
//
// The SwiGLU FFN activation was a pure-Go scalar loop with a per-call
// math.Exp — 0.97 s/step at seq 1500 in the X0 per-op table, and 14%
// of the remaining CPU samples in the X1K1 profile. Both ops now
// dispatch element-wise Metal kernels (vec_silu / vec_swiglu + _bwd,
// metal/kernels.go) when the input is Metal-resident, and a vectorized
// Accelerate path (acc_vsilu / acc_vswiglu + _bwd: one vForce sigmoid
// pass + one auto-vectorized combine loop) otherwise. Backward
// recomputes σ(x) instead of caching a sigmoid tensor per layer —
// one exp per lane beats keeping a (seq, 3072) cache alive in the
// autograd graph.
//
// Golden tests: silu_metal_test.go (kernel vs CPU reference at 1e-3
// abs per the plan tolerance table + numerical-grad 1e-2).

// siluPipelinesReady reports whether the K4 kernels were compiled.
func siluPipelinesReady() bool {
	if gpu == nil {
		return false
	}
	for _, k := range []string{"vec_silu", "vec_silu_bwd", "vec_swiglu", "vec_swiglu_bwd"} {
		if _, ok := gpu.pipelines[k]; !ok {
			return false
		}
	}
	return true
}

// SiLU returns x * sigmoid(x) element-wise (also called Swish).
// Used by Llama, Mistral, OpenMythos and most modern transformers
// inside the SwiGLU FFN.
//
//	silu(x) = x * σ(x)
//	silu'(x) = σ(x) * (1 + x * (1 - σ(x)))
func SiLU(a *Tensor) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(SiLU(promoteToF32(a)))
	}
	out := ZerosLike(a, a.shape...)

	if a.buf != nil && out.buf != nil && siluPipelinesReady() {
		gpu.Queue.Dispatch1D(gpu.pipe("vec_silu"), []*metal.Buffer{a.buf, out.buf}, a.Size())
		metalSiluDispatches.Add(1)
	} else {
		syncForCPU(a)
		accelerate.SiLU(a.data, out.data)
	}

	if GradEnabled() && a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "SiLU",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := zerosLikeEither(a.shape, grad, a)
				if a.buf != nil && grad.buf != nil && dx.buf != nil && siluPipelinesReady() {
					gpu.Queue.Dispatch1D(gpu.pipe("vec_silu_bwd"),
						[]*metal.Buffer{a.buf, grad.buf, dx.buf}, a.Size())
					metalSiluDispatches.Add(1)
				} else {
					syncForCPU(a, grad)
					accelerate.SiLUBwd(a.data, grad.data, dx.data)
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// SwiGLU is the gated activation used in Llama-style FFNs:
//
//	swiglu(gate, value) = SiLU(gate) * value
//	                    = gate * σ(gate) * value
//
// Typically the FFN has two parallel projections W_gate and W_up
// over the same input; this op fuses the gating into a single pass
// to avoid materialising the SiLU intermediate.
//
// Backward:
//
//	d/dgate  = value * σ(gate) * (1 + gate * (1 - σ(gate)))
//	d/dvalue = gate * σ(gate) = SiLU(gate)
func SwiGLU(gate, value *Tensor) *Tensor {
	if !sameShape(gate.shape, value.shape) {
		panic("gorch: SwiGLU requires gate and value to have the same shape")
	}
	requireSameDtype(gate, value, "SwiGLU")
	if gate.dtype == BFloat16 {
		return downcastToBF16(SwiGLU(promoteToF32(gate), promoteToF32(value)))
	}
	out := zerosLikeEither(gate.shape, gate, value)

	if gate.buf != nil && value.buf != nil && out.buf != nil && siluPipelinesReady() {
		gpu.Queue.Dispatch1D(gpu.pipe("vec_swiglu"),
			[]*metal.Buffer{gate.buf, value.buf, out.buf}, gate.Size())
		metalSiluDispatches.Add(1)
	} else {
		syncForCPU(gate, value)
		accelerate.SwiGLU(gate.data, value.data, out.data)
	}

	if GradEnabled() && (gate.requiresGrad || value.requiresGrad) {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "SwiGLU",
			inputs: []*Tensor{gate, value},
			backward: func(grad *Tensor) []*Tensor {
				dGate := zerosLikeEither(gate.shape, grad, gate)
				dValue := zerosLikeEither(value.shape, grad, value)
				if gate.buf != nil && value.buf != nil && grad.buf != nil &&
					dGate.buf != nil && dValue.buf != nil && siluPipelinesReady() {
					gpu.Queue.Dispatch1D(gpu.pipe("vec_swiglu_bwd"),
						[]*metal.Buffer{gate.buf, value.buf, grad.buf, dGate.buf, dValue.buf},
						gate.Size())
					metalSiluDispatches.Add(1)
				} else {
					syncForCPU(gate, value, grad)
					accelerate.SwiGLUBwd(gate.data, value.data, grad.data, dGate.data, dValue.data)
				}
				return []*Tensor{dGate, dValue}
			},
		}
	}
	return out
}
