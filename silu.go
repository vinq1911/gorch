//go:build darwin

package gorch

import "math"

// SiLU returns x * sigmoid(x) element-wise (also called Swish).
// Used by Llama, Mistral, OpenMythos and most modern transformers
// inside the SwiGLU FFN.
//
//	silu(x) = x * σ(x)
//	silu'(x) = σ(x) * (1 + x * (1 - σ(x)))
func SiLU(a *Tensor) *Tensor {
	out := Zeros(a.shape...)

	// Cache sigmoid for backward — SiLU's derivative needs it.
	// In NoGrad mode we skip the cache (no graph being built).
	needsBackward := GradEnabled() && a.requiresGrad
	var sig []float32
	if needsBackward {
		sig = make([]float32, len(a.data))
	}

	for i, x := range a.data {
		s := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		out.data[i] = x * s
		if needsBackward {
			sig[i] = s
		}
	}

	if needsBackward {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "SiLU",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(a.shape...)
				for i, x := range a.data {
					s := sig[i]
					dx.data[i] = grad.data[i] * s * (1 + x*(1-s))
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
	out := Zeros(gate.shape...)

	needsBackward := GradEnabled() && (gate.requiresGrad || value.requiresGrad)
	var sig []float32
	if needsBackward {
		sig = make([]float32, len(gate.data))
	}

	for i, g := range gate.data {
		s := float32(1.0 / (1.0 + math.Exp(float64(-g))))
		out.data[i] = g * s * value.data[i]
		if needsBackward {
			sig[i] = s
		}
	}

	if needsBackward {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "SwiGLU",
			inputs: []*Tensor{gate, value},
			backward: func(grad *Tensor) []*Tensor {
				dGate := Zeros(gate.shape...)
				dValue := Zeros(value.shape...)
				for i := range gate.data {
					gx := gate.data[i]
					vx := value.data[i]
					s := sig[i]
					gi := grad.data[i]
					// dy/dgate = value * σ(gate) * (1 + gate*(1 - σ(gate)))
					dGate.data[i] = gi * vx * s * (1 + gx*(1-s))
					// dy/dvalue = gate * σ(gate)
					dValue.data[i] = gi * gx * s
				}
				return []*Tensor{dGate, dValue}
			},
		}
	}
	return out
}
