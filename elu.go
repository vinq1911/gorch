//go:build darwin

package gorch

import "math"

// ELU returns the Exponential Linear Unit activation with alpha=1:
//
//	elu(x)  = x            if x > 0
//	        = exp(x) - 1   otherwise
//	elu'(x) = 1            if x > 0
//	        = exp(x)       otherwise (= elu(x) + 1)
//
// Mimi's SEANet encoder uses ELU between every conv.
func ELU(a *Tensor) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(ELU(promoteToF32(a)))
	}
	out := Zeros(a.shape...)
	for i, x := range a.data {
		if x > 0 {
			out.data[i] = x
		} else {
			out.data[i] = float32(math.Exp(float64(x))) - 1
		}
	}

	if GradEnabled() && a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "ELU",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(a.shape...)
				for i, x := range a.data {
					if x > 0 {
						dx.data[i] = grad.data[i]
					} else {
						// elu'(x) = exp(x) = elu(x) + 1
						dx.data[i] = grad.data[i] * (out.data[i] + 1)
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

const invSqrt2 = 0.7071067811865476   // 1/sqrt(2)
const invSqrt2Pi = 0.3989422804014327 // 1/sqrt(2*pi)

// GELUErf returns the exact Gaussian Error Linear Unit:
//
//	gelu(x) = 0.5 * x * (1 + erf(x/sqrt(2))) = x * Phi(x)
//
// This is PyTorch's GELU with approximate='none' — what Mimi's
// transformer ("gelu" hidden_act) uses. It differs from gorch's GELU
// (tanh approximation) by up to ~3e-4 per activation, enough to matter
// for tight end-to-end parity budgets.
//
//	gelu'(x) = Phi(x) + x * phi(x),  phi(x) = exp(-x^2/2)/sqrt(2*pi)
func GELUErf(a *Tensor) *Tensor {
	if a.dtype == BFloat16 {
		return downcastToBF16(GELUErf(promoteToF32(a)))
	}
	out := Zeros(a.shape...)
	for i, x := range a.data {
		phi := 0.5 * (1 + math.Erf(float64(x)*invSqrt2))
		out.data[i] = x * float32(phi)
	}

	if GradEnabled() && a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "GELUErf",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(a.shape...)
				for i, x := range a.data {
					xf := float64(x)
					cdf := 0.5 * (1 + math.Erf(xf*invSqrt2))
					pdf := invSqrt2Pi * math.Exp(-0.5*xf*xf)
					dx.data[i] = grad.data[i] * float32(cdf+xf*pdf)
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}
