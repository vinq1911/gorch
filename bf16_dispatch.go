//go:build darwin

package gorch

import "fmt"

// This file is the bf16 op-dispatch glue. Plan 0002 PR 2.
//
// Every existing F32 op grows a tiny dispatch block at the top of the
// form:
//
//	if a.dtype == BFloat16 {
//	    return downcastToBF16(OP(promoteToF32(a), ...))
//	}
//
// which routes bf16 inputs through upcast → run the existing f32
// implementation → downcast the result. Autograd is preserved through
// the upcast/downcast wrappers below, which install their own grad fns
// so gradient flows from the bf16 output, through the f32 inner op's
// graph, and back out to the bf16 source tensors.
//
// Mixed-dtype inputs panic. PyTorch promotes; gorch (for now) requires
// the caller to be explicit. This keeps tests and accounting simple
// while higher-priority bf16 plumbing lands. Promotion can be added
// later if a real use case needs it.

// upcastBF16 returns a fresh F32 tensor with t's bf16 values widened
// to f32 (lossless). Autograd: dL/dt_bf16 = round(dL/dt_f32 → bf16),
// so a gradient flowing back through this wrapper reaches the bf16
// source rounded to bf16 storage.
func upcastBF16(t *Tensor) *Tensor {
	if t.dtype != BFloat16 {
		panic("gorch: upcastBF16 expects a BF16 tensor")
	}
	out := &Tensor{
		dtype: Float32,
		data:  BF16ToF32Slice(t.data16),
		shape: copyShape(t.shape),
	}
	if GradEnabled() && t.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "UpcastBF16",
			inputs: []*Tensor{t},
			backward: func(grad *Tensor) []*Tensor {
				// grad arrives in f32; round it back to bf16 for the
				// bf16 source. The rounding loss is the bf16 storage
				// contract — same as PyTorch's bf16 mixed-precision.
				dt := &Tensor{
					dtype:  BFloat16,
					data16: F32ToBF16Slice(grad.data),
					shape:  copyShape(grad.shape),
				}
				return []*Tensor{dt}
			},
		}
	}
	return out
}

// downcastToBF16 returns a fresh BF16 tensor with t's f32 values
// rounded to bf16. Autograd: dL/dt_f32 = upcast(dL/dt_bf16), which is
// lossless (bf16 → f32 is a bit-shift).
func downcastToBF16(t *Tensor) *Tensor {
	if t.dtype != Float32 {
		panic("gorch: downcastToBF16 expects an F32 tensor")
	}
	out := &Tensor{
		dtype:  BFloat16,
		data16: F32ToBF16Slice(t.data),
		shape:  copyShape(t.shape),
	}
	if GradEnabled() && t.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "DowncastToBF16",
			inputs: []*Tensor{t},
			backward: func(grad *Tensor) []*Tensor {
				// grad is bf16; widen to f32 (lossless) for the f32
				// source.
				dt := &Tensor{
					dtype: Float32,
					data:  BF16ToF32Slice(grad.data16),
					shape: copyShape(grad.shape),
				}
				return []*Tensor{dt}
			},
		}
	}
	return out
}

// promoteToF32 returns t unchanged if it's already f32, or an
// autograd-aware upcast if bf16. Used inside op dispatch wrappers so
// the inner f32 op runs over f32 tensors regardless of the user's
// dtype choice.
func promoteToF32(t *Tensor) *Tensor {
	if t == nil || t.dtype == Float32 {
		return t
	}
	return upcastBF16(t)
}

// requireSameDtype panics if a and b have different dtypes. Mixed
// dtype is not supported in PR 2 — callers must explicitly upcast or
// downcast one side first.
func requireSameDtype(a, b *Tensor, opName string) {
	if a.dtype != b.dtype {
		panic(fmt.Sprintf("gorch: %s dtype mismatch: %s vs %s", opName, a.dtype, b.dtype))
	}
}
