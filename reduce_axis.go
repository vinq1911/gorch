//go:build darwin

package gorch

import "fmt"

// axisSplit computes the (outer, n, inner) decomposition for reducing
// over one axis of a's shape, plus the output shape with that axis
// removed (scalar results get shape (1,)).
func axisSplit(a *Tensor, axis int) (outer, n, inner int, outShape []int) {
	if axis < 0 || axis >= len(a.shape) {
		panic(fmt.Sprintf("gorch: axis %d out of range for shape %v", axis, a.shape))
	}
	outer, inner = 1, 1
	for i := 0; i < axis; i++ {
		outer *= a.shape[i]
	}
	n = a.shape[axis]
	for i := axis + 1; i < len(a.shape); i++ {
		inner *= a.shape[i]
	}
	outShape = make([]int, 0, len(a.shape)-1)
	outShape = append(outShape, a.shape[:axis]...)
	outShape = append(outShape, a.shape[axis+1:]...)
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	return outer, n, inner, outShape
}

// MeanAxis reduces one axis by arithmetic mean. The axis is removed
// from the output shape. Autograd broadcasts the gradient back along
// the reduced axis, scaled by 1/n.
func MeanAxis(a *Tensor, axis int) *Tensor {
	outer, n, inner, outShape := axisSplit(a, axis)
	out := Zeros(outShape...)
	syncForCPU(a)
	invN := 1 / float32(n)
	for o := 0; o < outer; o++ {
		for j := 0; j < n; j++ {
			src := a.data[(o*n+j)*inner : (o*n+j+1)*inner]
			dst := out.data[o*inner : (o+1)*inner]
			for i, v := range src {
				dst[i] += v
			}
		}
	}
	for i := range out.data {
		out.data[i] *= invN
	}

	if GradEnabled() && a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "MeanAxis",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				syncForCPU(a, grad)
				dx := Zeros(a.shape...)
				for o := 0; o < outer; o++ {
					gRow := grad.data[o*inner : (o+1)*inner]
					for j := 0; j < n; j++ {
						dst := dx.data[(o*n+j)*inner : (o*n+j+1)*inner]
						for i, g := range gRow {
							dst[i] = g * invN
						}
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// VarAxis reduces one axis by variance. With unbiased=true the sum of
// squared deviations is divided by n-1 (sample variance), otherwise by
// n (population variance — what Mimi's FSDD exporter pooling uses).
// The axis is removed from the output shape.
//
// Gradient: d var/d x_j = 2*(x_j - mean)/denom (the mean-dependence
// terms cancel), broadcast back along the reduced axis.
func VarAxis(a *Tensor, axis int, unbiased bool) *Tensor {
	outer, n, inner, outShape := axisSplit(a, axis)
	denom := n
	if unbiased {
		if n < 2 {
			panic("gorch: VarAxis with unbiased=true requires axis length >= 2")
		}
		denom = n - 1
	}
	invDenom := 1 / float32(denom)

	// Per-slice mean, kept for the backward closure.
	syncForCPU(a)
	mean := make([]float32, outer*inner)
	invN := 1 / float32(n)
	for o := 0; o < outer; o++ {
		for j := 0; j < n; j++ {
			src := a.data[(o*n+j)*inner : (o*n+j+1)*inner]
			dst := mean[o*inner : (o+1)*inner]
			for i, v := range src {
				dst[i] += v
			}
		}
	}
	for i := range mean {
		mean[i] *= invN
	}

	out := Zeros(outShape...)
	for o := 0; o < outer; o++ {
		mRow := mean[o*inner : (o+1)*inner]
		dst := out.data[o*inner : (o+1)*inner]
		for j := 0; j < n; j++ {
			src := a.data[(o*n+j)*inner : (o*n+j+1)*inner]
			for i, v := range src {
				d := v - mRow[i]
				dst[i] += d * d
			}
		}
	}
	for i := range out.data {
		out.data[i] *= invDenom
	}

	if GradEnabled() && a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "VarAxis",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				syncForCPU(a, grad)
				dx := Zeros(a.shape...)
				for o := 0; o < outer; o++ {
					gRow := grad.data[o*inner : (o+1)*inner]
					mRow := mean[o*inner : (o+1)*inner]
					for j := 0; j < n; j++ {
						src := a.data[(o*n+j)*inner : (o*n+j+1)*inner]
						dst := dx.data[(o*n+j)*inner : (o*n+j+1)*inner]
						for i, v := range src {
							dst[i] = gRow[i] * 2 * (v - mRow[i]) * invDenom
						}
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// MaxAxis reduces one axis by maximum. The axis is removed from the
// output shape. The gradient routes to the argmax position of each
// slice (first occurrence on ties), zero elsewhere.
func MaxAxis(a *Tensor, axis int) *Tensor {
	outer, n, inner, outShape := axisSplit(a, axis)
	out := Zeros(outShape...)
	syncForCPU(a)
	argmax := make([]int, outer*inner)
	for o := 0; o < outer; o++ {
		dst := out.data[o*inner : (o+1)*inner]
		arg := argmax[o*inner : (o+1)*inner]
		first := a.data[o*n*inner : o*n*inner+inner]
		copy(dst, first)
		for j := 1; j < n; j++ {
			src := a.data[(o*n+j)*inner : (o*n+j+1)*inner]
			for i, v := range src {
				if v > dst[i] {
					dst[i] = v
					arg[i] = j
				}
			}
		}
	}

	if GradEnabled() && a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "MaxAxis",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				syncForCPU(a, grad)
				dx := Zeros(a.shape...)
				for o := 0; o < outer; o++ {
					gRow := grad.data[o*inner : (o+1)*inner]
					arg := argmax[o*inner : (o+1)*inner]
					for i, g := range gRow {
						dx.data[(o*n+arg[i])*inner+i] = g
					}
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}
