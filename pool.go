//go:build darwin

package gorch

import "math"

// MaxPool2dForward applies 2D max pooling.
// input:  (batch, C, H, W)
// Returns: (batch, C, outH, outW)
// Also returns argmax indices for backward pass.
func MaxPool2dForward(input *Tensor, kernelSize, stride int) *Tensor {
	batch := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]
	outH := (H-kernelSize)/stride + 1
	outW := (W-kernelSize)/stride + 1

	outData := make([]float32, batch*C*outH*outW)
	// Store argmax indices for backward pass
	indices := make([]int, batch*C*outH*outW)

	for b := 0; b < batch; b++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					maxVal := float32(math.Inf(-1))
					maxIdx := 0
					for ky := 0; ky < kernelSize; ky++ {
						for kx := 0; kx < kernelSize; kx++ {
							ih := oh*stride + ky
							iw := ow*stride + kx
							idx := b*C*H*W + c*H*W + ih*W + iw
							if input.data[idx] > maxVal {
								maxVal = input.data[idx]
								maxIdx = idx
							}
						}
					}
					outIdx := b*C*outH*outW + c*outH*outW + oh*outW + ow
					outData[outIdx] = maxVal
					indices[outIdx] = maxIdx
				}
			}
		}
	}

	out := &Tensor{data: outData, shape: []int{batch, C, outH, outW}}

	if input.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "MaxPool2d",
			inputs: []*Tensor{input},
			backward: func(grad *Tensor) []*Tensor {
				dx := Zeros(input.shape...)
				for i, idx := range indices {
					dx.data[idx] += grad.data[i]
				}
				return []*Tensor{dx}
			},
		}
	}
	return out
}

// FlattenForward reshapes a (batch, C, H, W) tensor to (batch, C*H*W).
func FlattenForward(input *Tensor) *Tensor {
	batch := input.shape[0]
	features := 1
	for _, s := range input.shape[1:] {
		features *= s
	}

	// Data is already contiguous in memory, just change shape
	out := &Tensor{
		data:  make([]float32, len(input.data)),
		shape: []int{batch, features},
	}
	copy(out.data, input.data)

	if input.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Flatten",
			inputs: []*Tensor{input},
			backward: func(grad *Tensor) []*Tensor {
				dx := &Tensor{
					data:  make([]float32, len(grad.data)),
					shape: copyShape(input.shape),
				}
				copy(dx.data, grad.data)
				return []*Tensor{dx}
			},
		}
	}
	return out
}
