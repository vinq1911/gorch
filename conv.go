//go:build darwin

package gorch

import (
	"github.com/vinq1911/gorch/accelerate"
)

// im2col extracts image patches into columns for GEMM-based convolution.
// Input: (C, H, W) stored as flat slice in CHW order.
// Output: (C*kH*kW, outH*outW) written into the provided col buffer.
//
// This is the core data rearrangement that turns convolution into matrix multiply.
// Uses a tiled approach: the col buffer only needs to hold the full expansion
// for the given input, but is pre-allocated and reused across forward calls.
func im2col(input []float32, C, H, W, kH, kW, stride, pad int, col []float32) {
	outH := (H+2*pad-kH)/stride + 1
	outW := (W+2*pad-kW)/stride + 1
	colIdx := 0

	for c := 0; c < C; c++ {
		for ky := 0; ky < kH; ky++ {
			for kx := 0; kx < kW; kx++ {
				for oh := 0; oh < outH; oh++ {
					ih := oh*stride - pad + ky
					for ow := 0; ow < outW; ow++ {
						iw := ow*stride - pad + kx
						if ih >= 0 && ih < H && iw >= 0 && iw < W {
							col[colIdx] = input[c*H*W+ih*W+iw]
						} else {
							col[colIdx] = 0 // zero padding
						}
						colIdx++
					}
				}
			}
		}
	}
}

// col2im accumulates column data back into an image gradient.
// This is the reverse of im2col, used in the backward pass.
// Gradients are accumulated (not overwritten) into dx.
func col2im(col []float32, C, H, W, kH, kW, stride, pad int, dx []float32) {
	outH := (H+2*pad-kH)/stride + 1
	outW := (W+2*pad-kW)/stride + 1
	colIdx := 0

	for c := 0; c < C; c++ {
		for ky := 0; ky < kH; ky++ {
			for kx := 0; kx < kW; kx++ {
				for oh := 0; oh < outH; oh++ {
					ih := oh*stride - pad + ky
					for ow := 0; ow < outW; ow++ {
						iw := ow*stride - pad + kx
						if ih >= 0 && ih < H && iw >= 0 && iw < W {
							dx[c*H*W+ih*W+iw] += col[colIdx]
						}
						colIdx++
					}
				}
			}
		}
	}
}

// Conv2dForward computes convolution: output = input * weight + bias.
// input:  (batch, inC, H, W)
// weight: (outC, inC, kH, kW)
// bias:   (outC,) or nil
// Returns: (batch, outC, outH, outW)
//
// Uses im2col + BLAS sgemm. For 1x1 convolutions, skips im2col entirely.
func Conv2dForward(input *Tensor, weight *Tensor, bias *Tensor, stride, pad int) *Tensor {
	batch := input.shape[0]
	inC := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]
	outC := weight.shape[0]
	kH := weight.shape[2]
	kW := weight.shape[3]
	outH := (H+2*pad-kH)/stride + 1
	outW := (W+2*pad-kW)/stride + 1

	outData := make([]float32, batch*outC*outH*outW)

	// Weight reshaped: (outC, inC*kH*kW) — it's already stored this way
	wData := weight.data
	M := outC            // rows of weight matrix
	K := inC * kH * kW   // columns of weight matrix = rows of col matrix
	N := outH * outW     // columns of col matrix = spatial output size

	// 1x1 optimization: skip im2col, input is already in GEMM-ready shape
	is1x1 := kH == 1 && kW == 1 && stride == 1 && pad == 0

	// Pre-allocate scratch buffer for im2col (reused per sample)
	var colBuf []float32
	if !is1x1 {
		colBuf = make([]float32, K*N)
	}

	for b := 0; b < batch; b++ {
		inputSample := input.data[b*inC*H*W : (b+1)*inC*H*W]
		outputSample := outData[b*outC*outH*outW : (b+1)*outC*outH*outW]

		var colData []float32
		if is1x1 {
			// 1x1: input is already (inC, H*W) = (K, N), no expansion needed
			colData = inputSample
		} else {
			im2col(inputSample, inC, H, W, kH, kW, stride, pad, colBuf)
			colData = colBuf
		}

		// output = weight @ col  =>  (M, K) @ (K, N) = (M, N)
		accelerate.Sgemm(M, N, K, 1.0, wData, colData, 0.0, outputSample)

		// Fuse bias addition (ADR-003: single memory pass)
		if bias != nil {
			bData := bias.data
			for oc := 0; oc < outC; oc++ {
				row := outputSample[oc*outH*outW : (oc+1)*outH*outW]
				for i := range row {
					row[i] += bData[oc]
				}
			}
		}
	}

	out := &Tensor{data: outData, shape: []int{batch, outC, outH, outW}}

	// Autograd
	if input.requiresGrad || weight.requiresGrad || (bias != nil && bias.requiresGrad) {
		out.requiresGrad = true
		inputs := []*Tensor{input, weight}
		if bias != nil {
			inputs = append(inputs, bias)
		}

		out.gradFn = &GradFn{
			name:   "Conv2d",
			inputs: inputs,
			backward: func(grad *Tensor) []*Tensor {
				return conv2dBackward(grad, input, weight, bias, stride, pad)
			},
		}
	}
	return out
}

// conv2dBackward computes gradients for Conv2d.
func conv2dBackward(gradOutput *Tensor, input *Tensor, weight *Tensor, bias *Tensor, stride, pad int) []*Tensor {
	batch := input.shape[0]
	inC := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]
	outC := weight.shape[0]
	kH := weight.shape[2]
	kW := weight.shape[3]
	outH := gradOutput.shape[2]
	outW := gradOutput.shape[3]

	M := outC
	K := inC * kH * kW
	N := outH * outW

	is1x1 := kH == 1 && kW == 1 && stride == 1 && pad == 0

	var dInput *Tensor
	if input.requiresGrad {
		dInput = Zeros(input.shape...)
	}
	dWeight := Zeros(weight.shape...)

	var dBias *Tensor
	if bias != nil && bias.requiresGrad {
		dBias = Zeros(bias.shape...)
	}

	var colBuf []float32
	var dcolBuf []float32
	if !is1x1 {
		colBuf = make([]float32, K*N)
		if input.requiresGrad {
			dcolBuf = make([]float32, K*N)
		}
	}

	for b := 0; b < batch; b++ {
		gradSample := gradOutput.data[b*outC*outH*outW : (b+1)*outC*outH*outW]
		inputSample := input.data[b*inC*H*W : (b+1)*inC*H*W]

		// Get col data for this sample
		var colData []float32
		if is1x1 {
			colData = inputSample
		} else {
			im2col(inputSample, inC, H, W, kH, kW, stride, pad, colBuf)
			colData = colBuf
		}

		// dWeight += gradSample @ col^T  =>  (M, N) @ (N, K) = (M, K)
		// Using transB: gradSample(M,N) @ colData^T => SgemmTransB with B=colData(K,N)...
		// Actually: dW += grad @ col^T. grad is (M,N), col is (K,N), col^T is (N,K).
		// So dW(M,K) = grad(M,N) @ col^T(N,K) => standard Sgemm with transB on col.
		// But col is stored as (K,N), and we want col^T = (N,K).
		// SgemmTransB(M, K, N, ..., grad, col, ...) treats col as (K, N) and transposes it.
		accelerate.SgemmTransB(M, K, N, 1.0, gradSample, colData, 1.0, dWeight.data)

		// dInput: dcol = weight^T @ gradSample => (K, M) @ (M, N) = (K, N)
		if input.requiresGrad {
			if is1x1 {
				dcolBuf = dInput.data[b*inC*H*W : (b+1)*inC*H*W]
				accelerate.SgemmTransA(K, N, M, 1.0, weight.data, gradSample, 0.0, dcolBuf)
			} else {
				// Clear dcol buffer
				for i := range dcolBuf {
					dcolBuf[i] = 0
				}
				accelerate.SgemmTransA(K, N, M, 1.0, weight.data, gradSample, 0.0, dcolBuf)
				// col2im: accumulate dcol back into dInput
				col2im(dcolBuf, inC, H, W, kH, kW, stride, pad,
					dInput.data[b*inC*H*W:(b+1)*inC*H*W])
			}
		}

		// dBias = sum over spatial dims of gradOutput
		if dBias != nil {
			for oc := 0; oc < outC; oc++ {
				for i := 0; i < outH*outW; i++ {
					dBias.data[oc] += gradSample[oc*outH*outW+i]
				}
			}
		}
	}

	results := []*Tensor{dInput, dWeight}
	if dBias != nil {
		results = append(results, dBias)
	} else if bias != nil {
		results = append(results, Zeros(bias.shape...))
	}
	return results
}
