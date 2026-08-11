//go:build darwin

package gorch

import (
	"fmt"

	"github.com/vinq1911/gorch/accelerate"
)

// PadMode selects how Conv1dForward fills the padded region.
type PadMode int

const (
	// PadConstant pads with zeros (PyTorch pad_mode="constant").
	PadConstant PadMode = iota
	// PadReplicate repeats the edge sample into the padded region
	// (PyTorch pad_mode="replicate"). Mimi's downsample conv uses this.
	PadReplicate
)

// im2col1d extracts sliding windows into columns for GEMM-based 1-D
// convolution, with dilation support.
// Input: (C, L) stored as a flat slice in CL order (already padded —
// callers apply padding before im2col, so no bounds checks are needed).
// Output: (C*k, outL) written into the provided col buffer.
func im2col1d(input []float32, C, L, k, stride, dilation int, col []float32) {
	kEff := (k-1)*dilation + 1
	outL := (L-kEff)/stride + 1
	if stride == 1 {
		// Each (channel, tap) row of col is a contiguous slice of the
		// input — one memmove instead of outL scalar copies. The
		// stride-1 convs are the largest im2col volumes in the Mimi
		// SEANet encoder (k3/k1 resnet convs at 24 kHz lengths).
		colIdx := 0
		for c := 0; c < C; c++ {
			base := c * L
			for kx := 0; kx < k; kx++ {
				off := base + kx*dilation
				copy(col[colIdx:colIdx+outL], input[off:off+outL])
				colIdx += outL
			}
		}
		return
	}
	colIdx := 0
	for c := 0; c < C; c++ {
		base := c * L
		for kx := 0; kx < k; kx++ {
			off := base + kx*dilation
			for o := 0; o < outL; o++ {
				col[colIdx] = input[off+o*stride]
				colIdx++
			}
		}
	}
}

// col2im1d accumulates column data back into an input gradient.
// This is the reverse of im2col1d, used in the backward pass.
// Gradients are accumulated (not overwritten) into dx, which has
// the padded length L.
func col2im1d(col []float32, C, L, k, stride, dilation int, dx []float32) {
	kEff := (k-1)*dilation + 1
	outL := (L-kEff)/stride + 1
	colIdx := 0
	for c := 0; c < C; c++ {
		base := c * L
		for kx := 0; kx < k; kx++ {
			off := base + kx*dilation
			for o := 0; o < outL; o++ {
				dx[off+o*stride] += col[colIdx]
				colIdx++
			}
		}
	}
}

// pad1d copies one (C, L) sample into dst with asymmetric padding on
// the length axis: dst is (C, L+padLeft+padRight). PadConstant fills
// the pad region with zeros; PadReplicate repeats each channel's edge
// sample.
func pad1d(src []float32, C, L, padLeft, padRight int, mode PadMode, dst []float32) {
	Lp := L + padLeft + padRight
	for c := 0; c < C; c++ {
		srcRow := src[c*L : (c+1)*L]
		dstRow := dst[c*Lp : (c+1)*Lp]
		copy(dstRow[padLeft:padLeft+L], srcRow)
		var left, right float32
		if mode == PadReplicate {
			left = srcRow[0]
			right = srcRow[L-1]
		}
		for i := 0; i < padLeft; i++ {
			dstRow[i] = left
		}
		for i := 0; i < padRight; i++ {
			dstRow[padLeft+L+i] = right
		}
	}
}

// Conv1dForward computes 1-D convolution (cross-correlation, like
// PyTorch's Conv1d):
//
//	input:  (batch, inC, L)
//	weight: (outC, inC, k)
//	bias:   (outC,) or nil
//
// Asymmetric padding (padLeft, padRight) is applied explicitly before
// im2col — this is what causal convolutions need. Returns
// (batch, outC, outL) with outL = (L+padLeft+padRight-kEff)/stride + 1
// where kEff = (k-1)*dilation + 1.
//
// Groups are intentionally not supported: the Mimi *encoder* never
// uses them (only the decoder's upsample conv has upsample_groups=512),
// and no other gorch path needs them yet.
func Conv1dForward(input, weight, bias *Tensor, stride, dilation, padLeft, padRight int, mode PadMode) *Tensor {
	syncForCPU(input, weight, bias)
	if len(input.shape) != 3 {
		panic(fmt.Sprintf("gorch: Conv1dForward input must be (batch, inC, L), got %v", input.shape))
	}
	if len(weight.shape) != 3 {
		panic(fmt.Sprintf("gorch: Conv1dForward weight must be (outC, inC, k), got %v", weight.shape))
	}
	batch := input.shape[0]
	inC := input.shape[1]
	L := input.shape[2]
	outC := weight.shape[0]
	k := weight.shape[2]
	if weight.shape[1] != inC {
		panic(fmt.Sprintf("gorch: Conv1dForward channel mismatch: input inC=%d, weight inC=%d", inC, weight.shape[1]))
	}
	if bias != nil && bias.Size() != outC {
		panic(fmt.Sprintf("gorch: Conv1dForward bias size %d != outC %d", bias.Size(), outC))
	}
	if stride < 1 || dilation < 1 || padLeft < 0 || padRight < 0 {
		panic("gorch: Conv1dForward requires stride>=1, dilation>=1, non-negative padding")
	}

	kEff := (k-1)*dilation + 1
	Lp := L + padLeft + padRight
	if Lp < kEff {
		panic(fmt.Sprintf("gorch: Conv1dForward padded length %d < effective kernel %d", Lp, kEff))
	}
	outL := (Lp-kEff)/stride + 1

	outData := make([]float32, batch*outC*outL)

	// Weight reshaped: (outC, inC*k) — it's already stored this way.
	M := outC
	K := inC * k
	N := outL

	// Scratch buffers come from the shared pool: the SEANet resnet
	// convs run im2col over ~184 MB at 24 kHz input lengths, and a
	// fresh allocation per call dominated page-fault time (madvise)
	// in the Mimi encode profile.
	needPad := padLeft > 0 || padRight > 0
	var padBuf []float32
	if needPad {
		padBuf = AcquireFloat32(inC * Lp)
		defer ReleaseFloat32(padBuf)
	}
	colBuf := AcquireFloat32(K * N)
	defer ReleaseFloat32(colBuf)

	for b := 0; b < batch; b++ {
		inputSample := input.data[b*inC*L : (b+1)*inC*L]
		outputSample := outData[b*outC*outL : (b+1)*outC*outL]

		padded := inputSample
		if needPad {
			pad1d(inputSample, inC, L, padLeft, padRight, mode, padBuf)
			padded = padBuf
		}
		im2col1d(padded, inC, Lp, k, stride, dilation, colBuf)

		// output = weight @ col  =>  (M, K) @ (K, N) = (M, N)
		accelerate.Sgemm(M, N, K, 1.0, weight.data, colBuf, 0.0, outputSample)

		// Fused bias addition.
		if bias != nil {
			bData := bias.data
			for oc := 0; oc < outC; oc++ {
				row := outputSample[oc*outL : (oc+1)*outL]
				for i := range row {
					row[i] += bData[oc]
				}
			}
		}
	}

	out := &Tensor{data: outData, shape: []int{batch, outC, outL}}

	// Autograd
	if GradEnabled() && (input.requiresGrad || weight.requiresGrad || (bias != nil && bias.requiresGrad)) {
		out.requiresGrad = true
		inputs := []*Tensor{input, weight}
		if bias != nil {
			inputs = append(inputs, bias)
		}
		out.gradFn = &GradFn{
			name:   "Conv1d",
			inputs: inputs,
			backward: func(grad *Tensor) []*Tensor {
				return conv1dBackward(grad, input, weight, bias, stride, dilation, padLeft, padRight, mode)
			},
		}
	}
	return out
}

// conv1dBackward computes gradients for Conv1dForward.
func conv1dBackward(gradOutput, input, weight, bias *Tensor, stride, dilation, padLeft, padRight int, mode PadMode) []*Tensor {
	syncForCPU(gradOutput, input, weight, bias)
	batch := input.shape[0]
	inC := input.shape[1]
	L := input.shape[2]
	outC := weight.shape[0]
	k := weight.shape[2]
	outL := gradOutput.shape[2]
	Lp := L + padLeft + padRight

	M := outC
	K := inC * k
	N := outL

	var dInput *Tensor
	if input.requiresGrad {
		dInput = Zeros(input.shape...)
	}
	dWeight := Zeros(weight.shape...)
	var dBias *Tensor
	if bias != nil && bias.requiresGrad {
		dBias = Zeros(bias.shape...)
	}

	needPad := padLeft > 0 || padRight > 0
	var padBuf []float32
	if needPad {
		padBuf = make([]float32, inC*Lp)
	}
	colBuf := make([]float32, K*N)
	var dcolBuf, dpadBuf []float32
	if dInput != nil {
		dcolBuf = make([]float32, K*N)
		if needPad {
			dpadBuf = make([]float32, inC*Lp)
		}
	}

	for b := 0; b < batch; b++ {
		gradSample := gradOutput.data[b*outC*outL : (b+1)*outC*outL]
		inputSample := input.data[b*inC*L : (b+1)*inC*L]

		padded := inputSample
		if needPad {
			pad1d(inputSample, inC, L, padLeft, padRight, mode, padBuf)
			padded = padBuf
		}
		im2col1d(padded, inC, Lp, k, stride, dilation, colBuf)

		// dWeight += grad @ col^T  =>  (M, N) @ (N, K) = (M, K)
		accelerate.SgemmTransB(M, K, N, 1.0, gradSample, colBuf, 1.0, dWeight.data)

		// dInput: dcol = weight^T @ grad  =>  (K, M) @ (M, N) = (K, N)
		if dInput != nil {
			accelerate.SgemmTransA(K, N, M, 1.0, weight.data, gradSample, 0.0, dcolBuf)
			dInputSample := dInput.data[b*inC*L : (b+1)*inC*L]
			if needPad {
				for i := range dpadBuf {
					dpadBuf[i] = 0
				}
				col2im1d(dcolBuf, inC, Lp, k, stride, dilation, dpadBuf)
				// Fold the padded-region gradient back into the sample:
				// constant pad contributes nothing; replicate pad routes
				// pad-region gradient to the edge samples it copied.
				for c := 0; c < inC; c++ {
					dpadRow := dpadBuf[c*Lp : (c+1)*Lp]
					dRow := dInputSample[c*L : (c+1)*L]
					for i := 0; i < L; i++ {
						dRow[i] += dpadRow[padLeft+i]
					}
					if mode == PadReplicate {
						for i := 0; i < padLeft; i++ {
							dRow[0] += dpadRow[i]
						}
						for i := 0; i < padRight; i++ {
							dRow[L-1] += dpadRow[padLeft+L+i]
						}
					}
				}
			} else {
				col2im1d(dcolBuf, inC, L, k, stride, dilation, dInputSample)
			}
		}

		// dBias = sum over output positions of gradOutput.
		if dBias != nil {
			for oc := 0; oc < outC; oc++ {
				for i := 0; i < outL; i++ {
					dBias.data[oc] += gradSample[oc*outL+i]
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
