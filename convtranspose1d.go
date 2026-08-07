//go:build darwin

package gorch

import (
	"fmt"

	"github.com/vinq1911/gorch/accelerate"
)

// ConvTranspose1dForward computes 1-D transposed convolution, matching
// PyTorch's conv_transpose1d with padding=0, output_padding=0,
// dilation=1:
//
//	input:  (batch, inC, L)
//	weight: (inC, outC/groups, k)   — PyTorch ConvTranspose layout,
//	                                  NOT Conv1d's (outC, inC, k)
//	bias:   (outC,) or nil
//
// Returns (batch, outC, (L-1)*stride + k). No padding or trimming is
// applied — callers (nn.CausalConvTranspose1d) trim. inC must be
// divisible by groups; outC is weight.shape[1] * groups.
//
// Per batch and group the forward pass is one SgemmTransA
// (col = Wg^T @ Xg) followed by the existing col2im1d scatter-add —
// transposed-conv forward IS col2im, which conv1d.go already ships for
// its backward pass. The depthwise case (groups == inC == outC, Mimi's
// upsample) takes a direct per-channel loop instead of degenerate
// K=1 GEMMs. Bias is fused after.
//
// Inference-only: no autograd graph is built, and the call panics if
// gradient tracking is enabled while any input requires grad. Decision
// (plan 0007 §2.1): the only consumer is the frozen-weight Mimi
// decoder, ConvTranspose autograd is out of scope, and silently
// producing a detached output would corrupt a training graph — so fail
// loudly instead.
func ConvTranspose1dForward(input, weight, bias *Tensor, stride, groups int) *Tensor {
	if len(input.shape) != 3 {
		panic(fmt.Sprintf("gorch: ConvTranspose1dForward input must be (batch, inC, L), got %v", input.shape))
	}
	if len(weight.shape) != 3 {
		panic(fmt.Sprintf("gorch: ConvTranspose1dForward weight must be (inC, outC/groups, k), got %v", weight.shape))
	}
	batch := input.shape[0]
	inC := input.shape[1]
	L := input.shape[2]
	outCg := weight.shape[1]
	k := weight.shape[2]
	if stride < 1 {
		panic("gorch: ConvTranspose1dForward requires stride >= 1")
	}
	if groups < 1 || inC%groups != 0 {
		panic(fmt.Sprintf("gorch: ConvTranspose1dForward inC=%d not divisible by groups=%d", inC, groups))
	}
	if weight.shape[0] != inC {
		panic(fmt.Sprintf("gorch: ConvTranspose1dForward channel mismatch: input inC=%d, weight inC=%d", inC, weight.shape[0]))
	}
	outC := outCg * groups
	if bias != nil && bias.Size() != outC {
		panic(fmt.Sprintf("gorch: ConvTranspose1dForward bias size %d != outC %d", bias.Size(), outC))
	}
	if L < 1 {
		panic("gorch: ConvTranspose1dForward requires L >= 1")
	}
	if GradEnabled() && (input.requiresGrad || weight.requiresGrad || (bias != nil && bias.requiresGrad)) {
		panic("gorch: ConvTranspose1dForward is inference-only (frozen-weight Mimi decoder, plan 0007 §2.1): " +
			"ConvTranspose autograd is not implemented; wrap in NoGrad or clear requiresGrad")
	}

	outL := (L-1)*stride + k
	outData := make([]float32, batch*outC*outL)

	if groups == inC && groups == outC {
		convTranspose1dDepthwise(input.data, weight.data, batch, inC, L, k, stride, outData)
	} else {
		convTranspose1dGrouped(input.data, weight.data, batch, inC, L, outCg, k, stride, groups, outData)
	}

	if bias != nil {
		bData := bias.data
		for b := 0; b < batch; b++ {
			sample := outData[b*outC*outL : (b+1)*outC*outL]
			for oc := 0; oc < outC; oc++ {
				row := sample[oc*outL : (oc+1)*outL]
				for i := range row {
					row[i] += bData[oc]
				}
			}
		}
	}

	return &Tensor{data: outData, shape: []int{batch, outC, outL}}
}

// convTranspose1dDepthwise handles groups == inC == outC (one k-tap
// transposed filter per channel, weight (C, 1, k)): a direct
// scatter-add loop per channel — no GEMM. out must be zeroed,
// (batch, C, (L-1)*stride+k).
func convTranspose1dDepthwise(input, weight []float32, batch, C, L, k, stride int, out []float32) {
	outL := (L-1)*stride + k
	for b := 0; b < batch; b++ {
		for c := 0; c < C; c++ {
			w := weight[c*k : (c+1)*k]
			x := input[(b*C+c)*L : (b*C+c+1)*L]
			row := out[(b*C+c)*outL : (b*C+c+1)*outL]
			for t := 0; t < L; t++ {
				xv := x[t]
				dst := row[t*stride : t*stride+k]
				for kx, wv := range w {
					// The explicit float32 conversion forces the
					// product to round before the add: without it Go
					// may fuse this into an FMA on arm64 (the spec
					// only guarantees rounding at explicit
					// conversions), breaking bit-parity with the
					// GEMM path, which rounds the product in Sgemm
					// before col2im1d adds it.
					dst[kx] += float32(xv * wv)
				}
			}
		}
	}
}

// convTranspose1dGrouped handles the general case: per batch and group,
// col(outCg*k, L) = Wg^T @ Xg via SgemmTransA, then col2im1d
// scatter-adds col into the group's output rows. Both Wg
// ((inCg, outCg*k) — the weight rows of the group's input channels)
// and Xg ((inCg, L)) are contiguous sub-slices, so no repacking is
// needed. out must be zeroed, (batch, outCg*groups, (L-1)*stride+k).
func convTranspose1dGrouped(input, weight []float32, batch, inC, L, outCg, k, stride, groups int, out []float32) {
	inCg := inC / groups
	outC := outCg * groups
	outL := (L-1)*stride + k

	colBuf := AcquireFloat32(outCg * k * L)
	defer ReleaseFloat32(colBuf)

	for b := 0; b < batch; b++ {
		for g := 0; g < groups; g++ {
			wg := weight[g*inCg*outCg*k : (g+1)*inCg*outCg*k]
			xg := input[(b*inC+g*inCg)*L : (b*inC+(g+1)*inCg)*L]
			// col = Wg^T @ Xg  =>  (outCg*k, inCg) @ (inCg, L)
			accelerate.SgemmTransA(outCg*k, L, inCg, 1.0, wg, xg, 0.0, colBuf)
			dst := out[(b*outC+g*outCg)*outL : (b*outC+(g+1)*outCg)*outL]
			col2im1d(colBuf, outCg, outL, k, stride, 1, dst)
		}
	}
}
