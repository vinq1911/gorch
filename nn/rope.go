//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// RoPE — Rotary Position Embedding (Su et al. 2021).
//
// Applied to Q and K inside attention:
//
//	for each head, treat the headDim as pairs (x_2i, x_2i+1):
//	  rotated_2i   = x_2i * cos(θ) - x_2i+1 * sin(θ)
//	  rotated_2i+1 = x_2i * sin(θ) + x_2i+1 * cos(θ)
//
// Frequencies θ_i are precomputed from `base` (typically 10000) as:
//
//	θ_i,pos = pos / base^(2i/headDim)
//
// gorch's RoPE follows the standard production approach (nanoGPT,
// llama.cpp, vLLM, HuggingFace transformers): real-valued cos/sin
// tables computed on host once at construction time, applied as
// element-wise multiply + half-vector swap. Plan 0001 Phase 1 item 8;
// plan 0003 explicitly rejects the "complex-typed Metal shader" path
// the external Gemini advisory recommended.
//
// Llama-style RoPE rotates the *first half* paired with the *second
// half* of each head: for headDim=D, pair (x[0..D/2], x[D/2..D]). That's
// the convention used by Llama, Mistral, OpenMythos. The "interleaved"
// pair convention (x[2i], x[2i+1]) used by GPT-NeoX is a different
// layout — match Llama by default; the Style field flips between them.
//
// Inputs to Apply: shape (..., seqLen, headDim) flat-rank-3 or higher.
// The seqLen dimension is the second-to-last; headDim is the last.
type RoPE struct {
	HeadDim int
	MaxSeq  int
	Base    float32 // typically 10000 for Llama, 500000 for Llama-3
	Style   RopeStyle

	// Precomputed (maxSeq, headDim/2) tables.
	cos []float32
	sin []float32
}

// RopeStyle selects the half-rotation convention.
type RopeStyle int

const (
	// RopeLlama pairs x[i] with x[i + headDim/2] (Llama / Mistral /
	// DeepSeek / OpenMythos / nanoGPT).
	RopeLlama RopeStyle = iota
	// RopeGPTNeoX pairs x[2i] with x[2i+1] (interleaved). Less common
	// in modern stacks but supported for GPT-NeoX-style checkpoints.
	RopeGPTNeoX
)

// NewRoPE precomputes the cos/sin tables for sequence positions
// 0..maxSeq-1 across half of headDim.
//
// headDim must be even. base is 10_000 for most models, 500_000 for
// Llama-3.
func NewRoPE(headDim, maxSeq int, base float32, style RopeStyle) *RoPE {
	if headDim%2 != 0 {
		panic("gorch/nn: RoPE requires even headDim")
	}
	half := headDim / 2
	cos := make([]float32, maxSeq*half)
	sin := make([]float32, maxSeq*half)
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < half; i++ {
			// θ = pos / base^(2i/headDim) = pos * base^(-2i/headDim)
			invFreq := math.Pow(float64(base), -float64(2*i)/float64(headDim))
			theta := float64(pos) * invFreq
			cos[pos*half+i] = float32(math.Cos(theta))
			sin[pos*half+i] = float32(math.Sin(theta))
		}
	}
	return &RoPE{
		HeadDim: headDim,
		MaxSeq:  maxSeq,
		Base:    base,
		Style:   style,
		cos:     cos,
		sin:     sin,
	}
}

// Apply rotates x in place along the last (headDim) dimension based
// on the position index of each row along the second-to-last (seqLen)
// dimension. Input shape: (..., seqLen, headDim). startPos is the
// absolute position offset of row 0 (used by KV-cache decoding where
// the new tokens occupy positions cache.Len()..cache.Len()+seqLen-1).
//
// Returns a new tensor; does not modify x in place to keep the
// autograd contract clean (the caller's gradFn for x stays valid).
//
// Backward: rotation is its own inverse with sin negated. The grad
// for each pair (g0, g1) at position pos becomes:
//
//	dx_2i   = g_2i * cos + g_2i+1 * sin
//	dx_2i+1 = -g_2i * sin + g_2i+1 * cos
//
// (i.e., rotate the upstream grad by the same theta but with sin
// negated — equivalent to the conjugate / inverse rotation.)
func (r *RoPE) Apply(x *g.Tensor, startPos int) *g.Tensor {
	nd := x.Dim()
	if nd < 2 {
		panic("gorch/nn: RoPE.Apply requires rank ≥ 2")
	}
	shape := x.Shape()
	seqLen := shape[nd-2]
	headDim := shape[nd-1]
	if headDim != r.HeadDim {
		panic("gorch/nn: RoPE headDim mismatch")
	}
	if startPos+seqLen > r.MaxSeq {
		panic("gorch/nn: RoPE startPos+seqLen exceeds MaxSeq")
	}

	// Flatten everything before seqLen into one outer index.
	outer := 1
	for i := 0; i < nd-2; i++ {
		outer *= shape[i]
	}
	half := headDim / 2

	out := g.Zeros(shape...)
	xData := x.Data()
	outData := out.Data()
	cos := r.cos
	sin := r.sin

	rowStride := headDim
	seqStride := seqLen * headDim

	switch r.Style {
	case RopeLlama:
		for o := 0; o < outer; o++ {
			outerOff := o * seqStride
			for s := 0; s < seqLen; s++ {
				rowOff := outerOff + s*rowStride
				cs := (startPos + s) * half
				for i := 0; i < half; i++ {
					a := xData[rowOff+i]
					b := xData[rowOff+half+i]
					c := cos[cs+i]
					si := sin[cs+i]
					outData[rowOff+i] = a*c - b*si
					outData[rowOff+half+i] = a*si + b*c
				}
			}
		}
	case RopeGPTNeoX:
		for o := 0; o < outer; o++ {
			outerOff := o * seqStride
			for s := 0; s < seqLen; s++ {
				rowOff := outerOff + s*rowStride
				cs := (startPos + s) * half
				for i := 0; i < half; i++ {
					a := xData[rowOff+2*i]
					b := xData[rowOff+2*i+1]
					c := cos[cs+i]
					si := sin[cs+i]
					outData[rowOff+2*i] = a*c - b*si
					outData[rowOff+2*i+1] = a*si + b*c
				}
			}
		}
	}

	if g.GradEnabled() && x.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("RoPE", []*g.Tensor{x}, func(grad *g.Tensor) []*g.Tensor {
			dx := g.Zeros(shape...)
			gData := grad.Data()
			dxData := dx.Data()
			switch r.Style {
			case RopeLlama:
				for o := 0; o < outer; o++ {
					outerOff := o * seqStride
					for s := 0; s < seqLen; s++ {
						rowOff := outerOff + s*rowStride
						cs := (startPos + s) * half
						for i := 0; i < half; i++ {
							ga := gData[rowOff+i]
							gb := gData[rowOff+half+i]
							c := cos[cs+i]
							si := sin[cs+i]
							// inverse rotation: sin → -sin
							dxData[rowOff+i] = ga*c + gb*si
							dxData[rowOff+half+i] = -ga*si + gb*c
						}
					}
				}
			case RopeGPTNeoX:
				for o := 0; o < outer; o++ {
					outerOff := o * seqStride
					for s := 0; s < seqLen; s++ {
						rowOff := outerOff + s*rowStride
						cs := (startPos + s) * half
						for i := 0; i < half; i++ {
							ga := gData[rowOff+2*i]
							gb := gData[rowOff+2*i+1]
							c := cos[cs+i]
							si := sin[cs+i]
							dxData[rowOff+2*i] = ga*c + gb*si
							dxData[rowOff+2*i+1] = -ga*si + gb*c
						}
					}
				}
			}
			return []*g.Tensor{dx}
		})
	}
	return out
}
