//go:build darwin

package nn

import (
	"fmt"
	"math"

	g "github.com/vinq1911/gorch"
)

// CausalConv1d reproduces the offline semantics of transformers'
// MimiConv1d: left-pad by the full causal amount, right-pad only the
// extra needed to make the input length "ideal" for the stride.
// Input shape: (batch, inC, L), output shape: (batch, outC, outL).
type CausalConv1d struct {
	Weight   *g.Tensor // (outC, inC, k)
	Bias     *g.Tensor // (outC,) or nil
	Stride   int
	Dilation int
	PadMode  g.PadMode
}

// NewCausalConv1d creates a CausalConv1d with Kaiming initialization.
func NewCausalConv1d(inC, outC, k, stride, dilation int, bias bool, mode g.PadMode) *CausalConv1d {
	fanIn := float64(inC * k)
	scale := float32(math.Sqrt(2.0 / fanIn))
	w := g.RandN(outC, inC, k)
	for i := range w.Data() {
		w.Data()[i] *= scale
	}
	w.SetRequiresGrad(true)

	var b *g.Tensor
	if bias {
		b = g.Zeros(outC)
		b.SetRequiresGrad(true)
	}
	return &CausalConv1d{Weight: w, Bias: b, Stride: stride, Dilation: dilation, PadMode: mode}
}

// ceilDiv returns ceil(a/b) for b > 0. For a <= 0 Go's truncation
// toward zero already equals the ceiling.
func ceilDiv(a, b int) int {
	if a <= 0 {
		return a / b
	}
	return (a + b - 1) / b
}

// CausalPad1d computes the (padLeft, padRight) a causal Mimi conv
// applies for input length L, kernel k, stride and dilation:
//
//	kEff     = (k-1)*dilation + 1
//	padLeft  = kEff - stride                       (padding_total)
//	padRight = idealLength - L                     (extra_padding)
//	idealLength = ceil((L - kEff + padLeft)/stride)*stride + kEff - padLeft
//
// padRight is nonzero only when L is not stride-aligned; it must match
// HF's ceil-based formula exactly for offline parity.
func CausalPad1d(L, k, stride, dilation int) (padLeft, padRight int) {
	kEff := (k-1)*dilation + 1
	padTotal := kEff - stride
	if padTotal < 0 {
		panic(fmt.Sprintf("nn: CausalPad1d effective kernel %d < stride %d", kEff, stride))
	}
	ideal := ceilDiv(L-kEff+padTotal, stride)*stride + kEff - padTotal
	return padTotal, ideal - L
}

// Forward runs the offline causal convolution: pad (padTotal,
// extraPad) per CausalPad1d, then convolve.
func (c *CausalConv1d) Forward(x *g.Tensor) *g.Tensor {
	L := x.Shape()[2]
	k := c.Weight.Shape()[2]
	padL, padR := CausalPad1d(L, k, c.Stride, c.Dilation)
	return g.Conv1dForward(x, c.Weight, c.Bias, c.Stride, c.Dilation, padL, padR, c.PadMode)
}

func (c *CausalConv1d) Parameters() []*g.Tensor {
	if c.Bias == nil {
		return []*g.Tensor{c.Weight}
	}
	return []*g.Tensor{c.Weight, c.Bias}
}

// Conv1dStream holds the streaming left-context cache for one
// CausalConv1d — the equivalent of HF's MimiConv1dPaddingCache. The
// caller keeps one per layer and passes it to every ForwardStream
// call.
type Conv1dStream struct {
	ctx    []float32 // (inC, padTotal) left context, row-major
	ext    []float32 // reused (inC, padTotal+chunkL) scratch (plan risk 9)
	primed bool
}

// Reset clears the stream state so the next ForwardStream call reseeds
// the left context. The scratch buffer is kept for reuse.
func (s *Conv1dStream) Reset() {
	s.ctx = nil
	s.primed = false
}

// ForwardStream runs one streaming chunk through the causal conv.
// Contract:
//   - x is (1, inC, chunkL) and chunkL must be a multiple of Stride,
//     so extra (right) padding is never needed and consecutive chunks
//     tile exactly like the offline computation;
//   - on the first call the left context is seeded the way offline
//     padding works: zeros for PadConstant, the chunk's first sample
//     replicated for PadReplicate;
//   - after convolving, the last padTotal input columns are saved as
//     the next call's left context.
//
// Concatenating the outputs over chunks is bit-identical to the
// offline Forward on the concatenated (stride-aligned) input.
func (c *CausalConv1d) ForwardStream(x *g.Tensor, st *Conv1dStream) *g.Tensor {
	shape := x.Shape()
	if len(shape) != 3 || shape[0] != 1 {
		panic(fmt.Sprintf("nn: ForwardStream expects (1, inC, L), got %v", shape))
	}
	inC := shape[1]
	chunkL := shape[2]
	if chunkL%c.Stride != 0 {
		panic(fmt.Sprintf("nn: ForwardStream chunk length %d not a multiple of stride %d", chunkL, c.Stride))
	}
	k := c.Weight.Shape()[2]
	kEff := (k-1)*c.Dilation + 1
	padTotal := kEff - c.Stride
	if padTotal < 0 {
		panic(fmt.Sprintf("nn: ForwardStream effective kernel %d < stride %d", kEff, c.Stride))
	}

	if !st.primed {
		st.ctx = make([]float32, inC*padTotal)
		if c.PadMode == g.PadReplicate {
			// Mirror offline left-replicate: repeat the first sample.
			for ch := 0; ch < inC; ch++ {
				first := x.Data()[ch*chunkL]
				row := st.ctx[ch*padTotal : (ch+1)*padTotal]
				for i := range row {
					row[i] = first
				}
			}
		}
		st.primed = true
	}

	// Extended input: (1, inC, padTotal+chunkL) = ctx ++ chunk. The
	// scratch buffer persists across chunks to avoid per-chunk churn.
	extL := padTotal + chunkL
	if cap(st.ext) < inC*extL {
		st.ext = make([]float32, inC*extL)
	}
	extData := st.ext[:inC*extL]
	xData := x.Data()
	for ch := 0; ch < inC; ch++ {
		copy(extData[ch*extL:], st.ctx[ch*padTotal:(ch+1)*padTotal])
		copy(extData[ch*extL+padTotal:], xData[ch*chunkL:(ch+1)*chunkL])
	}
	ext := g.NewTensor(extData, 1, inC, extL)

	out := g.Conv1dForward(ext, c.Weight, c.Bias, c.Stride, c.Dilation, 0, 0, c.PadMode)

	// Save the last padTotal input columns as the next left context.
	for ch := 0; ch < inC; ch++ {
		copy(st.ctx[ch*padTotal:(ch+1)*padTotal], extData[ch*extL+extL-padTotal:(ch+1)*extL])
	}
	return out
}
