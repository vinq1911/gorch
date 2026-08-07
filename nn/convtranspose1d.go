//go:build darwin

package nn

import (
	"fmt"
	"math"

	g "github.com/vinq1911/gorch"
)

// CausalConvTranspose1d reproduces transformers' MimiConvTranspose1d
// with use_causal_conv=true, trim_right_ratio=1.0: run the transposed
// convolution with padding 0 (raw length (L-1)*stride + k), then trim
// kernel-stride columns from the right and none from the left —
// (batch, inC, L) → (batch, outC, L*stride).
//
// Inference-only, like ConvTranspose1dForward: the constructor does
// not mark the parameters as requiring grad (plan 0007 §2.1).
type CausalConvTranspose1d struct {
	Weight *g.Tensor // (inC, outC/groups, k) — PyTorch ConvTranspose layout
	Bias   *g.Tensor // (outC,) or nil
	Stride int
	Groups int
}

// NewCausalConvTranspose1d creates a CausalConvTranspose1d with
// Kaiming-style initialization (loader code overwrites the parameters
// with checkpoint weights).
func NewCausalConvTranspose1d(inC, outC, k, stride, groups int, bias bool) *CausalConvTranspose1d {
	if stride < 1 {
		panic("nn: NewCausalConvTranspose1d requires stride >= 1")
	}
	if groups < 1 || inC%groups != 0 || outC%groups != 0 {
		panic(fmt.Sprintf("nn: NewCausalConvTranspose1d inC=%d/outC=%d not divisible by groups=%d", inC, outC, groups))
	}
	if k < stride {
		panic(fmt.Sprintf("nn: NewCausalConvTranspose1d kernel %d < stride %d (causal trim k-stride would be negative)", k, stride))
	}
	fanIn := float64(inC / groups * k)
	scale := float32(math.Sqrt(2.0 / fanIn))
	w := g.RandN(inC, outC/groups, k)
	for i := range w.Data() {
		w.Data()[i] *= scale
	}
	var b *g.Tensor
	if bias {
		b = g.Zeros(outC)
	}
	return &CausalConvTranspose1d{Weight: w, Bias: b, Stride: stride, Groups: groups}
}

// Forward runs the offline causal transposed convolution:
// (batch, inC, L) → (batch, outC, L*stride), trimming the k-stride
// rightmost raw output columns (padding_right = padding_total with
// trim_right_ratio 1.0; padding_left = 0).
func (c *CausalConvTranspose1d) Forward(x *g.Tensor) *g.Tensor {
	raw := g.ConvTranspose1dForward(x, c.Weight, c.Bias, c.Stride, c.Groups)
	k := c.Weight.Shape()[2]
	trim := k - c.Stride
	if trim == 0 {
		return raw
	}
	shape := raw.Shape()
	batch, outC, rawL := shape[0], shape[1], shape[2]
	outL := rawL - trim
	data := make([]float32, batch*outC*outL)
	rawData := raw.Data()
	for r := 0; r < batch*outC; r++ {
		copy(data[r*outL:(r+1)*outL], rawData[r*rawL:r*rawL+outL])
	}
	return g.NewTensor(data, batch, outC, outL)
}

func (c *CausalConvTranspose1d) Parameters() []*g.Tensor {
	if c.Bias == nil {
		return []*g.Tensor{c.Weight}
	}
	return []*g.Tensor{c.Weight, c.Bias}
}

// ConvT1dStream holds the streaming overlap-add tail for one
// CausalConvTranspose1d: the last k-stride output columns of the
// previous chunk's raw (untrimmed, bias-free) transposed convolution —
// partial sums still awaiting contributions from future input frames.
type ConvT1dStream struct {
	tail []float32 // (outC, k-stride) pending bias-free partial sums
}

// Reset clears the stream state; the next ForwardStream call starts a
// fresh session with a zero tail (correct because padding_left = 0).
func (s *ConvT1dStream) Reset() {
	s.tail = nil
}

// ForwardStream runs one streaming chunk of S input frames.
// Contract (plan 0007 §2.2):
//   - compute the raw transposed conv WITHOUT bias →
//     (outC, (S-1)*stride + k);
//   - add the stored tail into the first k-stride columns;
//   - emit the first S*stride columns WITH bias added at emission;
//   - store the last k-stride columns (bias-free) as the new tail.
//
// The first chunk uses a zero tail. Adding bias only at emission
// prevents double-counting: every output column receives bias exactly
// once even though up to ceil(k/stride) chunks contribute partial sums
// to it. Concatenating the emitted chunks equals the offline Forward
// on the concatenated input.
func (c *CausalConvTranspose1d) ForwardStream(x *g.Tensor, st *ConvT1dStream) *g.Tensor {
	shape := x.Shape()
	if len(shape) != 3 || shape[0] != 1 {
		panic(fmt.Sprintf("nn: ConvT1dStream ForwardStream expects (1, inC, S), got %v", shape))
	}
	S := shape[2]
	k := c.Weight.Shape()[2]
	tailLen := k - c.Stride

	raw := g.ConvTranspose1dForward(x, c.Weight, nil, c.Stride, c.Groups)
	outC := raw.Shape()[1]
	rawL := raw.Shape()[2] // S*stride + tailLen
	rawData := raw.Data()

	if st.tail == nil {
		st.tail = make([]float32, outC*tailLen)
	} else if len(st.tail) != outC*tailLen {
		panic(fmt.Sprintf("nn: ConvT1dStream tail size %d does not match layer (outC=%d, k-stride=%d); missing Reset?",
			len(st.tail), outC, tailLen))
	}

	// Overlap-add the previous chunk's pending partial sums.
	for ch := 0; ch < outC; ch++ {
		row := rawData[ch*rawL : ch*rawL+tailLen]
		prev := st.tail[ch*tailLen : (ch+1)*tailLen]
		for i := range row {
			row[i] += prev[i]
		}
	}

	// Emit the first S*stride columns, bias at emission.
	emitL := S * c.Stride
	out := make([]float32, outC*emitL)
	for ch := 0; ch < outC; ch++ {
		row := out[ch*emitL : (ch+1)*emitL]
		copy(row, rawData[ch*rawL:ch*rawL+emitL])
		if c.Bias != nil {
			bv := c.Bias.Data()[ch]
			for i := range row {
				row[i] += bv
			}
		}
	}

	// Save the last k-stride raw columns (bias-free) as the new tail.
	// They may overlap the columns the old tail was added into (when
	// S*stride < k-stride); that is correct — partial sums accumulate
	// across as many chunks as contribute to a column before emission.
	for ch := 0; ch < outC; ch++ {
		copy(st.tail[ch*tailLen:(ch+1)*tailLen], rawData[ch*rawL+emitL:(ch+1)*rawL])
	}

	return g.NewTensor(out, 1, outC, emitL)
}
