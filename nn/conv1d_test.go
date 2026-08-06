//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestCausalPad1dTable: the pad computation must match a hand-computed
// (L, k, stride, dilation) → (padLeft, padRight) table covering every
// Mimi conv configuration (plan 0006 risk #1), including
// non-stride-aligned lengths where extra_padding > 0.
func TestCausalPad1dTable(t *testing.T) {
	cases := []struct {
		L, k, stride, dilation int
		padL, padR             int
	}{
		// k7 s1 — SEANet first conv; stride 1 never needs extra pad.
		{100, 7, 1, 1, 6, 0},
		{101, 7, 1, 1, 6, 0},
		// k8 s4 — SEANet downsampling conv.
		{100, 8, 4, 1, 4, 0},
		{101, 8, 4, 1, 4, 3},
		{103, 8, 4, 1, 4, 1},
		{2, 8, 4, 1, 4, 2}, // L < stride: ceil still lands on one frame
		// k10 s5.
		{100, 10, 5, 1, 5, 0},
		{102, 10, 5, 1, 5, 3},
		// k12 s6.
		{96, 12, 6, 1, 6, 0},
		{97, 12, 6, 1, 6, 5},
		// k16 s8.
		{64, 16, 8, 1, 8, 0},
		{65, 16, 8, 1, 8, 7},
		// k4 s2 — Mimi downsample (replicate pad, same geometry).
		{10, 4, 2, 1, 2, 0},
		{11, 4, 2, 1, 2, 1},
		// k3 s1 — residual-block conv.
		{37, 3, 1, 1, 2, 0},
		// k1 s1 — residual-block pointwise conv.
		{37, 1, 1, 1, 0, 0},
		// dilation > 1: kEff = 5.
		{20, 3, 1, 2, 4, 0},
		{21, 5, 2, 2, 7, 1}, // kEff=9, padTotal=7; L odd → extra 1
	}
	for _, c := range cases {
		padL, padR := CausalPad1d(c.L, c.k, c.stride, c.dilation)
		if padL != c.padL || padR != c.padR {
			t.Fatalf("CausalPad1d(L=%d,k=%d,s=%d,d=%d) = (%d,%d), want (%d,%d)",
				c.L, c.k, c.stride, c.dilation, padL, padR, c.padL, c.padR)
		}
	}
}

// TestCausalConv1dOutputLength: with causal padding, the output length
// is always ceil(L/stride) — the property the pad formula exists to
// guarantee.
func TestCausalConv1dOutputLength(t *testing.T) {
	cases := []struct{ L, k, stride int }{
		{100, 7, 1}, {101, 8, 4}, {102, 10, 5}, {97, 12, 6}, {65, 16, 8}, {11, 4, 2},
	}
	for _, c := range cases {
		conv := NewCausalConv1d(2, 3, c.k, c.stride, 1, true, g.PadConstant)
		out := conv.Forward(g.RandN(1, 2, c.L))
		wantL := (c.L + c.stride - 1) / c.stride
		if out.Shape()[2] != wantL {
			t.Fatalf("L=%d k=%d s=%d: outL=%d, want ceil(L/s)=%d", c.L, c.k, c.stride, out.Shape()[2], wantL)
		}
	}
}

// TestForwardStreamMatchesOffline: on stride-aligned input, chunked
// ForwardStream output concatenated over chunks must equal the offline
// Forward output, for both pad modes (plan 0006 §2.2).
func TestForwardStreamMatchesOffline(t *testing.T) {
	cases := []struct {
		name                        string
		inC, outC, k, stride, dilat int
		bias                        bool
		mode                        g.PadMode
		L, chunk                    int
	}{
		{"k8s4-constant-bias", 3, 4, 8, 4, 1, true, g.PadConstant, 48, 12},
		{"k7s1-constant-bias", 2, 2, 7, 1, 1, true, g.PadConstant, 30, 10},
		{"k4s2-replicate-nobias", 4, 4, 4, 2, 1, false, g.PadReplicate, 40, 8},
		{"k3s1d2-constant", 2, 3, 3, 1, 2, true, g.PadConstant, 24, 6},
		{"k1s1-constant", 3, 5, 1, 1, 1, true, g.PadConstant, 20, 5},
		{"k16s8-replicate-bias", 2, 3, 16, 8, 1, true, g.PadReplicate, 64, 16},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			conv := NewCausalConv1d(c.inC, c.outC, c.k, c.stride, c.dilat, c.bias, c.mode)
			x := g.RandN(1, c.inC, c.L)

			var offline *g.Tensor
			g.NoGrad(func() { offline = conv.Forward(x) })

			var st Conv1dStream
			outData := make([]float32, 0, offline.Size())
			outT := 0
			for start := 0; start < c.L; start += c.chunk {
				chunkData := make([]float32, c.inC*c.chunk)
				for ch := 0; ch < c.inC; ch++ {
					copy(chunkData[ch*c.chunk:], x.Data()[ch*c.L+start:ch*c.L+start+c.chunk])
				}
				var y *g.Tensor
				g.NoGrad(func() {
					y = conv.ForwardStream(g.NewTensor(chunkData, 1, c.inC, c.chunk), &st)
				})
				// Interleave: outputs are (1, outC, tChunk); gather per chunk.
				tChunk := y.Shape()[2]
				outData = append(outData, y.Data()...)
				outT += tChunk
			}
			if outT != offline.Shape()[2] {
				t.Fatalf("stream produced %d frames, offline %d", outT, offline.Shape()[2])
			}

			// Reassemble streamed chunks (each (outC, tChunk)) into (outC, T).
			streamed := make([]float32, c.outC*outT)
			pos := 0
			idx := 0
			for pos < outT {
				tChunk := c.chunk / c.stride
				for ch := 0; ch < c.outC; ch++ {
					copy(streamed[ch*outT+pos:], outData[idx+ch*tChunk:idx+(ch+1)*tChunk])
				}
				idx += c.outC * tChunk
				pos += tChunk
			}

			for i := range streamed {
				diff := math.Abs(float64(streamed[i] - offline.Data()[i]))
				if diff > 1e-6 {
					t.Fatalf("[%d] stream=%g offline=%g (diff %g)", i, streamed[i], offline.Data()[i], diff)
				}
			}
		})
	}
}

// TestForwardStreamRejectsMisalignedChunk: chunk length must be a
// multiple of the stride.
func TestForwardStreamRejectsMisalignedChunk(t *testing.T) {
	conv := NewCausalConv1d(1, 1, 8, 4, 1, true, g.PadConstant)
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for chunk length not a multiple of stride")
		}
	}()
	var st Conv1dStream
	conv.ForwardStream(g.RandN(1, 1, 10), &st)
}

// TestConv1dStreamReset: after Reset, the context is reseeded — the
// first post-reset chunk must match a fresh stream.
func TestConv1dStreamReset(t *testing.T) {
	conv := NewCausalConv1d(2, 2, 4, 2, 1, false, g.PadReplicate)
	x := g.RandN(1, 2, 8)

	var st Conv1dStream
	var first, again *g.Tensor
	g.NoGrad(func() {
		first = conv.ForwardStream(x, &st)
		conv.ForwardStream(g.RandN(1, 2, 8), &st) // advance state
		st.Reset()
		again = conv.ForwardStream(x, &st)
	})
	for i := range first.Data() {
		if first.Data()[i] != again.Data()[i] {
			t.Fatalf("[%d] post-reset %g != fresh %g", i, again.Data()[i], first.Data()[i])
		}
	}
}
