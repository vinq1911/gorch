//go:build darwin

package nn

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	g "github.com/vinq1911/gorch"
)

func randFill(rng *rand.Rand, t *g.Tensor) {
	d := t.Data()
	for i := range d {
		d[i] = float32(rng.NormFloat64())
	}
}

func maxAbsDiffF32(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

// TestCausalConvTranspose1dTrim: the causal trim (k-stride from the
// right, nothing from the left) must map L → exactly L*stride for all
// five Mimi transposed-conv geometries, and the trimmed output must be
// the prefix of the raw ConvTranspose1dForward output.
func TestCausalConvTranspose1dTrim(t *testing.T) {
	cases := []struct {
		name                         string
		inC, outC, k, stride, groups int
	}{
		{"k4s2g512", 512, 512, 4, 2, 512}, // upsample: depthwise, real channel count
		{"k16s8", 8, 4, 16, 8, 1},
		{"k12s6", 6, 3, 12, 6, 1},
		{"k10s5", 4, 2, 10, 5, 1},
		{"k8s4", 4, 2, 8, 4, 1},
	}
	rng := rand.New(rand.NewSource(3))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewCausalConvTranspose1d(tc.inC, tc.outC, tc.k, tc.stride, tc.groups, true)
			randFill(rng, c.Bias)
			for _, L := range []int{1, 5, 9} {
				x := g.Zeros(1, tc.inC, L)
				randFill(rng, x)
				out := c.Forward(x)
				s := out.Shape()
				if s[0] != 1 || s[1] != tc.outC || s[2] != L*tc.stride {
					t.Fatalf("L=%d: output shape %v, want [1 %d %d]", L, s, tc.outC, L*tc.stride)
				}
				raw := g.ConvTranspose1dForward(x, c.Weight, c.Bias, tc.stride, tc.groups)
				rawL := raw.Shape()[2]
				for ch := 0; ch < tc.outC; ch++ {
					for i := 0; i < L*tc.stride; i++ {
						if out.Data()[ch*L*tc.stride+i] != raw.Data()[ch*rawL+i] {
							t.Fatalf("L=%d ch=%d i=%d: trimmed output is not the raw prefix", L, ch, i)
						}
					}
				}
			}
		})
	}
}

// TestConvT1dStreamMatchesOffline: concatenating ForwardStream chunk
// outputs over random chunk splits (sizes 1..7) must equal the offline
// Forward on the concatenated input, for both biased and bias-free
// layers, across the Mimi geometries including the depthwise upsample.
// Streaming partitions the GEMM/scatter-add differently than offline,
// so f32 summation order differs; gate 1e-5 (values are O(1)).
func TestConvT1dStreamMatchesOffline(t *testing.T) {
	cases := []struct {
		name                         string
		inC, outC, k, stride, groups int
	}{
		{"k4s2g8depthwise", 8, 8, 4, 2, 8},
		{"k16s8", 6, 3, 16, 8, 1},
		{"k10s5", 4, 2, 10, 5, 1},
		{"k8s4g2", 4, 6, 8, 4, 2},
		{"k3s1", 2, 3, 3, 1, 1},
		{"k2s2_no_overlap", 3, 2, 2, 2, 1}, // k == stride: zero-length tail
	}
	for _, tc := range cases {
		for _, withBias := range []bool{false, true} {
			for seed := int64(0); seed < 3; seed++ {
				name := fmt.Sprintf("%s_bias=%v_seed%d", tc.name, withBias, seed)
				t.Run(name, func(t *testing.T) {
					rng := rand.New(rand.NewSource(seed*101 + 13))
					c := NewCausalConvTranspose1d(tc.inC, tc.outC, tc.k, tc.stride, tc.groups, withBias)
					if withBias {
						randFill(rng, c.Bias)
					}
					const T = 23
					x := g.Zeros(1, tc.inC, T)
					randFill(rng, x)
					want := c.Forward(x)

					st := &ConvT1dStream{}
					got := make([]float32, tc.outC*T*tc.stride)
					emitted := 0
					for pos := 0; pos < T; {
						S := 1 + rng.Intn(7)
						if pos+S > T {
							S = T - pos
						}
						chunk := g.Zeros(1, tc.inC, S)
						for ch := 0; ch < tc.inC; ch++ {
							copy(chunk.Data()[ch*S:(ch+1)*S], x.Data()[ch*T+pos:ch*T+pos+S])
						}
						out := c.ForwardStream(chunk, st)
						s := out.Shape()
						if s[0] != 1 || s[1] != tc.outC || s[2] != S*tc.stride {
							t.Fatalf("chunk at %d: output shape %v, want [1 %d %d]", pos, s, tc.outC, S*tc.stride)
						}
						for ch := 0; ch < tc.outC; ch++ {
							copy(got[ch*T*tc.stride+emitted:], out.Data()[ch*S*tc.stride:(ch+1)*S*tc.stride])
						}
						emitted += S * tc.stride
						pos += S
					}
					if emitted != T*tc.stride {
						t.Fatalf("emitted %d columns, want %d", emitted, T*tc.stride)
					}
					if d := maxAbsDiffF32(got, want.Data()); d > 1e-5 {
						t.Errorf("streaming vs offline max abs diff = %g, want <= 1e-5", d)
					}
				})
			}
		}
	}
}

// TestConvT1dStreamFirstChunkExact: a single chunk covering the whole
// input starts from a zero tail (correct because padding_left = 0) and
// must reproduce the offline Forward bit-exactly — the raw conv is the
// same call, and bias-at-emission adds the same f32 values the offline
// path fuses in.
func TestConvT1dStreamFirstChunkExact(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	c := NewCausalConvTranspose1d(4, 6, 8, 4, 2, true)
	randFill(rng, c.Bias)
	x := g.Zeros(1, 4, 11)
	randFill(rng, x)

	want := c.Forward(x)
	got := c.ForwardStream(x, &ConvT1dStream{})
	if d := maxAbsDiffF32(got.Data(), want.Data()); d != 0 {
		t.Errorf("single-chunk streaming differs from offline by %g, want bit-exact", d)
	}
}

// TestConvT1dStreamBiasOnce: with constant input, every interior
// output column (full kernel support, stride 1) has the same value —
// a double-counted bias at a chunk boundary would show up as a spike.
// Cross-column tolerance 1e-6: boundary columns sum the same taps in a
// different f32 order (part arrives pre-summed via the tail).
func TestConvT1dStreamBiasOnce(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	const inC, outC, k, T = 2, 3, 3, 12
	c := NewCausalConvTranspose1d(inC, outC, k, 1, 1, true)
	randFill(rng, c.Bias)

	x := g.Zeros(1, inC, T)
	for i := range x.Data() {
		x.Data()[i] = 1
	}

	st := &ConvT1dStream{}
	got := make([]float32, outC*T)
	emitted := 0
	for _, S := range []int{3, 1, 5, 3} {
		chunk := g.Zeros(1, inC, S)
		for ch := 0; ch < inC; ch++ {
			for i := 0; i < S; i++ {
				chunk.Data()[ch*S+i] = 1
			}
		}
		out := c.ForwardStream(chunk, st)
		for ch := 0; ch < outC; ch++ {
			copy(got[ch*T+emitted:], out.Data()[ch*S:(ch+1)*S])
		}
		emitted += S
	}

	for ch := 0; ch < outC; ch++ {
		ref := got[ch*T+k-1] // first column with full kernel support
		for i := k - 1; i < T; i++ {
			if d := math.Abs(float64(got[ch*T+i]) - float64(ref)); d > 1e-6 {
				t.Errorf("ch=%d col=%d: interior value %v != %v (diff %g) — bias applied more than once?",
					ch, i, got[ch*T+i], ref, d)
			}
		}
	}
}

// TestConvT1dStreamReset: after Reset the stream must reproduce a
// fresh session bit-exactly.
func TestConvT1dStreamReset(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	c := NewCausalConvTranspose1d(3, 4, 12, 6, 1, true)
	randFill(rng, c.Bias)

	chunks := make([]*g.Tensor, 3)
	for i := range chunks {
		chunks[i] = g.Zeros(1, 3, 2+i)
		randFill(rng, chunks[i])
	}
	run := func(st *ConvT1dStream) []float32 {
		var out []float32
		for _, ch := range chunks {
			out = append(out, c.ForwardStream(ch, st).Data()...)
		}
		return out
	}

	st := &ConvT1dStream{}
	first := run(st)
	st.Reset()
	second := run(st)
	fresh := run(&ConvT1dStream{})

	if d := maxAbsDiffF32(first, second); d != 0 {
		t.Errorf("post-Reset run differs from first run by %g, want bit-exact", d)
	}
	if d := maxAbsDiffF32(first, fresh); d != 0 {
		t.Errorf("fresh-state run differs from first run by %g, want bit-exact", d)
	}
}

// TestCausalConvTranspose1dParameters: Parameters must expose weight
// and (when present) bias.
func TestCausalConvTranspose1dParameters(t *testing.T) {
	c := NewCausalConvTranspose1d(4, 2, 8, 4, 1, true)
	if p := c.Parameters(); len(p) != 2 || p[0] != c.Weight || p[1] != c.Bias {
		t.Errorf("Parameters with bias = %d tensors, want [Weight Bias]", len(p))
	}
	nb := NewCausalConvTranspose1d(6, 6, 4, 2, 6, false)
	if p := nb.Parameters(); len(p) != 1 || p[0] != nb.Weight {
		t.Errorf("Parameters without bias = %d tensors, want [Weight]", len(p))
	}
	if nb.Bias != nil {
		t.Error("bias=false must leave Bias nil (the Mimi upsample has no bias key)")
	}
}
