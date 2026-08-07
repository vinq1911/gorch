//go:build darwin

package gorch

import (
	"math"
	"math/rand"
	"testing"
)

// naiveConvTranspose1d is a straightforward reference for
// ConvTranspose1dForward (PyTorch conv_transpose1d, padding=0,
// output_padding=0, dilation=1) with float64 accumulation:
//
//	out[b, g*outCg+ocg, t*stride+kx] +=
//	    x[b, g*inCg+icg, t] * w[g*inCg+icg, ocg, kx]
func naiveConvTranspose1d(input, weight, bias []float32, batch, inC, L, outCg, k, stride, groups int) []float32 {
	inCg := inC / groups
	outC := outCg * groups
	outL := (L-1)*stride + k
	acc := make([]float64, batch*outC*outL)
	for b := 0; b < batch; b++ {
		for g := 0; g < groups; g++ {
			for icg := 0; icg < inCg; icg++ {
				ic := g*inCg + icg
				for ocg := 0; ocg < outCg; ocg++ {
					oc := g*outCg + ocg
					for t := 0; t < L; t++ {
						xv := float64(input[(b*inC+ic)*L+t])
						for kx := 0; kx < k; kx++ {
							acc[(b*outC+oc)*outL+t*stride+kx] +=
								xv * float64(weight[(ic*outCg+ocg)*k+kx])
						}
					}
				}
			}
		}
	}
	out := make([]float32, len(acc))
	for i := range acc {
		v := acc[i]
		if bias != nil {
			v += float64(bias[i/outL%outC])
		}
		out[i] = float32(v)
	}
	return out
}

// TestConvTranspose1dVsNaive sweeps the six (k, stride, groups) combos
// the D0 fixtures pin (the five Mimi transposed-conv geometries plus a
// generic 2-group case), with batch 1 and 2 and with/without bias,
// against the float64 triple-loop reference.
func TestConvTranspose1dVsNaive(t *testing.T) {
	cases := []struct {
		name                         string
		inC, outC, k, stride, groups int
	}{
		{"k4s2g1", 3, 5, 4, 2, 1},
		{"k4s2depthwise", 6, 6, 4, 2, 6}, // the Mimi upsample shape (depthwise)
		{"k16s8g1", 4, 3, 16, 8, 1},
		{"k12s6g1", 3, 2, 12, 6, 1},
		{"k3s1g1", 2, 3, 3, 1, 1},
		{"k8s4g2", 4, 6, 8, 4, 2},
	}
	rng := rand.New(rand.NewSource(7))
	for _, tc := range cases {
		for _, batch := range []int{1, 2} {
			for _, withBias := range []bool{false, true} {
				name := tc.name
				if batch > 1 {
					name += "_batch2"
				}
				if withBias {
					name += "_bias"
				}
				t.Run(name, func(t *testing.T) {
					L := 9
					input := randTensor(rng, batch, tc.inC, L)
					weight := randTensor(rng, tc.inC, tc.outC/tc.groups, tc.k)
					var bias *Tensor
					var biasData []float32
					if withBias {
						bias = randTensor(rng, tc.outC)
						biasData = bias.Data()
					}

					got := ConvTranspose1dForward(input, weight, bias, tc.stride, tc.groups)

					outL := (L-1)*tc.stride + tc.k
					wantShape := []int{batch, tc.outC, outL}
					gs := got.Shape()
					if gs[0] != wantShape[0] || gs[1] != wantShape[1] || gs[2] != wantShape[2] {
						t.Fatalf("shape %v, want %v", gs, wantShape)
					}
					want := naiveConvTranspose1d(input.Data(), weight.Data(), biasData,
						batch, tc.inC, L, tc.outC/tc.groups, tc.k, tc.stride, tc.groups)
					if d := maxAbsDiff32(got.Data(), want); d > 1e-5 {
						t.Errorf("max abs diff vs naive = %g, want <= 1e-5", d)
					}
				})
			}
		}
	}
}

// TestConvTranspose1dDepthwiseMatchesGeneral: the depthwise fast path
// (direct per-channel loop) must agree with the general grouped
// GEMM+col2im path on the same tensors. The k4s2 case (the Mimi
// upsample, at most 2 overlapping taps per output column, and f32
// two-term addition is order-independent) must be bit-exact; k3s1
// (3 overlapping taps, so accumulation order can differ by one
// rounding) is gated at 1e-6.
func TestConvTranspose1dDepthwiseMatchesGeneral(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	cases := []struct {
		name      string
		k, stride int
		exact     bool
	}{
		{"k4s2", 4, 2, true},
		{"k3s1", 3, 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const C, L, batch = 8, 10, 2
			input := randTensor(rng, batch, C, L)
			weight := randTensor(rng, C, 1, tc.k)
			outL := (L-1)*tc.stride + tc.k

			fast := make([]float32, batch*C*outL)
			convTranspose1dDepthwise(input.Data(), weight.Data(), batch, C, L, tc.k, tc.stride, fast)
			general := make([]float32, batch*C*outL)
			convTranspose1dGrouped(input.Data(), weight.Data(), batch, C, L, 1, tc.k, tc.stride, C, general)

			d := maxAbsDiff32(fast, general)
			if tc.exact && d != 0 {
				t.Errorf("depthwise fast path differs from general path: max abs diff %g, want bit-exact", d)
			}
			if d > 1e-6 {
				t.Errorf("depthwise fast path differs from general path: max abs diff %g, want <= 1e-6", d)
			}

			// Public entry point must dispatch to the fast path for
			// groups == inC == outC.
			pub := ConvTranspose1dForward(input, weight, nil, tc.stride, C)
			if d := maxAbsDiff32(pub.Data(), fast); d != 0 {
				t.Errorf("public depthwise dispatch differs from fast path by %g", d)
			}
		})
	}
}

// TestConvTranspose1dInferenceOnly: the op must panic when gradient
// tracking is enabled and an input requires grad (frozen-weight
// decoder decision, plan 0007 §2.1), and must run fine for the same
// tensors under NoGrad.
func TestConvTranspose1dInferenceOnly(t *testing.T) {
	x := RandN(1, 2, 5)
	w := RandN(2, 3, 4)
	w.SetRequiresGrad(true)

	func() {
		defer func() {
			if recover() == nil {
				t.Error("expected panic with grad enabled and requiresGrad weight")
			}
		}()
		ConvTranspose1dForward(x, w, nil, 2, 1)
	}()

	NoGrad(func() {
		out := ConvTranspose1dForward(x, w, nil, 2, 1)
		if s := out.Shape(); s[2] != (5-1)*2+4 {
			t.Errorf("NoGrad output length %d, want %d", s[2], (5-1)*2+4)
		}
	})
}

func randTensor(rng *rand.Rand, shape ...int) *Tensor {
	tt := Zeros(shape...)
	d := tt.Data()
	for i := range d {
		d[i] = float32(rng.NormFloat64())
	}
	return tt
}

func maxAbsDiff32(a, b []float32) float64 {
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
