//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// naiveConv1d is a straightforward triple-loop reference for
// Conv1dForward, with virtual (non-materialised) padding.
func naiveConv1d(input, weight, bias []float32, batch, inC, L, outC, k, stride, dilation, padL, padR int, mode PadMode) []float32 {
	kEff := (k-1)*dilation + 1
	outL := (L+padL+padR-kEff)/stride + 1
	get := func(b, c, pos int) float32 {
		ip := pos - padL // position in the unpadded input
		if ip < 0 {
			if mode == PadReplicate {
				ip = 0
			} else {
				return 0
			}
		}
		if ip >= L {
			if mode == PadReplicate {
				ip = L - 1
			} else {
				return 0
			}
		}
		return input[(b*inC+c)*L+ip]
	}
	out := make([]float32, batch*outC*outL)
	for b := 0; b < batch; b++ {
		for oc := 0; oc < outC; oc++ {
			for o := 0; o < outL; o++ {
				var acc float32
				for ic := 0; ic < inC; ic++ {
					for kx := 0; kx < k; kx++ {
						w := weight[(oc*inC+ic)*k+kx]
						acc += w * get(b, ic, o*stride+kx*dilation)
					}
				}
				if bias != nil {
					acc += bias[oc]
				}
				out[(b*outC+oc)*outL+o] = acc
			}
		}
	}
	return out
}

// TestConv1dForwardMatchesNaive: Conv1dForward (im2col + Sgemm) must
// agree with the triple-loop reference across stride/dilation/padding
// combinations, both pad modes, with and without bias.
func TestConv1dForwardMatchesNaive(t *testing.T) {
	type cfg struct {
		batch, inC, L, outC, k, stride, dilation, padL, padR int
		mode                                                 PadMode
		bias                                                 bool
	}
	cases := []cfg{
		{2, 3, 12, 4, 3, 1, 1, 0, 0, PadConstant, true},
		{1, 2, 16, 5, 4, 2, 1, 2, 0, PadConstant, true},
		{2, 3, 15, 2, 3, 1, 2, 3, 1, PadConstant, false},
		{1, 1, 20, 3, 7, 1, 1, 6, 0, PadConstant, true},
		{2, 2, 18, 3, 8, 4, 1, 4, 2, PadConstant, true},
		{1, 4, 10, 4, 1, 1, 1, 0, 0, PadConstant, false},
		{2, 3, 12, 4, 3, 1, 1, 2, 1, PadReplicate, true},
		{1, 2, 14, 3, 4, 2, 1, 2, 0, PadReplicate, false},
		{2, 2, 16, 2, 3, 3, 2, 4, 2, PadReplicate, true},
		{1, 3, 9, 2, 5, 2, 1, 3, 2, PadReplicate, true},
	}
	for ci, c := range cases {
		input := RandN(c.batch, c.inC, c.L)
		weight := RandN(c.outC, c.inC, c.k)
		var bias *Tensor
		var biasData []float32
		if c.bias {
			bias = RandN(c.outC)
			biasData = bias.Data()
		}
		got := Conv1dForward(input, weight, bias, c.stride, c.dilation, c.padL, c.padR, c.mode)
		want := naiveConv1d(input.Data(), weight.Data(), biasData,
			c.batch, c.inC, c.L, c.outC, c.k, c.stride, c.dilation, c.padL, c.padR, c.mode)

		kEff := (c.k-1)*c.dilation + 1
		outL := (c.L+c.padL+c.padR-kEff)/c.stride + 1
		wantShape := []int{c.batch, c.outC, outL}
		if !sameShape(got.Shape(), wantShape) {
			t.Fatalf("case %d: shape %v, want %v", ci, got.Shape(), wantShape)
		}
		for i := range want {
			if math.Abs(float64(got.Data()[i]-want[i])) > 1e-4 {
				t.Fatalf("case %d [%d]: got %g, want %g", ci, i, got.Data()[i], want[i])
			}
		}
	}
}

// conv1dLoss builds a scalar loss with non-uniform per-element
// weighting so the numerical gradient check exercises every output
// position independently.
func conv1dLoss(input, weight, bias, mask *Tensor, stride, dilation, padL, padR int, mode PadMode) *Tensor {
	return Sum(Mul(Conv1dForward(input, weight, bias, stride, dilation, padL, padR, mode), mask))
}

// TestConv1dBackwardMatchesNumerical: analytic gradients for input,
// weight and bias vs central differences (repo convention, see
// reshape_batched_grad_test.go). Covers constant and replicate
// padding, stride and dilation.
func TestConv1dBackwardMatchesNumerical(t *testing.T) {
	type cfg struct {
		stride, dilation, padL, padR int
		mode                         PadMode
	}
	cases := []cfg{
		{1, 1, 2, 1, PadConstant},
		{2, 1, 2, 0, PadConstant},
		{1, 2, 4, 2, PadConstant},
		{1, 1, 2, 1, PadReplicate},
		{2, 1, 3, 2, PadReplicate},
	}
	const batch, inC, L, outC, k = 2, 2, 8, 3, 3
	for ci, c := range cases {
		input := RandN(batch, inC, L).SetRequiresGrad(true)
		weight := RandN(outC, inC, k).SetRequiresGrad(true)
		bias := RandN(outC).SetRequiresGrad(true)

		kEff := (k-1)*c.dilation + 1
		outL := (L+c.padL+c.padR-kEff)/c.stride + 1
		mask := RandN(batch, outC, outL)

		loss := conv1dLoss(input, weight, bias, mask, c.stride, c.dilation, c.padL, c.padR, c.mode)
		loss.Backward()
		dIn := append([]float32{}, input.Grad().Data()...)
		dW := append([]float32{}, weight.Grad().Data()...)
		dB := append([]float32{}, bias.Grad().Data()...)

		const h = 1e-3
		check := func(name string, x *Tensor, ana []float32) {
			t.Helper()
			for i := range x.Data() {
				orig := x.Data()[i]
				x.Data()[i] = orig + h
				yPlus := conv1dLoss(input, weight, bias, mask, c.stride, c.dilation, c.padL, c.padR, c.mode).Data()[0]
				x.Data()[i] = orig - h
				yMinus := conv1dLoss(input, weight, bias, mask, c.stride, c.dilation, c.padL, c.padR, c.mode).Data()[0]
				x.Data()[i] = orig
				num := (yPlus - yMinus) / (2 * h)
				if math.Abs(float64(ana[i]-num)) > 5e-2 {
					t.Fatalf("case %d %s[%d]: analytic=%g numeric=%g", ci, name, i, ana[i], num)
				}
			}
		}
		check("dInput", input, dIn)
		check("dWeight", weight, dW)
		check("dBias", bias, dB)
	}
}

// TestConv1dNilBias: nil bias must work in forward and backward.
func TestConv1dNilBias(t *testing.T) {
	input := RandN(1, 2, 10).SetRequiresGrad(true)
	weight := RandN(3, 2, 3).SetRequiresGrad(true)
	out := Conv1dForward(input, weight, nil, 1, 1, 2, 0, PadConstant)
	if !sameShape(out.Shape(), []int{1, 3, 10}) {
		t.Fatalf("shape %v, want [1 3 10]", out.Shape())
	}
	Sum(out).Backward()
	if input.Grad() == nil || weight.Grad() == nil {
		t.Fatal("nil-bias Conv1d lost gradients")
	}
}

// TestIm2col1dCol2im1dRoundTrip: col2im1d(im2col1d(x)) must count each
// input position exactly as often as it appears in a window.
func TestIm2col1dCol2im1dRoundTrip(t *testing.T) {
	const C, L, k, stride, dilation = 2, 11, 3, 2, 2
	kEff := (k-1)*dilation + 1
	outL := (L-kEff)/stride + 1
	input := RandN(C * L)
	col := make([]float32, C*k*outL)
	im2col1d(input.Data(), C, L, k, stride, dilation, col)

	// Accumulating ones through col2im1d yields per-position window counts.
	ones := make([]float32, len(col))
	for i := range ones {
		ones[i] = 1
	}
	counts := make([]float32, C*L)
	col2im1d(ones, C, L, k, stride, dilation, counts)
	for c := 0; c < C; c++ {
		for pos := 0; pos < L; pos++ {
			var want float32
			for kx := 0; kx < k; kx++ {
				rel := pos - kx*dilation
				if rel >= 0 && rel%stride == 0 && rel/stride < outL {
					want++
				}
			}
			if counts[c*L+pos] != want {
				t.Fatalf("count[%d,%d] = %g, want %g", c, pos, counts[c*L+pos], want)
			}
		}
	}
}
