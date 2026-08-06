//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// Plan 0002 PR 2 tests: every op that grew bf16 dispatch should
// produce results that match its f32 baseline within bf16 tolerance,
// and gradients should flow through bf16 tensors.

const (
	// bf16TolForward is the worst-case relative tolerance allowed
	// when comparing a bf16 forward result to its f32 baseline. bf16
	// has 7 mantissa bits, giving ≈ 1/128 ≈ 7.8e-3 of representation
	// error per value; reductions over many elements can compound
	// into ~5e-2 in the worst case (loss-style sums of dozens of
	// bf16 inputs). PyTorch's bf16 tests use 8e-3 absolute / 5e-2
	// rel; this matches.
	bf16TolForward = 5e-2
	// bf16TolGrad allows looser comparison for gradients: each grad
	// element is itself a sum of bf16 forward + bf16 grad, doubling
	// the rounding noise.
	bf16TolGrad = 8e-2
)

// fillRand fills s with deterministic pseudo-random values keyed by
// the seed so different ops can use independent inputs without state.
func fillRand(s []float32, seed int) {
	x := uint32(seed*2654435761 + 1)
	for i := range s {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		// Map to [-1, 1) range; small-magnitude values keep softmax /
		// log / exp inside their well-conditioned domain.
		s[i] = float32(int32(x))/float32(1<<31) // [-1, 1)
	}
}

// closeRel returns true if got/want are within tol relative or 1e-4
// absolute (whichever is larger). The absolute floor handles
// near-zero comparisons.
func closeRel(got, want float32, tol float64) bool {
	d := math.Abs(float64(got - want))
	denom := math.Abs(float64(want))
	if denom < 1e-4 {
		return d < 1e-3
	}
	return d/denom < tol
}

// assertCloseSlice compares two slices and fails on the first element
// that exceeds tol relative error. Returns the worst observed
// relative error for diagnostic purposes.
func assertCloseSlice(t *testing.T, got, want []float32, tol float64, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch %d vs %d", label, len(got), len(want))
	}
	worst := 0.0
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		denom := math.Abs(float64(want[i]))
		var rel float64
		if denom < 1e-4 {
			rel = d / 1e-4
		} else {
			rel = d / denom
		}
		if rel > worst {
			worst = rel
		}
		if !closeRel(got[i], want[i], tol) {
			t.Fatalf("%s: [%d] got=%g want=%g rel=%g (tol=%g)", label, i, got[i], want[i], rel, tol)
		}
	}
}

// runForwardF32 runs an op in f32 and returns the f32 output.
func runForwardF32(op func(*Tensor) *Tensor, src []float32, shape ...int) *Tensor {
	a := NewTensor(src, shape...)
	return op(a)
}

// runForwardBF16 runs the SAME op against a bf16 tensor and returns
// the result widened to f32 for comparison. The op must dispatch on
// bf16 internally (the dispatch wrappers added in PR 2).
func runForwardBF16(op func(*Tensor) *Tensor, src []float32, shape ...int) *Tensor {
	a := NewTensorBF16(F32ToBF16Slice(src), shape...)
	out := op(a)
	if out.Dtype() != BFloat16 {
		// Some ops (e.g. CrossEntropyLoss) intentionally return f32
		// from a bf16 input. That's the documented behaviour; widen
		// either way for comparison.
		return out
	}
	return out.ToF32()
}

func TestBF16ForwardParityUnary(t *testing.T) {
	const N = 32
	src := make([]float32, N)
	fillRand(src, 1)

	cases := []struct {
		name string
		op   func(*Tensor) *Tensor
	}{
		{"Neg", Neg},
		{"ReLU", ReLU},
		{"Sigmoid", Sigmoid},
		{"Tanh", Tanh},
		{"GELU", GELU},
		{"SiLU", SiLU},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f32 := runForwardF32(c.op, src, N)
			bf := runForwardBF16(c.op, src, N)
			assertCloseSlice(t, bf.Data(), f32.Data(), bf16TolForward, c.name)
		})
	}
}

func TestBF16ForwardParityBinary(t *testing.T) {
	const N = 16
	a := make([]float32, N)
	b := make([]float32, N)
	fillRand(a, 2)
	fillRand(b, 3)
	// Avoid div-by-zero for Div.
	for i := range b {
		if math.Abs(float64(b[i])) < 0.1 {
			b[i] += 0.5
		}
	}
	// Pre-round inputs to bf16 then back to f32 so the f32 baseline
	// operates on the same quantised inputs as the bf16 path. This
	// makes the test pin the *op*'s rounding contribution, not the
	// input quantisation error (which would dominate for small
	// values where one ULP at bf16 precision is several percent).
	aQ := BF16ToF32Slice(F32ToBF16Slice(a))
	bQ := BF16ToF32Slice(F32ToBF16Slice(b))

	cases := []struct {
		name string
		op   func(a, b *Tensor) *Tensor
	}{
		{"Add", Add},
		{"Sub", Sub},
		{"Mul", Mul},
		{"Div", Div},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			af32 := NewTensor(aQ, N)
			bf32 := NewTensor(bQ, N)
			outF := c.op(af32, bf32)

			abf := NewTensorBF16(F32ToBF16Slice(a), N)
			bbf := NewTensorBF16(F32ToBF16Slice(b), N)
			outBF := c.op(abf, bbf)
			assertCloseSlice(t, outBF.ToF32().Data(), outF.Data(), bf16TolForward, c.name)
		})
	}
}

func TestBF16ForwardParityMatMul(t *testing.T) {
	const M, K, N = 8, 16, 8
	a := make([]float32, M*K)
	b := make([]float32, K*N)
	fillRand(a, 4)
	fillRand(b, 5)

	af32 := NewTensor(a, M, K)
	bf32 := NewTensor(b, K, N)
	outF := MatMul(af32, bf32)

	abf := NewTensorBF16(F32ToBF16Slice(a), M, K)
	bbf := NewTensorBF16(F32ToBF16Slice(b), K, N)
	outBF := MatMul(abf, bbf)

	if outBF.Dtype() != BFloat16 {
		t.Fatalf("MatMul(bf16, bf16) returned dtype %v", outBF.Dtype())
	}
	assertCloseSlice(t, outBF.ToF32().Data(), outF.Data(), bf16TolForward, "MatMul")
}

func TestBF16ForwardParitySoftmax(t *testing.T) {
	const B, C = 4, 8
	src := make([]float32, B*C)
	fillRand(src, 6)

	outF := Softmax(NewTensor(src, B, C))
	outBF := Softmax(NewTensorBF16(F32ToBF16Slice(src), B, C))
	assertCloseSlice(t, outBF.ToF32().Data(), outF.Data(), bf16TolForward, "Softmax")
}

func TestBF16ForwardParityReductions(t *testing.T) {
	const N = 32
	src := make([]float32, N)
	fillRand(src, 7)

	outFSum := Sum(NewTensor(src, N))
	outBFSum := Sum(NewTensorBF16(F32ToBF16Slice(src), N))
	assertCloseSlice(t, outBFSum.ToF32().Data(), outFSum.Data(), bf16TolForward, "Sum")

	outFMean := Mean(NewTensor(src, N))
	outBFMean := Mean(NewTensorBF16(F32ToBF16Slice(src), N))
	assertCloseSlice(t, outBFMean.ToF32().Data(), outFMean.Data(), bf16TolForward, "Mean")
}

// TestBF16AutogradAdd: gradient through Add(bf16, bf16) should
// produce bf16 gradients that match their f32 baseline within
// tolerance.
func TestBF16AutogradAdd(t *testing.T) {
	const N = 16
	a := make([]float32, N)
	b := make([]float32, N)
	fillRand(a, 8)
	fillRand(b, 9)

	// f32 baseline: c = a + b, scalar = sum(c), backward.
	af := NewTensor(a, N).SetRequiresGrad(true)
	bf := NewTensor(b, N).SetRequiresGrad(true)
	cf := Add(af, bf)
	lossF := Sum(cf)
	lossF.Backward()

	abf := NewTensorBF16(F32ToBF16Slice(a), N).SetRequiresGrad(true)
	bbf := NewTensorBF16(F32ToBF16Slice(b), N).SetRequiresGrad(true)
	cbf := Add(abf, bbf)
	lossBF := Sum(cbf)
	lossBF.Backward()

	if abf.Grad() == nil {
		t.Fatal("bf16 a.grad is nil after backward")
	}
	if abf.Grad().Dtype() != BFloat16 {
		t.Fatalf("bf16 a.grad has dtype %v, want BF16", abf.Grad().Dtype())
	}
	assertCloseSlice(t, abf.Grad().ToF32().Data(), af.Grad().Data(), bf16TolGrad, "Add d/da")
	assertCloseSlice(t, bbf.Grad().ToF32().Data(), bf.Grad().Data(), bf16TolGrad, "Add d/db")
}

// TestBF16AutogradMatMul: gradient through MatMul(bf16, bf16) is the
// real-world case (linear layer + bf16 weights). Both d/da and d/db
// must match the f32 baseline.
func TestBF16AutogradMatMul(t *testing.T) {
	const M, K, N = 4, 6, 4
	a := make([]float32, M*K)
	b := make([]float32, K*N)
	fillRand(a, 10)
	fillRand(b, 11)

	af := NewTensor(a, M, K).SetRequiresGrad(true)
	bf := NewTensor(b, K, N).SetRequiresGrad(true)
	cf := MatMul(af, bf)
	lossF := Sum(cf)
	lossF.Backward()

	abf := NewTensorBF16(F32ToBF16Slice(a), M, K).SetRequiresGrad(true)
	bbf := NewTensorBF16(F32ToBF16Slice(b), K, N).SetRequiresGrad(true)
	cbf := MatMul(abf, bbf)
	lossBF := Sum(cbf)
	lossBF.Backward()

	if abf.Grad() == nil || bbf.Grad() == nil {
		t.Fatal("bf16 grads missing after MatMul backward")
	}
	assertCloseSlice(t, abf.Grad().ToF32().Data(), af.Grad().Data(), bf16TolGrad, "MatMul d/da")
	assertCloseSlice(t, bbf.Grad().ToF32().Data(), bf.Grad().Data(), bf16TolGrad, "MatMul d/db")
}

// TestBF16AutogradGELU: GELU has a non-trivial backward; this checks
// the up/downcast graph correctly threads the gradient through.
func TestBF16AutogradGELU(t *testing.T) {
	const N = 16
	src := make([]float32, N)
	fillRand(src, 12)

	af := NewTensor(src, N).SetRequiresGrad(true)
	yf := GELU(af)
	lossF := Sum(yf)
	lossF.Backward()

	abf := NewTensorBF16(F32ToBF16Slice(src), N).SetRequiresGrad(true)
	ybf := GELU(abf)
	lossBF := Sum(ybf)
	lossBF.Backward()

	if abf.Grad() == nil {
		t.Fatal("bf16 a.grad is nil after backward through GELU")
	}
	assertCloseSlice(t, abf.Grad().ToF32().Data(), af.Grad().Data(), bf16TolGrad, "GELU dx")
}

// TestBF16AutogradLinearChain: a small synthetic Linear-style chain
// (MatMul → AddBias → GELU → Sum) with bf16 weights. End-to-end check
// that the upcast/downcast bridges chain together without losing
// gradient.
func TestBF16AutogradLinearChain(t *testing.T) {
	const M, K, N = 4, 8, 8
	x := make([]float32, M*K)
	w := make([]float32, K*N)
	bv := make([]float32, N)
	fillRand(x, 13)
	fillRand(w, 14)
	fillRand(bv, 15)

	// f32 baseline.
	xF := NewTensor(x, M, K).SetRequiresGrad(true)
	wF := NewTensor(w, K, N).SetRequiresGrad(true)
	bF := NewTensor(bv, N).SetRequiresGrad(true)
	hF := MatMul(xF, wF)
	hF = AddBias(hF, bF)
	hF = GELU(hF)
	lossF := Sum(hF)
	lossF.Backward()

	// bf16 chain.
	xB := NewTensorBF16(F32ToBF16Slice(x), M, K).SetRequiresGrad(true)
	wB := NewTensorBF16(F32ToBF16Slice(w), K, N).SetRequiresGrad(true)
	bB := NewTensorBF16(F32ToBF16Slice(bv), N).SetRequiresGrad(true)
	hB := MatMul(xB, wB)
	hB = AddBias(hB, bB)
	hB = GELU(hB)
	lossB := Sum(hB)
	lossB.Backward()

	if wB.Grad() == nil || bB.Grad() == nil {
		t.Fatal("bf16 chain produced nil gradients on parameters")
	}
	if wB.Grad().Dtype() != BFloat16 {
		t.Fatalf("bf16 weight grad has dtype %v, want BF16", wB.Grad().Dtype())
	}
	// Forward parity:
	assertCloseSlice(t, hB.ToF32().Data(), hF.Data(), bf16TolForward, "chain forward")
	// Param grad parity. The chain compounds bf16 rounding noise
	// across three ops; allow the looser grad tolerance.
	assertCloseSlice(t, wB.Grad().ToF32().Data(), wF.Grad().Data(), bf16TolGrad, "chain dW")
	assertCloseSlice(t, bB.Grad().ToF32().Data(), bF.Grad().Data(), bf16TolGrad, "chain dBias")
}

// TestBF16DtypeMismatchPanics: mixed-dtype inputs must panic with a
// helpful message — PR 2 deliberately doesn't promote.
func TestBF16DtypeMismatchPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on mixed-dtype Add")
		}
	}()
	a := NewTensor([]float32{1, 2}, 2)
	b := NewTensorBF16(F32ToBF16Slice([]float32{3, 4}), 2)
	Add(a, b)
}

// TestBF16ReshapePreservesDtype: Reshape and ReshapeOp on a bf16
// tensor must produce a bf16 tensor (not silently corrupt to f32 with
// a nil .data slice).
func TestBF16ReshapePreservesDtype(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6}
	bf := NewTensorBF16(F32ToBF16Slice(src), 2, 3)

	r1 := bf.Reshape(3, 2)
	if r1.Dtype() != BFloat16 {
		t.Fatalf("Reshape lost dtype: got %v", r1.Dtype())
	}
	if r1.data16 == nil {
		t.Fatal("Reshape result has nil bf16 storage")
	}

	r2 := ReshapeOp(bf, 6)
	if r2.Dtype() != BFloat16 {
		t.Fatalf("ReshapeOp lost dtype: got %v", r2.Dtype())
	}
}

// TestBF16DetachPreservesDtype: Detach must keep the bf16 storage
// alive — otherwise inference goroutines holding bf16 weights would
// dereference nil.
func TestBF16DetachPreservesDtype(t *testing.T) {
	bf := NewTensorBF16(F32ToBF16Slice([]float32{1, 2, 3, 4}), 4).SetRequiresGrad(true)
	d := bf.Detach()
	if d.Dtype() != BFloat16 || d.data16 == nil {
		t.Fatalf("Detach dropped bf16 storage: dtype=%v, data16=%v", d.Dtype(), d.data16)
	}
	if d.RequiresGrad() {
		t.Fatal("Detach kept requires_grad=true")
	}
}

// TestBF16CrossEntropyAcceptsBF16: cross-entropy intentionally returns
// a fp32 loss when given bf16 logits (mixed-precision standard).
// This test pins the contract.
func TestBF16CrossEntropyAcceptsBF16(t *testing.T) {
	logits := []float32{1.5, -0.5, 0.2, -1.0, 0.7, 0.3}
	targets := []float32{0, 2}
	bf := NewTensorBF16(F32ToBF16Slice(logits), 2, 3).SetRequiresGrad(true)
	tt := NewTensor(targets, 2, 1)
	loss := CrossEntropyLoss(bf, tt)
	if loss.Dtype() != Float32 {
		t.Fatalf("CrossEntropyLoss(bf16) dtype = %v, want F32", loss.Dtype())
	}
	if math.IsNaN(float64(loss.Data()[0])) || math.IsInf(float64(loss.Data()[0]), 0) {
		t.Fatalf("CrossEntropyLoss returned %g", loss.Data()[0])
	}
	loss.Backward()
	if bf.Grad() == nil {
		t.Fatal("bf16 logits.Grad() is nil after CE backward")
	}
	if bf.Grad().Dtype() != BFloat16 {
		t.Fatalf("CE backward bf16 grad dtype = %v, want BF16", bf.Grad().Dtype())
	}
}
