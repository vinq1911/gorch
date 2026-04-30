//go:build darwin

package gorch

import (
	"math"
	"testing"
)

// TestBF16RoundTripExact: every f32 value with zero in its low 16
// mantissa bits is exactly representable in bf16. Round-tripping
// such values must be lossless.
func TestBF16RoundTripExact(t *testing.T) {
	cases := []float32{0, 1, -1, 0.5, -0.5, 256, 1e-10, -1e10}
	for _, v := range cases {
		bits := math.Float32bits(v)
		// Strip the low 16 bits so the value is exactly representable.
		exact := math.Float32frombits(bits &^ 0xffff)
		got := bf16ToF32(f32ToBF16(exact))
		if got != exact {
			t.Fatalf("exact value %g round-tripped to %g", exact, got)
		}
	}
}

// TestBF16RoundTripApprox: arbitrary f32 values lose at most ~3 bits
// of precision (relative error < 1/256 ≈ 0.4%) through a bf16 round.
// The 7-bit mantissa gives ~3 decimal digits of precision.
func TestBF16RoundTripApprox(t *testing.T) {
	cases := []float32{0.123456, -0.987654, 3.14159, 2.71828, 0.001234}
	for _, v := range cases {
		got := bf16ToF32(f32ToBF16(v))
		rel := math.Abs(float64(got-v)) / math.Abs(float64(v))
		if rel > 1.0/128.0 {
			t.Fatalf("%g → %g, rel error %g > 1/128", v, got, rel)
		}
	}
}

// TestBF16PreservesSpecialValues: zero, +inf, -inf must round-trip.
func TestBF16PreservesSpecialValues(t *testing.T) {
	cases := []float32{0, float32(math.Inf(1)), float32(math.Inf(-1))}
	for _, v := range cases {
		got := bf16ToF32(f32ToBF16(v))
		if math.IsInf(float64(v), 1) && !math.IsInf(float64(got), 1) {
			t.Fatalf("+Inf lost: got %g", got)
		}
		if math.IsInf(float64(v), -1) && !math.IsInf(float64(got), -1) {
			t.Fatalf("-Inf lost: got %g", got)
		}
		if v == 0 && got != 0 {
			t.Fatalf("0 lost: got %g", got)
		}
	}
}

func TestNewTensorBF16Constructor(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6}
	t16 := NewTensorBF16(F32ToBF16Slice(src), 2, 3)
	if t16.Dtype() != BFloat16 {
		t.Fatalf("dtype = %v, want BF16", t16.Dtype())
	}
	if t16.data != nil {
		t.Fatal("BF16 tensor's data (f32) field should be nil")
	}
	if len(t16.data16) != 6 {
		t.Fatalf("data16 length = %d, want 6", len(t16.data16))
	}
	// Shape preserved.
	if shape := t16.Shape(); shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}
}

func TestNewTensorBF16ShapeMismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	NewTensorBF16(make([]uint16, 6), 2, 4)
}

// TestToF32FromBF16: round-trip through ToBF16 + ToF32 should be
// equivalent to a single conversion pair.
func TestToF32FromBF16(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	bf := a.ToBF16()
	if bf.Dtype() != BFloat16 {
		t.Fatalf("ToBF16 produced dtype %v", bf.Dtype())
	}
	back := bf.ToF32()
	if back.Dtype() != Float32 {
		t.Fatalf("ToF32 produced dtype %v", back.Dtype())
	}
	for i, v := range a.Data() {
		got := back.Data()[i]
		// Whole integers ≤ 256 are exactly representable in bf16.
		if v != got {
			t.Fatalf("[%d] %g round-tripped to %g", i, v, got)
		}
	}
}

func TestToF32IsCopyNotAlias(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, 4)
	cp := a.ToF32()
	cp.Data()[0] = 999
	if a.Data()[0] == 999 {
		t.Fatal("ToF32 returned an alias, not a copy")
	}
}

func TestDTypeString(t *testing.T) {
	if Float32.String() != "F32" {
		t.Fatalf("Float32.String() = %q", Float32.String())
	}
	if BFloat16.String() != "BF16" {
		t.Fatalf("BFloat16.String() = %q", BFloat16.String())
	}
}
