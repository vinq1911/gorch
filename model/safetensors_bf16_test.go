//go:build darwin

package model

// Plan 0009 X3-B1: native bf16 safetensors round trip. Save must write
// bf16 tensors as dtype "BF16" bit-exactly (no widen/re-round cycle);
// LoadSafetensorsNative must keep the bf16 bits as data16;
// LoadSafetensors (the f32-compatibility loader) must widen the same
// file to f32.

import (
	"path/filepath"
	"testing"

	g "github.com/vinq1911/gorch"
)

func TestBF16SafetensorsNativeRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bf16_roundtrip.safetensors")

	// Values chosen so bf16 rounding is visible: 0.1 and 1e-3 are not
	// bf16-representable; 1.5 and -2 are.
	vals := []float32{0.1, 1.5, -2, 1e-3, 3.14159, -0.4567}
	bfBits := g.F32ToBF16Slice(vals)
	w := g.NewTensorBF16(bfBits, 2, 3)
	fvals := []float32{1, -2.5, 0.125, 9}
	fw := g.NewTensor(fvals, 4)

	if err := SaveSafetensors(path, map[string]*g.Tensor{"w_bf16": w, "w_f32": fw}); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Native load: bf16 stays bf16, bit-exact.
	sf, err := LoadSafetensorsNative(path)
	if err != nil {
		t.Fatalf("native load: %v", err)
	}
	got := sf.Tensors["w_bf16"]
	if got.Dtype() != g.BFloat16 {
		t.Fatalf("native load dtype = %v, want BF16", got.Dtype())
	}
	gotBits := got.Data16()
	if len(gotBits) != len(bfBits) {
		t.Fatalf("bf16 payload length %d, want %d", len(gotBits), len(bfBits))
	}
	for i := range bfBits {
		if gotBits[i] != bfBits[i] {
			t.Fatalf("bf16 bits[%d] = %#04x, want %#04x (round trip not bit-exact)", i, gotBits[i], bfBits[i])
		}
	}
	if sh := got.Shape(); sh[0] != 2 || sh[1] != 3 {
		t.Fatalf("bf16 shape = %v, want [2 3]", sh)
	}
	// F32 tensors are untouched by the native loader.
	gotF := sf.Tensors["w_f32"]
	if gotF.Dtype() != g.Float32 {
		t.Fatalf("native load f32 dtype = %v", gotF.Dtype())
	}
	for i, v := range fvals {
		if gotF.Data()[i] != v {
			t.Fatalf("f32[%d] = %g, want %g", i, gotF.Data()[i], v)
		}
	}

	// Compatibility load: the same file widens BF16 to F32.
	sfW, err := LoadSafetensors(path)
	if err != nil {
		t.Fatalf("compat load: %v", err)
	}
	wide := sfW.Tensors["w_bf16"]
	if wide.Dtype() != g.Float32 {
		t.Fatalf("compat load dtype = %v, want F32", wide.Dtype())
	}
	want := g.BF16ToF32Slice(bfBits)
	for i := range want {
		if wide.Data()[i] != want[i] {
			t.Fatalf("widened[%d] = %g, want %g", i, wide.Data()[i], want[i])
		}
	}
}
