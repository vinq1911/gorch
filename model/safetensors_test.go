//go:build darwin

package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	g "github.com/vinq1911/gorch"
)

func approx(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-4
}

func TestSaveThenLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	// Create tensors
	tensors := map[string]*g.Tensor{
		"weight": g.NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
		"bias":   g.NewTensor([]float32{0.1, 0.2, 0.3}, 3),
	}

	// Save
	if err := SaveSafetensors(path, tensors); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Verify file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("file is empty")
	}

	// Load
	loaded, err := LoadSafetensors(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Verify
	if len(loaded.Tensors) != 2 {
		t.Fatalf("loaded %d tensors, want 2", len(loaded.Tensors))
	}

	w := loaded.Tensors["weight"]
	if w == nil {
		t.Fatal("weight tensor not found")
	}
	if w.Shape()[0] != 2 || w.Shape()[1] != 3 {
		t.Fatalf("weight shape = %v, want [2,3]", w.Shape())
	}
	for i, want := range []float32{1, 2, 3, 4, 5, 6} {
		if !approx(w.Data()[i], want) {
			t.Fatalf("weight[%d] = %f, want %f", i, w.Data()[i], want)
		}
	}

	b := loaded.Tensors["bias"]
	if b == nil {
		t.Fatal("bias tensor not found")
	}
	for i, want := range []float32{0.1, 0.2, 0.3} {
		if !approx(b.Data()[i], want) {
			t.Fatalf("bias[%d] = %f, want %f", i, b.Data()[i], want)
		}
	}
}

func TestSaveLoadModelWeights(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")

	// Simulate model parameters
	params := []*g.Tensor{
		g.NewTensor([]float32{1, 2, 3, 4}, 2, 2),
		g.NewTensor([]float32{0.5, 0.5}, 2),
	}

	// Save with name mapping
	nameMap := map[int]string{0: "layer.weight", 1: "layer.bias"}
	if err := SaveModelWeights(path, params, nameMap); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Create fresh parameters (zeros)
	newParams := []*g.Tensor{
		g.Zeros(2, 2),
		g.Zeros(2),
	}

	// Load into new parameters
	loadMap := map[string]int{"layer.weight": 0, "layer.bias": 1}
	if err := LoadModelWeights(path, newParams, loadMap); err != nil {
		t.Fatalf("load: %v", err)
	}

	// Verify values were loaded
	for i, want := range []float32{1, 2, 3, 4} {
		if !approx(newParams[0].Data()[i], want) {
			t.Fatalf("weight[%d] = %f, want %f", i, newParams[0].Data()[i], want)
		}
	}
}

func TestFloat16Conversion(t *testing.T) {
	// Test known F16 values
	// F16 for 1.0 is 0x3C00
	// F16 for -2.0 is 0xC000
	// F16 for 0.5 is 0x3800
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x3C00, 1.0},
		{0xC000, -2.0},
		{0x3800, 0.5},
		{0x0000, 0.0},
		{0x7C00, float32(math.Inf(1))},
	}

	for _, tt := range tests {
		got := float16ToFloat32(tt.bits)
		if math.IsInf(float64(tt.want), 1) {
			if !math.IsInf(float64(got), 1) {
				t.Fatalf("f16(%04x) = %f, want +Inf", tt.bits, got)
			}
		} else if !approx(got, tt.want) {
			t.Fatalf("f16(%04x) = %f, want %f", tt.bits, got, tt.want)
		}
	}
}
