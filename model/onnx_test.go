//go:build darwin

package model

import (
	"path/filepath"
	"testing"

	"github.com/vinq1911/gorch/nn"
)

func TestPBVarintRoundTrip(t *testing.T) {
	cases := []uint64{0, 1, 127, 128, 16383, 16384, 1 << 32, 1 << 60}
	for _, v := range cases {
		var p pbBuf
		p.putVarint(v)
		r := newPBReader(p.Bytes())
		got, err := r.readVarint()
		if err != nil {
			t.Fatalf("read %d: %v", v, err)
		}
		if got != v {
			t.Fatalf("varint %d round-trip got %d", v, got)
		}
	}
}

func TestPBStringRoundTrip(t *testing.T) {
	var p pbBuf
	p.PutString(1, "hello")
	p.PutString(2, "world")
	r := newPBReader(p.Bytes())
	for i := 0; i < 2; i++ {
		field, wire, err := r.readField()
		if err != nil || wire != wireLenDelim {
			t.Fatalf("read field %d: wire=%d err=%v", i, wire, err)
		}
		s, err := r.readString()
		if err != nil {
			t.Fatal(err)
		}
		switch field {
		case 1:
			if s != "hello" {
				t.Fatalf("got %q", s)
			}
		case 2:
			if s != "world" {
				t.Fatalf("got %q", s)
			}
		}
	}
}

func TestExportLinearAndReimportInitializers(t *testing.T) {
	// Build a tiny Sequential with deterministic weights.
	l1 := nn.NewLinear(4, 8)
	l2 := nn.NewLinear(8, 2)
	seq := nn.NewSequential(l1, nn.NewReLU(), l2)

	// Stamp known values so we can verify round-trip.
	for i := range l1.Weight.Data() {
		l1.Weight.Data()[i] = float32(i) * 0.1
	}
	for i := range l2.Weight.Data() {
		l2.Weight.Data()[i] = -float32(i) * 0.5
	}
	for i := range l1.Bias.Data() {
		l1.Bias.Data()[i] = float32(i) + 1
	}

	path := filepath.Join(t.TempDir(), "model.onnx")
	if err := ExportSequentialToONNX(seq, []int{1, 4}, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	loaded, err := LoadONNX(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Should have 4 initializers (W, B for each Linear).
	if got := len(loaded.Names); got != 4 {
		t.Fatalf("expected 4 initializers, got %d (%v)", got, loaded.Names)
	}

	// Should have 3 nodes (Gemm, Relu, Gemm).
	if got := len(loaded.Nodes); got != 3 {
		t.Fatalf("expected 3 nodes, got %d", got)
	}
	wantOps := []string{"Gemm", "Relu", "Gemm"}
	for i, op := range wantOps {
		if loaded.Nodes[i].OpType != op {
			t.Fatalf("node %d: got %q want %q", i, loaded.Nodes[i].OpType, op)
		}
	}

	// Verify a known weight survived round-trip. The first Linear's
	// weight tensor is the first initializer in our exporter ordering.
	first := loaded.Tensors[loaded.Names[0]]
	if first == nil {
		t.Fatal("first initializer not found")
	}
	if first.Size() != l1.Weight.Size() {
		t.Fatalf("size mismatch: got %d want %d", first.Size(), l1.Weight.Size())
	}
	for i, want := range l1.Weight.Data() {
		got := first.Data()[i]
		if got != want {
			t.Fatalf("element %d: got %g want %g", i, got, want)
			break
		}
	}
}

func TestExportCNNRoundTripsInitializers(t *testing.T) {
	// Tiny CNN: Conv2d → ReLU → MaxPool → Flatten → Linear.
	conv := nn.NewConv2d(1, 4, 3, 1, 1) // (1,8,8) -> (4,8,8)
	pool := nn.NewMaxPool2d(2, 2)        // -> (4,4,4)
	lin := nn.NewLinear(4*4*4, 10)
	seq := nn.NewSequential(conv, nn.NewReLU(), pool, nn.NewFlatten(), lin)

	path := filepath.Join(t.TempDir(), "cnn.onnx")
	if err := ExportSequentialToONNX(seq, []int{1, 1, 8, 8}, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	loaded, err := LoadONNX(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// 4 initializers: conv.W, conv.B, lin.W, lin.B.
	if got := len(loaded.Names); got != 4 {
		t.Fatalf("got %d initializers (%v)", got, loaded.Names)
	}
	// Node ops in order.
	wantOps := []string{"Conv", "Relu", "MaxPool", "Flatten", "Gemm"}
	if len(loaded.Nodes) != len(wantOps) {
		t.Fatalf("got %d nodes, want %d", len(loaded.Nodes), len(wantOps))
	}
	for i, op := range wantOps {
		if loaded.Nodes[i].OpType != op {
			t.Fatalf("node %d: got %q want %q", i, loaded.Nodes[i].OpType, op)
		}
	}
}

func TestExportRejectsUnsupportedLayer(t *testing.T) {
	// LayerNorm is intentionally not in the v1 exporter.
	seq := nn.NewSequential(nn.NewLayerNorm(4))
	err := ExportSequentialToONNX(seq, []int{1, 4}, filepath.Join(t.TempDir(), "x.onnx"))
	if err == nil {
		t.Fatal("expected error for unsupported layer, got nil")
	}
}

func TestExportLinearShapeMismatchErrors(t *testing.T) {
	l := nn.NewLinear(4, 8)
	seq := nn.NewSequential(l)
	err := ExportSequentialToONNX(seq, []int{1, 999}, filepath.Join(t.TempDir(), "x.onnx"))
	if err == nil {
		t.Fatal("expected shape mismatch error")
	}
}
