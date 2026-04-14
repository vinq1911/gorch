//go:build darwin

package model

import (
	"testing"

	"github.com/vinq1911/gorch/nn"
)

func TestArgmax(t *testing.T) {
	logits := []float32{1, 5, 3, 2, 4}
	if argmax(logits) != 1 {
		t.Fatalf("argmax = %d, want 1", argmax(logits))
	}
}

func TestSampleGreedy(t *testing.T) {
	logits := []float32{1, 10, 3, 2}
	// Temperature 0 should be greedy
	// sample with very low temperature should pick index 1
	idx := sample(logits, 0.01, 0, 0)
	if idx != 1 {
		t.Fatalf("near-greedy sample = %d, want 1", idx)
	}
}

func TestKthLargest(t *testing.T) {
	data := []float32{3, 1, 4, 1, 5, 9, 2, 6}
	// 1st largest = 9
	if kthLargest(data, 1) != 9 {
		t.Fatalf("1st largest = %f, want 9", kthLargest(data, 1))
	}
	// 3rd largest = 5
	if kthLargest(data, 3) != 5 {
		t.Fatalf("3rd largest = %f, want 5", kthLargest(data, 3))
	}
}

func TestKVCache(t *testing.T) {
	kv := NewKVCache(2, 4, 8) // 2 layers, 4 heads, headDim=8
	if kv.Len() != 0 {
		t.Fatalf("initial len = %d, want 0", kv.Len())
	}

	// Add one token's worth of KV for layer 0, head 0
	key := make([]float32, 8)
	val := make([]float32, 8)
	for i := range key {
		key[i] = float32(i)
		val[i] = float32(i * 10)
	}
	kv.Append(0, 0, key, val)

	if kv.Len() != 1 {
		t.Fatalf("len after append = %d, want 1", kv.Len())
	}

	// Add another token
	kv.Append(0, 0, key, val)
	if kv.Len() != 2 {
		t.Fatalf("len after 2nd append = %d, want 2", kv.Len())
	}

	// Reset
	kv.Reset()
	if kv.Len() != 0 {
		t.Fatalf("len after reset = %d, want 0", kv.Len())
	}
}

func TestGenerateWithConfig(t *testing.T) {
	// Small model, greedy decode
	model := nn.NewGPT(20, 8, 2, 1, 32)
	input := []int{1, 5, 10}

	cfg := GreedyConfig(5)
	output := GenerateWithConfig(model, input, cfg)

	if len(output) != 8 { // 3 prompt + 5 new
		t.Fatalf("output len = %d, want 8", len(output))
	}
	// Greedy should be deterministic
	output2 := GenerateWithConfig(model, input, cfg)
	for i := range output {
		if output[i] != output2[i] {
			t.Fatal("greedy decode not deterministic")
		}
	}
}

func TestGenerateWithSampling(t *testing.T) {
	model := nn.NewGPT(20, 8, 2, 1, 32)
	input := []int{1, 5}

	cfg := DefaultGenerateConfig()
	cfg.MaxNewTokens = 10

	output := GenerateWithConfig(model, input, cfg)
	if len(output) != 12 {
		t.Fatalf("output len = %d, want 12", len(output))
	}
}
