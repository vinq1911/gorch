//go:build darwin && e2e

package e2e

// Plan 0009 X3-B5: memory accounting at the 0.6B geometry. The full
// frozen weight set (168320×1024 embedding table + 28 × the taBlock
// Linear weights) is allocated bf16 on Metal and the measured live
// buffer bytes must land at ≈1.23 GB — half the ≈2.46 GB the same set
// costs in f32 (also measured). Uses metal.LiveBufferBytes because Go's
// HeapAlloc cannot see MTLBuffer memory.

import (
	"runtime"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/metal"
)

// taWeightShapes is the 0.6B frozen weight set: embedding + per-layer
// projection matrices (norm gammas and biases are trainable/f32 and
// excluded — they are <0.1% of bytes).
func taWeightShapes() [][2]int {
	shapes := [][2]int{{taVocab, taHidden}} // embedding/lm_head table (tied)
	for l := 0; l < taLayers; l++ {
		shapes = append(shapes,
			[2]int{taQDim, taHidden},  // wq
			[2]int{taKVDim, taHidden}, // wk
			[2]int{taKVDim, taHidden}, // wv
			[2]int{taHidden, taQDim},  // wo
			[2]int{taInter, taHidden}, // gate
			[2]int{taInter, taHidden}, // up
			[2]int{taHidden, taInter}, // down
		)
	}
	return shapes
}

func TestBF16MemoryAccounting(t *testing.T) {
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	dev := g.MetalDev()
	shapes := taWeightShapes()
	totalParams := 0
	for _, s := range shapes {
		totalParams += s[0] * s[1]
	}
	t.Logf("0.6B frozen weight set: %d params (%.1fM)", totalParams, float64(totalParams)/1e6)

	measure := func(alloc func(rows, cols int) *g.Tensor) float64 {
		taFlushGC()
		base := metal.LiveBufferBytes()
		tensors := make([]*g.Tensor, 0, len(shapes))
		for _, s := range shapes {
			tensors = append(tensors, alloc(s[0], s[1]))
		}
		live := float64(metal.LiveBufferBytes()-base) / 1e9
		runtime.KeepAlive(tensors)
		tensors = nil
		taFlushGC()
		time.Sleep(50 * time.Millisecond)
		taFlushGC()
		return live
	}

	// f32 set: expect ≈ totalParams × 4 B ≈ 2.46 GB.
	f32GB := measure(func(rows, cols int) *g.Tensor {
		return g.ZerosOnMetal(dev, rows, cols)
	})
	// bf16 set: expect ≈ totalParams × 2 B ≈ 1.23 GB.
	bf16GB := measure(func(rows, cols int) *g.Tensor {
		return g.NewTensorBF16OnMetal(dev, make([]uint16, rows*cols), rows, cols)
	})

	wantF32 := float64(totalParams) * 4 / 1e9
	wantBF16 := float64(totalParams) * 2 / 1e9
	t.Logf("measured Metal-resident weights: f32 %.3f GB (expect %.3f), bf16 %.3f GB (expect %.3f), ratio %.3f",
		f32GB, wantF32, bf16GB, wantBF16, f32GB/bf16GB)

	// ±5% envelopes (Metal page rounding on 176 buffers is far below this).
	if f32GB < wantF32*0.95 || f32GB > wantF32*1.05 {
		t.Errorf("f32 weight set measured %.3f GB, want %.3f GB ±5%%", f32GB, wantF32)
	}
	if bf16GB < wantBF16*0.95 || bf16GB > wantBF16*1.05 {
		t.Errorf("bf16 weight set measured %.3f GB, want %.3f GB ±5%% (plan gate: ≈1.23 GB)", bf16GB, wantBF16)
	}
	if r := f32GB / bf16GB; r < 1.9 || r > 2.1 {
		t.Errorf("f32/bf16 byte ratio %.3f, want ≈2", r)
	}
}
