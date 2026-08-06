//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestEncodeBatchSingleMatchesEncode verifies that EncodeBatch with
// a batch of size 1 produces the same hidden states as the existing
// per-sequence Encode (within fp32 noise from a different batched
// matmul ordering).
func TestEncodeBatchSingleMatchesEncode(t *testing.T) {
	gpt := NewGPT(32, 16, 2, 2, 16)
	tokens := []int{1, 2, 3, 4, 5}

	want := gpt.Encode(tokens) // (5, 16)
	got := gpt.EncodeBatch([][]int{tokens})

	wantData := want.Data()
	gotData := got.Data()
	if got.Shape()[0] != 1 || got.Shape()[1] != len(tokens) || got.Shape()[2] != gpt.Dim {
		t.Fatalf("EncodeBatch shape = %v, want [1 5 16]", got.Shape())
	}
	if len(wantData) != len(gotData) {
		t.Fatalf("size mismatch: %d vs %d", len(wantData), len(gotData))
	}
	for i, w := range wantData {
		d := math.Abs(float64(w - gotData[i]))
		if d > 1e-3 {
			t.Fatalf("token %d dim %d: want %g got %g (diff %g)",
				i/gpt.Dim, i%gpt.Dim, w, gotData[i], d)
		}
	}
}

// TestEncodeBatchUniformLengthMatchesPerSequence: a batch of three
// equal-length sequences must produce the same hidden states for each
// sequence as if each were encoded alone (length mask still gates
// nothing because every position is "real" when lengths are equal,
// so this is the pure-batched-matmul correctness check).
func TestEncodeBatchUniformLengthMatchesPerSequence(t *testing.T) {
	gpt := NewGPT(32, 16, 2, 2, 16)
	batch := [][]int{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}

	got := gpt.EncodeBatch(batch)
	gotData := got.Data()

	for b, seq := range batch {
		want := gpt.Encode(seq)
		wantData := want.Data()
		offset := b * len(seq) * gpt.Dim
		batchSlice := gotData[offset : offset+len(seq)*gpt.Dim]
		for i, w := range wantData {
			d := math.Abs(float64(w - batchSlice[i]))
			if d > 1e-3 {
				t.Fatalf("seq %d pos %d dim %d: want %g got %g",
					b, i/gpt.Dim, i%gpt.Dim, w, batchSlice[i])
			}
		}
	}
}

// TestEncodeBatchVariableLengthMatchesPerSequence: padded short
// sequences must produce identical hidden states (within fp32 noise)
// to a single-sequence Encode of the un-padded version, in the rows
// 0..len-1. This is the real test of the length mask.
func TestEncodeBatchVariableLengthMatchesPerSequence(t *testing.T) {
	gpt := NewGPT(32, 16, 2, 2, 16)
	batch := [][]int{
		{1, 2, 3, 4, 5}, // len 5
		{7, 8},          // len 2
		{11, 12, 13},    // len 3
	}

	got := gpt.EncodeBatch(batch) // shape (3, 5, 16)
	gotData := got.Data()
	maxLen := 5
	dim := gpt.Dim

	for b, seq := range batch {
		want := gpt.Encode(seq) // (len(seq), dim)
		wantData := want.Data()
		for i := 0; i < len(seq); i++ {
			offset := (b*maxLen + i) * dim
			batchRow := gotData[offset : offset+dim]
			wantRow := wantData[i*dim : (i+1)*dim]
			for k := 0; k < dim; k++ {
				d := math.Abs(float64(wantRow[k] - batchRow[k]))
				if d > 5e-3 {
					t.Fatalf("seq %d (len %d) pos %d dim %d: want %g got %g (diff %g)",
						b, len(seq), i, k, wantRow[k], batchRow[k], d)
				}
			}
		}
	}
}

// TestEncodeBatchPadDoesNotLeak: changing the pad-token contents must
// not change the hidden states at real positions. This is the bit
// that justifies the length mask: pad keys/queries cannot influence
// real positions.
func TestEncodeBatchPadDoesNotLeak(t *testing.T) {
	gpt := NewGPT(32, 16, 2, 2, 16)
	short := []int{1, 2, 3}                 // len 3
	long := []int{1, 2, 3, 17, 18, 19}      // same prefix, then in-vocab noise

	// Real-positions of the short sequence in a length-6 batch.
	gotShort := gpt.EncodeBatch([][]int{short, long}).Data()
	standalone := gpt.Encode(short).Data()

	dim := gpt.Dim
	for i := 0; i < len(short); i++ {
		// Position i of batch row 0.
		batchRow := gotShort[(0*6+i)*dim : (0*6+i+1)*dim]
		stdRow := standalone[i*dim : (i+1)*dim]
		for k := 0; k < dim; k++ {
			d := math.Abs(float64(batchRow[k] - stdRow[k]))
			if d > 5e-3 {
				t.Fatalf("pad leakage at pos %d dim %d: standalone=%g batched=%g",
					i, k, stdRow[k], batchRow[k])
			}
		}
	}
	_ = g.Zeros // keep import live in any pruning
}

// TestEncodeBatchPanicOnTooLong panics if any sequence is over MaxSeq.
func TestEncodeBatchPanicOnTooLong(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for oversized sequence")
		}
	}()
	gpt := NewGPT(8, 4, 2, 1, 4)
	gpt.EncodeBatch([][]int{{1, 2, 3, 4, 5}}) // exceeds MaxSeq=4
}
