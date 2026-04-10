//go:build darwin

package data

import (
	"testing"
)

// simpleDataset is a toy dataset for testing the DataLoader.
type simpleDataset struct {
	n int
}

func (d *simpleDataset) Len() int                                { return d.n }
func (d *simpleDataset) InputShape() []int                       { return []int{2} }
func (d *simpleDataset) TargetShape() []int                      { return []int{1} }
func (d *simpleDataset) Get(i int) ([]float32, []float32) {
	return []float32{float32(i), float32(i * 10)}, []float32{float32(i % 2)}
}

func TestDataLoaderBasic(t *testing.T) {
	ds := &simpleDataset{n: 10}
	dl := NewDataLoader(ds, 3, false)

	dl.Reset()
	var totalSamples int
	batches := 0
	for {
		inp, tgt := dl.Next()
		if inp == nil {
			break
		}
		batches++
		totalSamples += inp.Shape()[0]
		if inp.Shape()[1] != 2 {
			t.Fatalf("input feature dim = %d, want 2", inp.Shape()[1])
		}
		if tgt.Shape()[1] != 1 {
			t.Fatalf("target dim = %d, want 1", tgt.Shape()[1])
		}
	}
	if totalSamples != 10 {
		t.Fatalf("total samples = %d, want 10", totalSamples)
	}
	if batches != 4 { // ceil(10/3) = 4
		t.Fatalf("batches = %d, want 4", batches)
	}
}

func TestDataLoaderShuffle(t *testing.T) {
	ds := &simpleDataset{n: 100}
	dl := NewDataLoader(ds, 10, true)

	// Run two epochs and check that order differs
	dl.Reset()
	var first []float32
	for {
		inp, _ := dl.Next()
		if inp == nil {
			break
		}
		first = append(first, inp.Data()[0]) // first element of each batch
	}

	dl.Reset()
	var second []float32
	for {
		inp, _ := dl.Next()
		if inp == nil {
			break
		}
		second = append(second, inp.Data()[0])
	}

	// With 100 samples shuffled, it's extremely unlikely to get the same order
	same := true
	for i := range first {
		if first[i] != second[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("two shuffled epochs produced identical order (extremely unlikely)")
	}
}

func TestDataLoaderBatches(t *testing.T) {
	ds := &simpleDataset{n: 7}
	dl := NewDataLoader(ds, 3, false)
	if dl.Batches() != 3 { // ceil(7/3) = 3
		t.Fatalf("Batches() = %d, want 3", dl.Batches())
	}
}
