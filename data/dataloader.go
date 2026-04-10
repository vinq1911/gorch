//go:build darwin

// Package data provides dataset and data loading utilities for gorch.
package data

import (
	"math/rand"

	g "github.com/vinq1911/gorch"
)

// Dataset represents a collection of input-target pairs.
type Dataset interface {
	Len() int
	Get(index int) (input, target []float32)
	InputShape() []int  // shape of a single input (excluding batch dim)
	TargetShape() []int // shape of a single target (excluding batch dim)
}

// DataLoader batches and optionally shuffles a Dataset.
type DataLoader struct {
	dataset   Dataset
	batchSize int
	shuffle   bool
	indices   []int
	pos       int
}

// NewDataLoader creates a DataLoader for the given dataset.
func NewDataLoader(dataset Dataset, batchSize int, shuffle bool) *DataLoader {
	indices := make([]int, dataset.Len())
	for i := range indices {
		indices[i] = i
	}
	return &DataLoader{
		dataset:   dataset,
		batchSize: batchSize,
		shuffle:   shuffle,
		indices:   indices,
	}
}

// Reset resets the DataLoader for a new epoch, optionally shuffling.
func (dl *DataLoader) Reset() {
	dl.pos = 0
	if dl.shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}
}

// Next returns the next batch of inputs and targets.
// Returns nil, nil when the epoch is exhausted.
func (dl *DataLoader) Next() (inputs, targets *g.Tensor) {
	if dl.pos >= len(dl.indices) {
		return nil, nil
	}

	end := dl.pos + dl.batchSize
	if end > len(dl.indices) {
		end = len(dl.indices)
	}
	batch := dl.indices[dl.pos:end]
	dl.pos = end

	actualBatch := len(batch)
	inShape := dl.dataset.InputShape()
	tgtShape := dl.dataset.TargetShape()

	inSize := 1
	for _, s := range inShape {
		inSize *= s
	}
	tgtSize := 1
	for _, s := range tgtShape {
		tgtSize *= s
	}

	inData := make([]float32, actualBatch*inSize)
	tgtData := make([]float32, actualBatch*tgtSize)

	for i, idx := range batch {
		inp, tgt := dl.dataset.Get(idx)
		copy(inData[i*inSize:(i+1)*inSize], inp)
		copy(tgtData[i*tgtSize:(i+1)*tgtSize], tgt)
	}

	// Build shape: [batchSize, ...inputShape]
	inTensorShape := append([]int{actualBatch}, inShape...)
	tgtTensorShape := append([]int{actualBatch}, tgtShape...)

	return g.NewTensor(inData, inTensorShape...), g.NewTensor(tgtData, tgtTensorShape...)
}

// Batches returns the total number of batches per epoch.
func (dl *DataLoader) Batches() int {
	n := dl.dataset.Len()
	return (n + dl.batchSize - 1) / dl.batchSize
}
