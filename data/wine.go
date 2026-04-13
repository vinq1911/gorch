//go:build darwin

package data

import (
	"fmt"
	"os"
	"path/filepath"
)

const wineURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

// LoadWineQuality loads the UCI Wine Quality (Red) dataset.
// 1,599 samples, 11 numeric features, label is quality score (3-8).
// Labels are remapped to 0-based contiguous classes.
// Returns the full dataset; use TrainTestSplit() to split.
func LoadWineQuality(dir string) (*TabularDataset, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "winequality-red.csv")
	if err := downloadIfMissing(path, wineURL); err != nil {
		return nil, fmt.Errorf("download wine data: %w", err)
	}

	ds, err := LoadCSV(path, ";", -1, true, nil)
	if err != nil {
		return nil, fmt.Errorf("parse wine csv: %w", err)
	}

	// Remap labels to 0-based contiguous (quality scores are 3-8)
	labelSet := make(map[int]bool)
	for _, l := range ds.labels {
		labelSet[l] = true
	}
	remap := make(map[int]int)
	idx := 0
	for l := 0; l <= 10; l++ { // iterate in order
		if labelSet[l] {
			remap[l] = idx
			idx++
		}
	}
	for i, l := range ds.labels {
		ds.labels[i] = remap[l]
	}
	ds.numClasses = idx

	return ds, nil
}
