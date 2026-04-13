//go:build darwin

package data

import (
	"fmt"
	"os"
	"path/filepath"
)

const breastCancerURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

// LoadBreastCancer loads the UCI Breast Cancer Wisconsin (Diagnostic) dataset.
// 569 samples, 30 numeric features, 2 classes (Malignant/Benign).
// Format: ID, diagnosis (M/B), 30 features. No header.
func LoadBreastCancer(dir string) (*TabularDataset, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "wdbc.data")
	if err := downloadIfMissing(path, breastCancerURL); err != nil {
		return nil, fmt.Errorf("download breast cancer data: %w", err)
	}

	// Column 1 is diagnosis (M/B), columns 2-31 are features, column 0 is ID (skip)
	// We'll parse manually since label is column 1 (string) and col 0 is non-numeric
	ds, err := LoadCSV(path, ",", 1, false, map[string]int{"M": 1, "B": 0})
	if err != nil {
		return nil, fmt.Errorf("parse breast cancer csv: %w", err)
	}

	// Remove the ID column (first feature column is the ID which got parsed as a feature)
	for i := range ds.features {
		ds.features[i] = ds.features[i][1:] // drop first element (ID)
	}
	ds.inputDim = len(ds.features[0])

	return ds, nil
}

// BreastCancerClassNames returns class names for display.
var BreastCancerClassNames = []string{"Benign", "Malignant"}
