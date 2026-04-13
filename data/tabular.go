//go:build darwin

package data

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// TabularDataset holds a generic numeric tabular dataset.
type TabularDataset struct {
	features   [][]float32
	labels     []int
	numClasses int
	inputDim   int
}

func (d *TabularDataset) Len() int                         { return len(d.features) }
func (d *TabularDataset) InputShape() []int                 { return []int{d.inputDim} }
func (d *TabularDataset) TargetShape() []int                { return []int{1} }
func (d *TabularDataset) NumClasses() int                   { return d.numClasses }
func (d *TabularDataset) Get(i int) ([]float32, []float32) {
	return d.features[i], []float32{float32(d.labels[i])}
}

// LoadCSV loads a numeric CSV file.
// sep: separator character (e.g., ",", ";")
// labelCol: column index of the label (-1 for last column)
// skipHeader: skip first row
// labelMap: optional map from string label to int class (nil = auto-detect from ints)
func LoadCSV(path string, sep string, labelCol int, skipHeader bool, labelMap map[string]int) (*TabularDataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	if skipHeader && scanner.Scan() {
		// skip header line
	}

	var allFeatures [][]float32
	var allLabels []int
	autoLabelMap := make(map[string]int)
	nextLabel := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.Split(line, sep)

		// Determine label column
		lc := labelCol
		if lc < 0 {
			lc = len(fields) - 1
		}

		// Parse label
		labelStr := strings.TrimSpace(fields[lc])
		var label int
		if labelMap != nil {
			var ok bool
			label, ok = labelMap[labelStr]
			if !ok {
				return nil, fmt.Errorf("unknown label %q", labelStr)
			}
		} else {
			// Try int parse first, else auto-map
			if v, err := strconv.Atoi(labelStr); err == nil {
				label = v
			} else {
				if _, ok := autoLabelMap[labelStr]; !ok {
					autoLabelMap[labelStr] = nextLabel
					nextLabel++
				}
				label = autoLabelMap[labelStr]
			}
		}

		// Parse features (all columns except label)
		feats := make([]float32, 0, len(fields)-1)
		for i, field := range fields {
			if i == lc {
				continue
			}
			v, err := strconv.ParseFloat(strings.TrimSpace(field), 32)
			if err != nil {
				return nil, fmt.Errorf("parse error row %d col %d: %q: %w", len(allFeatures), i, field, err)
			}
			feats = append(feats, float32(v))
		}

		allFeatures = append(allFeatures, feats)
		allLabels = append(allLabels, label)
	}

	if len(allFeatures) == 0 {
		return nil, fmt.Errorf("no data rows found")
	}

	// Compute number of classes
	maxLabel := 0
	for _, l := range allLabels {
		if l > maxLabel {
			maxLabel = l
		}
	}

	return &TabularDataset{
		features:   allFeatures,
		labels:     allLabels,
		numClasses: maxLabel + 1,
		inputDim:   len(allFeatures[0]),
	}, nil
}

// Normalize standardizes each feature column to zero mean, unit variance.
func (d *TabularDataset) Normalize() {
	n := len(d.features)
	dim := d.inputDim

	means := make([]float64, dim)
	stds := make([]float64, dim)

	// Compute means
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			means[j] += float64(d.features[i][j])
		}
	}
	for j := range means {
		means[j] /= float64(n)
	}

	// Compute stds
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			diff := float64(d.features[i][j]) - means[j]
			stds[j] += diff * diff
		}
	}
	for j := range stds {
		stds[j] = math.Sqrt(stds[j] / float64(n))
		if stds[j] < 1e-8 {
			stds[j] = 1 // avoid div by zero
		}
	}

	// Apply
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			d.features[i][j] = float32((float64(d.features[i][j]) - means[j]) / stds[j])
		}
	}
}

// TrainTestSplit splits the dataset into training and test sets.
// testFraction is the fraction of data to use for testing (e.g., 0.2 for 80/20 split).
// Shuffles before splitting.
func (d *TabularDataset) TrainTestSplit(testFraction float64) (*TabularDataset, *TabularDataset) {
	n := d.Len()
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(n, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	splitIdx := int(float64(n) * (1 - testFraction))

	train := &TabularDataset{
		inputDim:   d.inputDim,
		numClasses: d.numClasses,
	}
	test := &TabularDataset{
		inputDim:   d.inputDim,
		numClasses: d.numClasses,
	}

	for i, idx := range indices {
		if i < splitIdx {
			train.features = append(train.features, d.features[idx])
			train.labels = append(train.labels, d.labels[idx])
		} else {
			test.features = append(test.features, d.features[idx])
			test.labels = append(test.labels, d.labels[idx])
		}
	}
	return train, test
}
