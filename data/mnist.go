//go:build darwin

package data

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const mnistBaseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

var mnistFiles = map[string]string{
	"train-images": "train-images-idx3-ubyte.gz",
	"train-labels": "train-labels-idx1-ubyte.gz",
	"test-images":  "t10k-images-idx3-ubyte.gz",
	"test-labels":  "t10k-labels-idx1-ubyte.gz",
}

// MNISTDataset holds MNIST images and labels.
type MNISTDataset struct {
	images [][]float32 // each image: 784 float32s in [0,1]
	labels []int
}

// LoadMNIST loads the MNIST dataset from disk (downloading if needed).
// dir is the cache directory. train selects train vs test split.
func LoadMNIST(dir string, train bool) (*MNISTDataset, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	var imgKey, lblKey string
	if train {
		imgKey, lblKey = "train-images", "train-labels"
	} else {
		imgKey, lblKey = "test-images", "test-labels"
	}

	images, err := loadMNISTImages(dir, imgKey)
	if err != nil {
		return nil, fmt.Errorf("load images: %w", err)
	}
	labels, err := loadMNISTLabels(dir, lblKey)
	if err != nil {
		return nil, fmt.Errorf("load labels: %w", err)
	}

	return &MNISTDataset{images: images, labels: labels}, nil
}

func (d *MNISTDataset) Len() int                          { return len(d.images) }
func (d *MNISTDataset) InputShape() []int                  { return []int{784} }
func (d *MNISTDataset) TargetShape() []int                 { return []int{1} }
func (d *MNISTDataset) Get(i int) ([]float32, []float32) {
	return d.images[i], []float32{float32(d.labels[i])}
}

func loadMNISTImages(dir, key string) ([][]float32, error) {
	path := filepath.Join(dir, mnistFiles[key])
	if err := downloadIfMissing(path, mnistBaseURL+mnistFiles[key]); err != nil {
		return nil, err
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magic, count, rows, cols int32
	binary.Read(gz, binary.BigEndian, &magic)
	binary.Read(gz, binary.BigEndian, &count)
	binary.Read(gz, binary.BigEndian, &rows)
	binary.Read(gz, binary.BigEndian, &cols)

	if magic != 2051 {
		return nil, fmt.Errorf("bad image magic: %d", magic)
	}

	imgSize := int(rows * cols)
	images := make([][]float32, count)
	buf := make([]byte, imgSize)

	for i := 0; i < int(count); i++ {
		if _, err := io.ReadFull(gz, buf); err != nil {
			return nil, err
		}
		img := make([]float32, imgSize)
		for j, b := range buf {
			img[j] = float32(b) / 255.0
		}
		images[i] = img
	}
	return images, nil
}

func loadMNISTLabels(dir, key string) ([]int, error) {
	path := filepath.Join(dir, mnistFiles[key])
	if err := downloadIfMissing(path, mnistBaseURL+mnistFiles[key]); err != nil {
		return nil, err
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magic, count int32
	binary.Read(gz, binary.BigEndian, &magic)
	binary.Read(gz, binary.BigEndian, &count)

	if magic != 2049 {
		return nil, fmt.Errorf("bad label magic: %d", magic)
	}

	buf := make([]byte, count)
	if _, err := io.ReadFull(gz, buf); err != nil {
		return nil, err
	}

	labels := make([]int, count)
	for i, b := range buf {
		labels[i] = int(b)
	}
	return labels, nil
}

func downloadIfMissing(path, url string) error {
	if _, err := os.Stat(path); err == nil {
		return nil // already exists
	}

	fmt.Printf("Downloading %s ...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}
