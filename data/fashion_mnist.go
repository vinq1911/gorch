//go:build darwin

package data

import "fmt"

// Fashion-MNIST uses the same IDX file format as MNIST, just a different URL.
const fashionMNISTBaseURL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

var fashionMNISTFiles = map[string]string{
	"train-images": "train-images-idx3-ubyte.gz",
	"train-labels": "train-labels-idx1-ubyte.gz",
	"test-images":  "t10k-images-idx3-ubyte.gz",
	"test-labels":  "t10k-labels-idx1-ubyte.gz",
}

// FashionMNIST class names.
var FashionMNISTClasses = []string{
	"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
}

// LoadFashionMNIST loads the Fashion-MNIST dataset from disk (downloading if needed).
// dir is the cache directory. train selects train vs test split.
func LoadFashionMNIST(dir string, train bool) (*MNISTDataset, error) {
	var imgKey, lblKey string
	if train {
		imgKey, lblKey = "train-images", "train-labels"
	} else {
		imgKey, lblKey = "test-images", "test-labels"
	}

	// Reuse MNIST loading functions — same IDX file format, different URL.
	images, err := loadIDXImages(dir, fashionMNISTBaseURL, fashionMNISTFiles[imgKey])
	if err != nil {
		return nil, fmt.Errorf("load fashion images: %w", err)
	}
	labels, err := loadIDXLabels(dir, fashionMNISTBaseURL, fashionMNISTFiles[lblKey])
	if err != nil {
		return nil, fmt.Errorf("load fashion labels: %w", err)
	}

	return &MNISTDataset{images: images, labels: labels}, nil
}
