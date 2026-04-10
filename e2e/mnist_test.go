//go:build darwin && e2e

// Package e2e contains end-to-end integration tests.
// Run with: CGO_ENABLED=1 go test ./e2e/ -tags e2e -v -timeout 10m
package e2e

import (
	"fmt"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// TestMNISTTraining trains a 2-layer MLP on MNIST and verifies >90% accuracy.
// This downloads MNIST data on first run (~11 MB).
func TestMNISTTraining(t *testing.T) {
	cacheDir := t.TempDir()

	// Load data
	t.Log("Loading MNIST training data...")
	trainSet, err := data.LoadMNIST(cacheDir, true)
	if err != nil {
		t.Fatalf("failed to load MNIST train: %v", err)
	}
	t.Logf("Training samples: %d", trainSet.Len())

	testSet, err := data.LoadMNIST(cacheDir, false)
	if err != nil {
		t.Fatalf("failed to load MNIST test: %v", err)
	}
	t.Logf("Test samples: %d", testSet.Len())

	// Model: 784 -> 128 -> ReLU -> 10
	model := nn.NewSequential(
		nn.NewLinear(784, 128),
		nn.NewReLU(),
		nn.NewLinear(128, 10),
	)

	opt := optim.NewAdam(model.Parameters(), 0.001)
	trainLoader := data.NewDataLoader(trainSet, 64, true)

	epochs := 3
	start := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		trainLoader.Reset()
		var epochLoss float32
		batches := 0

		for {
			inputs, targets := trainLoader.Next()
			if inputs == nil {
				break
			}

			opt.ZeroGrad()
			logits := model.Forward(inputs)
			loss := g.CrossEntropyLoss(logits, targets)
			loss.Backward()
			opt.Step()

			epochLoss += loss.Data()[0]
			batches++
		}

		avgLoss := epochLoss / float32(batches)
		elapsed := time.Since(start).Seconds()
		t.Logf("Epoch %d/%d  loss=%.4f  elapsed=%.1fs", epoch+1, epochs, avgLoss, elapsed)
	}

	// Evaluate on test set
	testLoader := data.NewDataLoader(testSet, 256, false)
	testLoader.Reset()
	correct := 0
	total := 0

	for {
		inputs, targets := testLoader.Next()
		if inputs == nil {
			break
		}

		logits := model.Forward(inputs)
		preds := logits.Data()
		tgts := targets.Data()
		batch := inputs.Shape()[0]

		for i := 0; i < batch; i++ {
			// argmax
			maxIdx := 0
			maxVal := preds[i*10]
			for j := 1; j < 10; j++ {
				if preds[i*10+j] > maxVal {
					maxVal = preds[i*10+j]
					maxIdx = j
				}
			}
			if maxIdx == int(tgts[i]) {
				correct++
			}
			total++
		}
	}

	accuracy := float64(correct) / float64(total) * 100
	t.Logf("Test accuracy: %.2f%% (%d/%d)", accuracy, correct, total)

	if accuracy < 90.0 {
		t.Fatalf("MNIST accuracy %.2f%% is below 90%% threshold", accuracy)
	}
}

// TestMNISTSanity is a fast smoke test with a small subset.
// Verifies the full pipeline works without downloading all of MNIST.
func TestMNISTSanity(t *testing.T) {
	// Create synthetic "MNIST-like" data: 100 samples, 784 features, 10 classes
	model := nn.NewSequential(
		nn.NewLinear(784, 32),
		nn.NewReLU(),
		nn.NewLinear(32, 10),
	)

	opt := optim.NewAdam(model.Parameters(), 0.01)

	// Generate random data and fit it (just testing the pipeline doesn't crash)
	batchSize := 20
	for epoch := 0; epoch < 10; epoch++ {
		opt.ZeroGrad()
		x := g.RandN(batchSize, 784)
		// Random labels 0-9
		labels := make([]float32, batchSize)
		for i := range labels {
			labels[i] = float32(i % 10)
		}
		targets := g.NewTensor(labels, batchSize, 1)

		logits := model.Forward(x)
		loss := g.CrossEntropyLoss(logits, targets)
		loss.Backward()
		opt.Step()

		if epoch == 0 {
			fmt.Printf("  Sanity check initial loss: %.4f\n", loss.Data()[0])
		}
	}
	fmt.Println("  MNIST sanity pipeline: OK")
}
