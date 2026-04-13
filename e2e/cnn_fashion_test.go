//go:build darwin && e2e

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

// TestCNNFashionMNIST trains a CNN on Fashion-MNIST.
// Architecture: Conv(1→16,3x3) → ReLU → MaxPool(2) → Conv(16→32,3x3) → ReLU → MaxPool(2) → Flatten → Linear(800→10)
func TestCNNFashionMNIST(t *testing.T) {
	cacheDir := t.TempDir()

	t.Log("Loading Fashion-MNIST (2D)...")
	trainSet, err := data.LoadFashionMNIST(cacheDir, true)
	if err != nil {
		t.Fatalf("load train: %v", err)
	}
	testSet, err := data.LoadFashionMNIST(cacheDir, false)
	if err != nil {
		t.Fatalf("load test: %v", err)
	}

	// Wrap as 2D datasets for CNN input (1, 28, 28)
	trainSet2D := trainSet.As2D()
	testSet2D := testSet.As2D()

	// CNN model
	// Input: (batch, 1, 28, 28)
	// Conv1: (batch, 16, 28, 28) with pad=1
	// Pool1: (batch, 16, 14, 14)
	// Conv2: (batch, 32, 14, 14) with pad=1
	// Pool2: (batch, 32, 7, 7)
	// Flatten: (batch, 32*7*7) = (batch, 1568)
	// Linear: (batch, 10)
	model := nn.NewSequential(
		nn.NewConv2d(1, 16, 3, 1, 1),  // (1,28,28) -> (16,28,28)
		nn.NewReLU(),
		nn.NewMaxPool2d(2, 2),          // -> (16,14,14)
		nn.NewConv2d(16, 32, 3, 1, 1),  // -> (32,14,14)
		nn.NewReLU(),
		nn.NewMaxPool2d(2, 2),          // -> (32,7,7)
		nn.NewFlatten(),                // -> (1568)
		nn.NewLinear(1568, 10),         // -> (10)
	)

	paramCount := 0
	for _, p := range model.Parameters() {
		paramCount += p.Size()
	}
	t.Logf("CNN parameters: %d", paramCount)

	opt := optim.NewAdam(model.Parameters(), 0.001)
	trainLoader := data.NewDataLoader(trainSet2D, 64, true)

	epochs := 5
	start := time.Now()
	var losses []float32

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
		losses = append(losses, avgLoss)
		elapsed := time.Since(start)
		t.Logf("Epoch %d/%d  loss=%.4f  elapsed=%v", epoch+1, epochs, avgLoss, elapsed.Round(time.Second))
	}

	// Evaluate
	testLoader := data.NewDataLoader(testSet2D, 256, false)
	testLoader.Reset()
	correct, total := 0, 0
	var perClassHit, perClassTotal [10]int

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
			maxIdx := 0
			maxVal := preds[i*10]
			for j := 1; j < 10; j++ {
				if preds[i*10+j] > maxVal {
					maxVal = preds[i*10+j]
					maxIdx = j
				}
			}
			cls := int(tgts[i])
			perClassTotal[cls]++
			if maxIdx == cls {
				correct++
				perClassHit[cls]++
			}
			total++
		}
	}

	accuracy := float64(correct) / float64(total) * 100
	trainTime := time.Since(start)

	t.Logf("CNN Test accuracy: %.2f%% (%d/%d) in %v", accuracy, correct, total, trainTime.Round(time.Second))
	t.Log("Per-class accuracy:")
	for i, name := range data.FashionMNISTClasses {
		pct := float64(perClassHit[i]) / float64(perClassTotal[i]) * 100
		t.Logf("  %s: %.1f%% (%d/%d)", name, pct, perClassHit[i], perClassTotal[i])
	}

	// Print comparison
	fmt.Printf("\n  CNN vs MLP Comparison:\n")
	fmt.Printf("  MLP (784→128→10):           88.1%% accuracy, ~2s\n")
	fmt.Printf("  CNN (Conv16→Conv32→FC):      %.1f%% accuracy, %v\n", accuracy, trainTime.Round(time.Second))

	if accuracy < 88.0 {
		t.Fatalf("CNN accuracy %.2f%% below 88%% threshold", accuracy)
	}
}
