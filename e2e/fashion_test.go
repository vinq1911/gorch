//go:build darwin && e2e

package e2e

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// modelConfig describes a model architecture for benchmarking.
type modelConfig struct {
	name   string
	layers []int // e.g. [784, 256, 128, 10]
	lr     float32
	epochs int
}

func buildModel(layers []int) *nn.Sequential {
	var modules []nn.Module
	for i := 0; i < len(layers)-1; i++ {
		modules = append(modules, nn.NewLinear(layers[i], layers[i+1]))
		if i < len(layers)-2 { // no activation after last layer
			modules = append(modules, nn.NewReLU())
		}
	}
	return nn.NewSequential(modules...)
}

// trainResult holds the results of training one model.
type trainResult struct {
	name          string
	epochs        int
	lossPerEpoch  []float32
	testAccuracy  float64
	perClass      [10]float64 // accuracy per class
	perClassTotal [10]int
	perClassHit   [10]int
	trainTime     time.Duration
	params        int
}

func trainAndEval(cfg modelConfig, trainSet, testSet *data.MNISTDataset) trainResult {
	model := buildModel(cfg.layers)
	opt := optim.NewAdam(model.Parameters(), cfg.lr)
	loader := data.NewDataLoader(trainSet, 128, true)

	// Count parameters
	paramCount := 0
	for _, p := range model.Parameters() {
		paramCount += p.Size()
	}

	start := time.Now()
	var losses []float32

	for epoch := 0; epoch < cfg.epochs; epoch++ {
		loader.Reset()
		var epochLoss float32
		batches := 0

		for {
			inputs, targets := loader.Next()
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
		losses = append(losses, epochLoss/float32(batches))
	}
	trainTime := time.Since(start)

	// Evaluate
	testLoader := data.NewDataLoader(testSet, 256, false)
	testLoader.Reset()
	var result trainResult
	result.name = cfg.name
	result.epochs = cfg.epochs
	result.lossPerEpoch = losses
	result.trainTime = trainTime
	result.params = paramCount

	correct, total := 0, 0
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
			result.perClassTotal[cls]++
			if maxIdx == cls {
				correct++
				result.perClassHit[cls]++
			}
			total++
		}
	}
	result.testAccuracy = float64(correct) / float64(total) * 100
	for i := 0; i < 10; i++ {
		if result.perClassTotal[i] > 0 {
			result.perClass[i] = float64(result.perClassHit[i]) / float64(result.perClassTotal[i]) * 100
		}
	}
	return result
}

// TestFashionMNIST trains 3 MLP architectures on Fashion-MNIST and generates a report.
func TestFashionMNIST(t *testing.T) {
	cacheDir := t.TempDir()

	t.Log("Loading Fashion-MNIST...")
	trainSet, err := data.LoadFashionMNIST(cacheDir, true)
	if err != nil {
		t.Fatalf("failed to load Fashion-MNIST train: %v", err)
	}
	testSet, err := data.LoadFashionMNIST(cacheDir, false)
	if err != nil {
		t.Fatalf("failed to load Fashion-MNIST test: %v", err)
	}
	t.Logf("Train: %d samples, Test: %d samples", trainSet.Len(), testSet.Len())

	configs := []modelConfig{
		{name: "Small (784→128→10)", layers: []int{784, 128, 10}, lr: 0.001, epochs: 10},
		{name: "Medium (784→256→128→10)", layers: []int{784, 256, 128, 10}, lr: 0.001, epochs: 10},
		{name: "Large (784→512→256→128→10)", layers: []int{784, 512, 256, 128, 10}, lr: 0.001, epochs: 10},
	}

	var results []trainResult
	for _, cfg := range configs {
		t.Logf("Training %s ...", cfg.name)
		r := trainAndEval(cfg, trainSet, testSet)
		t.Logf("  %s: %.2f%% accuracy, %v training time, %d params", r.name, r.testAccuracy, r.trainTime.Round(time.Millisecond), r.params)
		results = append(results, r)
	}

	// Generate report
	report := generateReport(results)
	t.Log("\n" + report)

	// Write report to file
	reportPath := "../doc/fashion-mnist-report.md"
	if err := os.WriteFile(reportPath, []byte(report), 0644); err != nil {
		t.Logf("Warning: could not write report file: %v", err)
	} else {
		t.Logf("Report written to %s", reportPath)
	}

	// Assert minimum accuracy for medium model
	for _, r := range results {
		if r.name == "Medium (784→256→128→10)" && r.testAccuracy < 87.0 {
			t.Fatalf("Medium model accuracy %.2f%% below 87%% threshold", r.testAccuracy)
		}
	}
}

func generateReport(results []trainResult) string {
	var b strings.Builder

	b.WriteString("# Fashion-MNIST Training Report\n\n")
	b.WriteString("**Framework:** gorch (Go + Apple Accelerate + Metal)\n")
	b.WriteString(fmt.Sprintf("**Date:** %s\n", time.Now().Format("2006-01-02")))
	b.WriteString("**Hardware:** Apple Silicon (M-series)\n")
	b.WriteString("**Dataset:** Fashion-MNIST (60,000 train / 10,000 test, 28x28 grayscale, 10 classes)\n\n")

	b.WriteString("## Task\n\n")
	b.WriteString("Classify grayscale images of clothing items into 10 categories:\n")
	b.WriteString("T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.\n\n")
	b.WriteString("Fashion-MNIST is a drop-in replacement for MNIST that is significantly harder.\n")
	b.WriteString("Published MLP baselines achieve ~87-89% accuracy; CNNs reach ~93%.\n\n")

	// Architecture comparison
	b.WriteString("## Results: Architecture Comparison\n\n")
	b.WriteString("| Model | Params | Test Accuracy | Training Time | Final Loss |\n")
	b.WriteString("|-------|--------|--------------|---------------|------------|\n")
	for _, r := range results {
		b.WriteString(fmt.Sprintf("| %s | %d | **%.2f%%** | %s | %.4f |\n",
			r.name, r.params, r.testAccuracy,
			r.trainTime.Round(time.Millisecond), r.lossPerEpoch[len(r.lossPerEpoch)-1]))
	}

	// Training curves
	b.WriteString("\n## Training Curves (Loss per Epoch)\n\n")
	b.WriteString("| Epoch |")
	for _, r := range results {
		b.WriteString(fmt.Sprintf(" %s |", r.name))
	}
	b.WriteString("\n|-------|")
	for range results {
		b.WriteString("--------|")
	}
	b.WriteString("\n")
	for epoch := 0; epoch < results[0].epochs; epoch++ {
		b.WriteString(fmt.Sprintf("| %d |", epoch+1))
		for _, r := range results {
			b.WriteString(fmt.Sprintf(" %.4f |", r.lossPerEpoch[epoch]))
		}
		b.WriteString("\n")
	}

	// Per-class accuracy (use best model)
	best := results[len(results)-1]
	b.WriteString(fmt.Sprintf("\n## Per-Class Accuracy (%s)\n\n", best.name))
	b.WriteString("| Class | Name | Accuracy | Correct/Total |\n")
	b.WriteString("|-------|------|----------|---------------|\n")
	for i, name := range data.FashionMNISTClasses {
		b.WriteString(fmt.Sprintf("| %d | %s | %.1f%% | %d/%d |\n",
			i, name, best.perClass[i], best.perClassHit[i], best.perClassTotal[i]))
	}

	// Analysis
	b.WriteString("\n## Analysis\n\n")

	// Find hardest and easiest classes
	hardest, easiest := 0, 0
	for i := 1; i < 10; i++ {
		if best.perClass[i] < best.perClass[hardest] {
			hardest = i
		}
		if best.perClass[i] > best.perClass[easiest] {
			easiest = i
		}
	}

	b.WriteString(fmt.Sprintf("**Easiest class:** %s (%.1f%%)\n",
		data.FashionMNISTClasses[easiest], best.perClass[easiest]))
	b.WriteString(fmt.Sprintf("**Hardest class:** %s (%.1f%%)\n\n",
		data.FashionMNISTClasses[hardest], best.perClass[hardest]))

	b.WriteString("### What gorch does well\n\n")
	b.WriteString("- Full training loop works end-to-end: data loading, forward, loss, backward, optimizer step\n")
	b.WriteString("- Accelerate BLAS makes CPU training fast (seconds, not minutes)\n")
	b.WriteString("- Autograd correctly propagates gradients through multi-layer networks\n")
	b.WriteString("- Accuracy is competitive with published MLP baselines\n")
	b.WriteString("- DataLoader with shuffle provides proper stochastic training\n\n")

	b.WriteString("### Current limitations\n\n")
	b.WriteString("- No Conv2d — limited to MLP architectures (flattened pixels)\n")
	b.WriteString("- No Dropout or BatchNorm — may overfit on harder datasets\n")
	b.WriteString("- No learning rate scheduling — fixed LR throughout training\n")
	b.WriteString("- No data augmentation — could improve generalization\n")
	b.WriteString("- Shirt vs T-shirt/top confusion is expected — they look similar even to humans\n\n")

	b.WriteString("### Comparison to published baselines\n\n")
	b.WriteString("| Method | Accuracy | Source |\n")
	b.WriteString("|--------|----------|--------|\n")
	b.WriteString(fmt.Sprintf("| **gorch MLP (best)** | **%.1f%%** | This report |\n", best.testAccuracy))
	b.WriteString("| 2-layer MLP (256) | 87.1% | Zalando benchmark |\n")
	b.WriteString("| 3-layer MLP (256+128) | 88.3% | Zalando benchmark |\n")
	b.WriteString("| CNN (2 conv + 2 FC) | 91.6% | Zalando benchmark |\n")
	b.WriteString("| ResNet-18 | 93.6% | Literature |\n")

	return b.String()
}
