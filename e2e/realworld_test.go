//go:build darwin && e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/optim"
)

// datasetResult holds results for one dataset with multiple model configs.
type datasetResult struct {
	Name        string
	Samples     int
	Features    int
	Classes     int
	ClassNames  []string
	Models      []modelResult
}

type modelResult struct {
	Name         string
	Layers       []int
	Params       int
	Epochs       int
	LR           float32
	TrainTime    time.Duration
	TestAccuracy float64
	LossPerEpoch []float32
	PerClass     []classResult
}

type classResult struct {
	Name    string
	Correct int
	Total   int
	Pct     float64
}

func trainModel(cfg modelConfig, trainSet, testSet data.Dataset, numClasses int, classNames []string) modelResult {
	model := buildModel(cfg.layers)
	opt := optim.NewAdam(model.Parameters(), cfg.lr)
	loader := data.NewDataLoader(trainSet, 32, true)

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
	correct, total := 0, 0
	perClassHit := make([]int, numClasses)
	perClassTotal := make([]int, numClasses)

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
			maxVal := preds[i*numClasses]
			for j := 1; j < numClasses; j++ {
				if preds[i*numClasses+j] > maxVal {
					maxVal = preds[i*numClasses+j]
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

	var perClass []classResult
	for i := 0; i < numClasses; i++ {
		name := fmt.Sprintf("Class %d", i)
		if i < len(classNames) {
			name = classNames[i]
		}
		pct := 0.0
		if perClassTotal[i] > 0 {
			pct = float64(perClassHit[i]) / float64(perClassTotal[i]) * 100
		}
		perClass = append(perClass, classResult{Name: name, Correct: perClassHit[i], Total: perClassTotal[i], Pct: pct})
	}

	return modelResult{
		Name:         cfg.name,
		Layers:       cfg.layers,
		Params:       paramCount,
		Epochs:       cfg.epochs,
		LR:           cfg.lr,
		TrainTime:    trainTime,
		TestAccuracy: float64(correct) / float64(total) * 100,
		LossPerEpoch: losses,
		PerClass:     perClass,
	}
}

// TestRealWorldDatasets trains on Wine Quality and Breast Cancer datasets.
func TestRealWorldDatasets(t *testing.T) {
	cacheDir := t.TempDir()
	var allResults []datasetResult

	// ===================== Wine Quality =====================
	t.Log("=== Wine Quality (Red) ===")
	wineDS, err := data.LoadWineQuality(cacheDir)
	if err != nil {
		t.Fatalf("load wine: %v", err)
	}
	wineDS.Normalize()
	wineTrain, wineTest := wineDS.TrainTestSplit(0.2)
	t.Logf("Wine: %d train, %d test, %d features, %d classes",
		wineTrain.Len(), wineTest.Len(), wineDS.InputShape()[0], wineDS.NumClasses())

	wineClassNames := []string{"Quality 3", "Quality 4", "Quality 5", "Quality 6", "Quality 7", "Quality 8"}
	wineConfigs := []modelConfig{
		{name: "Small (11→32→6)", layers: []int{11, 32, wineDS.NumClasses()}, lr: 0.001, epochs: 50},
		{name: "Medium (11→64→32→6)", layers: []int{11, 64, 32, wineDS.NumClasses()}, lr: 0.001, epochs: 50},
		{name: "Large (11→128→64→32→6)", layers: []int{11, 128, 64, 32, wineDS.NumClasses()}, lr: 0.001, epochs: 50},
	}

	var wineModels []modelResult
	for _, cfg := range wineConfigs {
		t.Logf("  Training %s ...", cfg.name)
		r := trainModel(cfg, wineTrain, wineTest, wineDS.NumClasses(), wineClassNames)
		t.Logf("  %s: %.2f%% accuracy, %v", r.Name, r.TestAccuracy, r.TrainTime.Round(time.Millisecond))
		wineModels = append(wineModels, r)
	}

	allResults = append(allResults, datasetResult{
		Name:       "Wine Quality (Red)",
		Samples:    wineDS.Len(),
		Features:   wineDS.InputShape()[0],
		Classes:    wineDS.NumClasses(),
		ClassNames: wineClassNames,
		Models:     wineModels,
	})

	// ===================== Breast Cancer =====================
	t.Log("=== Breast Cancer Wisconsin ===")
	bcDS, err := data.LoadBreastCancer(cacheDir)
	if err != nil {
		t.Fatalf("load breast cancer: %v", err)
	}
	bcDS.Normalize()
	bcTrain, bcTest := bcDS.TrainTestSplit(0.2)
	t.Logf("BC: %d train, %d test, %d features, %d classes",
		bcTrain.Len(), bcTest.Len(), bcDS.InputShape()[0], bcDS.NumClasses())

	bcConfigs := []modelConfig{
		{name: "Small (30→16→2)", layers: []int{30, 16, 2}, lr: 0.001, epochs: 50},
		{name: "Medium (30→64→32→2)", layers: []int{30, 64, 32, 2}, lr: 0.001, epochs: 50},
		{name: "Large (30→128→64→32→2)", layers: []int{30, 128, 64, 32, 2}, lr: 0.0005, epochs: 80},
	}

	var bcModels []modelResult
	for _, cfg := range bcConfigs {
		t.Logf("  Training %s ...", cfg.name)
		r := trainModel(cfg, bcTrain, bcTest, bcDS.NumClasses(), data.BreastCancerClassNames)
		t.Logf("  %s: %.2f%% accuracy, %v", r.Name, r.TestAccuracy, r.TrainTime.Round(time.Millisecond))
		bcModels = append(bcModels, r)
	}

	allResults = append(allResults, datasetResult{
		Name:       "Breast Cancer Wisconsin (Diagnostic)",
		Samples:    bcDS.Len(),
		Features:   bcDS.InputShape()[0],
		Classes:    bcDS.NumClasses(),
		ClassNames: data.BreastCancerClassNames,
		Models:     bcModels,
	})

	// ===================== Fashion-MNIST (reference) =====================
	t.Log("=== Fashion-MNIST (reference) ===")
	fTrain, err := data.LoadFashionMNIST(cacheDir, true)
	if err != nil {
		t.Fatalf("load fashion train: %v", err)
	}
	fTest, err := data.LoadFashionMNIST(cacheDir, false)
	if err != nil {
		t.Fatalf("load fashion test: %v", err)
	}

	fashionCfg := modelConfig{name: "MLP (784→256→128→10)", layers: []int{784, 256, 128, 10}, lr: 0.001, epochs: 10}
	t.Logf("  Training %s ...", fashionCfg.name)
	fashionResult := trainModel(fashionCfg, fTrain, fTest, 10, data.FashionMNISTClasses)
	t.Logf("  %s: %.2f%% accuracy, %v", fashionResult.Name, fashionResult.TestAccuracy, fashionResult.TrainTime.Round(time.Millisecond))

	allResults = append(allResults, datasetResult{
		Name:       "Fashion-MNIST",
		Samples:    fTrain.Len() + fTest.Len(),
		Features:   784,
		Classes:    10,
		ClassNames: data.FashionMNISTClasses,
		Models:     []modelResult{fashionResult},
	})

	// ===================== Write JSON for report generator =====================
	jsonPath := "../doc/realworld_results.json"
	jsonBytes, _ := json.MarshalIndent(allResults, "", "  ")
	if err := os.WriteFile(jsonPath, jsonBytes, 0644); err != nil {
		t.Logf("Warning: could not write JSON: %v", err)
	}

	// ===================== Print summary =====================
	t.Log("\n========== SUMMARY ==========")
	for _, ds := range allResults {
		t.Logf("\n%s (%d samples, %d features, %d classes)", ds.Name, ds.Samples, ds.Features, ds.Classes)
		for _, m := range ds.Models {
			t.Logf("  %-35s  %.2f%%  %v  %d params", m.Name, m.TestAccuracy, m.TrainTime.Round(time.Millisecond), m.Params)
		}
	}

	// Write markdown report too
	mdReport := generateRealWorldReport(allResults)
	mdPath := "../doc/realworld-report.md"
	if err := os.WriteFile(mdPath, []byte(mdReport), 0644); err != nil {
		t.Logf("Warning: could not write markdown: %v", err)
	}
}

func generateRealWorldReport(results []datasetResult) string {
	var b strings.Builder

	b.WriteString("# Real-World Training Report\n\n")
	b.WriteString(fmt.Sprintf("**Framework:** gorch (Go + Apple Accelerate + Metal)\n"))
	b.WriteString(fmt.Sprintf("**Date:** %s\n", time.Now().Format("2006-01-02")))
	b.WriteString("**Hardware:** Apple Silicon (M-series)\n\n")

	// Executive summary
	b.WriteString("## Executive Summary\n\n")
	b.WriteString("Three real-world datasets tested to validate gorch as a production-capable ML framework:\n\n")
	b.WriteString("| Dataset | Task | Best Accuracy | Training Time | Published Baseline |\n")
	b.WriteString("|---------|------|--------------|---------------|-------------------|\n")
	for _, ds := range results {
		best := ds.Models[0]
		for _, m := range ds.Models[1:] {
			if m.TestAccuracy > best.TestAccuracy {
				best = m
			}
		}
		baseline := ""
		switch {
		case strings.Contains(ds.Name, "Wine"):
			baseline = "~55-65% MLP"
		case strings.Contains(ds.Name, "Breast"):
			baseline = "~95-97% MLP"
		case strings.Contains(ds.Name, "Fashion"):
			baseline = "~88% MLP"
		}
		b.WriteString(fmt.Sprintf("| %s | %d-class | **%.1f%%** | %v | %s |\n",
			ds.Name, ds.Classes, best.TestAccuracy, best.TrainTime.Round(time.Millisecond), baseline))
	}

	// Per-dataset details
	for _, ds := range results {
		b.WriteString(fmt.Sprintf("\n---\n\n## %s\n\n", ds.Name))
		b.WriteString(fmt.Sprintf("- **Samples:** %d\n", ds.Samples))
		b.WriteString(fmt.Sprintf("- **Features:** %d\n", ds.Features))
		b.WriteString(fmt.Sprintf("- **Classes:** %d\n\n", ds.Classes))

		// Architecture comparison
		b.WriteString("### Architecture Comparison\n\n")
		b.WriteString("| Model | Params | Accuracy | Training Time | Final Loss |\n")
		b.WriteString("|-------|--------|----------|---------------|------------|\n")
		for _, m := range ds.Models {
			b.WriteString(fmt.Sprintf("| %s | %d | **%.2f%%** | %v | %.4f |\n",
				m.Name, m.Params, m.TestAccuracy, m.TrainTime.Round(time.Millisecond), m.LossPerEpoch[len(m.LossPerEpoch)-1]))
		}

		// Per-class for best model
		best := ds.Models[0]
		for _, m := range ds.Models[1:] {
			if m.TestAccuracy > best.TestAccuracy {
				best = m
			}
		}
		if len(best.PerClass) > 0 {
			b.WriteString(fmt.Sprintf("\n### Per-Class Accuracy (%s)\n\n", best.Name))
			b.WriteString("| Class | Accuracy | Correct/Total |\n")
			b.WriteString("|-------|----------|---------------|\n")
			for _, pc := range best.PerClass {
				if pc.Total > 0 {
					b.WriteString(fmt.Sprintf("| %s | %.1f%% | %d/%d |\n", pc.Name, pc.Pct, pc.Correct, pc.Total))
				}
			}
		}

		// Loss curve
		b.WriteString(fmt.Sprintf("\n### Training Loss Curve (%s)\n\n", best.Name))
		b.WriteString("| Epoch | Loss |\n")
		b.WriteString("|-------|------|\n")
		step := 1
		if len(best.LossPerEpoch) > 20 {
			step = len(best.LossPerEpoch) / 10
		}
		for i, l := range best.LossPerEpoch {
			if i%step == 0 || i == len(best.LossPerEpoch)-1 {
				b.WriteString(fmt.Sprintf("| %d | %.4f |\n", i+1, l))
			}
		}
	}

	return b.String()
}
