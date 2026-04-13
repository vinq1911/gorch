//go:build darwin && e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// benchResult captures one benchmark result.
type benchResult struct {
	Category    string  `json:"category"`
	Name        string  `json:"name"`
	Accuracy    float64 `json:"accuracy,omitempty"`
	Loss        float32 `json:"loss,omitempty"`
	TrainTime   string  `json:"train_time"`
	TrainTimeMs float64 `json:"train_time_ms"`
	Params      int     `json:"params"`
	Epochs      int     `json:"epochs"`
	Details     string  `json:"details,omitempty"`
}

// TestComprehensiveBenchmark runs all gorch capabilities on real data and generates a report.
func TestComprehensiveBenchmark(t *testing.T) {
	cacheDir := t.TempDir()
	var results []benchResult

	// ================================================================
	// 1. MNIST digit classification (MLP)
	// ================================================================
	t.Log("=== 1. MNIST Digit Classification ===")
	{
		trainSet, err := data.LoadMNIST(cacheDir, true)
		if err != nil {
			t.Fatalf("load MNIST: %v", err)
		}
		testSet, err := data.LoadMNIST(cacheDir, false)
		if err != nil {
			t.Fatalf("load MNIST test: %v", err)
		}

		mdl := nn.NewSequential(
			nn.NewLinear(784, 128), nn.NewReLU(),
			nn.NewLinear(128, 10),
		)
		opt := optim.NewAdam(mdl.Parameters(), 0.001)
		loader := data.NewDataLoader(trainSet, 64, true)

		start := time.Now()
		var finalLoss float32
		for epoch := 0; epoch < 3; epoch++ {
			loader.Reset()
			var el float32
			b := 0
			for {
				inp, tgt := loader.Next()
				if inp == nil {
					break
				}
				opt.ZeroGrad()
				loss := g.CrossEntropyLoss(mdl.Forward(inp), tgt)
				loss.Backward()
				opt.Step()
				el += loss.Data()[0]
				b++
			}
			finalLoss = el / float32(b)
		}
		trainTime := time.Since(start)

		acc := evaluate(mdl, testSet, 10)
		t.Logf("  MNIST: %.2f%%, %v, loss=%.4f", acc, trainTime.Round(time.Millisecond), finalLoss)
		results = append(results, benchResult{
			Category: "Image Classification", Name: "MNIST (MLP 784→128→10)",
			Accuracy: acc, Loss: finalLoss, TrainTime: trainTime.Round(time.Millisecond).String(),
			TrainTimeMs: float64(trainTime.Milliseconds()), Params: countParams(mdl), Epochs: 3,
			Details: "Handwritten digit recognition, 60K train / 10K test",
		})
	}

	// ================================================================
	// 2. Fashion-MNIST (MLP)
	// ================================================================
	t.Log("=== 2. Fashion-MNIST (MLP) ===")
	{
		trainSet, err := data.LoadFashionMNIST(cacheDir, true)
		if err != nil {
			t.Fatalf("load Fashion: %v", err)
		}
		testSet, err := data.LoadFashionMNIST(cacheDir, false)
		if err != nil {
			t.Fatalf("load Fashion test: %v", err)
		}

		mdl := nn.NewSequential(
			nn.NewLinear(784, 256), nn.NewReLU(),
			nn.NewLinear(256, 128), nn.NewReLU(),
			nn.NewLinear(128, 10),
		)
		opt := optim.NewAdam(mdl.Parameters(), 0.001)
		loader := data.NewDataLoader(trainSet, 128, true)

		start := time.Now()
		var finalLoss float32
		for epoch := 0; epoch < 10; epoch++ {
			loader.Reset()
			var el float32
			b := 0
			for {
				inp, tgt := loader.Next()
				if inp == nil {
					break
				}
				opt.ZeroGrad()
				loss := g.CrossEntropyLoss(mdl.Forward(inp), tgt)
				loss.Backward()
				opt.Step()
				el += loss.Data()[0]
				b++
			}
			finalLoss = el / float32(b)
		}
		trainTime := time.Since(start)

		acc := evaluate(mdl, testSet, 10)
		t.Logf("  Fashion MLP: %.2f%%, %v", acc, trainTime.Round(time.Millisecond))
		results = append(results, benchResult{
			Category: "Image Classification", Name: "Fashion-MNIST (MLP 784→256→128→10)",
			Accuracy: acc, Loss: finalLoss, TrainTime: trainTime.Round(time.Millisecond).String(),
			TrainTimeMs: float64(trainTime.Milliseconds()), Params: countParams(mdl), Epochs: 10,
			Details: "Clothing classification, 60K train / 10K test, 10 classes",
		})
	}

	// ================================================================
	// 3. Fashion-MNIST (CNN)
	// ================================================================
	t.Log("=== 3. Fashion-MNIST (CNN) ===")
	{
		trainSet, err := data.LoadFashionMNIST(cacheDir, true)
		if err != nil {
			t.Fatalf("load Fashion: %v", err)
		}
		testSet, err := data.LoadFashionMNIST(cacheDir, false)
		if err != nil {
			t.Fatalf("load Fashion test: %v", err)
		}

		trainSet2D := trainSet.As2D()
		testSet2D := testSet.As2D()

		mdl := nn.NewSequential(
			nn.NewConv2d(1, 16, 3, 1, 1), nn.NewReLU(), nn.NewMaxPool2d(2, 2),
			nn.NewConv2d(16, 32, 3, 1, 1), nn.NewReLU(), nn.NewMaxPool2d(2, 2),
			nn.NewFlatten(),
			nn.NewLinear(1568, 10),
		)
		opt := optim.NewAdam(mdl.Parameters(), 0.001)
		loader := data.NewDataLoader(trainSet2D, 64, true)

		start := time.Now()
		var finalLoss float32
		for epoch := 0; epoch < 3; epoch++ {
			loader.Reset()
			var el float32
			b := 0
			for {
				inp, tgt := loader.Next()
				if inp == nil {
					break
				}
				opt.ZeroGrad()
				loss := g.CrossEntropyLoss(mdl.Forward(inp), tgt)
				loss.Backward()
				opt.Step()
				el += loss.Data()[0]
				b++
			}
			finalLoss = el / float32(b)
		}
		trainTime := time.Since(start)

		acc := evaluateDataset(mdl, testSet2D, 10)
		t.Logf("  Fashion CNN: %.2f%%, %v", acc, trainTime.Round(time.Millisecond))
		results = append(results, benchResult{
			Category: "Image Classification", Name: "Fashion-MNIST (CNN Conv16→Conv32→FC)",
			Accuracy: acc, Loss: finalLoss, TrainTime: trainTime.Round(time.Millisecond).String(),
			TrainTimeMs: float64(trainTime.Milliseconds()), Params: countParams(mdl), Epochs: 3,
			Details: "2-layer CNN, 3 epochs only (vs 5 in dedicated test)",
		})
	}

	// ================================================================
	// 4. Breast Cancer Wisconsin
	// ================================================================
	t.Log("=== 4. Breast Cancer Wisconsin ===")
	{
		ds, err := data.LoadBreastCancer(cacheDir)
		if err != nil {
			t.Fatalf("load BC: %v", err)
		}
		ds.Normalize()
		trainSet, testSet := ds.TrainTestSplit(0.2)

		mdl := nn.NewSequential(
			nn.NewLinear(30, 64), nn.NewReLU(),
			nn.NewLinear(64, 32), nn.NewReLU(),
			nn.NewLinear(32, 2),
		)
		opt := optim.NewAdam(mdl.Parameters(), 0.001)
		loader := data.NewDataLoader(trainSet, 32, true)

		start := time.Now()
		var finalLoss float32
		for epoch := 0; epoch < 50; epoch++ {
			loader.Reset()
			var el float32
			b := 0
			for {
				inp, tgt := loader.Next()
				if inp == nil {
					break
				}
				opt.ZeroGrad()
				loss := g.CrossEntropyLoss(mdl.Forward(inp), tgt)
				loss.Backward()
				opt.Step()
				el += loss.Data()[0]
				b++
			}
			finalLoss = el / float32(b)
		}
		trainTime := time.Since(start)

		acc := evaluate(mdl, testSet, 2)
		t.Logf("  Breast Cancer: %.2f%%, %v", acc, trainTime.Round(time.Millisecond))
		results = append(results, benchResult{
			Category: "Medical", Name: "Breast Cancer Wisconsin (MLP 30→64→32→2)",
			Accuracy: acc, Loss: finalLoss, TrainTime: trainTime.Round(time.Millisecond).String(),
			TrainTimeMs: float64(trainTime.Milliseconds()), Params: countParams(mdl), Epochs: 50,
			Details: "Binary classification (benign/malignant), 569 samples, 30 features",
		})
	}

	// ================================================================
	// 5. Wine Quality
	// ================================================================
	t.Log("=== 5. Wine Quality ===")
	{
		ds, err := data.LoadWineQuality(cacheDir)
		if err != nil {
			t.Fatalf("load wine: %v", err)
		}
		ds.Normalize()
		trainSet, testSet := ds.TrainTestSplit(0.2)

		mdl := nn.NewSequential(
			nn.NewLinear(11, 128), nn.NewReLU(),
			nn.NewLinear(128, 64), nn.NewReLU(),
			nn.NewLinear(64, 32), nn.NewReLU(),
			nn.NewLinear(32, ds.NumClasses()),
		)
		opt := optim.NewAdam(mdl.Parameters(), 0.001)
		loader := data.NewDataLoader(trainSet, 32, true)

		start := time.Now()
		var finalLoss float32
		for epoch := 0; epoch < 50; epoch++ {
			loader.Reset()
			var el float32
			b := 0
			for {
				inp, tgt := loader.Next()
				if inp == nil {
					break
				}
				opt.ZeroGrad()
				loss := g.CrossEntropyLoss(mdl.Forward(inp), tgt)
				loss.Backward()
				opt.Step()
				el += loss.Data()[0]
				b++
			}
			finalLoss = el / float32(b)
		}
		trainTime := time.Since(start)

		acc := evaluate(mdl, testSet, ds.NumClasses())
		t.Logf("  Wine Quality: %.2f%%, %v", acc, trainTime.Round(time.Millisecond))
		results = append(results, benchResult{
			Category: "Tabular", Name: "Wine Quality (MLP 11→128→64→32→6)",
			Accuracy: acc, Loss: finalLoss, TrainTime: trainTime.Round(time.Millisecond).String(),
			TrainTimeMs: float64(trainTime.Milliseconds()), Params: countParams(mdl), Epochs: 50,
			Details: "6-class wine score prediction, 1599 samples, 11 features",
		})
	}

	// ================================================================
	// 6. GPT character-level training
	// ================================================================
	t.Log("=== 6. GPT Character-Level Training ===")
	{
		corpus := "to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles"
		tok := model.NewSimpleTokenizer(corpus)

		gpt := nn.NewGPT(tok.VocabSize(), 64, 4, 2, 256)
		opt := optim.NewAdam(gpt.Parameters(), 0.003)
		sched := optim.NewWarmupCosineScheduler(0.003, 0.0001, 20, 200, opt.SetLR)

		ids := tok.Encode(corpus)
		seqLen := len(ids) - 1

		start := time.Now()
		var initialLoss, finalLoss float32
		for step := 0; step < 200; step++ {
			opt.ZeroGrad()
			logits := gpt.Forward(ids[:seqLen])

			targets := make([]float32, seqLen)
			for i := 0; i < seqLen; i++ {
				targets[i] = float32(ids[i+1])
			}
			loss := g.CrossEntropyLoss(logits, g.NewTensor(targets, seqLen, 1))
			loss.Backward()
			opt.Step()
			sched.Step()

			if step == 0 {
				initialLoss = loss.Data()[0]
			}
			finalLoss = loss.Data()[0]

			// Zero grads on all params for next step
			for _, p := range gpt.Parameters() {
				p.ZeroGrad()
			}
		}
		trainTime := time.Since(start)

		t.Logf("  GPT: loss %.4f → %.4f, %v, %d params",
			initialLoss, finalLoss, trainTime.Round(time.Millisecond), gpt.CountParameters())

		results = append(results, benchResult{
			Category: "Language Model", Name: "GPT (char-level, 2-layer, dim=64)",
			Loss: finalLoss, TrainTime: trainTime.Round(time.Millisecond).String(),
			TrainTimeMs: float64(trainTime.Milliseconds()), Params: gpt.CountParameters(), Epochs: 200,
			Details: fmt.Sprintf("Shakespeare-style text, vocab=%d, loss %.4f→%.4f, warmup+cosine LR", tok.VocabSize(), initialLoss, finalLoss),
		})
	}

	// ================================================================
	// 7. Model save/load round-trip
	// ================================================================
	t.Log("=== 7. Save/Load Round-Trip ===")
	{
		mdl := nn.NewSequential(
			nn.NewLinear(10, 20), nn.NewReLU(),
			nn.NewLinear(20, 5),
		)
		params := mdl.Parameters()
		savePath := filepath.Join(cacheDir, "test_model.safetensors")

		nameMap := make(map[int]string)
		loadMap := make(map[string]int)
		for i := range params {
			name := fmt.Sprintf("param.%d", i)
			nameMap[i] = name
			loadMap[name] = i
		}

		// Save
		start := time.Now()
		err := model.SaveModelWeights(savePath, params, nameMap)
		if err != nil {
			t.Fatalf("save: %v", err)
		}
		saveTime := time.Since(start)

		// Create fresh model and load
		mdl2 := nn.NewSequential(
			nn.NewLinear(10, 20), nn.NewReLU(),
			nn.NewLinear(20, 5),
		)
		params2 := mdl2.Parameters()

		start = time.Now()
		err = model.LoadModelWeights(savePath, params2, loadMap)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		loadTime := time.Since(start)

		// Verify exact match
		match := true
		for i := range params {
			for j := range params[i].Data() {
				if params[i].Data()[j] != params2[i].Data()[j] {
					match = false
					break
				}
			}
		}

		status := "PASS"
		if !match {
			status = "FAIL"
			t.Fatal("save/load round-trip mismatch")
		}

		t.Logf("  Save: %v, Load: %v, Match: %s", saveTime.Round(time.Microsecond), loadTime.Round(time.Microsecond), status)
		results = append(results, benchResult{
			Category: "Model I/O", Name: "Safetensors Save/Load Round-Trip",
			TrainTime: fmt.Sprintf("save=%v load=%v", saveTime.Round(time.Microsecond), loadTime.Round(time.Microsecond)),
			TrainTimeMs: float64(saveTime.Microseconds() + loadTime.Microseconds()) / 1000.0,
			Params: countParams(mdl),
			Details: fmt.Sprintf("4 tensors, exact value match: %s", status),
		})
	}

	// ================================================================
	// Write results JSON
	// ================================================================
	jsonBytes, _ := json.MarshalIndent(results, "", "  ")
	jsonPath := filepath.Join("..", "doc", "comprehensive_results.json")
	os.WriteFile(jsonPath, jsonBytes, 0644)

	// ================================================================
	// Print summary
	// ================================================================
	t.Log("\n========== COMPREHENSIVE BENCHMARK SUMMARY ==========")
	t.Logf("%-45s %10s %10s %8s", "Benchmark", "Accuracy", "Time", "Params")
	t.Log(strings.Repeat("-", 80))
	for _, r := range results {
		accStr := "—"
		if r.Accuracy > 0 {
			accStr = fmt.Sprintf("%.2f%%", r.Accuracy)
		}
		t.Logf("%-45s %10s %10s %8d", r.Name, accStr, r.TrainTime, r.Params)
	}

	var totalMs float64
	for _, r := range results {
		totalMs += r.TrainTimeMs
	}
	t.Logf("\nTotal training time: %.1fs", totalMs/1000)
	t.Logf("Total benchmarks: %d", len(results))
}

// evaluate runs a Sequential model on a dataset and returns accuracy.
func evaluate(mdl *nn.Sequential, ds data.Dataset, numClasses int) float64 {
	return evaluateDataset(mdl, ds, numClasses)
}

func evaluateDataset(mdl *nn.Sequential, ds data.Dataset, numClasses int) float64 {
	loader := data.NewDataLoader(ds, 256, false)
	loader.Reset()
	correct, total := 0, 0
	for {
		inp, tgt := loader.Next()
		if inp == nil {
			break
		}
		logits := mdl.Forward(inp)
		preds := logits.Data()
		tgts := tgt.Data()
		batch := inp.Shape()[0]
		for i := 0; i < batch; i++ {
			maxIdx := 0
			maxVal := preds[i*numClasses]
			for j := 1; j < numClasses; j++ {
				if preds[i*numClasses+j] > maxVal {
					maxVal = preds[i*numClasses+j]
					maxIdx = j
				}
			}
			if maxIdx == int(tgts[i]) {
				correct++
			}
			total++
		}
	}
	return float64(correct) / float64(total) * 100
}

func countParams(mdl *nn.Sequential) int {
	n := 0
	for _, p := range mdl.Parameters() {
		n += p.Size()
	}
	return n
}
