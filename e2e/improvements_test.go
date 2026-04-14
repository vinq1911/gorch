//go:build darwin && e2e

package e2e

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

type improvResult struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Accuracy    float64 `json:"accuracy"`
	FinalLoss   float32 `json:"final_loss"`
	TrainTime   string  `json:"train_time"`
	TrainTimeMs int64   `json:"train_time_ms"`
	Params      int     `json:"params"`
	Epochs      int     `json:"epochs"`
}

// TestImprovements benchmarks BatchNorm, Dropout, GELU, LR scheduling
// on Fashion-MNIST to measure their impact.
func TestImprovements(t *testing.T) {
	cacheDir := t.TempDir()

	trainSet, err := data.LoadFashionMNIST(cacheDir, true)
	if err != nil {
		t.Fatalf("load fashion train: %v", err)
	}
	testSet, err := data.LoadFashionMNIST(cacheDir, false)
	if err != nil {
		t.Fatalf("load fashion test: %v", err)
	}

	var results []improvResult

	// ===== 1. Baseline: plain MLP =====
	t.Log("=== 1. Baseline MLP ===")
	{
		mdl := nn.NewSequential(
			nn.NewLinear(784, 256), nn.NewReLU(),
			nn.NewLinear(256, 128), nn.NewReLU(),
			nn.NewLinear(128, 10),
		)
		r := trainAndMeasure(t, "Baseline MLP (ReLU)", "784→256→128→10, Adam 0.001, no regularization",
			mdl, trainSet, testSet, 0.001, 10, nil)
		results = append(results, r)
	}

	// ===== 2. MLP + GELU =====
	t.Log("=== 2. MLP + GELU ===")
	{
		mdl := nn.NewSequential(
			nn.NewLinear(784, 256), &geluModule{},
			nn.NewLinear(256, 128), &geluModule{},
			nn.NewLinear(128, 10),
		)
		r := trainAndMeasure(t, "MLP + GELU", "Same architecture, GELU instead of ReLU",
			mdl, trainSet, testSet, 0.001, 10, nil)
		results = append(results, r)
	}

	// ===== 3. MLP + BatchNorm =====
	t.Log("=== 3. MLP + BatchNorm ===")
	{
		mdl := nn.NewSequential(
			nn.NewLinear(784, 256), nn.NewBatchNorm1d(256), nn.NewReLU(),
			nn.NewLinear(256, 128), nn.NewBatchNorm1d(128), nn.NewReLU(),
			nn.NewLinear(128, 10),
		)
		r := trainAndMeasure(t, "MLP + BatchNorm", "BatchNorm after each Linear, before activation",
			mdl, trainSet, testSet, 0.001, 10, nil)
		results = append(results, r)
	}

	// ===== 4. MLP + Dropout =====
	t.Log("=== 4. MLP + Dropout ===")
	{
		d1 := nn.NewDropout(0.2)
		d2 := nn.NewDropout(0.2)
		mdl := nn.NewSequential(
			nn.NewLinear(784, 256), nn.NewReLU(), d1,
			nn.NewLinear(256, 128), nn.NewReLU(), d2,
			nn.NewLinear(128, 10),
		)
		r := trainAndMeasure(t, "MLP + Dropout(0.2)", "Dropout after each activation",
			mdl, trainSet, testSet, 0.001, 10, func() { d1.Eval(); d2.Eval() })
		results = append(results, r)
	}

	// ===== 5. MLP + BatchNorm + Dropout + GELU =====
	t.Log("=== 5. Full improvements ===")
	{
		bn1 := nn.NewBatchNorm1d(256)
		bn2 := nn.NewBatchNorm1d(128)
		d1 := nn.NewDropout(0.2)
		d2 := nn.NewDropout(0.2)
		mdl := nn.NewSequential(
			nn.NewLinear(784, 256), bn1, &geluModule{}, d1,
			nn.NewLinear(256, 128), bn2, &geluModule{}, d2,
			nn.NewLinear(128, 10),
		)
		r := trainAndMeasure(t, "Full (BN+GELU+Dropout)", "BatchNorm + GELU + Dropout(0.2)",
			mdl, trainSet, testSet, 0.001, 10, func() { bn1.Eval(); bn2.Eval(); d1.Eval(); d2.Eval() })
		results = append(results, r)
	}

	// ===== 6. Full + Cosine LR =====
	t.Log("=== 6. Full + Cosine LR ===")
	{
		bn1 := nn.NewBatchNorm1d(256)
		bn2 := nn.NewBatchNorm1d(128)
		d1 := nn.NewDropout(0.2)
		d2 := nn.NewDropout(0.2)
		mdl := nn.NewSequential(
			nn.NewLinear(784, 256), bn1, &geluModule{}, d1,
			nn.NewLinear(256, 128), bn2, &geluModule{}, d2,
			nn.NewLinear(128, 10),
		)
		opt := optim.NewAdam(mdl.Parameters(), 0.002)
		sched := optim.NewCosineAnnealingLR(opt, 0.002, 0.0001, 10, opt.SetLR)
		loader := data.NewDataLoader(trainSet, 128, true)

		start := time.Now()
		var finalLoss float32
		for epoch := 0; epoch < 10; epoch++ {
			loader.Reset()
			var el float32; b := 0
			for {
				inp, tgt := loader.Next()
				if inp == nil { break }
				opt.ZeroGrad()
				loss := g.CrossEntropyLoss(mdl.Forward(inp), tgt)
				loss.Backward()
				opt.Step()
				el += loss.Data()[0]; b++
			}
			finalLoss = el / float32(b)
			sched.Step()
		}
		trainTime := time.Since(start)

		// Eval mode
		bn1.Eval(); bn2.Eval(); d1.Eval(); d2.Eval()
		acc := evalAccuracy(mdl, testSet, 10)
		t.Logf("  Full+Cosine: %.2f%%, %v", acc, trainTime.Round(time.Millisecond))

		results = append(results, improvResult{
			Name: "Full + Cosine LR", Description: "BN+GELU+Dropout + CosineAnnealing(0.002→0.0001)",
			Accuracy: acc, FinalLoss: finalLoss, TrainTime: trainTime.Round(time.Millisecond).String(),
			TrainTimeMs: trainTime.Milliseconds(), Params: countParams(mdl), Epochs: 10,
		})
	}

	// Save results
	jsonBytes, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("../doc/improvements_results.json", jsonBytes, 0644)

	// Summary
	t.Log("\n========== IMPROVEMENT COMPARISON ==========")
	t.Logf("%-30s %10s %10s %10s", "Config", "Accuracy", "Time", "vs Base")
	t.Log(strings.Repeat("-", 65))
	baseAcc := results[0].Accuracy
	for _, r := range results {
		delta := r.Accuracy - baseAcc
		sign := "+"
		if delta < 0 { sign = "" }
		t.Logf("%-30s %9.2f%% %10s %9s%.1f%%", r.Name, r.Accuracy, r.TrainTime, sign, delta)
	}
}

func trainAndMeasure(t *testing.T, name, desc string, mdl *nn.Sequential, trainSet, testSet *data.MNISTDataset, lr float32, epochs int, evalSetup func()) improvResult {
	opt := optim.NewAdam(mdl.Parameters(), lr)
	loader := data.NewDataLoader(trainSet, 128, true)

	start := time.Now()
	var finalLoss float32
	for epoch := 0; epoch < epochs; epoch++ {
		loader.Reset()
		var el float32; b := 0
		for {
			inp, tgt := loader.Next()
			if inp == nil { break }
			opt.ZeroGrad()
			loss := g.CrossEntropyLoss(mdl.Forward(inp), tgt)
			loss.Backward()
			opt.Step()
			el += loss.Data()[0]; b++
		}
		finalLoss = el / float32(b)
	}
	trainTime := time.Since(start)

	if evalSetup != nil {
		evalSetup()
	}
	acc := evalAccuracy(mdl, testSet, 10)
	t.Logf("  %s: %.2f%%, %v", name, acc, trainTime.Round(time.Millisecond))

	return improvResult{
		Name: name, Description: desc, Accuracy: acc, FinalLoss: finalLoss,
		TrainTime: trainTime.Round(time.Millisecond).String(), TrainTimeMs: trainTime.Milliseconds(),
		Params: countParams(mdl), Epochs: epochs,
	}
}

func evalAccuracy(mdl *nn.Sequential, testSet data.Dataset, numClasses int) float64 {
	loader := data.NewDataLoader(testSet, 256, false)
	loader.Reset()
	correct, total := 0, 0
	for {
		inp, tgt := loader.Next()
		if inp == nil { break }
		logits := mdl.Forward(inp)
		preds := logits.Data(); tgts := tgt.Data()
		batch := inp.Shape()[0]
		for i := 0; i < batch; i++ {
			maxIdx := 0; maxVal := preds[i*numClasses]
			for j := 1; j < numClasses; j++ {
				if preds[i*numClasses+j] > maxVal { maxVal = preds[i*numClasses+j]; maxIdx = j }
			}
			if maxIdx == int(tgts[i]) { correct++ }
			total++
		}
	}
	return float64(correct) / float64(total) * 100
}

// geluModule wraps GELU as a Module.
type geluModule struct{}
func (gm *geluModule) Forward(x *g.Tensor) *g.Tensor { return g.GELU(x) }
func (gm *geluModule) Parameters() []*g.Tensor        { return nil }
