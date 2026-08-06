//go:build darwin && e2e

package e2e

import (
	"os"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// embDataset adapts a (N, D) embedding matrix + (N, 1) labels to data.Dataset.
type embDataset struct {
	x, y *g.Tensor
	dim  int
}

func (d *embDataset) Len() int           { return d.x.Shape()[0] }
func (d *embDataset) InputShape() []int  { return []int{d.dim} }
func (d *embDataset) TargetShape() []int { return []int{1} }
func (d *embDataset) Get(i int) (input, target []float32) {
	return d.x.Data()[i*d.dim : (i+1)*d.dim], d.y.Data()[i : i+1]
}

type headResult struct {
	accuracy  float64
	correct   int
	total     int
	params    int
	trainTime time.Duration
	perClip   time.Duration
}

// trainHead trains a small MLP classifier on frozen embeddings and
// evaluates on the held-out split.
func trainHead(t *testing.T, trainSet, testSet *embDataset, classes, epochs int) headResult {
	dim := trainSet.dim
	dropout := nn.NewDropout(0.2)
	classifier := nn.NewSequential(
		nn.NewLinear(dim, 256),
		nn.NewReLU(),
		dropout,
		nn.NewLinear(256, classes),
	)
	params := classifier.Parameters()
	nParams := 0
	for _, p := range params {
		nParams += p.Size()
	}
	opt := optim.NewAdamW(params, 1e-3, 0.01)
	loader := data.NewDataLoader(trainSet, 64, true)

	start := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		loader.Reset()
		var epochLoss float32
		batches := 0
		for {
			inputs, targets := loader.Next()
			if inputs == nil {
				break
			}
			opt.ZeroGrad()
			loss := g.CrossEntropyLoss(classifier.Forward(inputs), targets)
			loss.Backward()
			optim.ClipGradNorm(params, 1.0)
			opt.Step()
			epochLoss += loss.Data()[0]
			batches++
		}
		if (epoch+1)%10 == 0 {
			t.Logf("  epoch %d/%d loss %.4f", epoch+1, epochs, epochLoss/float32(batches))
		}
	}
	trainTime := time.Since(start)

	dropout.Eval()
	evalStart := time.Now()
	correct := 0
	testLoader := data.NewDataLoader(testSet, 256, false)
	testLoader.Reset()
	for {
		inputs, targets := testLoader.Next()
		if inputs == nil {
			break
		}
		logits := classifier.Forward(inputs)
		preds, tgts := logits.Data(), targets.Data()
		for i := 0; i < inputs.Shape()[0]; i++ {
			argmax := 0
			for j := 1; j < classes; j++ {
				if preds[i*classes+j] > preds[i*classes+argmax] {
					argmax = j
				}
			}
			if argmax == int(tgts[i]) {
				correct++
			}
		}
	}
	evalTime := time.Since(evalStart)

	return headResult{
		accuracy:  float64(correct) / float64(testSet.Len()) * 100,
		correct:   correct,
		total:     testSet.Len(),
		params:    nParams,
		trainTime: trainTime,
		perClip:   evalTime / time.Duration(testSet.Len()),
	}
}

// TestMimiFSDD trains audio classifiers on frozen Mimi codec embeddings
// (1024-dim mean+std-pooled latents, see audio/export_fsdd_mimi.py) and
// evaluates on the held-out FSDD test split. Two heads over the same
// features: spoken-digit content (10-way, transcription-like) and
// speaker identity (6-way, paralinguistic — the property sentiment and
// language detection rely on). This is the gorch half of the
// Mimi -> gorch audio pipeline: the frozen codec runs elsewhere,
// gorch owns classifier training and inference.
func TestMimiFSDD(t *testing.T) {
	path := os.Getenv("GORCH_MIMI_FSDD")
	if path == "" {
		path = "../audio/fsdd_mimi.safetensors"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("embeddings not found at %s — run audio/export_fsdd_mimi.py first", path)
	}

	sf, err := model.LoadSafetensors(path)
	if err != nil {
		t.Fatalf("load embeddings: %v", err)
	}
	trainX, testX := sf.Tensors["train_x"], sf.Tensors["test_x"]
	dim := trainX.Shape()[1]
	t.Logf("train %d, test %d, dim %d", trainX.Shape()[0], testX.Shape()[0], dim)

	t.Log("digit head (spoken content, 10 classes):")
	digit := trainHead(t,
		&embDataset{x: trainX, y: sf.Tensors["train_y"], dim: dim},
		&embDataset{x: testX, y: sf.Tensors["test_y"], dim: dim},
		10, 30)
	t.Logf("digit accuracy: %.2f%% (%d/%d), %d params, train %v, inference %v/clip",
		digit.accuracy, digit.correct, digit.total, digit.params,
		digit.trainTime.Round(time.Millisecond), digit.perClip)

	t.Log("speaker head (paralinguistic, 6 classes):")
	speaker := trainHead(t,
		&embDataset{x: trainX, y: sf.Tensors["train_spk"], dim: dim},
		&embDataset{x: testX, y: sf.Tensors["test_spk"], dim: dim},
		6, 30)
	t.Logf("speaker accuracy: %.2f%% (%d/%d), train %v",
		speaker.accuracy, speaker.correct, speaker.total, speaker.trainTime.Round(time.Millisecond))

	// Hardest split: train digits on 5 speakers, test on a fully
	// held-out 6th voice (speaker-independent generalization).
	heldOut := float32(5)
	allX := append(append([]float32{}, trainX.Data()...), testX.Data()...)
	allDigit := append(append([]float32{}, sf.Tensors["train_y"].Data()...), sf.Tensors["test_y"].Data()...)
	allSpk := append(append([]float32{}, sf.Tensors["train_spk"].Data()...), sf.Tensors["test_spk"].Data()...)
	var trX, teX, trY, teY []float32
	for i, spk := range allSpk {
		row := allX[i*dim : (i+1)*dim]
		if spk == heldOut {
			teX = append(teX, row...)
			teY = append(teY, allDigit[i])
		} else {
			trX = append(trX, row...)
			trY = append(trY, allDigit[i])
		}
	}
	t.Logf("speaker-independent digit head (train %d clips / 5 voices, test %d clips / 1 unseen voice):",
		len(trY), len(teY))
	si := trainHead(t,
		&embDataset{x: g.NewTensor(trX, len(trY), dim), y: g.NewTensor(trY, len(trY), 1), dim: dim},
		&embDataset{x: g.NewTensor(teX, len(teY), dim), y: g.NewTensor(teY, len(teY), 1), dim: dim},
		10, 30)
	t.Logf("speaker-independent digit accuracy: %.2f%% (%d/%d)", si.accuracy, si.correct, si.total)

	if digit.accuracy < 90.0 {
		t.Fatalf("digit accuracy %.2f%% below 90%% threshold", digit.accuracy)
	}
	if speaker.accuracy < 90.0 {
		t.Fatalf("speaker accuracy %.2f%% below 90%% threshold", speaker.accuracy)
	}
	if si.accuracy < 70.0 {
		t.Fatalf("speaker-independent accuracy %.2f%% below 70%% threshold", si.accuracy)
	}
}
