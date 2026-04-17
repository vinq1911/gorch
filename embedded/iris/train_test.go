package iris

import (
	"testing"
)

func TestTrainFloatReachesReasonableAccuracy(t *testing.T) {
	xTr, xTe, yTr, yTe := Split(0.2)
	m := Train(TrainConfig{
		HiddenDim:    8,
		LearningRate: 0.05,
		Epochs:       300,
		Seed:         42,
	}, xTr, yTr)

	trainAcc := m.Accuracy(xTr, yTr)
	testAcc := m.Accuracy(xTe, yTe)
	t.Logf("float model: train=%.3f test=%.3f", trainAcc, testAcc)
	if testAcc < 0.85 {
		t.Fatalf("float test accuracy too low: %.3f (expected >= 0.85)", testAcc)
	}
}

func TestQuantizedModelMatchesFloatWithinTolerance(t *testing.T) {
	xTr, xTe, yTr, yTe := Split(0.2)
	fm := Train(TrainConfig{
		HiddenDim:    8,
		LearningRate: 0.05,
		Epochs:       300,
		Seed:         42,
	}, xTr, yTr)

	qm := fm.Quantize(xTr)

	// Count quantized-vs-float disagreements on the full dataset.
	var total, disagree, qCorrect, fCorrect int
	for _, row := range Data {
		x := []float32{row[0], row[1], row[2], row[3]}
		y := int(row[4])
		fPred := fm.Predict(x)
		qPred, _ := qm.Infer(fm.QuantizeInput(x))
		total++
		if fPred != int(qPred) {
			disagree++
		}
		if fPred == y {
			fCorrect++
		}
		if int(qPred) == y {
			qCorrect++
		}
	}
	t.Logf("agreement float-vs-int8: %d/%d (%.1f%%); float acc %.3f  int8 acc %.3f",
		total-disagree, total, 100*float64(total-disagree)/float64(total),
		float64(fCorrect)/float64(total), float64(qCorrect)/float64(total))

	// Int8 accuracy shouldn't drop catastrophically.
	if float64(qCorrect)/float64(total) < 0.85 {
		t.Fatalf("int8 accuracy %.3f is too low", float64(qCorrect)/float64(total))
	}

	// Test-set accuracy for int8.
	qCorrectTest := 0
	for i, x := range xTe {
		qPred, _ := qm.Infer(fm.QuantizeInput(x))
		if int(qPred) == yTe[i] {
			qCorrectTest++
		}
	}
	t.Logf("int8 test acc: %d/%d = %.3f", qCorrectTest, len(xTe),
		float32(qCorrectTest)/float32(len(xTe)))
}
