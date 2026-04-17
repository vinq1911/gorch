package iris

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/vinq1911/gorch/embedded"
)

// FloatModel is a small MLP trained in float32 so we have an accuracy
// baseline and calibration forward pass. Architecture: InputDim -> HiddenDim
// (ReLU) -> OutputDim (logits).
type FloatModel struct {
	InputDim, HiddenDim, OutputDim int

	// Row-major weights: W1[h*InputDim + i], W2[o*HiddenDim + h].
	W1, B1 []float32
	W2, B2 []float32

	// Feature normalization: the deployed int8 model receives
	// round((x - FeatMean) / FeatStd / InputScale) as its int8 input.
	FeatMean  []float32
	FeatStd   []float32
	InputScale float32
}

// TrainConfig is the knobs for SGD training.
type TrainConfig struct {
	HiddenDim    int
	LearningRate float32
	Epochs       int
	Seed         int64
}

// Train runs plain SGD with softmax cross-entropy loss. Returns the
// trained float model including normalization parameters computed from
// the training set.
func Train(cfg TrainConfig, xTr [][]float32, yTr []int) *FloatModel {
	if len(xTr) == 0 {
		panic("iris: empty training set")
	}
	inputDim := len(xTr[0])
	outputDim := 3 // iris classes

	r := rand.New(rand.NewSource(cfg.Seed))
	m := &FloatModel{
		InputDim:  inputDim,
		HiddenDim: cfg.HiddenDim,
		OutputDim: outputDim,
	}

	// Per-feature mean / std over the training set.
	m.FeatMean = make([]float32, inputDim)
	m.FeatStd = make([]float32, inputDim)
	for _, x := range xTr {
		for i, v := range x {
			m.FeatMean[i] += v
		}
	}
	for i := range m.FeatMean {
		m.FeatMean[i] /= float32(len(xTr))
	}
	for _, x := range xTr {
		for i, v := range x {
			d := v - m.FeatMean[i]
			m.FeatStd[i] += d * d
		}
	}
	for i := range m.FeatStd {
		m.FeatStd[i] = float32(math.Sqrt(float64(m.FeatStd[i] / float32(len(xTr)))))
		if m.FeatStd[i] < 1e-6 {
			m.FeatStd[i] = 1
		}
	}

	// Precompute normalized training set.
	xNorm := make([][]float32, len(xTr))
	for i, x := range xTr {
		xNorm[i] = m.normalize(x)
	}

	// He init for the ReLU layer, smaller scale for the classifier head.
	he1 := float32(math.Sqrt(2.0 / float64(inputDim)))
	he2 := float32(math.Sqrt(2.0 / float64(cfg.HiddenDim)))
	m.W1 = make([]float32, cfg.HiddenDim*inputDim)
	m.B1 = make([]float32, cfg.HiddenDim)
	m.W2 = make([]float32, outputDim*cfg.HiddenDim)
	m.B2 = make([]float32, outputDim)
	for i := range m.W1 {
		m.W1[i] = float32(r.NormFloat64()) * he1
	}
	for i := range m.W2 {
		m.W2[i] = float32(r.NormFloat64()) * he2
	}

	// Plain SGD, full-batch gradient step per "epoch" is fine for 90 samples.
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		// Shuffle sample indices so per-sample updates don't cycle.
		perm := r.Perm(len(xNorm))
		for _, idx := range perm {
			m.sgdStep(xNorm[idx], yTr[idx], cfg.LearningRate)
		}
	}

	// Input quantization scale, computed on the normalized training set.
	var maxAbs float32
	for _, x := range xNorm {
		for _, v := range x {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
	}
	if maxAbs < 1e-6 {
		maxAbs = 1
	}
	m.InputScale = maxAbs / 127

	return m
}

func (m *FloatModel) normalize(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = (v - m.FeatMean[i]) / m.FeatStd[i]
	}
	return out
}

// QuantizeInput maps a raw float feature vector to the int8 input the
// kernel expects (normalized, then scaled and saturated).
func (m *FloatModel) QuantizeInput(x []float32) []int8 {
	out := make([]int8, len(x))
	xn := m.normalize(x)
	for i, v := range xn {
		q := int32(math.Round(float64(v / m.InputScale)))
		if q > 127 {
			q = 127
		}
		if q < -128 {
			q = -128
		}
		out[i] = int8(q)
	}
	return out
}

// Forward runs the float model on a pre-normalized input.
func (m *FloatModel) Forward(xNorm []float32) (hidden, logits []float32) {
	hidden = make([]float32, m.HiddenDim)
	for h := 0; h < m.HiddenDim; h++ {
		s := m.B1[h]
		for i := 0; i < m.InputDim; i++ {
			s += m.W1[h*m.InputDim+i] * xNorm[i]
		}
		if s < 0 {
			s = 0
		}
		hidden[h] = s
	}
	logits = make([]float32, m.OutputDim)
	for o := 0; o < m.OutputDim; o++ {
		s := m.B2[o]
		for h := 0; h < m.HiddenDim; h++ {
			s += m.W2[o*m.HiddenDim+h] * hidden[h]
		}
		logits[o] = s
	}
	return
}

// Predict returns the argmax class for a raw (unnormalized) feature vector.
func (m *FloatModel) Predict(x []float32) int {
	_, logits := m.Forward(m.normalize(x))
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// sgdStep does one SGD update on a single (normalized) sample.
func (m *FloatModel) sgdStep(xNorm []float32, y int, lr float32) {
	hidden, logits := m.Forward(xNorm)

	// Softmax probabilities
	maxL := logits[0]
	for _, v := range logits[1:] {
		if v > maxL {
			maxL = v
		}
	}
	var sum float32
	probs := make([]float32, m.OutputDim)
	for i, v := range logits {
		p := float32(math.Exp(float64(v - maxL)))
		probs[i] = p
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}

	// dL/dlogits = probs - one_hot(y)
	dLogits := make([]float32, m.OutputDim)
	for i := range dLogits {
		dLogits[i] = probs[i]
	}
	dLogits[y] -= 1

	// Gradients for W2, B2 and backprop to hidden.
	dHidden := make([]float32, m.HiddenDim)
	for o := 0; o < m.OutputDim; o++ {
		dl := dLogits[o]
		for h := 0; h < m.HiddenDim; h++ {
			dHidden[h] += m.W2[o*m.HiddenDim+h] * dl
			m.W2[o*m.HiddenDim+h] -= lr * dl * hidden[h]
		}
		m.B2[o] -= lr * dl
	}

	// ReLU backward.
	for h := 0; h < m.HiddenDim; h++ {
		if hidden[h] == 0 {
			dHidden[h] = 0
		}
	}

	// Gradients for W1, B1.
	for h := 0; h < m.HiddenDim; h++ {
		dh := dHidden[h]
		for i := 0; i < m.InputDim; i++ {
			m.W1[h*m.InputDim+i] -= lr * dh * xNorm[i]
		}
		m.B1[h] -= lr * dh
	}
}

// Accuracy returns the classification accuracy on (x, y).
func (m *FloatModel) Accuracy(xs [][]float32, ys []int) float32 {
	if len(xs) == 0 {
		return 0
	}
	correct := 0
	for i, x := range xs {
		if m.Predict(x) == ys[i] {
			correct++
		}
	}
	return float32(correct) / float32(len(xs))
}

// Quantize converts the float model to an int8 embedded.TinyModel, using
// calibX as the calibration set to find per-tensor activation scales.
// The returned model mirrors the firmware's forward pass bit-for-bit.
func (m *FloatModel) Quantize(calibX [][]float32) *embedded.TinyModel {
	// Calibrate: run each sample forward, track max-abs of hidden (post-ReLU)
	// and logits. We do NOT need a logit scale since the final layer is
	// LinearI32 (no requant) — argmax on the raw int32 accumulator is correct.
	var hiddenMaxAbs float32
	for _, x := range calibX {
		hidden, _ := m.Forward(m.normalize(x))
		for _, v := range hidden {
			a := v
			if a < 0 {
				a = -a
			}
			if a > hiddenMaxAbs {
				hiddenMaxAbs = a
			}
		}
	}
	if hiddenMaxAbs < 1e-6 {
		hiddenMaxAbs = 1
	}
	hiddenScale := hiddenMaxAbs / 127

	// Weight scales: symmetric per-tensor, max-abs / 127.
	w1Scale := maxAbsF32(m.W1) / 127
	w2Scale := maxAbsF32(m.W2) / 127
	if w1Scale < 1e-12 {
		w1Scale = 1e-12
	}
	if w2Scale < 1e-12 {
		w2Scale = 1e-12
	}

	// Layer 0 accumulator units: w1Scale * InputScale.
	// Output desired units: hiddenScale.  M0 real = (w1Scale * InputScale) / hiddenScale.
	// Convert to Q0.31 M + shift.
	m0Real := float64(w1Scale) * float64(m.InputScale) / float64(hiddenScale)
	M0, S0 := quantizeMultiplier(m0Real)

	// Quantize W1, B1.
	W1q := quantizeWeightsI8(m.W1, w1Scale)
	B1q := quantizeBiasI32(m.B1, float64(w1Scale)*float64(m.InputScale))

	// Quantize W2, B2. Accumulator units for L2 are w2Scale * hiddenScale.
	W2q := quantizeWeightsI8(m.W2, w2Scale)
	B2q := quantizeBiasI32(m.B2, float64(w2Scale)*float64(hiddenScale))

	return &embedded.TinyModel{
		InputDim: m.InputDim,
		Layers: []embedded.Layer{
			{
				Kind:   embedded.KindLinearI8,
				InDim:  m.InputDim,
				OutDim: m.HiddenDim,
				W:      W1q,
				B:      B1q,
				M:      M0,
				S:      S0,
			},
			{Kind: embedded.KindReLU, InDim: m.HiddenDim, OutDim: m.HiddenDim},
			{
				Kind:   embedded.KindLinearI32,
				InDim:  m.HiddenDim,
				OutDim: m.OutputDim,
				W:      W2q,
				B:      B2q,
			},
		},
	}
}

func maxAbsF32(v []float32) float32 {
	var m float32
	for _, x := range v {
		a := x
		if a < 0 {
			a = -a
		}
		if a > m {
			m = a
		}
	}
	return m
}

func quantizeWeightsI8(W []float32, scale float32) []int8 {
	out := make([]int8, len(W))
	for i, w := range W {
		q := int32(math.Round(float64(w / scale)))
		if q > 127 {
			q = 127
		}
		if q < -128 {
			q = -128
		}
		out[i] = int8(q)
	}
	return out
}

func quantizeBiasI32(B []float32, accScale float64) []int32 {
	out := make([]int32, len(B))
	for i, b := range B {
		q := math.Round(float64(b) / accScale)
		if q > float64(math.MaxInt32) {
			q = float64(math.MaxInt32)
		}
		if q < float64(math.MinInt32) {
			q = float64(math.MinInt32)
		}
		out[i] = int32(q)
	}
	return out
}

// quantizeMultiplier expresses a real-valued scale M (typically in (0, 1))
// as a Q0.31 multiplier plus a right-shift so that
//     saturate((acc * M + 2^(S-1)) >> S)
// approximates acc * realScale. Matches gm1_requant in gm1_avr.h.
func quantizeMultiplier(realScale float64) (int32, uint8) {
	if realScale <= 0 {
		return 0, 0
	}
	// Scale realScale into [0.5, 1.0) and remember the normalization.
	n := 0
	x := realScale
	for x >= 1.0 {
		x /= 2
		n++
	}
	for x < 0.5 {
		x *= 2
		n--
	}
	// x is in [0.5, 1). M = round(x * 2^31). Shift = 31 - n.
	M := int64(math.Round(x * float64(int64(1)<<31)))
	if M >= int64(1)<<31 {
		M = int64(1)<<31 - 1
	}
	shift := 31 - n
	if shift < 0 {
		// realScale >= 2^31 -- unreachable for our models.
		panic(fmt.Sprintf("quantizeMultiplier: scale %g too large", realScale))
	}
	if shift > 63 {
		shift = 63
	}
	return int32(M), uint8(shift)
}
