//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// BatchNorm1d implements 1D batch normalization for (batch, features) tensors.
// During training, normalizes using batch mean/variance and updates running stats.
// During evaluation, uses running mean/variance.
type BatchNorm1d struct {
	Weight     *g.Tensor // gamma (features,)
	Bias       *g.Tensor // beta (features,)
	RunMean    []float32 // running mean (not a grad tensor)
	RunVar     []float32 // running variance
	Momentum   float32
	Eps        float32
	Features   int
	Training   bool
}

// NewBatchNorm1d creates a BatchNorm1d layer.
func NewBatchNorm1d(features int) *BatchNorm1d {
	w := g.Ones(features)
	w.SetRequiresGrad(true)
	b := g.Zeros(features)
	b.SetRequiresGrad(true)

	runVar := make([]float32, features)
	for i := range runVar {
		runVar[i] = 1.0
	}

	return &BatchNorm1d{
		Weight:   w,
		Bias:     b,
		RunMean:  make([]float32, features),
		RunVar:   runVar,
		Momentum: 0.1,
		Eps:      1e-5,
		Features: features,
		Training: true,
	}
}

// Forward applies batch normalization.
func (bn *BatchNorm1d) Forward(x *g.Tensor) *g.Tensor {
	if x.Dim() != 2 {
		panic("gorch: BatchNorm1d requires 2-D tensor (batch, features)")
	}
	batch := x.Shape()[0]
	N := bn.Features
	xData := x.Data()
	wData := bn.Weight.Data()
	bData := bn.Bias.Data()

	outData := make([]float32, batch*N)

	var mean, variance []float32
	var xNorm []float32

	if bn.Training {
		// Compute batch mean and variance
		mean = make([]float32, N)
		variance = make([]float32, N)
		for j := 0; j < N; j++ {
			var m float64
			for i := 0; i < batch; i++ {
				m += float64(xData[i*N+j])
			}
			mean[j] = float32(m / float64(batch))
		}
		for j := 0; j < N; j++ {
			var v float64
			for i := 0; i < batch; i++ {
				d := float64(xData[i*N+j]) - float64(mean[j])
				v += d * d
			}
			variance[j] = float32(v / float64(batch))
		}

		// Update running stats
		for j := 0; j < N; j++ {
			bn.RunMean[j] = (1-bn.Momentum)*bn.RunMean[j] + bn.Momentum*mean[j]
			bn.RunVar[j] = (1-bn.Momentum)*bn.RunVar[j] + bn.Momentum*variance[j]
		}
	} else {
		mean = bn.RunMean
		variance = bn.RunVar
	}

	// Normalize and apply affine
	xNorm = make([]float32, batch*N)
	for j := 0; j < N; j++ {
		invStd := float32(1.0 / math.Sqrt(float64(variance[j]+bn.Eps)))
		for i := 0; i < batch; i++ {
			normalized := (xData[i*N+j] - mean[j]) * invStd
			xNorm[i*N+j] = normalized
			outData[i*N+j] = normalized*wData[j] + bData[j]
		}
	}

	out := g.NewTensor(outData, batch, N)

	if x.RequiresGrad() || bn.Weight.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("BatchNorm1d", []*g.Tensor{x, bn.Weight, bn.Bias}, func(grad *g.Tensor) []*g.Tensor {
			gData := grad.Data()
			dx := g.Zeros(batch, N)
			dw := g.Zeros(N)
			db := g.Zeros(N)
			dxData := dx.Data()
			dwData := dw.Data()
			dbData := db.Data()

			for j := 0; j < N; j++ {
				invStd := float32(1.0 / math.Sqrt(float64(variance[j]+bn.Eps)))

				// dBias, dWeight
				for i := 0; i < batch; i++ {
					dbData[j] += gData[i*N+j]
					dwData[j] += gData[i*N+j] * xNorm[i*N+j]
				}

				// dx
				var sumDxhat, sumDxhatXnorm float64
				for i := 0; i < batch; i++ {
					dxhat := float64(gData[i*N+j] * wData[j])
					sumDxhat += dxhat
					sumDxhatXnorm += dxhat * float64(xNorm[i*N+j])
				}
				meanDxhat := sumDxhat / float64(batch)
				meanDxhatXnorm := sumDxhatXnorm / float64(batch)

				for i := 0; i < batch; i++ {
					dxhat := float64(gData[i*N+j] * wData[j])
					dxData[i*N+j] = float32((dxhat - meanDxhat - float64(xNorm[i*N+j])*meanDxhatXnorm) * float64(invStd))
				}
			}
			return []*g.Tensor{dx, dw, db}
		})
	}
	return out
}

func (bn *BatchNorm1d) Parameters() []*g.Tensor {
	return []*g.Tensor{bn.Weight, bn.Bias}
}

// Train sets training mode.
func (bn *BatchNorm1d) Train() { bn.Training = true }

// Eval sets evaluation mode (use running stats).
func (bn *BatchNorm1d) Eval() { bn.Training = false }
