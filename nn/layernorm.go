//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// LayerNorm normalizes the last dimension of a 2-D tensor.
// Input: (M, dim), output: (M, dim).
// Computes: y = gamma * (x - mean) / sqrt(var + eps) + beta
type LayerNorm struct {
	Weight *g.Tensor // gamma (dim,)
	Bias   *g.Tensor // beta (dim,)
	Dim    int
	Eps    float32
}

// NewLayerNorm creates a LayerNorm layer.
func NewLayerNorm(dim int) *LayerNorm {
	w := g.Ones(dim)
	w.SetRequiresGrad(true)
	b := g.Zeros(dim)
	b.SetRequiresGrad(true)
	return &LayerNorm{Weight: w, Bias: b, Dim: dim, Eps: 1e-5}
}

// Forward normalizes the last dimension.
func (ln *LayerNorm) Forward(x *g.Tensor) *g.Tensor {
	if x.Dim() != 2 {
		panic("gorch: LayerNorm requires 2-D tensor")
	}
	M := x.Shape()[0]
	N := x.Shape()[1]
	if N != ln.Dim {
		panic("gorch: LayerNorm dim mismatch")
	}

	xData := x.Data()
	wData := ln.Weight.Data()
	bData := ln.Bias.Data()
	outData := make([]float32, M*N)

	// Store normalized values for backward
	xNorm := make([]float32, M*N)
	invStd := make([]float32, M)

	for i := 0; i < M; i++ {
		row := xData[i*N : (i+1)*N]

		// Mean
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(N)

		// Variance
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(N)

		inv := float32(1.0 / math.Sqrt(variance+float64(ln.Eps)))
		invStd[i] = inv

		for j := 0; j < N; j++ {
			normalized := (row[j] - float32(mean)) * inv
			xNorm[i*N+j] = normalized
			outData[i*N+j] = normalized*wData[j] + bData[j]
		}
	}

	out := g.NewTensor(outData, M, N)

	if x.RequiresGrad() || ln.Weight.RequiresGrad() || ln.Bias.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("LayerNorm", []*g.Tensor{x, ln.Weight, ln.Bias}, func(grad *g.Tensor) []*g.Tensor {
			gData := grad.Data()
			dx := g.Zeros(M, N)
			dw := g.Zeros(N)
			db := g.Zeros(N)
			dxData := dx.Data()
			dwData := dw.Data()
			dbData := db.Data()

			for i := 0; i < M; i++ {
				// dBias
				for j := 0; j < N; j++ {
					dbData[j] += gData[i*N+j]
				}
				// dWeight
				for j := 0; j < N; j++ {
					dwData[j] += gData[i*N+j] * xNorm[i*N+j]
				}
				// dx: gradient of layer norm
				// dxhat = grad * weight
				// dx = invStd * (dxhat - mean(dxhat) - xnorm * mean(dxhat * xnorm))
				var sumDxhat, sumDxhatXnorm float64
				for j := 0; j < N; j++ {
					dxhat := float64(gData[i*N+j] * wData[j])
					sumDxhat += dxhat
					sumDxhatXnorm += dxhat * float64(xNorm[i*N+j])
				}
				meanDxhat := sumDxhat / float64(N)
				meanDxhatXnorm := sumDxhatXnorm / float64(N)

				for j := 0; j < N; j++ {
					dxhat := float64(gData[i*N+j] * wData[j])
					dxData[i*N+j] = float32((dxhat - meanDxhat - float64(xNorm[i*N+j])*meanDxhatXnorm) * float64(invStd[i]))
				}
			}
			return []*g.Tensor{dx, dw, db}
		})
	}
	return out
}

func (ln *LayerNorm) Parameters() []*g.Tensor {
	return []*g.Tensor{ln.Weight, ln.Bias}
}
