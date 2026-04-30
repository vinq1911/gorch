//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// RMSNorm normalises the last dimension of a 2-D tensor by its
// root-mean-square value, then scales by a learnable gamma. Unlike
// LayerNorm there's no mean subtraction and no bias — that's the whole
// design. Used by Llama, Mistral, DeepSeek, OpenMythos and every modern
// decoder transformer.
//
//	y_i = (x_i / sqrt(mean(x²) + eps)) * gamma_i
type RMSNorm struct {
	Weight *g.Tensor // gamma (dim,)
	Dim    int
	Eps    float32
}

// NewRMSNorm creates an RMSNorm layer with gamma initialised to ones.
func NewRMSNorm(dim int) *RMSNorm {
	w := g.Ones(dim)
	w.SetRequiresGrad(true)
	return &RMSNorm{Weight: w, Dim: dim, Eps: 1e-6}
}

// Forward normalises the last dimension and scales by the learnable gamma.
// Input shape: (M, dim). Output: (M, dim).
func (rn *RMSNorm) Forward(x *g.Tensor) *g.Tensor {
	if x.Dim() != 2 {
		panic("gorch/nn: RMSNorm requires 2-D tensor")
	}
	M := x.Shape()[0]
	N := x.Shape()[1]
	if N != rn.Dim {
		panic("gorch/nn: RMSNorm dim mismatch")
	}

	xData := x.Data()
	wData := rn.Weight.Data()

	// In NoGrad mode, skip allocating the per-row inverse-RMS cache
	// since the backward closure isn't built.
	needsBackward := g.GradEnabled() && (x.RequiresGrad() || rn.Weight.RequiresGrad())

	out := g.Zeros(M, N)
	outData := out.Data()

	var invRMS []float32 // per-row 1/sqrt(mean(x²) + eps), kept for backward
	if needsBackward {
		invRMS = make([]float32, M)
	}

	for i := 0; i < M; i++ {
		row := xData[i*N : (i+1)*N]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(N)+float64(rn.Eps)))
		if needsBackward {
			invRMS[i] = inv
		}
		for j := 0; j < N; j++ {
			outData[i*N+j] = row[j] * inv * wData[j]
		}
	}

	if needsBackward {
		out.SetRequiresGrad(true)
		out.SetGradFn("RMSNorm", []*g.Tensor{x, rn.Weight}, func(grad *g.Tensor) []*g.Tensor {
			gData := grad.Data()
			dx := g.Zeros(M, N)
			dw := g.Zeros(N)
			dxData := dx.Data()
			dwData := dw.Data()

			invN := 1.0 / float64(N)
			for i := 0; i < M; i++ {
				row := xData[i*N : (i+1)*N]
				gRow := gData[i*N : (i+1)*N]
				inv := invRMS[i]

				// dW[j] += grad[i,j] * (x[i,j] * inv)
				// dx[i,j] = inv * (gamma[j]*grad[i,j] - normalised[i,j] * mean(gamma * grad * normalised))
				// where normalised[i,j] = x[i,j] * inv.
				//
				// Compact form derived from y = x / rms * gamma:
				//   sumDot = (1/N) * sum_k (gamma[k] * grad[k] * normalised[k])
				//   dx[j] = inv * (gamma[j]*grad[j] - normalised[j] * sumDot)
				var sumDot float64
				for j := 0; j < N; j++ {
					nj := float64(row[j]) * float64(inv)
					sumDot += float64(wData[j]) * float64(gRow[j]) * nj
					dwData[j] += gRow[j] * row[j] * inv
				}
				sumDot *= invN

				for j := 0; j < N; j++ {
					nj := float64(row[j]) * float64(inv)
					dxData[i*N+j] = float32(float64(inv) * (float64(wData[j])*float64(gRow[j]) - nj*sumDot))
				}
			}
			return []*g.Tensor{dx, dw}
		})
	}
	return out
}

func (rn *RMSNorm) Parameters() []*g.Tensor {
	return []*g.Tensor{rn.Weight}
}
