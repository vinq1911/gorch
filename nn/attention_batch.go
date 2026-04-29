//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
)

// ForwardBatched runs self-attention on a batch of sequences sharing
// a common max length. x is (B*S, D) flat; the caller supplies the
// batch size, sequence length, and per-sequence true lengths so that
// pad positions are masked out of the attention softmax.
//
// Returns (B*S, D). Inference-only — no autograd. Issue #9.
func (mha *MultiHeadAttention) ForwardBatched(x *g.Tensor, batchSize, seqLen int, lengths []int) *g.Tensor {
	if len(lengths) != batchSize {
		panic("gorch/nn: lengths must have batchSize entries")
	}
	if x.Shape()[0] != batchSize*seqLen {
		panic("gorch/nn: x first dim must equal batchSize*seqLen")
	}
	numHeads := mha.NumHeads
	headDim := mha.HeadDim

	// Project: (B*S, D) → (B*S, D)
	q := mha.Wq.Forward(x)
	k := mha.Wk.Forward(x)
	v := mha.Wv.Forward(x)

	// Reshape to (B*numHeads, S, headDim) packed: outer index is
	// b*numHeads+h, then seqLen rows of headDim each.
	qHeads := reshapeToHeadsBatched(q, batchSize, seqLen, numHeads, headDim)
	kHeads := reshapeToHeadsBatched(k, batchSize, seqLen, numHeads, headDim)
	vHeads := reshapeToHeadsBatched(v, batchSize, seqLen, numHeads, headDim)

	// Batched Q @ K^T → (B*numHeads, S, S)
	totalHeads := batchSize * numHeads
	scores := g.BatchedMatMulTransB(qHeads, kHeads, totalHeads, seqLen, seqLen, headDim)

	// Scale
	invScale := float32(1.0 / sqrtFloat32(float32(headDim)))
	scoresData := scores.Data()
	for i := range scoresData {
		scoresData[i] *= invScale
	}

	// Apply combined causal + length mask per batch.
	for b := 0; b < batchSize; b++ {
		l := lengths[b]
		if l > seqLen {
			l = seqLen
		}
		for h := 0; h < numHeads; h++ {
			block := scoresData[(b*numHeads+h)*seqLen*seqLen : (b*numHeads+h+1)*seqLen*seqLen]
			for i := 0; i < seqLen; i++ {
				row := block[i*seqLen : (i+1)*seqLen]
				// Mask keys beyond the true sequence length.
				for j := l; j < seqLen; j++ {
					row[j] = -1e9
				}
				// Causal: query at i can't attend to keys j>i.
				if mha.Causal {
					for j := i + 1; j < seqLen; j++ {
						row[j] = -1e9
					}
				}
			}
			// Softmax per row, in place.
			softmaxInPlace(block, seqLen)
		}
	}

	// Batched attn @ V → (B*numHeads, S, headDim)
	attnOut := g.BatchedMatMul(scores, vHeads, totalHeads, seqLen, headDim, seqLen)

	// Reshape back to (B*S, D).
	concat := reshapeFromHeadsBatched(attnOut, batchSize, seqLen, numHeads, headDim)

	return mha.Wo.Forward(concat)
}

// reshapeToHeadsBatched rearranges (B*S, D) → packed (B*numHeads, S, headDim).
// Outer index is b*numHeads + h; inside is seqLen rows of headDim.
func reshapeToHeadsBatched(x *g.Tensor, batch, seq, numHeads, headDim int) *g.Tensor {
	dim := numHeads * headDim
	xData := x.Data()
	outData := make([]float32, batch*numHeads*seq*headDim)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seq; s++ {
				srcOff := (b*seq+s)*dim + h*headDim
				dstOff := ((b*numHeads+h)*seq + s) * headDim
				copy(outData[dstOff:dstOff+headDim], xData[srcOff:srcOff+headDim])
			}
		}
	}
	out := g.NewTensor(outData, batch*numHeads*seq, headDim)
	if x.IsOnMetal() {
		if dev := g.MetalDev(); dev != nil {
			out.ToMetal(dev)
		}
	}
	return out
}

// reshapeFromHeadsBatched undoes reshapeToHeadsBatched.
func reshapeFromHeadsBatched(x *g.Tensor, batch, seq, numHeads, headDim int) *g.Tensor {
	dim := numHeads * headDim
	xData := x.Data()
	outData := make([]float32, batch*seq*dim)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seq; s++ {
				srcOff := ((b*numHeads+h)*seq + s) * headDim
				dstOff := (b*seq+s)*dim + h*headDim
				copy(outData[dstOff:dstOff+headDim], xData[srcOff:srcOff+headDim])
			}
		}
	}
	return g.NewTensor(outData, batch*seq, dim)
}

