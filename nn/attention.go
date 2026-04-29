//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// MultiHeadAttention implements multi-head self-attention.
// Uses batched matmul to process all heads in one GPU dispatch.
//
// The Causal field selects between causal (decoder, GPT-style) and
// bidirectional (encoder, BERT-style) self-attention. It defaults to
// true via NewMultiHeadAttention so that pre-existing GPT-style code
// keeps the same behaviour. Set Causal=false on the constructed module
// before the first Forward to make it bidirectional.
type MultiHeadAttention struct {
	Wq       *Linear
	Wk       *Linear
	Wv       *Linear
	Wo       *Linear
	NumHeads int
	HeadDim  int
	Dim      int
	Causal   bool
}

// NewMultiHeadAttention creates a multi-head attention module with the
// causal-mask default (decoder/GPT-style). Use NewMultiHeadAttentionBi
// for an encoder-style bidirectional layer.
func NewMultiHeadAttention(dim, numHeads int) *MultiHeadAttention {
	if dim%numHeads != 0 {
		panic("gorch: dim must be divisible by numHeads")
	}
	headDim := dim / numHeads
	return &MultiHeadAttention{
		Wq:       NewLinear(dim, dim),
		Wk:       NewLinear(dim, dim),
		Wv:       NewLinear(dim, dim),
		Wo:       NewLinear(dim, dim),
		NumHeads: numHeads,
		HeadDim:  headDim,
		Dim:      dim,
		Causal:   true,
	}
}

// NewMultiHeadAttentionBi creates a multi-head attention module
// configured for bidirectional (BERT-style) self-attention. No causal
// mask is applied; every position can attend to every other position.
func NewMultiHeadAttentionBi(dim, numHeads int) *MultiHeadAttention {
	mha := NewMultiHeadAttention(dim, numHeads)
	mha.Causal = false
	return mha
}

// Forward computes multi-head self-attention.
// x: (seq, dim). Returns: (seq, dim).
//
// Uses batched matmul: all heads computed in 2 GPU dispatches instead of 24 CPU matmuls.
// Data layout for batched ops: (numHeads, seq, headDim) packed contiguously.
func (mha *MultiHeadAttention) Forward(x *g.Tensor, seqLen int) *g.Tensor {
	numHeads := mha.NumHeads
	headDim := mha.HeadDim
	dim := mha.Dim

	// Project: (seq, dim) → (seq, dim) for Q, K, V
	q := mha.Wq.Forward(x) // (seq, dim)
	k := mha.Wk.Forward(x)
	v := mha.Wv.Forward(x)

	// Use batched matmul for inference (no grad required on inputs)
	// Fall back to per-head loop for training (has autograd)
	useBatched := !x.RequiresGrad()

	var concat *g.Tensor

	if useBatched {
		// === Batched path: 2 GPU dispatches instead of 24 CPU matmuls ===
		qHeads := reshapeToHeads(q, seqLen, numHeads, headDim)
		kHeads := reshapeToHeads(k, seqLen, numHeads, headDim)
		vHeads := reshapeToHeads(v, seqLen, numHeads, headDim)

		// Batched Q @ K^T → (numHeads, seq, seq)
		scores := g.BatchedMatMulTransB(qHeads, kHeads, numHeads, seqLen, seqLen, headDim)

		// Scale, mask, softmax (in-place on unified memory)
		invScale := float32(1.0 / float64(headDim))
		scoresData := scores.Data()
		for i := range scoresData {
			scoresData[i] *= invScale
		}
		var mask []bool
		if mha.Causal {
			mask = g.CausalMask(seqLen)
		}
		for h := 0; h < numHeads; h++ {
			offset := h * seqLen * seqLen
			if mask != nil {
				for i, m := range mask {
					if m {
						scoresData[offset+i] = -1e9
					}
				}
			}
			softmaxInPlace(scoresData[offset:offset+seqLen*seqLen], seqLen)
		}

		// Batched attn @ V → (numHeads, seq, headDim)
		attnOut := g.BatchedMatMul(scores, vHeads, numHeads, seqLen, headDim, seqLen)
		concat = reshapeFromHeads(attnOut, seqLen, numHeads, headDim)

	} else {
		// === Per-head loop: supports autograd for training ===
		headOutputs := make([]*g.Tensor, numHeads)
		for h := 0; h < numHeads; h++ {
			qh := extractHead(q, seqLen, dim, h, headDim)
			kh := extractHead(k, seqLen, dim, h, headDim)
			vh := extractHead(v, seqLen, dim, h, headDim)

			kT := g.Transpose2D(kh)
			scores := g.ScaledMatMul(qh, kT, float32(headDim))
			if mha.Causal {
				mask := g.CausalMask(seqLen)
				scores = g.MaskFill(scores, mask, -1e9)
			}
			attnWeights := g.Softmax(scores)
			headOutputs[h] = g.MatMul(attnWeights, vh)
		}
		concat = concatHeadsLoop(headOutputs, seqLen, numHeads, headDim)
	}

	return mha.Wo.Forward(concat)
}

// extractHead extracts one attention head's columns from a (seq, dim) tensor.
func extractHead(x *g.Tensor, seq, dim, headIdx, headDim int) *g.Tensor {
	out := g.Zeros(seq, headDim)
	xData := x.Data()
	outData := out.Data()
	offset := headIdx * headDim
	for i := 0; i < seq; i++ {
		copy(outData[i*headDim:(i+1)*headDim], xData[i*dim+offset:i*dim+offset+headDim])
	}
	if x.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("ExtractHead", []*g.Tensor{x}, func(grad *g.Tensor) []*g.Tensor {
			dx := g.Zeros(seq, dim)
			dxData := dx.Data()
			gData := grad.Data()
			for i := 0; i < seq; i++ {
				for j := 0; j < headDim; j++ {
					dxData[i*dim+offset+j] += gData[i*headDim+j]
				}
			}
			return []*g.Tensor{dx}
		})
	}
	return out
}

// concatHeadsLoop concatenates head outputs back into (seq, dim) with autograd.
func concatHeadsLoop(heads []*g.Tensor, seq, numHeads, headDim int) *g.Tensor {
	dim := numHeads * headDim
	out := g.Zeros(seq, dim)
	outData := out.Data()
	for h, head := range heads {
		hData := head.Data()
		offset := h * headDim
		for i := 0; i < seq; i++ {
			copy(outData[i*dim+offset:i*dim+offset+headDim], hData[i*headDim:(i+1)*headDim])
		}
	}
	anyGrad := false
	for _, h := range heads {
		if h.RequiresGrad() { anyGrad = true; break }
	}
	if anyGrad {
		out.SetRequiresGrad(true)
		inputs := make([]*g.Tensor, len(heads))
		copy(inputs, heads)
		out.SetGradFn("ConcatHeads", inputs, func(grad *g.Tensor) []*g.Tensor {
			gData := grad.Data()
			grads := make([]*g.Tensor, numHeads)
			for h := 0; h < numHeads; h++ {
				dh := g.Zeros(seq, headDim)
				dhData := dh.Data()
				offset := h * headDim
				for i := 0; i < seq; i++ {
					copy(dhData[i*headDim:(i+1)*headDim], gData[i*dim+offset:i*dim+offset+headDim])
				}
				grads[h] = dh
			}
			return grads
		})
	}
	return out
}

// reshapeToHeads rearranges (seq, dim) → packed (numHeads, seq, headDim).
// Input layout: row i has [head0_0..head0_hd, head1_0..head1_hd, ...]
// Output layout: [head0_row0, head0_row1, ..., head1_row0, head1_row1, ...]
func reshapeToHeads(x *g.Tensor, seq, numHeads, headDim int) *g.Tensor {
	dim := numHeads * headDim
	xData := x.Data()
	outData := make([]float32, numHeads*seq*headDim)

	for h := 0; h < numHeads; h++ {
		for s := 0; s < seq; s++ {
			srcOff := s*dim + h*headDim
			dstOff := h*seq*headDim + s*headDim
			copy(outData[dstOff:dstOff+headDim], xData[srcOff:srcOff+headDim])
		}
	}

	out := g.NewTensor(outData, numHeads*seq, headDim)

	// If input is on Metal, move output to Metal for batched MPS
	if x.IsOnMetal() {
		if dev := g.MetalDev(); dev != nil {
			out.ToMetal(dev)
		}
	}

	if x.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("ReshapeToHeads", []*g.Tensor{x}, func(grad *g.Tensor) []*g.Tensor {
			return []*g.Tensor{reshapeFromHeads(grad, seq, numHeads, headDim)}
		})
	}
	return out
}

// reshapeFromHeads rearranges packed (numHeads, seq, headDim) → (seq, dim).
func reshapeFromHeads(x *g.Tensor, seq, numHeads, headDim int) *g.Tensor {
	dim := numHeads * headDim
	xData := x.Data()
	outData := make([]float32, seq*dim)

	for h := 0; h < numHeads; h++ {
		for s := 0; s < seq; s++ {
			srcOff := h*seq*headDim + s*headDim
			dstOff := s*dim + h*headDim
			copy(outData[dstOff:dstOff+headDim], xData[srcOff:srcOff+headDim])
		}
	}

	out := g.NewTensor(outData, seq, dim)

	if x.RequiresGrad() {
		out.SetRequiresGrad(true)
		out.SetGradFn("ReshapeFromHeads", []*g.Tensor{x}, func(grad *g.Tensor) []*g.Tensor {
			return []*g.Tensor{reshapeToHeads(grad, seq, numHeads, headDim)}
		})
	}
	return out
}

// softmaxInPlace applies softmax to a (rows, cols) block in-place.
func softmaxInPlace(data []float32, cols int) {
	rows := len(data) / cols
	for i := 0; i < rows; i++ {
		row := data[i*cols : (i+1)*cols]

		// Numerical stability: subtract max
		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}

		var sum float32
		for j := range row {
			row[j] = float32(exp64(float64(row[j] - maxVal)))
			sum += row[j]
		}
		for j := range row {
			row[j] /= sum
		}
	}
}

func exp64(x float64) float64 {
	return math.Exp(x)
}

// ForwardCached computes self-attention for `x` against a KV cache.
// x is the new tokens (1 row for incremental decoding, len(prompt) on
// prefill). The cache slot at layerIdx is updated with the new K/V.
// posOffset is the absolute position of the first new token before
// this call (i.e., cache.Len() prior to appending).
//
// Causal masking uses absolute positions: a query at absolute position
// p may attend to keys at absolute positions 0..p inclusive. For
// incremental decoding (1 new token), no rows of the mask are active.
//
// This path is inference-only — it does not record an autograd tape.
func (mha *MultiHeadAttention) ForwardCached(x *g.Tensor, cache *KVCache, layerIdx, posOffset int) *g.Tensor {
	numHeads := mha.NumHeads
	headDim := mha.HeadDim
	dim := mha.Dim
	newSeq := x.Shape()[0]

	// Project new Q/K/V — same as the standard forward.
	q := mha.Wq.Forward(x)
	k := mha.Wk.Forward(x)
	v := mha.Wv.Forward(x)

	// Move off Metal — cached compute happens on CPU. On Apple Silicon
	// this is a no-op on the data (unified memory), it just clears the
	// buf pointer so we get a normal CPU slice for the loops below.
	if k.IsOnMetal() {
		k.ToCPU()
	}
	if v.IsOnMetal() {
		v.ToCPU()
	}
	if q.IsOnMetal() {
		q.ToCPU()
	}

	// Append the new K/V to the cache. After this, the cache holds
	// totalSeq = posOffset + newSeq tokens.
	cache.Append(layerIdx, k.Data(), v.Data())
	totalSeq := cache.Len()
	cachedK := cache.Keys[layerIdx]   // flat (totalSeq, dim)
	cachedV := cache.Values[layerIdx] // flat (totalSeq, dim)

	qData := q.Data()
	out := g.Zeros(newSeq, dim)
	outData := out.Data()

	invScale := float32(1.0 / sqrtFloat32(float32(headDim)))

	// Per-head loop — no batched matmul because totalSeq ≠ newSeq.
	// For incremental decoding newSeq=1 so this is one matmul per
	// head per token, dominated by the (1, headDim) @ (headDim, totalSeq)
	// score and (1, totalSeq) @ (totalSeq, headDim) attended sum.
	scores := make([]float32, newSeq*totalSeq)
	for h := 0; h < numHeads; h++ {
		headOff := h * headDim
		// scores[i, j] = q[i, h, :] · k[j, h, :] * invScale
		for i := 0; i < newSeq; i++ {
			qRow := qData[i*dim+headOff : i*dim+headOff+headDim]
			for j := 0; j < totalSeq; j++ {
				kRow := cachedK[j*dim+headOff : j*dim+headOff+headDim]
				var sum float32
				for d := 0; d < headDim; d++ {
					sum += qRow[d] * kRow[d]
				}
				scores[i*totalSeq+j] = sum * invScale
			}
		}

		// Causal mask in absolute coordinates and softmax per row.
		if mha.Causal {
			for i := 0; i < newSeq; i++ {
				absPos := posOffset + i
				row := scores[i*totalSeq : (i+1)*totalSeq]
				for j := absPos + 1; j < totalSeq; j++ {
					row[j] = -1e9
				}
			}
		}
		softmaxInPlace(scores, totalSeq)

		// out[i, h, :] = sum_j scores[i, j] * v[j, h, :]
		for i := 0; i < newSeq; i++ {
			outRow := outData[i*dim+headOff : i*dim+headOff+headDim]
			scoreRow := scores[i*totalSeq : (i+1)*totalSeq]
			for j := 0; j < totalSeq; j++ {
				if scoreRow[j] == 0 {
					continue
				}
				vRow := cachedV[j*dim+headOff : j*dim+headOff+headDim]
				w := scoreRow[j]
				for d := 0; d < headDim; d++ {
					outRow[d] += w * vRow[d]
				}
			}
		}
	}

	return mha.Wo.Forward(out)
}

func sqrtFloat32(x float32) float32 {
	// Avoid pulling math just for this — Newton-Raphson style.
	// Use math.Sqrt via the existing import in this file is cleaner.
	return float32(math.Sqrt(float64(x)))
}

func (mha *MultiHeadAttention) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, mha.Wq.Parameters()...)
	params = append(params, mha.Wk.Parameters()...)
	params = append(params, mha.Wv.Parameters()...)
	params = append(params, mha.Wo.Parameters()...)
	return params
}
