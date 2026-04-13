//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
)

// Embedding is a lookup table that maps integer IDs to dense vectors.
type Embedding struct {
	Weight *g.Tensor // (vocabSize, embedDim)
	Dim    int
}

// NewEmbedding creates an Embedding with random normal initialization.
func NewEmbedding(vocabSize, embedDim int) *Embedding {
	w := g.RandN(vocabSize, embedDim)
	// Scale by 0.02 (GPT-2 style init)
	for i := range w.Data() {
		w.Data()[i] *= 0.02
	}
	w.SetRequiresGrad(true)
	return &Embedding{Weight: w, Dim: embedDim}
}

// Forward looks up embeddings for the given token IDs.
// Returns (len(ids), embedDim).
func (e *Embedding) Forward(ids []int) *g.Tensor {
	return g.EmbeddingLookup(e.Weight, ids)
}

func (e *Embedding) Parameters() []*g.Tensor {
	return []*g.Tensor{e.Weight}
}
