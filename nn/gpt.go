//go:build darwin

package nn

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/metal"
)

// GPT is a decoder-only transformer language model.
// Architecture: token embedding + positional embedding → N transformer blocks → layer norm → linear head.
type GPT struct {
	TokenEmbed *Embedding
	PosEmbed   *Embedding
	Blocks     []*TransformerBlock
	FinalNorm  *LayerNorm
	LMHead     *Linear // hidden → vocab (output projection)
	Dim        int
	NumLayers  int
	NumHeads   int
	MaxSeq     int
	VocabSize  int
}

// NewGPT creates a GPT model with the given hyperparameters.
func NewGPT(vocabSize, dim, numHeads, numLayers, maxSeq int) *GPT {
	blocks := make([]*TransformerBlock, numLayers)
	for i := range blocks {
		blocks[i] = NewTransformerBlock(dim, numHeads)
	}

	return &GPT{
		TokenEmbed: NewEmbedding(vocabSize, dim),
		PosEmbed:   NewEmbedding(maxSeq, dim),
		Blocks:     blocks,
		FinalNorm:  NewLayerNorm(dim),
		LMHead:     NewLinear(dim, vocabSize),
		Dim:        dim,
		NumLayers:  numLayers,
		NumHeads:   numHeads,
		MaxSeq:     maxSeq,
		VocabSize:  vocabSize,
	}
}

// Forward runs the GPT model.
// tokenIDs: flat list of token IDs for a single sequence.
// Returns: (seqLen, vocabSize) logits.
func (gpt *GPT) Forward(tokenIDs []int) *g.Tensor {
	seqLen := len(tokenIDs)
	if seqLen > gpt.MaxSeq {
		panic("gorch: sequence length exceeds MaxSeq")
	}

	// Token embeddings: (seq, dim)
	tokEmb := gpt.TokenEmbed.Forward(tokenIDs)

	// Position embeddings: (seq, dim)
	posIDs := make([]int, seqLen)
	for i := range posIDs {
		posIDs[i] = i
	}
	posEmb := gpt.PosEmbed.Forward(posIDs)

	// Combine: x = tok_emb + pos_emb
	x := g.Add(tokEmb, posEmb) // (seq, dim)

	// Transformer blocks
	for _, block := range gpt.Blocks {
		x = block.Forward(x, seqLen)
	}

	// Final layer norm
	x = gpt.FinalNorm.Forward(x)

	// Language model head: (seq, dim) → (seq, vocab)
	logits := gpt.LMHead.Forward(x)

	return logits
}

// Parameters returns all learnable parameters.
func (gpt *GPT) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, gpt.TokenEmbed.Parameters()...)
	params = append(params, gpt.PosEmbed.Parameters()...)
	for _, block := range gpt.Blocks {
		params = append(params, block.Parameters()...)
	}
	params = append(params, gpt.FinalNorm.Parameters()...)
	params = append(params, gpt.LMHead.Parameters()...)
	return params
}

// CountParameters returns the total number of learnable parameters.
func (gpt *GPT) CountParameters() int {
	total := 0
	for _, p := range gpt.Parameters() {
		total += p.Size()
	}
	return total
}

// ToMetal moves all model weights to Metal GPU.
func (gpt *GPT) ToMetal(dev *metal.Device) {
	gpt.TokenEmbed.Weight.ToMetal(dev)
	gpt.PosEmbed.Weight.ToMetal(dev)

	for _, block := range gpt.Blocks {
		block.Attn.Wq.ToMetal(dev)
		block.Attn.Wk.ToMetal(dev)
		block.Attn.Wv.ToMetal(dev)
		block.Attn.Wo.ToMetal(dev)
		block.FFN1.ToMetal(dev)
		block.FFN2.ToMetal(dev)
		// LayerNorm weights stay on CPU — they're small and
		// operate element-wise on unified memory anyway
	}

	gpt.LMHead.ToMetal(dev)
}

// ToCPU moves all model weights back to CPU.
func (gpt *GPT) ToCPU() {
	gpt.TokenEmbed.Weight.ToCPU()
	gpt.PosEmbed.Weight.ToCPU()

	for _, block := range gpt.Blocks {
		block.Attn.Wq.ToCPU()
		block.Attn.Wk.ToCPU()
		block.Attn.Wv.ToCPU()
		block.Attn.Wo.ToCPU()
		block.FFN1.ToCPU()
		block.FFN2.ToCPU()
	}

	gpt.LMHead.ToCPU()
}
