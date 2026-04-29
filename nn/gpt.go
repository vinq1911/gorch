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
	// TiedLMHead reports whether LMHead.Weight aliases TokenEmbed.Weight.
	// When true, Parameters() returns the shared tensor only once and
	// gradient updates from both the embedding lookup and the output
	// projection accumulate into the same buffer (HF GPT-2 behaviour).
	TiedLMHead bool
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

// Encode runs the GPT model up to (but not including) the language-model
// head and returns the (seqLen, dim) hidden states after the final layer
// norm. Useful for embeddings, retrieval, classification heads, and any
// downstream task that does not need next-token logits.
func (gpt *GPT) Encode(tokenIDs []int) *g.Tensor {
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
	return gpt.FinalNorm.Forward(x)
}

// Forward runs the GPT model.
// tokenIDs: flat list of token IDs for a single sequence.
// Returns: (seqLen, vocabSize) logits.
func (gpt *GPT) Forward(tokenIDs []int) *g.Tensor {
	hidden := gpt.Encode(tokenIDs)
	// Language model head: (seq, dim) → (seq, vocab)
	return gpt.LMHead.Forward(hidden)
}

// ForwardCached runs the GPT model on `newTokenIDs` against a KV
// cache. The first call should pass the full prompt to populate the
// cache (a prefill); subsequent calls pass one new token each. The
// position embedding offset is taken from cache.Len() before this
// call, so callers do not need to track it.
//
// Returns logits of shape (len(newTokenIDs), vocabSize). On
// incremental calls (1 new token) only the new row is materialised.
func (gpt *GPT) ForwardCached(newTokenIDs []int, cache *KVCache) *g.Tensor {
	posOffset := cache.Len()
	newSeq := len(newTokenIDs)
	if posOffset+newSeq > gpt.MaxSeq {
		panic("gorch: cached sequence length exceeds MaxSeq")
	}

	tokEmb := gpt.TokenEmbed.Forward(newTokenIDs)

	posIDs := make([]int, newSeq)
	for i := range posIDs {
		posIDs[i] = posOffset + i
	}
	posEmb := gpt.PosEmbed.Forward(posIDs)

	x := g.Add(tokEmb, posEmb)
	for layerIdx, block := range gpt.Blocks {
		x = block.ForwardCached(x, cache, layerIdx, posOffset)
	}
	x = gpt.FinalNorm.Forward(x)
	return gpt.LMHead.Forward(x)
}

// TieLMHeadToEmbedding aliases the language-model head's weight to
// the token-embedding weight. Both modules then share the same
// underlying buffer — what HuggingFace GPT-2 does as
// `lm_head.weight = wte.weight`. Gradient updates to the LM head
// from the cross-entropy loss and updates to the embedding from
// the lookup path both accumulate into the shared tensor.
//
// Parameters() de-duplicates the alias so the optimizer sees one
// parameter slot, not two.
//
// Idempotent — calling it twice is a no-op.
func (gpt *GPT) TieLMHeadToEmbedding() {
	if gpt.TiedLMHead {
		return
	}
	if gpt.LMHead.Weight.Size() != gpt.TokenEmbed.Weight.Size() {
		panic("gorch/nn: cannot tie LMHead to embedding — sizes differ")
	}
	gpt.LMHead.Weight = gpt.TokenEmbed.Weight
	gpt.TiedLMHead = true
}

// Parameters returns all learnable parameters. When TiedLMHead is
// true, the shared embedding/LM-head weight appears once.
func (gpt *GPT) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, gpt.TokenEmbed.Parameters()...)
	params = append(params, gpt.PosEmbed.Parameters()...)
	for _, block := range gpt.Blocks {
		params = append(params, block.Parameters()...)
	}
	params = append(params, gpt.FinalNorm.Parameters()...)
	if gpt.TiedLMHead {
		// LMHead.Weight aliases TokenEmbed.Weight, already in the
		// list. Append only the bias.
		params = append(params, gpt.LMHead.Bias)
	} else {
		params = append(params, gpt.LMHead.Parameters()...)
	}
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
