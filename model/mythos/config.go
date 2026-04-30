//go:build darwin

// Package mythos implements the OpenMythos recurrent-depth transformer
// architecture in gorch. Plan 0001 Phase 2.
//
// Architecture (loose summary; see plan 0001 for the long version):
//
//   tokens → Embedding → Prelude (N standard blocks)
//          → Recurrent loop: for t in 1..MaxLoopIters:
//                h_{t+1} = lti(h_t, e) + Block(h_t, e)
//          → Coda (N standard blocks)
//          → final RMSNorm → LM head
//
// v1 = `mythos_tiny` (~5–10 M params). Bigger configs (mythos_1b,
// mythos_8b, mythos_1t) are out of scope until distributed training,
// activation checkpointing, and bf16 land — tracked in plan 0001
// Phase 5.
package mythos

// Config captures every shape parameter for an OpenMythos model.
//
// The defaults track the OpenMythos config dataclass field-by-field;
// drop-in TinyMythos / Mythos1B / Mythos8B presets below mirror the
// canonical sizes from the source repo.
type Config struct {
	// Architecture
	VocabSize int // tokenizer vocabulary
	Dim       int // hidden size

	// Attention
	NumHeads     int  // number of query heads
	NumKVHeads   int  // number of key/value heads (≤ NumHeads, must divide it)
	MaxSeqLen    int  // RoPE cache length and causal-mask seq cap
	UseMLA       bool // false → GQA; true → MLA. v1 ships GQA only.
	RopeBaseFreq float32

	// Recurrent depth
	PreludeLayers   int // standard blocks run once before the loop
	CodaLayers      int // standard blocks run once after the loop
	MaxLoopIters    int // recurrent block iterations (v1: fixed; ACT defers)
	LTIDampInit     float32

	// Mixture of Experts
	NumExperts         int
	NumExpertsPerToken int // top-K
	ExpertDim          int

	// RMSNorm
	NormEps float32

	// Training (default values; overridable per-run)
	Dropout float32
}

// TinyConfig is the v1 target: ~5–10 M parameters; trains end-to-end
// on TinyStories on a single Apple Silicon Mac in a day. Numbers
// match the table in plan 0001's "v1 scope decision" section.
//
// vocabSize is provided by the caller — TinyStories' BPE vocab is
// ~5k, GPT-2's is 50257. Both work; pass whichever the data loader
// returns.
func TinyConfig(vocabSize int) Config {
	return Config{
		VocabSize: vocabSize,
		Dim:       128,

		NumHeads:     4,
		NumKVHeads:   2,
		MaxSeqLen:    512,
		UseMLA:       false,
		RopeBaseFreq: 10000,

		PreludeLayers: 2,
		CodaLayers:    2,
		MaxLoopIters:  4,
		LTIDampInit:   0.5,

		NumExperts:         4,
		NumExpertsPerToken: 2,
		ExpertDim:          256,

		NormEps: 1e-6,
		Dropout: 0,
	}
}

// HeadDim returns the per-head dimensionality (Dim / NumHeads).
func (c Config) HeadDim() int { return c.Dim / c.NumHeads }
