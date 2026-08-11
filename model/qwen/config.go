//go:build darwin

// Package qwen implements native inference for Qwen3-family causal
// language models (plan doc/plans/0008-qwen-voice-lora.md, M0).
//
// The block is config-driven: every architectural difference between
// Qwen3-0.6B and Qwen2.5-0.5B-Instruct (explicit head_dim, QK-norm,
// attention biases, geometry) is a Config field, so the fallback swap
// is a config change plus new golden fixtures — not a rewrite.
package qwen

// Config carries every architecture fact from plan 0008 §0.2.
type Config struct {
	HiddenSize       int     // hidden_size
	NumLayers        int     // num_hidden_layers
	NumQueryHeads    int     // num_attention_heads
	NumKVHeads       int     // num_key_value_heads (GQA)
	HeadDim          int     // explicit per-head dim; Qwen3 has 128 ≠ hidden/heads
	IntermediateSize int     // SwiGLU MLP intermediate dim
	VocabSize        int     // 151,936 for the whole Qwen2.5/3 family
	MaxSeq           int     // max positions (RoPE table size)
	RopeTheta        float32 // rope base; 1e6 for Qwen2.5/3
	RMSNormEps       float32 // 1e-6
	AttnBias         bool    // q/k/v biases (Qwen2.5 true, Qwen3 false)
	UseQKNorm        bool    // per-head q_norm/k_norm RMSNorm (Qwen3 only)
	TiedEmbeddings   bool    // lm_head aliased to embed_tokens
	EOSTokens        []int   // generation stop ids
}

// InnerDim is the attention inner dimension numQ·headDim (2048 for
// Qwen3-0.6B — larger than the 1024 hidden dim).
func (c Config) InnerDim() int { return c.NumQueryHeads * c.HeadDim }

// KVDim is the K/V projection width numKV·headDim — the per-layer
// KV-cache row size.
func (c Config) KVDim() int { return c.NumKVHeads * c.HeadDim }

// Qwen3_0_6B returns the verified Qwen3-0.6B configuration
// (plan 0008 §0.2, cross-checked against the HF config.json).
func Qwen3_0_6B() Config {
	return Config{
		HiddenSize:       1024,
		NumLayers:        28,
		NumQueryHeads:    16,
		NumKVHeads:       8,
		HeadDim:          128,
		IntermediateSize: 3072,
		VocabSize:        151936,
		MaxSeq:           40960,
		RopeTheta:        1_000_000,
		RMSNormEps:       1e-6,
		AttnBias:         false,
		UseQKNorm:        true,
		TiedEmbeddings:   true,
		EOSTokens:        []int{151645, 151643}, // <|im_end|>, <|endoftext|>
	}
}

// Qwen25_0_5B_Instruct returns the Qwen2.5-0.5B-Instruct fallback
// configuration (plan 0008 §1.1 fallback trigger). head_dim is the
// derived 896/14 = 64; q/k/v projections carry biases; no QK-norm.
func Qwen25_0_5B_Instruct() Config {
	return Config{
		HiddenSize:       896,
		NumLayers:        24,
		NumQueryHeads:    14,
		NumKVHeads:       2,
		HeadDim:          64,
		IntermediateSize: 4864,
		VocabSize:        151936,
		MaxSeq:           32768,
		RopeTheta:        1_000_000,
		RMSNormEps:       1e-6,
		AttnBias:         true,
		UseQKNorm:        false,
		TiedEmbeddings:   true,
		EOSTokens:        []int{151645, 151643},
	}
}
