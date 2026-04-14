//go:build darwin

package model

import (
	"fmt"
	"os"
	"path/filepath"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// GPT2Config holds GPT-2 architecture parameters.
type GPT2Config struct {
	VocabSize int
	Dim       int
	NumHeads  int
	NumLayers int
	MaxSeq    int
}

// GPT2Small returns the config for openai-community/gpt2 (124M params).
func GPT2Small() GPT2Config {
	return GPT2Config{VocabSize: 50257, Dim: 768, NumHeads: 12, NumLayers: 12, MaxSeq: 1024}
}

// TinyStories1M returns the config for roneneldan/TinyStories-1M.
func TinyStories1M() GPT2Config {
	return GPT2Config{VocabSize: 50257, Dim: 64, NumHeads: 16, NumLayers: 8, MaxSeq: 1024}
}

const hfBaseURL = "https://huggingface.co/"

// DownloadGPT2 downloads model files for a HuggingFace GPT-2 model.
func DownloadGPT2(modelName, dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	base := hfBaseURL + modelName + "/resolve/main/"
	files := []string{"model.safetensors", "vocab.json", "merges.txt"}

	for _, f := range files {
		path := filepath.Join(dir, f)
		if err := downloadIfMissing(path, base+f); err != nil {
			return fmt.Errorf("download %s: %w", f, err)
		}
	}
	return nil
}

// LoadGPT2 loads a pretrained GPT-2 model from safetensors.
// Handles the GPT-2 Conv1D convention (transposed weights) and fused QKV.
func LoadGPT2(dir string, cfg GPT2Config) (*nn.GPT, error) {
	sfPath := filepath.Join(dir, "model.safetensors")
	sf, err := LoadSafetensors(sfPath)
	if err != nil {
		return nil, fmt.Errorf("load safetensors: %w", err)
	}

	model := nn.NewGPT(cfg.VocabSize, cfg.Dim, cfg.NumHeads, cfg.NumLayers, cfg.MaxSeq)

	// Load token and position embeddings
	if err := copyTensor(sf, "wte.weight", model.TokenEmbed.Weight); err != nil {
		return nil, err
	}
	if err := copyTensor(sf, "wpe.weight", model.PosEmbed.Weight); err != nil {
		return nil, err
	}

	// Load transformer blocks
	for i := 0; i < cfg.NumLayers; i++ {
		block := model.Blocks[i]
		prefix := fmt.Sprintf("h.%d.", i)

		// LayerNorm 1 (pre-attention)
		if err := copyTensor(sf, prefix+"ln_1.weight", block.Norm1.Weight); err != nil {
			return nil, err
		}
		if err := copyTensor(sf, prefix+"ln_1.bias", block.Norm1.Bias); err != nil {
			return nil, err
		}

		// Attention: GPT-2 fuses Q,K,V into one c_attn projection
		// c_attn.weight is (dim, 3*dim) in Conv1D convention = (dim, 3*dim)
		// We need to split into Wq, Wk, Wv and transpose each
		if err := loadFusedQKV(sf, prefix+"attn.c_attn", block.Attn, cfg.Dim); err != nil {
			return nil, err
		}

		// Output projection
		if err := loadConv1DLinear(sf, prefix+"attn.c_proj", block.Attn.Wo); err != nil {
			return nil, err
		}

		// LayerNorm 2 (pre-FFN)
		if err := copyTensor(sf, prefix+"ln_2.weight", block.Norm2.Weight); err != nil {
			return nil, err
		}
		if err := copyTensor(sf, prefix+"ln_2.bias", block.Norm2.Bias); err != nil {
			return nil, err
		}

		// FFN
		if err := loadConv1DLinear(sf, prefix+"mlp.c_fc", block.FFN1); err != nil {
			return nil, err
		}
		if err := loadConv1DLinear(sf, prefix+"mlp.c_proj", block.FFN2); err != nil {
			return nil, err
		}
	}

	// Final LayerNorm
	if err := copyTensor(sf, "ln_f.weight", model.FinalNorm.Weight); err != nil {
		return nil, err
	}
	if err := copyTensor(sf, "ln_f.bias", model.FinalNorm.Bias); err != nil {
		return nil, err
	}

	// LM Head: GPT-2 ties weights with token embeddings (wte.weight)
	// Copy wte.weight into LMHead.Weight
	if wte, ok := sf.Tensors["wte.weight"]; ok {
		// LMHead.Weight shape is (vocab, dim), wte is (vocab, dim)
		// But our Linear expects (out, in) = (vocab, dim) — same!
		copy(model.LMHead.Weight.Data(), wte.Data())
	}

	fmt.Printf("Loaded GPT-2 model: %d layers, dim=%d, heads=%d, params=%d\n",
		cfg.NumLayers, cfg.Dim, cfg.NumHeads, model.CountParameters())

	return model, nil
}

// copyTensor copies data from safetensors into a gorch tensor.
func copyTensor(sf *SafetensorsFile, name string, dst *g.Tensor) error {
	src, ok := sf.Tensors[name]
	if !ok {
		return fmt.Errorf("tensor %q not found in safetensors", name)
	}
	if src.Size() != dst.Size() {
		return fmt.Errorf("size mismatch for %q: file=%d model=%d", name, src.Size(), dst.Size())
	}
	copy(dst.Data(), src.Data())
	return nil
}

// loadConv1DLinear loads a GPT-2 Conv1D layer into a gorch Linear.
// Conv1D stores weights as (in, out); gorch Linear expects (out, in).
// So we need to transpose.
func loadConv1DLinear(sf *SafetensorsFile, prefix string, linear *nn.Linear) error {
	wName := prefix + ".weight"
	bName := prefix + ".bias"

	wt, ok := sf.Tensors[wName]
	if !ok {
		return fmt.Errorf("tensor %q not found", wName)
	}

	// Conv1D: (in, out). Linear: (out, in). Transpose.
	wShape := wt.Shape()
	inDim, outDim := wShape[0], wShape[1]
	wData := wt.Data()
	dst := linear.Weight.Data()

	if inDim*outDim != linear.Weight.Size() {
		return fmt.Errorf("weight size mismatch for %q: %dx%d=%d, model=%d",
			wName, inDim, outDim, inDim*outDim, linear.Weight.Size())
	}

	// Transpose: dst[o*inDim + i] = wData[i*outDim + o]
	for i := 0; i < inDim; i++ {
		for o := 0; o < outDim; o++ {
			dst[o*inDim+i] = wData[i*outDim+o]
		}
	}

	// Bias
	bt, ok := sf.Tensors[bName]
	if !ok {
		return fmt.Errorf("tensor %q not found", bName)
	}
	copy(linear.Bias.Data(), bt.Data())

	return nil
}

// loadFusedQKV splits GPT-2's fused c_attn into separate Wq, Wk, Wv.
// c_attn.weight is (dim, 3*dim) in Conv1D convention.
// c_attn.bias is (3*dim,).
func loadFusedQKV(sf *SafetensorsFile, prefix string, attn *nn.MultiHeadAttention, dim int) error {
	wName := prefix + ".weight"
	bName := prefix + ".bias"

	wt, ok := sf.Tensors[wName]
	if !ok {
		return fmt.Errorf("tensor %q not found", wName)
	}
	bt, ok := sf.Tensors[bName]
	if !ok {
		return fmt.Errorf("tensor %q not found", bName)
	}

	wData := wt.Data() // shape: (dim, 3*dim) in Conv1D
	bData := bt.Data() // shape: (3*dim,)

	// Split into Q, K, V sections (each dim columns)
	// Conv1D: row i, cols 0..dim-1 = Q, cols dim..2*dim-1 = K, cols 2*dim..3*dim-1 = V
	// Then transpose each from (in, out) to (out, in)

	qW := attn.Wq.Weight.Data() // (dim, dim)
	kW := attn.Wk.Weight.Data()
	vW := attn.Wv.Weight.Data()

	for i := 0; i < dim; i++ {
		for o := 0; o < dim; o++ {
			// Conv1D row i, col o (Q), col dim+o (K), col 2*dim+o (V)
			// Transpose to (out, in): dst[o*dim + i]
			qW[o*dim+i] = wData[i*(3*dim)+o]
			kW[o*dim+i] = wData[i*(3*dim)+dim+o]
			vW[o*dim+i] = wData[i*(3*dim)+2*dim+o]
		}
	}

	// Bias: split into Q, K, V
	qB := attn.Wq.Bias.Data()
	kB := attn.Wk.Bias.Data()
	vB := attn.Wv.Bias.Data()
	copy(qB, bData[0:dim])
	copy(kB, bData[dim:2*dim])
	copy(vB, bData[2*dim:3*dim])

	return nil
}

// Generate produces text by autoregressively sampling from a GPT model.
// Uses greedy decoding (argmax).
func Generate(model *nn.GPT, tokenIDs []int, maxNewTokens int) []int {
	result := make([]int, len(tokenIDs))
	copy(result, tokenIDs)

	for i := 0; i < maxNewTokens; i++ {
		// Truncate to max sequence length
		input := result
		if len(input) > model.MaxSeq {
			input = input[len(input)-model.MaxSeq:]
		}

		// Forward pass
		logits := model.Forward(input) // (seq, vocab)

		// Get logits for last position
		lastLogits := logits.Data()[(len(input)-1)*model.VocabSize : len(input)*model.VocabSize]

		// Argmax
		maxIdx := 0
		maxVal := lastLogits[0]
		for j := 1; j < model.VocabSize; j++ {
			if lastLogits[j] > maxVal {
				maxVal = lastLogits[j]
				maxIdx = j
			}
		}

		result = append(result, maxIdx)
	}

	return result
}
