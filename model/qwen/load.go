//go:build darwin

package qwen

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/nn"
)

const checkpointURL = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors"

// minCheckpointBytes is a sanity floor for the Qwen3-0.6B checkpoint
// (actual size ≈ 1.19 GB bf16). A smaller file is a truncated download.
const minCheckpointBytes = 1000 << 20

// FindCheckpoint locates a local Qwen3-0.6B model.safetensors: the
// QWEN3_MODEL env var if set, else the HF hub cache
// (~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*).
func FindCheckpoint() (string, error) {
	if p := os.Getenv("QWEN3_MODEL"); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("QWEN3_MODEL=%s: %w", p, err)
		}
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	pattern := filepath.Join(home,
		".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}
	for _, m := range matches {
		if fi, err := os.Stat(m); err == nil && fi.Size() >= minCheckpointBytes {
			return m, nil
		}
	}
	return "", fmt.Errorf("no Qwen3-0.6B checkpoint found (set QWEN3_MODEL, populate the HF cache, or call Download): %s", pattern)
}

// Download fetches the Qwen3-0.6B checkpoint into dir and returns the
// local path (mimi.Download / downloadIfMissing precedent). Skips the
// download if a plausible file is already present.
func Download(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "model.safetensors")

	if fi, err := os.Stat(path); err == nil {
		if fi.Size() >= minCheckpointBytes {
			return path, nil
		}
		return "", fmt.Errorf("existing %s is %d bytes (< %d): truncated download? remove it and retry",
			path, fi.Size(), minCheckpointBytes)
	}

	fmt.Printf("Downloading %s ...\n", checkpointURL)
	resp, err := http.Get(checkpointURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("HTTP %d for %s", resp.StatusCode, checkpointURL)
	}

	tmp, err := os.CreateTemp(dir, "model.safetensors.download-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())

	n, err := io.Copy(tmp, resp.Body)
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return "", err
	}
	if n < minCheckpointBytes {
		return "", fmt.Errorf("downloaded %d bytes (< %d): truncated response", n, minCheckpointBytes)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return "", err
	}
	return path, nil
}

// Load reads an HF Qwen safetensors checkpoint (bf16 decoded to f32 by
// the streaming reader) into a model built from cfg, following the
// Mimi loader discipline: every expected key must exist with the
// expected shape, unexpected keys are errors, and all problems are
// reported together.
//
// HF nn.Linear stores weights as (out, in) — the same layout as gorch
// Linear.Weight, so weights are aliased with NO transpose (unlike the
// GPT-2 Conv1D loader). Tied head: with tie_word_embeddings the head is
// aliased to embed_tokens; a materialised lm_head.weight (some exports
// carry one — the Qwen3-0.6B file does) is tolerated only if it decodes
// identically to embed_tokens. Missing bias tensors (Qwen3 has no
// attention biases) stay frozen zeros. All parameters are frozen after
// loading — M1's LoRA/vocab-extension owns trainable state.
func Load(path string, cfg Config) (*Model, error) {
	return load(path, cfg, false)
}

// LoadTruncated loads only the first cfg.NumLayers layers from a
// deeper checkpoint, ignoring the deeper layers' tensors — the
// depth-truncation knob for the M1 overfit gate (plan 0008 §3.6):
// training MECHANICS are gated on real pretrained weights even when a
// full-depth CPU run would be impractically slow.
func LoadTruncated(path string, cfg Config) (*Model, error) {
	return load(path, cfg, true)
}

func load(path string, cfg Config, allowDeeperLayers bool) (*Model, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}

	m := New(cfg)
	consumed := map[string]bool{}
	var problems []string

	take := func(name string, want ...int) *g.Tensor {
		t, ok := sf.Tensors[name]
		if !ok {
			problems = append(problems, "missing: "+name)
			return nil
		}
		consumed[name] = true
		if !shapeEq(t.Shape(), want) {
			problems = append(problems, fmt.Sprintf("shape: %s is %v, want %v", name, t.Shape(), want))
			return nil
		}
		return t
	}
	assign := func(dst **g.Tensor, name string, want ...int) {
		if t := take(name, want...); t != nil {
			*dst = t
		}
	}
	copyBias := func(l *nn.Linear, name string, want int) {
		if t := take(name, want); t != nil {
			copy(l.Bias.Data(), t.Data())
		}
	}

	H, I := cfg.HiddenSize, cfg.IntermediateSize
	inner, kv := cfg.InnerDim(), cfg.KVDim()

	assign(&m.Embed.Weight, "model.embed_tokens.weight", cfg.VocabSize, H)
	assign(&m.Norm.Weight, "model.norm.weight", H)

	for i, blk := range m.Blocks {
		p := fmt.Sprintf("model.layers.%d.", i)
		assign(&blk.NormAttn.Weight, p+"input_layernorm.weight", H)
		assign(&blk.NormFFN.Weight, p+"post_attention_layernorm.weight", H)
		assign(&blk.Attn.Wq.Weight, p+"self_attn.q_proj.weight", inner, H)
		assign(&blk.Attn.Wk.Weight, p+"self_attn.k_proj.weight", kv, H)
		assign(&blk.Attn.Wv.Weight, p+"self_attn.v_proj.weight", kv, H)
		assign(&blk.Attn.Wo.Weight, p+"self_attn.o_proj.weight", H, inner)
		if cfg.UseQKNorm {
			assign(&blk.Attn.QNorm.Weight, p+"self_attn.q_norm.weight", cfg.HeadDim)
			assign(&blk.Attn.KNorm.Weight, p+"self_attn.k_norm.weight", cfg.HeadDim)
		}
		if cfg.AttnBias {
			copyBias(blk.Attn.Wq, p+"self_attn.q_proj.bias", inner)
			copyBias(blk.Attn.Wk, p+"self_attn.k_proj.bias", kv)
			copyBias(blk.Attn.Wv, p+"self_attn.v_proj.bias", kv)
		}
		assign(&blk.FFN.Wgate.Weight, p+"mlp.gate_proj.weight", I, H)
		assign(&blk.FFN.Wup.Weight, p+"mlp.up_proj.weight", I, H)
		assign(&blk.FFN.Wdown.Weight, p+"mlp.down_proj.weight", H, I)
	}

	// Tied LM head: alias (the Model always reads Embed.Weight); a
	// materialised lm_head.weight must decode identically.
	if cfg.TiedEmbeddings {
		if lm, ok := sf.Tensors["lm_head.weight"]; ok {
			consumed["lm_head.weight"] = true
			if !shapeEq(lm.Shape(), []int{cfg.VocabSize, H}) {
				problems = append(problems, fmt.Sprintf("shape: lm_head.weight is %v, want %v", lm.Shape(), []int{cfg.VocabSize, H}))
			} else if emb, ok := sf.Tensors["model.embed_tokens.weight"]; ok {
				a, b := lm.Data(), emb.Data()
				for j := range a {
					if a[j] != b[j] {
						problems = append(problems,
							fmt.Sprintf("tied-head violation: lm_head.weight differs from model.embed_tokens.weight at element %d (%v vs %v)", j, a[j], b[j]))
						break
					}
				}
			}
		}
	} else {
		problems = append(problems, "untied lm_head is not implemented (all Qwen2.5/3 small models tie embeddings)")
	}

	for _, name := range sf.Names {
		if !consumed[name] {
			if allowDeeperLayers && isDeeperLayerKey(name, cfg.NumLayers) {
				continue
			}
			problems = append(problems, "unexpected key: "+name)
		}
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		if len(problems) > 40 {
			problems = append(problems[:40], fmt.Sprintf("... and %d more", len(problems)-40))
		}
		return nil, fmt.Errorf("qwen: checkpoint %s does not match config:\n  %s",
			path, joinLines(problems))
	}

	// Inference model: freeze everything (M1 LoRA owns trainable state).
	for _, p := range m.Parameters() {
		p.SetRequiresGrad(false)
	}
	return m, nil
}

// LoadPretrained locates (FindCheckpoint) and loads Qwen3-0.6B.
func LoadPretrained() (*Model, error) {
	path, err := FindCheckpoint()
	if err != nil {
		return nil, err
	}
	return Load(path, Qwen3_0_6B())
}

// isDeeperLayerKey reports whether name addresses a transformer layer
// at index ≥ numLayers ("model.layers.{i}.…").
func isDeeperLayerKey(name string, numLayers int) bool {
	const p = "model.layers."
	if !strings.HasPrefix(name, p) {
		return false
	}
	rest := name[len(p):]
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return false
	}
	idx, err := strconv.Atoi(rest[:dot])
	return err == nil && idx >= numLayers
}

func shapeEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func joinLines(lines []string) string {
	out := ""
	for i, l := range lines {
		if i > 0 {
			out += "\n  "
		}
		out += l
	}
	return out
}
