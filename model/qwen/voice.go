//go:build darwin

package qwen

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// VoiceConfig sizes the trainable surgery applied on top of a frozen
// base model (plan 0008 §3.1–3.2).
type VoiceConfig struct {
	LoRARank   int     // r (plan default 16; overfit gate uses 8)
	LoRAAlpha  float32 // α (plan default 32)
	LoRALayers int     // adapt the TOP n layers; 0 = all layers
	NumExt     int     // appended vocab rows; 0 = NumExtTokens (16,400)
}

// VoiceModel is the M1 training composition: a frozen Qwen base with
// LoRA adapters on q/k/v/o + gate/up/down and a split-tensor extended
// embedding whose appended rows are the only other trainable state.
type VoiceModel struct {
	Base  *Model
	Embed *nn.ExtendedEmbedding
	Cfg   VoiceConfig

	adapters     []*nn.LoRALinear
	adapterNames []string
}

// NewVoiceModel freezes every base parameter, wraps the projection
// layers of the top Cfg.LoRALayers blocks with LoRA adapters, and
// replaces the embedding/tied head with an ExtendedEmbedding.
func NewVoiceModel(m *Model, vc VoiceConfig) *VoiceModel {
	if vc.LoRARank <= 0 {
		vc.LoRARank = 16
	}
	if vc.LoRAAlpha == 0 {
		vc.LoRAAlpha = 32
	}
	if vc.LoRALayers <= 0 || vc.LoRALayers > len(m.Blocks) {
		vc.LoRALayers = len(m.Blocks)
	}
	if vc.NumExt <= 0 {
		vc.NumExt = NumExtTokens
	}

	for _, p := range m.Parameters() {
		p.SetRequiresGrad(false)
	}

	vm := &VoiceModel{
		Base:  m,
		Embed: nn.NewExtendedEmbedding(m.Embed.Weight, vc.NumExt),
		Cfg:   vc,
	}

	add := func(name string, l *nn.LoRALinear) *nn.LoRALinear {
		vm.adapters = append(vm.adapters, l)
		vm.adapterNames = append(vm.adapterNames, name)
		return l
	}
	first := len(m.Blocks) - vc.LoRALayers
	for i := first; i < len(m.Blocks); i++ {
		blk := m.Blocks[i]
		p := fmt.Sprintf("lora.layers.%d.", i)
		blk.Attn.LoRAQ = add(p+"q_proj", nn.NewLoRALinear(blk.Attn.Wq, vc.LoRARank, vc.LoRAAlpha))
		blk.Attn.LoRAK = add(p+"k_proj", nn.NewLoRALinear(blk.Attn.Wk, vc.LoRARank, vc.LoRAAlpha))
		blk.Attn.LoRAV = add(p+"v_proj", nn.NewLoRALinear(blk.Attn.Wv, vc.LoRARank, vc.LoRAAlpha))
		blk.Attn.LoRAO = add(p+"o_proj", nn.NewLoRALinear(blk.Attn.Wo, vc.LoRARank, vc.LoRAAlpha))
		blk.FFN.LoRAGate = add(p+"gate_proj", nn.NewLoRALinear(blk.FFN.Wgate, vc.LoRARank, vc.LoRAAlpha))
		blk.FFN.LoRAUp = add(p+"up_proj", nn.NewLoRALinear(blk.FFN.Wup, vc.LoRARank, vc.LoRAAlpha))
		blk.FFN.LoRADown = add(p+"down_proj", nn.NewLoRALinear(blk.FFN.Wdown, vc.LoRARank, vc.LoRAAlpha))
	}
	return vm
}

// ForwardHidden runs embeddings → blocks → final norm with autograd
// active, returning the (seq, hidden) hidden states. Token ids may
// span the extended vocabulary.
func (vm *VoiceModel) ForwardHidden(tokens []int) *g.Tensor {
	if len(tokens) == 0 {
		panic("qwen: empty token slice")
	}
	h := vm.Embed.Forward(tokens)
	for _, blk := range vm.Base.Blocks {
		h = blk.Forward(h, 0)
	}
	return vm.Base.Norm.Forward(h)
}

// ForwardCached is the KV-cached decode path over the extended vocab:
// returns logits (1, ExtVocab) for the last position. Inference-only.
func (vm *VoiceModel) ForwardCached(tokens []int, cache *nn.KVCache) *g.Tensor {
	if len(tokens) == 0 {
		panic("qwen: empty token slice")
	}
	var logits *g.Tensor
	g.NoGrad(func() {
		posOffset := cache.Len()
		h := vm.Embed.Forward(tokens)
		for i, blk := range vm.Base.Blocks {
			h = blk.ForwardCached(h, cache, i, posOffset)
		}
		seq := h.Shape()[0]
		dim := h.Shape()[1]
		last := g.NewTensor(h.Data()[(seq-1)*dim:seq*dim], 1, dim)
		last = vm.Base.Norm.Forward(last)
		logits = vm.Embed.Logits(last)
	})
	return logits
}

// GenerateGreedy produces up to maxNew tokens after prompt with
// KV-cached greedy decoding over the extended vocabulary, stopping
// when one of stop is produced (stop token excluded from the result).
// Returns only the generated ids.
func (vm *VoiceModel) GenerateGreedy(prompt []int, maxNew int, stop []int) []int {
	cache := nn.NewKVCache(vm.Base.Cfg.NumLayers, vm.Base.Cfg.KVDim())
	logits := vm.ForwardCached(prompt, cache)
	var out []int
	for i := 0; i < maxNew; i++ {
		next := argmaxF32(logits.Data())
		if isStop(next, stop) {
			break
		}
		out = append(out, next)
		if cache.Len() >= vm.Base.Cfg.MaxSeq {
			break
		}
		logits = vm.ForwardCached([]int{next}, cache)
	}
	return out
}

// TrainableParams returns the trainable tensors and their checkpoint
// names, LoRA factors first (…"/A", …"/B") and the extended-embedding
// rows ("ext_embed") last — the order the trainer's two LR groups and
// the checkpoint format rely on.
func (vm *VoiceModel) TrainableParams() (names []string, params []*g.Tensor) {
	for i, l := range vm.adapters {
		names = append(names, vm.adapterNames[i]+"/A", vm.adapterNames[i]+"/B")
		params = append(params, l.A, l.B)
	}
	names = append(names, "ext_embed")
	params = append(params, vm.Embed.Ext)
	return names, params
}

// LoRAParams returns only the adapter factors (the first LR group).
func (vm *VoiceModel) LoRAParams() []*g.Tensor {
	var out []*g.Tensor
	for _, l := range vm.adapters {
		out = append(out, l.A, l.B)
	}
	return out
}

// Adapters exposes the wrapped LoRA layers (Merge/Unmerge etc.).
func (vm *VoiceModel) Adapters() []*nn.LoRALinear { return vm.adapters }
