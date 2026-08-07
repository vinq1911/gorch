//go:build darwin

package mimi

import (
	"fmt"
	"math"
	"sort"
	"strings"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/nn"
)

// hfConvPrefixes returns the HF checkpoint key prefix (without
// ".weight"/".bias") for each conv in SEANet.Convs order, per the plan
// §4.6 name map: encoder.layers.{0,3,6,9,12,14}.conv and
// encoder.layers.{1,4,7,10}.block.{1,3}.conv.
func hfConvPrefixes(s *SEANet) []string {
	prefixes := make([]string, 0, len(s.Convs))
	prefixes = append(prefixes, "encoder.layers.0.conv")
	for i := 0; i < s.stages(); i++ {
		blockIdx := 1 + 3*i
		downIdx := 3 + 3*i
		prefixes = append(prefixes,
			fmt.Sprintf("encoder.layers.%d.block.1.conv", blockIdx),
			fmt.Sprintf("encoder.layers.%d.block.3.conv", blockIdx),
			fmt.Sprintf("encoder.layers.%d.conv", downIdx))
	}
	prefixes = append(prefixes, fmt.Sprintf("encoder.layers.%d.conv", 2+3*s.stages()))
	return prefixes
}

// LoadSEANet loads the SEANet encoder weights from a kyutai/mimi
// model.safetensors checkpoint. It verifies every expected key exists
// with the expected shape and fails listing all missing, misshapen,
// and unexpected keys. Keys under decoder.*, quantizer.*,
// encoder_transformer.*, downsample.* (and the other non-encoder
// families) are ignored until later phases.
func LoadSEANet(path string) (*SEANet, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}
	return loadSEANetFrom(sf)
}

func loadSEANetFrom(sf *model.SafetensorsFile) (*SEANet, error) {
	s := NewSEANet(DefaultConfig())
	prefixes := hfConvPrefixes(s)

	consumed := map[string]bool{}
	var problems []string

	for i, prefix := range prefixes {
		conv := s.Convs[i]

		w, keys, ok := convWeight(sf.Tensors, prefix)
		if !ok {
			problems = append(problems, fmt.Sprintf("missing: %s.weight (no weight-norm g/v fallback keys either)", prefix))
		} else {
			for _, k := range keys {
				consumed[k] = true
			}
			if !shapeEq(w.Shape(), conv.Weight.Shape()) {
				problems = append(problems, fmt.Sprintf("shape: %s.weight is %v, want %v", prefix, w.Shape(), conv.Weight.Shape()))
			} else {
				conv.Weight = w
			}
		}

		biasKey := prefix + ".bias"
		b, ok := sf.Tensors[biasKey]
		if !ok {
			problems = append(problems, "missing: "+biasKey)
		} else {
			consumed[biasKey] = true
			if !shapeEq(b.Shape(), conv.Bias.Shape()) {
				problems = append(problems, fmt.Sprintf("shape: %s is %v, want %v", biasKey, b.Shape(), conv.Bias.Shape()))
			} else {
				conv.Bias = b
			}
		}
	}

	// Every encoder.* key must have been consumed; other families are
	// loaded by their own loaders (transformer/downsample by Load,
	// quantizer by LoadQuantizer, decoder-side keys by LoadDecoder) and
	// deliberately ignored here.
	var unexpected []string
	for _, name := range sf.Names {
		if strings.HasPrefix(name, "encoder.") && !consumed[name] {
			unexpected = append(unexpected, name)
		}
	}
	sort.Strings(unexpected)
	for _, name := range unexpected {
		problems = append(problems, "unexpected: "+name)
	}

	if len(problems) > 0 {
		return nil, fmt.Errorf("mimi: SEANet checkpoint mismatch (%d problems):\n  %s",
			len(problems), strings.Join(problems, "\n  "))
	}
	return s, nil
}

// Load builds the full pre-quantizer Encoder (SEANet + transformer +
// downsample) from a kyutai/mimi model.safetensors checkpoint, with
// the same fail-loudly key/shape validation as LoadSEANet. Keys under
// decoder.*, decoder_transformer.*, upsample.* and quantizer.* are
// ignored here — they are validated by LoadDecoder and LoadQuantizer
// (LoadFull runs all loaders off one parse and asserts full coverage).
func Load(path string) (*Encoder, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}
	return loadEncoderFrom(sf)
}

func loadEncoderFrom(sf *model.SafetensorsFile) (*Encoder, error) {
	seanet, err := loadSEANetFrom(sf)
	if err != nil {
		return nil, err
	}
	e := NewEncoder(seanet.Cfg)
	e.SEANet = seanet

	consumed := map[string]bool{}
	var problems []string

	// take fetches a checkpoint tensor, validates its shape and hands
	// it to assign; missing keys and shape mismatches are collected.
	take := func(key string, want []int, assign func(*g.Tensor)) {
		t, ok := sf.Tensors[key]
		if !ok {
			problems = append(problems, "missing: "+key)
			return
		}
		consumed[key] = true
		if !shapeEq(t.Shape(), want) {
			problems = append(problems, fmt.Sprintf("shape: %s is %v, want %v", key, t.Shape(), want))
			return
		}
		assign(t)
	}

	dim := e.Cfg.HiddenSize
	loadTransformerLayers(take, "encoder_transformer.", e.Layers[:], e.Cfg)

	take("downsample.conv.weight", []int{dim, dim, 4}, func(t *g.Tensor) { e.Downsample.Weight = t })

	// Every encoder_transformer.* and downsample.* key must have been
	// consumed (encoder.* was validated by loadSEANetFrom).
	var unexpected []string
	for _, name := range sf.Names {
		if (strings.HasPrefix(name, "encoder_transformer.") || strings.HasPrefix(name, "downsample.")) && !consumed[name] {
			unexpected = append(unexpected, name)
		}
	}
	sort.Strings(unexpected)
	for _, name := range unexpected {
		problems = append(problems, "unexpected: "+name)
	}

	if len(problems) > 0 {
		return nil, fmt.Errorf("mimi: encoder checkpoint mismatch (%d problems):\n  %s",
			len(problems), strings.Join(problems, "\n  "))
	}
	return e, nil
}

// loadTransformerLayers loads the 12-key parameter set of each Mimi
// transformer layer under prefix ("encoder_transformer." or
// "decoder_transformer." — the checkpoint layer structure is
// bit-for-bit identical between the two, plan 0007 §0.2.3) via the
// caller's take closure, which handles missing-key/shape collection
// and consumed-key bookkeeping.
func loadTransformerLayers(take func(key string, want []int, assign func(*g.Tensor)), prefix string, layers []*Layer, cfg Config) {
	dim, inter := cfg.HiddenSize, cfg.Intermediate
	for i, l := range layers {
		l := l
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		for _, proj := range []struct {
			key  string
			dst  *nn.Linear
			want []int
		}{
			{p + "self_attn.q_proj.weight", l.Wq, []int{dim, dim}},
			{p + "self_attn.k_proj.weight", l.Wk, []int{dim, dim}},
			{p + "self_attn.v_proj.weight", l.Wv, []int{dim, dim}},
			{p + "self_attn.o_proj.weight", l.Wo, []int{dim, dim}},
			{p + "mlp.fc1.weight", l.Fc1, []int{inter, dim}},
			{p + "mlp.fc2.weight", l.Fc2, []int{dim, inter}},
		} {
			proj := proj
			take(proj.key, proj.want, func(t *g.Tensor) { proj.dst.Weight = t })
		}
		take(p+"input_layernorm.weight", []int{dim}, func(t *g.Tensor) { l.Norm1.Weight = t })
		take(p+"input_layernorm.bias", []int{dim}, func(t *g.Tensor) { l.Norm1.Bias = t })
		take(p+"post_attention_layernorm.weight", []int{dim}, func(t *g.Tensor) { l.Norm2.Weight = t })
		take(p+"post_attention_layernorm.bias", []int{dim}, func(t *g.Tensor) { l.Norm2.Bias = t })
		take(p+"self_attn_layer_scale.scale", []int{dim}, func(t *g.Tensor) { l.AttnScale = t })
		take(p+"mlp_layer_scale.scale", []int{dim}, func(t *g.Tensor) { l.MlpScale = t })
	}
}

// LoadDecoder loads the full Mimi decoder (upsample + transformer +
// SEANet decoder) from a kyutai/mimi model.safetensors checkpoint,
// with the same fail-loudly key/shape validation as Load: every
// decoder.*, decoder_transformer.* and upsample.* key must be consumed
// with its expected shape.
func LoadDecoder(path string) (*Decoder, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}
	return loadDecoderFrom(sf)
}

func loadDecoderFrom(sf *model.SafetensorsFile) (*Decoder, error) {
	d := NewDecoder(DefaultConfig())

	consumed := map[string]bool{}
	var problems []string

	take := func(key string, want []int, assign func(*g.Tensor)) {
		t, ok := sf.Tensors[key]
		if !ok {
			problems = append(problems, "missing: "+key)
			return
		}
		consumed[key] = true
		if !shapeEq(t.Shape(), want) {
			problems = append(problems, fmt.Sprintf("shape: %s is %v, want %v", key, t.Shape(), want))
			return
		}
		assign(t)
	}

	// takeConv loads a plain conv's weight and bias by checkpoint
	// prefix, with the weight-norm g/v fallback (plain convs only —
	// transposed weight-norm fusion is out of scope and would fail
	// loudly as missing + unexpected keys, plan 0007 §3.2).
	takeConv := func(prefix string, conv *nn.CausalConv1d) {
		w, keys, ok := convWeight(sf.Tensors, prefix)
		if !ok {
			problems = append(problems, fmt.Sprintf("missing: %s.weight (no weight-norm g/v fallback keys either)", prefix))
		} else {
			for _, k := range keys {
				consumed[k] = true
			}
			if !shapeEq(w.Shape(), conv.Weight.Shape()) {
				problems = append(problems, fmt.Sprintf("shape: %s.weight is %v, want %v", prefix, w.Shape(), conv.Weight.Shape()))
			} else {
				conv.Weight = w
			}
		}
		take(prefix+".bias", conv.Bias.Shape(), func(t *g.Tensor) { conv.Bias = t })
	}

	// Depthwise 12.5→25 Hz upsample: (inC, outC/groups, k) ConvTranspose
	// layout, no bias key exists (verified, plan 0007 §0.2.1).
	take("upsample.conv.weight", []int{d.Cfg.HiddenSize, 1, 4}, func(t *g.Tensor) { d.Upsample.Weight = t })

	loadTransformerLayers(take, "decoder_transformer.", d.Layers[:], d.Cfg)

	takeConv("decoder.layers.0.conv", d.SEANet.Init)
	for s := range d.SEANet.Ups {
		up := d.SEANet.Ups[s]
		p := fmt.Sprintf("decoder.layers.%d.conv", 2+3*s)
		// ConvTranspose weight is (inC, outC, k) — the four non-square
		// SEANet transposes make this shape check catch a transposed
		// layout immediately (plan 0007 §8 risk 1).
		take(p+".weight", up.Weight.Shape(), func(t *g.Tensor) { up.Weight = t })
		take(p+".bias", up.Bias.Shape(), func(t *g.Tensor) { up.Bias = t })
		blk := fmt.Sprintf("decoder.layers.%d.block.", 3+3*s)
		takeConv(blk+"1.conv", d.SEANet.Res[s][0])
		takeConv(blk+"3.conv", d.SEANet.Res[s][1])
	}
	takeConv("decoder.layers.14.conv", d.SEANet.Final)

	// Every decoder-side key must have been consumed.
	var unexpected []string
	for _, name := range sf.Names {
		if (strings.HasPrefix(name, "decoder.") || strings.HasPrefix(name, "decoder_transformer.") ||
			strings.HasPrefix(name, "upsample.")) && !consumed[name] {
			unexpected = append(unexpected, name)
		}
	}
	sort.Strings(unexpected)
	for _, name := range unexpected {
		problems = append(problems, "unexpected: "+name)
	}

	if len(problems) > 0 {
		return nil, fmt.Errorf("mimi: decoder checkpoint mismatch (%d problems):\n  %s",
			len(problems), strings.Join(problems, "\n  "))
	}
	return d, nil
}

// mimiKeyFamilies are the checkpoint key prefixes claimed by the three
// sub-loaders: encoder.*, encoder_transformer.*, downsample.* by Load;
// quantizer.* by LoadQuantizer; decoder.*, decoder_transformer.*,
// upsample.* by LoadDecoder.
var mimiKeyFamilies = []string{
	"encoder.", "encoder_transformer.", "downsample.",
	"quantizer.",
	"decoder.", "decoder_transformer.", "upsample.",
}

// LoadFull loads the encoder, quantizer and decoder from a single
// checkpoint parse (the safetensors read dominates load time). It also
// asserts the consumed-key union covers the whole checkpoint (all 350
// tensors of kyutai/mimi): each sub-loader fail-loudly consumes every
// key inside its prefix families, and LoadFull rejects any key outside
// all families — so success proves every checkpoint tensor was
// consumed by exactly one loader (plan 0007 §3.2, risk 7).
func LoadFull(path string) (*Encoder, *Quantizer, *Decoder, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, nil, nil, err
	}
	var stray []string
	for _, name := range sf.Names {
		claimed := false
		for _, fam := range mimiKeyFamilies {
			if strings.HasPrefix(name, fam) {
				claimed = true
				break
			}
		}
		if !claimed {
			stray = append(stray, name)
		}
	}
	if len(stray) > 0 {
		sort.Strings(stray)
		return nil, nil, nil, fmt.Errorf("mimi: checkpoint has %d keys outside every loader's families:\n  %s",
			len(stray), strings.Join(stray, "\n  "))
	}
	e, err := loadEncoderFrom(sf)
	if err != nil {
		return nil, nil, nil, err
	}
	q, err := loadQuantizerFrom(sf)
	if err != nil {
		return nil, nil, nil, err
	}
	d, err := loadDecoderFrom(sf)
	if err != nil {
		return nil, nil, nil, err
	}
	return e, q, d, nil
}

// convWeight fetches a conv weight by prefix. Primary path: the plain
// fused "<prefix>.weight" (what kyutai/mimi ships). Fallback: fuse
// weight-norm parameter pairs — "<prefix>.weight_g"/"weight_v" or
// "<prefix>.parametrizations.weight.original0/original1" — as
// w = g · v/‖v‖ with the norm over dims (1,2) per output channel.
// Returns the weight, the checkpoint keys consumed, and whether any
// style was found.
func convWeight(tensors map[string]*g.Tensor, prefix string) (*g.Tensor, []string, bool) {
	if w, ok := tensors[prefix+".weight"]; ok {
		return w, []string{prefix + ".weight"}, true
	}
	for _, pair := range [][2]string{
		{prefix + ".weight_g", prefix + ".weight_v"},
		{prefix + ".parametrizations.weight.original0", prefix + ".parametrizations.weight.original1"},
	} {
		gt, ok1 := tensors[pair[0]]
		vt, ok2 := tensors[pair[1]]
		if ok1 && ok2 {
			return fuseWeightNorm(gt, vt), []string{pair[0], pair[1]}, true
		}
	}
	return nil, nil, false
}

// fuseWeightNorm computes w[o,:,:] = g[o] * v[o,:,:] / ‖v[o,:,:]‖₂.
// gt may be (outC,), (outC,1) or (outC,1,1) — PyTorch stores
// (outC,1,1); only its outC leading dimension is used. vt is
// (outC, inC, k).
func fuseWeightNorm(gt, vt *g.Tensor) *g.Tensor {
	vShape := vt.Shape()
	if len(vShape) != 3 {
		panic(fmt.Sprintf("mimi: fuseWeightNorm expects 3-D v, got %v", vShape))
	}
	outC := vShape[0]
	if len(gt.Data()) != outC {
		panic(fmt.Sprintf("mimi: fuseWeightNorm g has %d elements, want %d", len(gt.Data()), outC))
	}
	row := vShape[1] * vShape[2]
	v := vt.Data()
	out := make([]float32, len(v))
	for o := 0; o < outC; o++ {
		var ss float64
		for i := o * row; i < (o+1)*row; i++ {
			ss += float64(v[i]) * float64(v[i])
		}
		scale := float32(float64(gt.Data()[o]) / math.Sqrt(ss))
		for i := o * row; i < (o+1)*row; i++ {
			out[i] = v[i] * scale
		}
	}
	return g.NewTensor(out, vShape...)
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
