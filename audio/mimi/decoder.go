//go:build darwin

package mimi

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// SEANetDecoder is the Mimi SEANet decoder (plan 0007 §0.2.4): the
// 15-layer causal stack mapping the (1, 512, T25) post-transformer
// latent at 25 Hz to a (1, 1, 960·T25) waveform at 24 kHz (stride
// product 8·6·5·4 = 960, ratios iterated in config order — NOT
// reversed like the encoder). Layout, in checkpoint order:
//
//	Init                 conv 512→1024 k7 s1     decoder.layers.0
//	per stage i in 0..3:
//	  ELU
//	  Ups[i]             ConvTranspose           decoder.layers.{2,5,8,11}
//	                     (k16s8 1024→512, k12s6 512→256,
//	                      k10s5 256→128, k8s4 128→64, all with bias)
//	  Res[i]             resnet block            decoder.layers.{3,6,9,12}
//	                     (ELU→conv k3 C→C/2→ELU→conv k1 C/2→C,
//	                      identity shortcut)
//	ELU
//	Final                conv 64→1 k3 s1         decoder.layers.14
//
// Unlike the encoder, the resnet block comes AFTER the strided
// (transposed) conv in each stage.
type SEANetDecoder struct {
	Init  *nn.CausalConv1d             // 512→1024 k7 s1
	Ups   [4]*nn.CausalConvTranspose1d // k16s8, k12s6, k10s5, k8s4 (bias=true)
	Res   [4][2]*nn.CausalConv1d       // per-stage resnet convs (k3 C→C/2, k1 C/2→C)
	Final *nn.CausalConv1d             // 64→1 k3 s1
}

// NewSEANetDecoder builds the decoder stack with randomly initialized
// weights (LoadDecoder replaces them with checkpoint tensors).
// Inference-only: no parameter has requires-grad set.
func NewSEANetDecoder(cfg Config) *SEANetDecoder {
	if cfg.NumResidualLayers != 1 {
		panic(fmt.Sprintf("mimi: NewSEANetDecoder supports NumResidualLayers=1, got %d", cfg.NumResidualLayers))
	}
	d := &SEANetDecoder{}
	if len(cfg.UpsamplingRatios) != len(d.Ups) {
		panic(fmt.Sprintf("mimi: NewSEANetDecoder supports %d stages, got %d ratios", len(d.Ups), len(cfg.UpsamplingRatios)))
	}
	mult := 1 << len(cfg.UpsamplingRatios) // 16
	d.Init = nn.NewCausalConv1d(cfg.HiddenSize, mult*cfg.NumFilters, cfg.KernelSize, 1, 1, true, g.PadConstant)
	for i, ratio := range cfg.UpsamplingRatios {
		dim := mult * cfg.NumFilters
		// Strided upsampling transposed conv: kernel 2*ratio, stride
		// ratio, with bias (only the 12.5→25 Hz upsample is bias-free).
		d.Ups[i] = nn.NewCausalConvTranspose1d(dim, dim/2, ratio*2, ratio, 1, true)
		// Resnet block: ELU → conv k3 (dim/2 → dim/2/Compress) → ELU →
		// conv k1 back, identity shortcut. Dilation is always 1
		// (NumResidualLayers = 1).
		d.Res[i][0] = nn.NewCausalConv1d(dim/2, dim/2/cfg.Compress, cfg.ResidualKernelSize, 1, 1, true, g.PadConstant)
		d.Res[i][1] = nn.NewCausalConv1d(dim/2/cfg.Compress, dim/2, 1, 1, 1, true, g.PadConstant)
		mult /= 2
	}
	d.Final = nn.NewCausalConv1d(cfg.NumFilters, 1, cfg.LastKernelSize, 1, 1, true, g.PadConstant)
	for _, p := range d.Parameters() {
		p.SetRequiresGrad(false)
	}
	return d
}

// Parameters returns every parameter tensor of the stack.
func (s *SEANetDecoder) Parameters() []*g.Tensor {
	params := s.Init.Parameters()
	for i := range s.Ups {
		params = append(params, s.Ups[i].Parameters()...)
		params = append(params, s.Res[i][0].Parameters()...)
		params = append(params, s.Res[i][1].Parameters()...)
	}
	return append(params, s.Final.Parameters()...)
}

// Forward runs the offline decoder stack: (1, 512, T25) →
// (1, 1, 960·T25). Callers wrap in g.NoGrad.
func (s *SEANetDecoder) Forward(x *g.Tensor) *g.Tensor {
	x = s.Init.Forward(x)
	for i := range s.Ups {
		x = g.ELU(x)
		x = s.Ups[i].Forward(x)
		x = resnetForward(x, s.Res[i][0], s.Res[i][1])
	}
	x = g.ELU(x)
	return s.Final.Forward(x)
}

// Decoder is the full Mimi decoder (plan 0007 §0.2): quantized latent
// at 12.5 Hz → depthwise ConvTranspose upsample to 25 Hz → 8
// transformer layers (the decoder_transformer — structurally identical
// to the encoder's, so the same Layer type is reused unchanged) →
// SEANet decoder → 24 kHz waveform. Note the order: the upsample comes
// BEFORE the transformer, which runs at 25 Hz.
type Decoder struct {
	Upsample *nn.CausalConvTranspose1d // 512→512 k4 s2 groups=512, no bias (depthwise)
	Layers   [8]*Layer                 // decoder_transformer — same Layer type as the encoder
	Rope     *nn.RoPE
	SEANet   *SEANetDecoder
	Cfg      Config
}

// NewDecoder builds the decoder with randomly initialized weights
// (LoadDecoder replaces them with checkpoint tensors). Inference-only.
func NewDecoder(cfg Config) *Decoder {
	d := &Decoder{
		Rope:   nn.NewRoPE(cfg.HeadDim, cfg.MaxPositions, cfg.RopeTheta, nn.RopeLlama),
		SEANet: NewSEANetDecoder(cfg),
		Cfg:    cfg,
	}
	if cfg.NumLayers != len(d.Layers) {
		panic(fmt.Sprintf("mimi: NumLayers = %d, want %d", cfg.NumLayers, len(d.Layers)))
	}
	for i := range d.Layers {
		d.Layers[i] = NewLayer(cfg)
	}
	// Upsample geometry is fixed by the checkpoint like the encoder's
	// downsample: 12.5 → 25 Hz means kernel 2·(25/12.5) = 4, stride 2,
	// depthwise (groups = upsample_groups = HiddenSize), bias-free
	// (verified: no upsample.conv.bias key exists).
	d.Upsample = nn.NewCausalConvTranspose1d(cfg.HiddenSize, cfg.HiddenSize, 4, 2, cfg.HiddenSize, false)
	d.Upsample.Weight.SetRequiresGrad(false)
	return d
}

// DecodeLatent runs the offline decoder on a (T, 512) quantized latent
// (as produced by Quantizer.Decode) and returns exactly 1920·T waveform
// samples at 24 kHz. Attention uses the plain causal mask — matching
// HF's offline reference exactly (HF's offline decoder does NOT apply
// the 250-frame sliding window; the same caveat as Encoder.Encode).
func (d *Decoder) DecodeLatent(latent *g.Tensor) []float32 {
	return d.decode(latent, 0)
}

// DecodeLatentWindowed is DecodeLatent with the strict 250-frame
// sliding-window causal mask (the intended Mimi semantics, and the
// streaming reference). It differs from DecodeLatent only for inputs
// longer than 125 tokens (>250 transformer frames at 25 Hz).
func (d *Decoder) DecodeLatentWindowed(latent *g.Tensor) []float32 {
	return d.decode(latent, d.Cfg.SlidingWindow)
}

// Decode is the codes→PCM convenience: Quantizer.Decode followed by
// DecodeLatent. codes is (numQuantizers, T) as returned by
// Quantizer.Encode; the result is 1920·T samples at 24 kHz.
func (d *Decoder) Decode(q *Quantizer, codes [][]int) []float32 {
	return d.DecodeLatent(q.Decode(codes))
}

func (d *Decoder) decode(latent *g.Tensor, window int) []float32 {
	shape := latent.Shape()
	if len(shape) != 2 || shape[1] != d.Cfg.HiddenSize {
		panic(fmt.Sprintf("mimi: DecodeLatent latent shape %v, want (T, %d)", shape, d.Cfg.HiddenSize))
	}
	if frames := 2 * shape[0]; frames > d.Cfg.MaxPositions {
		panic(fmt.Sprintf("mimi: DecodeLatent %d tokens = %d transformer frames > MaxPositions %d",
			shape[0], frames, d.Cfg.MaxPositions))
	}
	var out []float32
	g.NoGrad(func() {
		x := transposeTC(latent)  // (1, 512, T)
		x = d.Upsample.Forward(x) // (1, 512, 2T)
		h := transposeCT(x)       // (2T, 512)
		for _, l := range d.Layers {
			h = l.Forward(h, d.Rope, window)
		}
		wav := d.SEANet.Forward(transposeTC(h)) // (1, 1, 1920T)
		out = wav.Data()
	})
	return out
}
