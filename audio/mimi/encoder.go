//go:build darwin

package mimi

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// Encoder is the full pre-quantizer Mimi encoder (plan §4.4):
// SEANet (24 kHz PCM → 25 Hz, 512-d) → 8 transformer layers →
// causal downsample conv (512→512, k4, s2, no bias, replicate pad)
// → 12.5 Hz latent.
type Encoder struct {
	SEANet     *SEANet
	Layers     [8]*Layer
	Rope       *nn.RoPE
	Downsample *nn.CausalConv1d
	Cfg        Config
}

// NewEncoder builds the encoder with randomly initialized weights
// (Load replaces them with checkpoint tensors). Inference-only.
func NewEncoder(cfg Config) *Encoder {
	e := &Encoder{
		SEANet: NewSEANet(cfg),
		Rope:   nn.NewRoPE(cfg.HeadDim, cfg.MaxPositions, cfg.RopeTheta, nn.RopeLlama),
		Cfg:    cfg,
	}
	if cfg.NumLayers != len(e.Layers) {
		panic(fmt.Sprintf("mimi: NumLayers = %d, want %d", cfg.NumLayers, len(e.Layers)))
	}
	for i := range e.Layers {
		e.Layers[i] = NewLayer(cfg)
	}
	e.Downsample = nn.NewCausalConv1d(cfg.HiddenSize, cfg.HiddenSize, 4, 2, 1, false, g.PadReplicate)
	e.Downsample.Weight.SetRequiresGrad(false)
	return e
}

// Encode runs the offline encoder on 24 kHz mono PCM and returns the
// (T, 512) pre-quantizer latent at 12.5 Hz. Attention uses the plain
// causal mask — matching HF's offline reference exactly (HF's offline
// encoder does NOT apply the 250-frame sliding window; see
// audio/export_mimi_fixtures.py).
func (e *Encoder) Encode(pcm []float32) *g.Tensor {
	return e.encode(pcm, 0)
}

// EncodeWindowed is Encode with the strict 250-frame sliding-window
// causal mask (the intended Mimi semantics, and Phase 5's streaming
// reference). It differs from Encode only for inputs longer than
// 10 s (>250 frames at 25 Hz).
func (e *Encoder) EncodeWindowed(pcm []float32) *g.Tensor {
	return e.encode(pcm, e.Cfg.SlidingWindow)
}

func (e *Encoder) encode(pcm []float32, window int) *g.Tensor {
	var out *g.Tensor
	g.NoGrad(func() {
		x := e.SEANet.Forward(g.NewTensor(pcm, 1, 1, len(pcm))) // (1, 512, T25)
		h := transposeCT(x)                                     // (T25, 512)
		h = e.forwardTransformer(h, window)
		out = transposeCT(e.Downsample.Forward(transposeTC(h))) // (T12.5, 512)
	})
	return out
}

// forwardTransformer runs the 8 transformer layers on h (T, 512).
// window <= 0 means plain causal attention.
func (e *Encoder) forwardTransformer(h *g.Tensor, window int) *g.Tensor {
	for _, l := range e.Layers {
		h = l.Forward(h, e.Rope, window)
	}
	return h
}

// transposeCT flattens a (1, C, T) or (C, T) tensor to (T, C).
func transposeCT(x *g.Tensor) *g.Tensor {
	shape := x.Shape()
	C, T := shape[len(shape)-2], shape[len(shape)-1]
	return g.Permute(x.Reshape(C, T), []int{1, 0})
}

// transposeTC maps a (T, C) tensor to (1, C, T) for the conv stack.
func transposeTC(x *g.Tensor) *g.Tensor {
	T, C := x.Shape()[0], x.Shape()[1]
	return g.Permute(x, []int{1, 0}).Reshape(1, C, T)
}
