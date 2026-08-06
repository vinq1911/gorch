//go:build darwin

package mimi

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// SEANet is the Mimi SEANet encoder: the 14-conv causal stack of plan
// §0.2 mapping 24 kHz mono PCM (1, 1, L) to a (1, 512, T) latent at
// 25 Hz (stride product 4*5*6*8 = 960).
//
// Convs holds the convolutions in checkpoint order:
//
//	[0]              initial conv           encoder.layers.0
//	[1+3i], [2+3i]   resnet block stage i   encoder.layers.{1+3i}.block.{1,3}
//	[3+3i]           strided down conv      encoder.layers.{3+3i}
//	[len-1]          final conv             encoder.layers.{2+3*stages}
//
// for i in 0..3 (stages iterate cfg.UpsamplingRatios reversed:
// 4, 5, 6, 8). ELUs sit inside the resnet blocks, before each down
// conv, and before the final conv — matching the §0.2 index table.
type SEANet struct {
	Convs []*nn.CausalConv1d
	Cfg   Config
}

// NewSEANet builds the encoder stack with randomly initialized weights
// (LoadSEANet replaces them with checkpoint tensors). Inference-only:
// no parameter has requires-grad set.
func NewSEANet(cfg Config) *SEANet {
	if cfg.NumResidualLayers != 1 {
		panic(fmt.Sprintf("mimi: NewSEANet supports NumResidualLayers=1, got %d", cfg.NumResidualLayers))
	}
	var convs []*nn.CausalConv1d
	add := func(inC, outC, k, stride int) {
		c := nn.NewCausalConv1d(inC, outC, k, stride, 1, true, g.PadConstant)
		for _, p := range c.Parameters() {
			p.SetRequiresGrad(false)
		}
		convs = append(convs, c)
	}

	mult := 1
	add(1, mult*cfg.NumFilters, cfg.KernelSize, 1)
	for i := len(cfg.UpsamplingRatios) - 1; i >= 0; i-- {
		ratio := cfg.UpsamplingRatios[i]
		dim := mult * cfg.NumFilters
		// Resnet block: ELU → conv k3 (dim→dim/2) → ELU → conv k1
		// (dim/2→dim), identity shortcut. Dilation is always
		// DilationGrowthRate**0 = 1 (NumResidualLayers = 1).
		add(dim, dim/cfg.Compress, cfg.ResidualKernelSize, 1)
		add(dim/cfg.Compress, dim, 1, 1)
		// Strided downsampling conv: kernel 2*ratio, stride ratio.
		add(dim, dim*2, ratio*2, ratio)
		mult *= 2
	}
	add(mult*cfg.NumFilters, cfg.HiddenSize, cfg.LastKernelSize, 1)

	return &SEANet{Convs: convs, Cfg: cfg}
}

// stages returns the number of downsampling stages encoded in Convs.
func (s *SEANet) stages() int {
	return (len(s.Convs) - 2) / 3
}

// Forward runs the offline encoder: (1, 1, L) → (1, 512, ceil-cascade
// of L over the strides; 25 Hz for 24 kHz input). Callers wrap in
// g.NoGrad.
func (s *SEANet) Forward(x *g.Tensor) *g.Tensor {
	x = s.Convs[0].Forward(x)
	for i := 0; i < s.stages(); i++ {
		b1, b2, down := s.Convs[1+3*i], s.Convs[2+3*i], s.Convs[3+3*i]
		x = resnetForward(x, b1, b2)
		x = g.ELU(x)
		x = down.Forward(x)
	}
	x = g.ELU(x)
	return s.Convs[len(s.Convs)-1].Forward(x)
}

// resnetForward is one SEANet resnet block with identity shortcut:
// x + conv2(ELU(conv1(ELU(x)))).
func resnetForward(x *g.Tensor, conv1, conv2 *nn.CausalConv1d) *g.Tensor {
	h := g.ELU(x)
	h = conv1.Forward(h)
	h = g.ELU(h)
	h = conv2.Forward(h)
	return g.Add(x, h)
}

// NewStreamStates returns one fresh Conv1dStream per conv, in Convs
// order, for use with ForwardStream. The caller owns the slice; reuse
// it across chunks of one session and Reset (or reallocate) between
// sessions.
func (s *SEANet) NewStreamStates() []*nn.Conv1dStream {
	states := make([]*nn.Conv1dStream, len(s.Convs))
	for i := range states {
		states[i] = &nn.Conv1dStream{}
	}
	return states
}

// ForwardStream runs one streaming chunk through the encoder using the
// per-conv left-context caches in states (from NewStreamStates). For
// the canonical 80 ms chunk (1, 1, 1920) it returns (1, 512, 2).
// Chunk lengths must be multiples of the total stride (960) so every
// conv sees stride-aligned input; concatenating chunk outputs is then
// bit-identical to the offline Forward on the concatenated input.
func (s *SEANet) ForwardStream(x *g.Tensor, states []*nn.Conv1dStream) *g.Tensor {
	if len(states) != len(s.Convs) {
		panic(fmt.Sprintf("mimi: ForwardStream got %d states, want %d", len(states), len(s.Convs)))
	}
	x = s.Convs[0].ForwardStream(x, states[0])
	for i := 0; i < s.stages(); i++ {
		b1, b2, down := s.Convs[1+3*i], s.Convs[2+3*i], s.Convs[3+3*i]
		h := g.ELU(x)
		h = b1.ForwardStream(h, states[1+3*i])
		h = g.ELU(h)
		h = b2.ForwardStream(h, states[2+3*i])
		x = g.Add(x, h)
		x = g.ELU(x)
		x = down.ForwardStream(x, states[3+3*i])
	}
	x = g.ELU(x)
	last := len(s.Convs) - 1
	return s.Convs[last].ForwardStream(x, states[last])
}
