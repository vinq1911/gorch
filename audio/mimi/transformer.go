//go:build darwin

package mimi

import (
	"fmt"
	"math"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// Layer is one Mimi encoder-transformer layer (plan §4.3): pre-norm
// MHA (8 heads × head_dim 64, RoPE θ=10000, bias-free projections)
// and a bias-free 512→2048→512 exact-GELU MLP, each sublayer gated by
// a learned per-channel layer scale:
//
//	x = x + AttnScale ⊙ Attn(Norm1(x))
//	x = x + MlpScale  ⊙ Fc2(GELUErf(Fc1(Norm2(x))))
//
// There is no final norm after the last layer (verified against the
// checkpoint: zero non-layer encoder_transformer.* keys).
type Layer struct {
	Wq, Wk, Wv, Wo *nn.Linear    // bias tensors stay zero (checkpoint has none)
	Norm1, Norm2   *nn.LayerNorm // eps 1e-5, with bias
	Fc1, Fc2       *nn.Linear    // bias tensors stay zero
	AttnScale      *g.Tensor     // (dim,) self_attn_layer_scale.scale
	MlpScale       *g.Tensor     // (dim,) mlp_layer_scale.scale

	numHeads int
	headDim  int
}

// NewLayer builds a randomly initialized layer with the cfg geometry
// (Load replaces every parameter with checkpoint tensors).
// Inference-only: no parameter has requires-grad set.
func NewLayer(cfg Config) *Layer {
	if cfg.NumHeads*cfg.HeadDim != cfg.HiddenSize {
		panic(fmt.Sprintf("mimi: NumHeads*HeadDim = %d, want HiddenSize %d",
			cfg.NumHeads*cfg.HeadDim, cfg.HiddenSize))
	}
	l := &Layer{
		Wq:        nn.NewLinear(cfg.HiddenSize, cfg.HiddenSize),
		Wk:        nn.NewLinear(cfg.HiddenSize, cfg.HiddenSize),
		Wv:        nn.NewLinear(cfg.HiddenSize, cfg.HiddenSize),
		Wo:        nn.NewLinear(cfg.HiddenSize, cfg.HiddenSize),
		Norm1:     nn.NewLayerNorm(cfg.HiddenSize),
		Norm2:     nn.NewLayerNorm(cfg.HiddenSize),
		Fc1:       nn.NewLinear(cfg.HiddenSize, cfg.Intermediate),
		Fc2:       nn.NewLinear(cfg.Intermediate, cfg.HiddenSize),
		AttnScale: g.Zeros(cfg.HiddenSize),
		MlpScale:  g.Zeros(cfg.HiddenSize),
		numHeads:  cfg.NumHeads,
		headDim:   cfg.HeadDim,
	}
	l.Norm1.Eps = cfg.NormEps
	l.Norm2.Eps = cfg.NormEps
	for _, p := range l.parameters() {
		p.SetRequiresGrad(false)
	}
	// The checkpoint stores no attention/MLP biases; zero the randomly
	// unused-but-added Linear biases so Forward adds exact zeros.
	for _, lin := range []*nn.Linear{l.Wq, l.Wk, l.Wv, l.Wo, l.Fc1, l.Fc2} {
		for i := range lin.Bias.Data() {
			lin.Bias.Data()[i] = 0
		}
	}
	return l
}

func (l *Layer) parameters() []*g.Tensor {
	params := []*g.Tensor{l.AttnScale, l.MlpScale}
	params = append(params, l.Norm1.Parameters()...)
	params = append(params, l.Norm2.Parameters()...)
	for _, lin := range []*nn.Linear{l.Wq, l.Wk, l.Wv, l.Wo, l.Fc1, l.Fc2} {
		params = append(params, lin.Parameters()...)
	}
	return params
}

// attnMask returns the flat (T, T) bool mask (true = masked) for
// causal attention with an optional sliding window: key j is visible
// to query i iff j <= i and (window <= 0 or i-j < window).
//
// window <= 0 selects the plain-causal mask, which is what HF's
// offline Mimi encoder actually applies (its offline sdpa path ignores
// sliding_window — verified empirically, see
// audio/export_mimi_fixtures.py). window = cfg.SlidingWindow (250)
// is the strict Mimi semantics that HF streaming approximates.
func attnMask(T, window int) []bool {
	mask := make([]bool, T*T)
	for i := 0; i < T; i++ {
		row := mask[i*T : (i+1)*T]
		for j := i + 1; j < T; j++ {
			row[j] = true
		}
		if window > 0 {
			for j := 0; j <= i-window; j++ {
				row[j] = true
			}
		}
	}
	return mask
}

// Forward runs the layer offline on x of shape (T, dim). rope is the
// shared encoder RoPE table (RopeLlama, θ=10000), applied to Q and K
// at absolute positions 0..T-1. window <= 0 means plain causal (HF
// offline behavior); window > 0 additionally hides keys older than
// `window` positions.
func (l *Layer) Forward(x *g.Tensor, rope *nn.RoPE, window int) *g.Tensor {
	T := x.Shape()[0]
	nH, hD := l.numHeads, l.headDim
	dim := nH * hD

	// --- Attention sublayer ---
	h := l.Norm1.Forward(x)
	q := l.Wq.Forward(h)
	k := l.Wk.Forward(h)
	v := l.Wv.Forward(h)

	// (T, dim) → (nH, T, hD) for batched per-head matmuls.
	qH := g.Permute(q.Reshape(T, nH, hD), []int{1, 0, 2})
	kH := g.Permute(k.Reshape(T, nH, hD), []int{1, 0, 2})
	vH := g.Permute(v.Reshape(T, nH, hD), []int{1, 0, 2})

	// RoPE on Q and K (not V) at absolute positions.
	qH = rope.Apply(qH, 0)
	kH = rope.Apply(kH, 0)

	// Scores (nH, T, T), scaled by 1/√headDim.
	scores := g.BatchedMatMulTransB(qH, kH, nH, T, T, hD)
	invScale := float32(1.0 / math.Sqrt(float64(hD)))
	scaled := g.Mul(scores, g.Full(invScale, scores.Shape()...))

	// Causal (+ optional sliding-window) mask, tiled over heads.
	base := attnMask(T, window)
	full := make([]bool, nH*T*T)
	for hIdx := 0; hIdx < nH; hIdx++ {
		copy(full[hIdx*T*T:(hIdx+1)*T*T], base)
	}
	masked := g.MaskFill(scaled.Reshape(nH*T, T), full, -1e9)

	probs := g.Softmax(masked).Reshape(nH, T, T)

	// attn @ V → (nH, T, hD) → (T, dim) → output projection.
	attn := g.BatchedMatMul(probs, vH, nH, T, hD, T)
	concat := g.Permute(attn, []int{1, 0, 2}).Reshape(T, dim)
	x = g.Add(x, g.MulB(l.Wo.Forward(concat), l.AttnScale))

	// --- MLP sublayer ---
	m := l.Norm2.Forward(x)
	m = l.Fc1.Forward(m)
	m = g.GELUErf(m)
	m = l.Fc2.Forward(m)
	return g.Add(x, g.MulB(m, l.MlpScale))
}
