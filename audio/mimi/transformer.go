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

// WindowKV is the per-layer streaming K/V cache (plan §4.3): a sliding
// buffer of the last <=window K/V rows per head with their absolute
// positions. RoPE is applied BEFORE caching (rotate-then-cache, the
// same order as HF and as the offline path's absolute positions), so
// eviction is the only maintenance needed.
//
// Retention: after appending S new rows, the next chunk's oldest query
// (position p) may still see key p-window+1, so Append keeps up to
// window-1 past rows before adding the new ones. During attention the
// buffer therefore holds at most window-1+S rows and the strict window
// is enforced by the position mask in ForwardCached — unlike HF
// streaming, which keeps 249 past keys but skips the mask (251
// effective keys), this matches EncodeWindowed exactly.
type WindowKV struct {
	numHeads int
	headDim  int
	window   int

	count     int   // rows currently held (per head)
	positions []int // absolute positions of the held rows

	// (numHeads, count, headDim) tensors holding the cached rows,
	// mutated in place once the steady-state count is reached (the
	// shape only changes while the cache is still filling), so the
	// per-chunk path allocates nothing (plan risk 9).
	kT, vT *g.Tensor
}

// NewWindowKV creates an empty cache for numHeads heads of headDim with
// the given sliding window (>0).
func NewWindowKV(numHeads, headDim, window int) *WindowKV {
	if window <= 0 {
		panic(fmt.Sprintf("mimi: WindowKV requires window > 0, got %d", window))
	}
	return &WindowKV{numHeads: numHeads, headDim: headDim, window: window}
}

// Len returns the number of K/V rows currently held (per head).
func (c *WindowKV) Len() int { return c.count }

// Reset empties the cache for session reuse.
func (c *WindowKV) Reset() {
	c.count = 0
	c.positions = c.positions[:0]
}

// Append adds S freshly RoPE-rotated K/V rows (shape (numHeads, S,
// headDim)) at absolute positions startPos..startPos+S-1, first
// evicting all but the newest window-1 past rows.
func (c *WindowKV) Append(kNew, vNew *g.Tensor, startPos int) {
	shape := kNew.Shape()
	if len(shape) != 3 || shape[0] != c.numHeads || shape[2] != c.headDim {
		panic(fmt.Sprintf("mimi: WindowKV.Append got shape %v, want (%d, S, %d)", shape, c.numHeads, c.headDim))
	}
	S := shape[1]
	keep := c.count
	if maxPast := c.window - 1; keep > maxPast {
		keep = maxPast
	}
	drop := c.count - keep
	newCount := keep + S
	rd := c.headDim // row length

	if c.kT == nil || c.count != newCount {
		// Cache still filling (or chunk size changed): move to freshly
		// shaped tensors. Happens only until steady state, then never
		// again for the session.
		nk, nv := g.Zeros(c.numHeads, newCount, rd), g.Zeros(c.numHeads, newCount, rd)
		if keep > 0 {
			ok, ov := c.kT.Data(), c.vT.Data()
			nkd, nvd := nk.Data(), nv.Data()
			for h := 0; h < c.numHeads; h++ {
				dst := h * newCount * rd
				src := (h*c.count + drop) * rd
				copy(nkd[dst:dst+keep*rd], ok[src:src+keep*rd])
				copy(nvd[dst:dst+keep*rd], ov[src:src+keep*rd])
			}
		}
		c.kT, c.vT = nk, nv
	} else if drop > 0 {
		// Steady state: shift each head block left by drop rows in
		// place (allocation-free).
		kd, vd := c.kT.Data(), c.vT.Data()
		for h := 0; h < c.numHeads; h++ {
			base := h * newCount * rd
			copy(kd[base:base+keep*rd], kd[base+drop*rd:base+c.count*rd])
			copy(vd[base:base+keep*rd], vd[base+drop*rd:base+c.count*rd])
		}
	}

	kd, vd := c.kT.Data(), c.vT.Data()
	nkd, nvd := kNew.Data(), vNew.Data()
	for h := 0; h < c.numHeads; h++ {
		dst := h*newCount*rd + keep*rd
		copy(kd[dst:dst+S*rd], nkd[h*S*rd:(h+1)*S*rd])
		copy(vd[dst:dst+S*rd], nvd[h*S*rd:(h+1)*S*rd])
	}

	c.positions = append(c.positions[:0], c.positions[drop:c.count]...)
	for i := 0; i < S; i++ {
		c.positions = append(c.positions, startPos+i)
	}
	c.count = newCount
}

// ForwardCached is the streaming counterpart of Forward: it processes
// x of shape (S, dim) — S new tokens at absolute positions
// startPos..startPos+S-1 — attending over cache (this layer's
// WindowKV) plus the new tokens, under the same strict sliding-window
// causal rule as Forward with window = cache.window: key position j is
// visible to query position i iff j <= i and i-j < window. The offline
// Forward stays untouched; concatenating ForwardCached outputs over
// chunks reproduces Forward(x, rope, window) on the full sequence.
func (l *Layer) ForwardCached(x *g.Tensor, rope *nn.RoPE, cache *WindowKV, startPos int) *g.Tensor {
	S := x.Shape()[0]
	nH, hD := l.numHeads, l.headDim
	dim := nH * hD

	// --- Attention sublayer ---
	h := l.Norm1.Forward(x)
	q := l.Wq.Forward(h)
	k := l.Wk.Forward(h)
	v := l.Wv.Forward(h)

	qH := g.Permute(q.Reshape(S, nH, hD), []int{1, 0, 2})
	kH := g.Permute(k.Reshape(S, nH, hD), []int{1, 0, 2})
	vH := g.Permute(v.Reshape(S, nH, hD), []int{1, 0, 2})

	// RoPE at absolute positions, then cache (rotate-then-cache).
	qH = rope.Apply(qH, startPos)
	kH = rope.Apply(kH, startPos)
	cache.Append(kH, vH, startPos)

	count := cache.Len()
	kAll, vAll := cache.kT, cache.vT

	// Scores (nH, S, count), scaled by 1/√headDim (same op sequence as
	// Forward for bit-parity).
	scores := g.BatchedMatMulTransB(qH, kAll, nH, S, count, hD)
	invScale := float32(1.0 / math.Sqrt(float64(hD)))
	scaled := g.Mul(scores, g.Full(invScale, scores.Shape()...))

	// Position-based causal + sliding-window mask, tiled over heads.
	base := make([]bool, S*count)
	for i := 0; i < S; i++ {
		qPos := startPos + i
		row := base[i*count : (i+1)*count]
		for j, kPos := range cache.positions {
			row[j] = kPos > qPos || qPos-kPos >= cache.window
		}
	}
	full := make([]bool, nH*S*count)
	for hIdx := 0; hIdx < nH; hIdx++ {
		copy(full[hIdx*S*count:(hIdx+1)*S*count], base)
	}
	masked := g.MaskFill(scaled.Reshape(nH*S, count), full, -1e9)

	probs := g.Softmax(masked).Reshape(nH, S, count)

	attn := g.BatchedMatMul(probs, vAll, nH, S, hD, count)
	concat := g.Permute(attn, []int{1, 0, 2}).Reshape(S, dim)
	x = g.Add(x, g.MulB(l.Wo.Forward(concat), l.AttnScale))

	// --- MLP sublayer ---
	m := l.Norm2.Forward(x)
	m = l.Fc1.Forward(m)
	m = g.GELUErf(m)
	m = l.Fc2.Forward(m)
	return g.Add(x, g.MulB(m, l.MlpScale))
}
