# Plan 0006: native Mimi encoder inference in gorch

**Status:** P0–P7 complete — offline encoder golden-verified,
WAV/resample native, streaming at 8.8 ms/80 ms chunk
(`audio/mimi/stream.go`), the E2E FSDD acceptance gate
(`e2e/mimi_native_fsdd_test.go`) passes with zero Python, and the RVQ
quantizer (`audio/mimi/quantizer.go`) produces discrete Mimi tokens
matching HF's `model.encode(...).audio_codes` exactly (all 32 and 8
codebooks, golden-verified end to end).
**Branch:** `claude/audio-classifier-mimi-gorch-82b2f9` (planning notes;
implementation lands on dedicated feature branches per phase).
**Last updated:** 2026-08-06
**Goal:** run the Mimi (kyutai/mimi) audio-codec ENCODER natively in Go —
frozen-weight inference only — to remove Python from the audio feature
pipeline proven in `doc/mimi-audio-report.md`. Training the encoder is
out of scope. Targets: offline ≤1e-3 relative error vs the transformers
reference, streaming 80 ms chunks bit-equal to offline, <10 ms/chunk on
M4 CPU (Python baseline: 43 ms), FSDD classifier parity (≥96%
speaker-independent with Go-computed embeddings).

## 0. Ground truth established during exploration

### 0.1 What gorch has today (verified in source)

| Capability | Where | Status for this project |
|---|---|---|
| f32 CPU tensors, Accelerate BLAS (`Sgemm`/`SgemmTransA/B`, `VAdd`, `Tanh`, …) | `tensor.go`, `accelerate/accelerate.go` | Usable as-is |
| Conv2d via im2col+sgemm, symmetric zero pad only, **no dilation, no groups, no asymmetric pad** | `conv.go` (`im2col`, `Conv2dForward`), `nn/module.go` (`Conv2d`) | Pattern to copy for Conv1d |
| MHA (`nn/attention.go`) — batched offline path + `ForwardCached` KV path, causal mask, correct 1/√d scaling; **no RoPE inside, no sliding window, Linear always has bias** | `nn/attention.go` | Not directly reusable for Mimi |
| GQA with optional RoPE applied to Q/K, `Permute`/`Reshape`-based head handling — **offline only, no KV-cache path, no sliding window** | `nn/gqa.go` | Best pattern reference for the Mimi attention |
| RoPE with `RopeLlama` (half-split) and `RopeGPTNeoX` styles, arbitrary base, `startPos` offset | `nn/rope.go` | Directly reusable (Mimi = HF `rotate_half` = `RopeLlama`, base 10000) |
| KVCache (append-only per layer, **no sliding-window eviction**) | `nn/kvcache.go` | Needs a windowed variant |
| LayerNorm (2-D `(M,dim)`, eps configurable, default 1e-5) | `nn/layernorm.go` | Reusable as-is |
| GELU — **tanh approximation only** | `ops.go:292` | Mimi uses exact-erf `gelu`; see §2.3 |
| ELU | — | **Missing** |
| Conv1d, causal/asymmetric padding, dilation | — | **Missing** |
| Axis-wise Mean/Max/Std | — | **Missing** (only full-tensor `Sum`/`Mean`, `ops.go:511/532`) |
| WAV I/O, resampler | — | **Missing** (`data/` has MNIST/tabular loaders only) |
| safetensors load (F32/F16/BF16), save, HF download helper | `model/safetensors.go`, `model/download.go`, `model/gpt2_loader.go` | Reusable as-is |
| Everything is `//go:build darwin` (cgo Accelerate) | all files | New files must carry the same tag |

### 0.2 Mimi encoder facts — verified against the live transformers
source and the actual kyutai/mimi safetensors header (350 tensors) +
config.json. Three assumptions from earlier notes are wrong, and each
correction simplifies the work:

1. **No weight-norm fusion is needed for the published checkpoint.**
   The kyutai/mimi safetensors stores plain fused weights —
   `encoder.layers.0.conv.weight [64,1,7]`, `...conv.bias [64]`, etc.
   There are **no** `weight_g`/`weight_v` or
   `parametrizations.weight.original0/1` keys anywhere. The loader
   should still *detect* the g/v patterns and fuse
   (`w = g · v/‖v‖`, norm over dims 1,2 per output channel) as a
   fallback for other checkpoints, but the primary path is a straight
   copy.
2. **Dilation is always 1.** Residual blocks use
   `dilation = dilation_growth_rate**j` for `j in
   range(num_residual_layers)` and `num_residual_layers = 1`, so
   dilation is only ever `2⁰ = 1`. Implement the dilation parameter in
   Conv1d anyway (~5 lines in im2col), but no Mimi path exercises it
   beyond 1.
3. **Mimi's transformer MLP uses exact-erf GELU** (`hidden_act:
   "gelu"` → PyTorch `GELUActivation`, `approximate='none'`), while
   gorch's `GELU` is the tanh approximation. Max abs deviation ≈ 3e-4
   per activation across 8 layers — enough to threaten a ≤1e-3
   end-to-end target. Add an exact `GELUErf` (Go has `math.Erf`).

Verified architecture:

**SEANet encoder** — `encoder.layers` list with ELUs occupying indices
(indices matter for weight names):

| Idx | Layer | Weights |
|---|---|---|
| 0 | CausalConv1d 1→64, k7, s1 | `encoder.layers.0.conv.{weight,bias}` |
| 1 | ResnetBlock(64): ELU → conv 64→32 k3 d1 → ELU → conv 32→64 k1; identity shortcut (`use_conv_shortcut: false`) | `encoder.layers.1.block.{1,3}.conv.{weight,bias}` |
| 2 | ELU (no params) | |
| 3 | CausalConv1d 64→128, k8, s4 | `encoder.layers.3.conv.*` |
| 4 | ResnetBlock(128): 128→64 k3, 64→128 k1 | `encoder.layers.4.block.{1,3}.conv.*` |
| 5 | ELU | |
| 6 | CausalConv1d 128→256, k10, s5 | `encoder.layers.6.conv.*` |
| 7 | ResnetBlock(256) | `encoder.layers.7.block.{1,3}.conv.*` |
| 8 | ELU | |
| 9 | CausalConv1d 256→512, k12, s6 | `encoder.layers.9.conv.*` |
| 10 | ResnetBlock(512) | `encoder.layers.10.block.{1,3}.conv.*` |
| 11 | ELU | |
| 12 | CausalConv1d 512→1024, k16, s8 | `encoder.layers.12.conv.*` |
| 13 | ELU | |
| 14 | CausalConv1d 1024→512, k3, s1 | `encoder.layers.14.conv.*` |

Stride product 4·5·6·8 = 960 → 24 kHz in, 25 Hz out ("encodec frame
rate"). All convs `bias=true`, `pad_mode="constant"` (zeros), causal.

**Causal padding semantics** (`MimiConv1d.forward`):
`kEff = (k-1)·dilation + 1`; `padding_total = kEff - stride`; left pad
= `padding_total`, right pad = `extra_padding` where `extra =
ideal_length - length`, `ideal_length = ceil((length - kEff +
padding_total)/stride)·stride + kEff - padding_total`. The right pad
only fires when the input length isn't stride-aligned; it must be
reproduced exactly for offline parity.

**Encoder transformer** (`encoder_transformer.layers.{0..7}`): 8
layers, d=512, 8 heads, head_dim 64 (`num_key_value_heads` = 8, i.e.
plain MHA), RoPE θ=10000 (HF rotate_half = gorch `RopeLlama`), sliding
window 250, causal, `nn.LayerNorm` eps 1e-5 (with bias), MLP
512→2048→512 **bias-free**, exact GELU, attention projections
**bias-free** (`attention_bias: false`), per-sublayer **layer scale**:
`out = residual + scale ⊙ sublayer(x)` with learned `(512,)` vectors
`self_attn_layer_scale.scale` / `mlp_layer_scale.scale`. **No final
norm** (verified: zero non-layer `encoder_transformer.*` keys). No
positional embedding other than RoPE. `max_position_embeddings: 8000`
(= 320 s at 25 Hz).

**Downsample**: `downsample.conv.weight [512,512,4]` — causal conv
512→512, k4, s2, **no bias**, **`pad_mode="replicate"`** (not zeros!).
25 Hz → 12.5 Hz.

**Quantizer** (deferred, §6):
`quantizer.{semantic,acoustic}_residual_vector_quantizer.{input_proj,output_proj}.weight`
are 1×1 convs 512↔256, no bias, plus per-layer EMA codebooks stored as
`codebook.embed_sum [2048,256]` and `codebook.cluster_usage [2048]` —
the embedding matrix is `embed_sum / clamp(cluster_usage, ε)`, not
stored directly.

**Decision: the classifier pipeline needs only the pre-quantizer
latent** (`audio/export_fsdd_mimi.py` confirms — it never touches the
quantizer). RVQ is Phase 7, optional, only needed for discrete Mimi
tokens (e.g. token-LM work). This defers ~100 tensors and the
EMA-codebook subtlety.

**Streaming chunk math**: 80 ms = 1920 samples = 2·960 → each chunk
yields exactly 2 SEANet frames → 2 transformer tokens → 1 downsampled
frame at 12.5 Hz. Because 1920 is a multiple of every cumulative
stride, `extra_padding = 0` at every conv during streaming, so
streaming and offline agree exactly (bit-for-bit, same summation
order) if left-context caches are seeded the way offline padding
works: zeros for constant-pad convs, first-frame replication for the
replicate-pad downsample.

## 1. Package layout

```
gorch/
├── conv1d.go                      # NEW  core Conv1d op (root package)
├── conv1d_test.go                 # NEW
├── elu.go                         # NEW  ELU + GELUErf
├── reduce_axis.go                 # NEW  MeanAxis / VarAxis / MaxAxis
├── reduce_axis_test.go            # NEW
├── nn/
│   ├── conv1d.go                  # NEW  CausalConv1d module + streaming state
│   └── conv1d_test.go             # NEW
├── audio/                         # becomes a Go package
│   ├── wav.go                     # NEW  WAV reader (PCM 16/24/32 + float32/64)
│   ├── wav_test.go                # NEW
│   ├── resample.go                # NEW  polyphase Kaiser-windowed-sinc resampler
│   ├── resample_test.go           # NEW
│   ├── export_mimi_fixtures.py    # NEW  golden-fixture generator (run once)
│   ├── testdata/                  # NEW  fixture .safetensors + .wav files
│   └── mimi/
│       ├── config.go              # NEW  MimiConfig with kyutai/mimi defaults
│       ├── seanet.go              # NEW  SEANet encoder stack
│       ├── transformer.go         # NEW  Mimi transformer layer (offline + cached)
│       ├── encoder.go             # NEW  SEANet → transformer → downsample
│       ├── stream.go              # NEW  StreamingEncoder (conv caches + windowed KV)
│       ├── load.go                # NEW  checkpoint loader / name map / g·v fusion fallback
│       ├── quantizer.go           # NEW  (Phase 7, optional) RVQ
│       └── *_test.go              # NEW  unit + golden tests
└── e2e/
    └── mimi_native_fsdd_test.go   # NEW  end-to-end parity vs 97% baseline
```

All new `.go` files need `//go:build darwin`. Generic ops (Conv1d,
ELU, axis reductions) go in root/`nn` (already wanted per
`doc/mimi-audio-report.md` §4); everything Mimi-specific stays in
`audio/mimi` so `nn/` stays model-agnostic.

## 2. New ops — signatures, placement, LOC

### 2.1 `conv1d.go` (root package) — ~280 LOC + ~250 test

```go
type PadMode int
const (
    PadConstant PadMode = iota // zeros
    PadReplicate               // repeat edge sample
)

// im2col1d: input (C, L) → col (C*k, outL), with dilation.
func im2col1d(input []float32, C, L, k, stride, dilation int, col []float32)
func col2im1d(col []float32, C, L, k, stride, dilation int, dx []float32) // backward

// Conv1dForward computes 1-D convolution.
//   input:  (batch, inC, L)   weight: (outC, inC, k)   bias: (outC,) or nil
// Padding applied explicitly (asymmetric) before im2col; returns (batch, outC, outL).
func Conv1dForward(input, weight, bias *Tensor, stride, dilation, padLeft, padRight int, mode PadMode) *Tensor
```

Mirror `Conv2dForward` (`conv.go:74`): pad into a reused scratch
`(inC, L+padLeft+padRight)` buffer (replicate mode copies edge values
into the pad region), `im2col1d`, `accelerate.Sgemm(outC, outL,
inC*k, ...)`, fused bias add. Include the autograd branch (col2im1d
for dInput, `SgemmTransB` for dWeight) following the Conv2d pattern —
~60 extra lines, closes the "Conv1d for temporal CNNs" gap; the Mimi
path always runs under `NoGrad`. Skip `groups` (encoder never needs
it; only the *decoder's* upsample uses `upsample_groups=512` — note in
a comment).

Do **not** route Conv1d through Conv2d-with-H=1: Conv2dForward has no
dilation and only symmetric padding; a native 1-D loop is simpler than
working around both.

### 2.2 `nn/conv1d.go` — ~220 LOC

```go
// CausalConv1d reproduces transformers' MimiConv1d semantics.
type CausalConv1d struct {
    Weight   *g.Tensor // (outC, inC, k)
    Bias     *g.Tensor // (outC,) or nil
    Stride, Dilation int
    PadMode  g.PadMode
}
func NewCausalConv1d(inC, outC, k, stride, dilation int, bias bool, mode g.PadMode) *CausalConv1d

// Forward (offline): padTotal = (k-1)*dilation + 1 - stride,
// extraPad per the Mimi formula, pads (padTotal, extraPad), convolves.
func (c *CausalConv1d) Forward(x *g.Tensor) *g.Tensor

// Streaming: caller keeps one Conv1dStream per layer.
type Conv1dStream struct { ctx []float32; primed bool } // (inC, padTotal) left context
func (c *CausalConv1d) ForwardStream(x *g.Tensor, st *Conv1dStream) *g.Tensor
```

`ForwardStream` contract: chunk length must be a multiple of `Stride`;
prepend `ctx` (zeros on first call for `PadConstant`; for
`PadReplicate`, seed with the chunk's first sample repeated — matching
offline left-replicate), convolve with no extra padding, then save the
last `padTotal` *input* columns as next context. Equivalent to HF's
`MimiConv1dPaddingCache`.

Per-layer context sizes for the checkpoint: k7s1→6, res k3→2, res
k1→0, k8s4→4, k10s5→5, k12s6→6, k16s8→8, final k3→2, downsample k4s2→2.

### 2.3 Activations — `elu.go` (root) — ~90 LOC

```go
func ELU(a *Tensor) *Tensor      // alpha=1: x>0 ? x : exp(x)-1  (+autograd)
func GELUErf(a *Tensor) *Tensor  // 0.5*x*(1+erf(x/√2)) — exact; Mimi's "gelu"
```

Follow the `unaryOp`/`SiLU` pattern (`silu.go:13`). CPU loop is fine;
vectorize later via `accelerate` if profiling demands.

### 2.4 Axis reductions — `reduce_axis.go` (root) — ~180 LOC

```go
func MeanAxis(a *Tensor, axis int) *Tensor // shape with axis removed
func VarAxis(a *Tensor, axis int, unbiased bool) *Tensor
func MaxAxis(a *Tensor, axis int) *Tensor
```

Needed so FSDD parity can do mean+std pooling `(T,512) → (1024,)` in
Go exactly as `export_fsdd_mimi.py::pool` (exporter uses
`unbiased=False`). With autograd (grad broadcast back along the axis).

## 3. Audio I/O

### 3.1 `audio/wav.go` — ~230 LOC

```go
type WAV struct { SampleRate int; Channels int; Samples []float32 }
func ReadWAV(path string) (*WAV, error)
func ReadWAVReader(r io.ReadSeeker) (*WAV, error)
func (w *WAV) Mono() []float32 // channel average
```

Pure Go, stdlib only: RIFF chunk walk (tolerate `LIST`/`fact`/junk
before `data`), `fmt ` formats 1 (PCM int 16/24/32; 24-bit is 3-byte
LE sign-extended), 3 (IEEE float32/float64), and 0xFFFE
(WAVE_FORMAT_EXTENSIBLE — dispatch on SubFormat GUID). Normalize ints
to [-1,1) by dividing by 2^(bits-1) — matches `soundfile`'s float32
read used by the Python exporter, so FSDD parity holds.

### 3.2 `audio/resample.go` — ~200 LOC

```go
// Resample converts srIn → srOut via polyphase windowed-sinc (rational ratio).
func Resample(x []float32, srIn, srOut int) []float32
```

Match `scipy.signal.resample_poly` defaults (the 97% baseline was
trained on scipy-resampled audio): reduce `up/down` by gcd; FIR
lowpass `firwin(2*half_len+1, fc)` with `half_len = 10*max(up,down)`,
cutoff `fc = 1/max(up,down)` (normalized to Nyquist), **Kaiser window
β=5.0**, gain-scaled by `up`; zero-phase alignment (scipy trims
`half_len`, output length `ceil(len(x)*up/down)`). Efficient polyphase
evaluation (compute only needed phases — don't literally upsample by
80 for 44.1 kHz). Ratios to test: 8k→24k (×3), 16k→24k (3/2), 48k→24k
(1/2), 44.1k→24k (80/147). Validate against scipy fixtures at rel err
≤1e-4.

## 4. `audio/mimi` — the encoder

### 4.1 `config.go` (~60 LOC)

```go
type Config struct {
    SampleRate, NumFilters, KernelSize, LastKernelSize, ResidualKernelSize int
    UpsamplingRatios []int      // [8,6,5,4]; encoder iterates reversed
    DilationGrowthRate, NumResidualLayers, Compress int
    HiddenSize, NumLayers, NumHeads, HeadDim, Intermediate int
    RopeTheta float32; SlidingWindow int; NormEps float32; MaxPositions int
}
func DefaultConfig() Config // kyutai/mimi values from §0.2
```

### 4.2 `seanet.go` (~200 LOC)

```go
type resnetBlock struct{ conv1, conv2 *nn.CausalConv1d } // ELU→conv k3→ELU→conv k1, identity shortcut
type SEANet struct {
    Convs []*nn.CausalConv1d // 14 convs in checkpoint order (§0.2 table)
}
// Forward: (1, 1, L) → (1, 512, T25). Offline.
// ForwardStream: chunk (1, 1, 1920) with []Conv1dStream → (1, 512, 2)
```

ELU/residual adds are plain `g.ELU` / `g.Add`. Keep a flat conv list
*plus* block structure so the loader can address by HF layer index.

### 4.3 `transformer.go` (~260 LOC)

Don't reuse `nn.TransformerBlock` (fixed 4×dim FFN, tanh-GELU, no
layer scale, no RoPE) or `nn.MultiHeadAttention` (no RoPE). Model the
layer on the `nn.GQA` batched pattern (`Permute`/`Reshape` head
handling, `g.BatchedMatMulTransB` scores):

```go
type Layer struct {
    Wq, Wk, Wv, Wo *nn.Linear      // bias tensors stay zero (checkpoint has none)
    Norm1, Norm2   *nn.LayerNorm   // eps 1e-5
    Fc1, Fc2       *nn.Linear
    AttnScale, MlpScale *g.Tensor  // (512,)
}
// Forward(x (T,512), rope *nn.RoPE, window int) — offline, sliding-window causal mask:
//   key j visible to query i iff i-window < j <= i
// ForwardCached(x, cache *WindowKV, rope, startPos) — streaming, 2 tokens/step
```

Residual wiring: `x = x + AttnScale ⊙ Attn(Norm1(x))`;
`x = x + MlpScale ⊙ Fc2(GELUErf(Fc1(Norm2(x))))`. Layer scale is an
element-wise broadcast multiply over the channel dim (`g.MulB` or a
20-line helper). RoPE: `nn.NewRoPE(64, cfg.MaxPositions /*8000*/,
10000, nn.RopeLlama)` applied to Q and K per head at absolute
positions — the `gqa.go:95` pattern. No final norm after layer 8.

`WindowKV` (~60 LOC): per-layer ring buffer keeping the last ≤250 K/V
rows *plus absolute positions* (RoPE applied pre-cache, so only
eviction is needed — HF does the same: rotate-then-cache). A local
ring buffer avoids touching `nn.KVCache` semantics used by GPT tests.

### 4.4 `encoder.go` (~130 LOC)

```go
type Encoder struct {
    SEANet     *SEANet
    Layers     [8]*Layer
    Rope       *nn.RoPE
    Downsample *nn.CausalConv1d // 512→512 k4 s2, no bias, PadReplicate
    Cfg        Config
}
// Encode offline: pcm 24 kHz mono → (T, 512) latent at 12.5 Hz.
// SEANet (1,512,T25) → transpose (T25,512) → 8 layers → transpose → downsample.
func (e *Encoder) Encode(pcm []float32) *g.Tensor
```

Wrap in `g.NoGrad`. Transposes via `g.Permute` or a flat copy helper.

### 4.5 `stream.go` (~200 LOC)

```go
type Stream struct {
    enc        *Encoder
    convStates []nn.Conv1dStream // one per SEANet conv + downsample (15 total)
    kv         [8]*WindowKV
    pos        int               // absolute 25 Hz frame position (RoPE index)
    dsParity   bool              // downsample stride-2 phase, if chunks ever ≠ 2 frames
}
func (e *Encoder) NewStream() *Stream
// Push accepts exactly 1920 samples (80 ms) and returns one (1, 512) frame,
// bit-identical to the corresponding offline frame.
func (s *Stream) Push(chunk []float32) *g.Tensor
func (s *Stream) Reset()
```

Session cap: `pos < 8000` (320 s), mirroring
`max_position_embeddings`; error past it (same failure mode as HF).

### 4.6 `load.go` (~230 LOC)

```go
func Load(path string) (*Encoder, error)  // path to model.safetensors
func Download(dir string) (string, error) // reuse model/download.go pattern → kyutai/mimi
```

Name map (Go field ← HF key):

| Go destination | HF key(s) |
|---|---|
| `SEANet.Convs[0].Weight/Bias` | `encoder.layers.0.conv.{weight,bias}` |
| resnet block convs | `encoder.layers.{1,4,7,10}.block.{1,3}.conv.{weight,bias}` |
| downsampling convs | `encoder.layers.{3,6,9,12}.conv.{weight,bias}` |
| final conv | `encoder.layers.14.conv.{weight,bias}` |
| `Layers[i].Wq.Weight` etc. | `encoder_transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight` (biases: zeros) |
| `Layers[i].Norm1/Norm2` | `...input_layernorm.{weight,bias}`, `...post_attention_layernorm.{weight,bias}` |
| `Layers[i].Fc1/Fc2.Weight` | `...mlp.fc{1,2}.weight` (biases: zeros) |
| `Layers[i].AttnScale/MlpScale` | `...self_attn_layer_scale.scale`, `...mlp_layer_scale.scale` |
| `Downsample.Weight` | `downsample.conv.weight` (no bias) |

Loader rules: verify every expected key exists with expected shape
(fail loudly, print missing/unexpected); ignore `decoder.*`,
`decoder_transformer.*`, `upsample.*`, `quantizer.*` (until Phase 7).
Fallback fusion: if `X.conv.weight` absent but `X.conv.weight_g` +
`weight_v` (or `parametrizations.weight.original0/1`) exist, compute
`w[o,:,:] = g[o,0,0] * v[o,:,:] / ‖v[o,:,:]‖₂` before assigning.
`nn.Linear` weights are `(out,in)` row-major exactly like HF
`nn.Linear.weight` — direct copy, no transpose (unlike the GPT-2
Conv1D transposition in `gpt2_loader.go`).

## 5. Test strategy and fixtures

### 5.1 Fixture generator — `audio/export_mimi_fixtures.py` (~150 LOC)

Run once against transformers (pin version in a comment; report used
4.57). Deterministic inputs: (a) seeded 2 s chirp+noise at 24 kHz;
(b) one real FSDD clip resampled by the Go-matching pipeline (also
store the 8 kHz original); (c) a 12 s signal (>250 frames at 25 Hz)
to exercise the sliding-window mask offline. Writes
`audio/testdata/mimi_fixtures.safetensors` containing, per input:

| Fixture tensor | Stage | Shape |
|---|---|---|
| `{sig}_pcm` | input | `(L,)` |
| `{sig}_seanet` | `model.encoder(x)` | `(512, T25)` |
| `{sig}_layer0` | after transformer layer 0 (hook) | `(T25, 512)` — first-divergence debugging |
| `{sig}_transformer` | `model.encoder_transformer(...)` | `(T25, 512)` |
| `{sig}_latent` | after `model.downsample` | `(512, T12.5)` |
| `{sig}_stream_latent` | chunked encode, 1920-sample chunks, padding-cache path | `(512, T12.5)` |
| `resample_{8k,16k,44k,48k}_{in,out}` | scipy `resample_poly` pairs | — |
| `{sig}_pooled` | mean+std pool per exporter | `(1024,)` |

Also print `sorted(state_dict.keys())` diffed against the loader's
expectation list.

### 5.2 Test pyramid

1. **Op unit tests**: Conv1d vs naive triple-loop reference across
   stride/dilation/pad combos incl. replicate; Conv1d(k×1) vs
   `Conv2dForward` equivalence; autograd numerical-gradient checks
   (repo convention, see `reshape_batched_grad_test.go`); ELU/GELUErf
   vs closed-form; MeanAxis/VarAxis vs manual.
2. **Golden stage tests** (`audio/mimi/golden_test.go`): load real
   checkpoint (skip unless `MIMI_MODEL` env var or cached download
   exists), assert per-stage vs fixtures. Metric: max relative error
   `|a−b| / (|b| + 1e-5)`. Budgets: post-SEANet ≤1e-4,
   post-transformer ≤5e-4, final ≤1e-3. The `layer0` fixture localizes
   transformer divergence (RoPE convention, mask, layer scale).
3. **Streaming ≡ offline** (`stream_test.go`): 2 s fixture offline vs
   25 × 80 ms `Push`es; max abs diff ≤1e-5 (should be ~exact), and
   vs the Python `stream_latent` fixture at ≤1e-3.
4. **Sliding-window test**: 12 s fixture (300 frames > 250 window) —
   the only case where the window mask changes offline output; catches
   an unmasked implementation that short-clip tests would pass.
5. **Resampler tests**: vs scipy fixture pairs, rel err ≤1e-4; plus
   spectral sanity (3 kHz tone survives 8k→24k; energy above old
   Nyquist < −60 dB).
6. **E2E FSDD parity** (`e2e/mimi_native_fsdd_test.go`, tags
   `darwin && e2e`): given `FSDD_DIR`, run WAV→resample→`Encode`→
   mean+std pool in pure Go for all 3000 clips, then reuse `trainHead`
   from `e2e/mimi_fsdd_test.go`. Accept: digit ≥99%, speaker ≥99%,
   speaker-independent ≥96% (97.0% has run-to-run jitter); also assert
   Go-vs-Python pooled features rel err ≤1e-3 on a subsample — the
   real parity claim.
7. **Benchmarks**: `BenchmarkMimiEncode10s` (target: beat Python's
   334 ms) and `BenchmarkMimiStreamChunk` (target **<10 ms per 80 ms
   chunk** vs Python's 43 ms; expected ~2–6 ms — per-chunk work is
   ~0.1–0.2 GFLOP against Accelerate, dominated by the k16 512→1024
   conv and 8×2-token attention).

## 6. Phasing, dependencies, effort

| Phase | Content | Depends on | Effort |
|---|---|---|---|
| **P0** | Fixture generator + checkpoint `Download` helper + loader key-list dump | — | 0.5–1 d |
| **P1** | Core ops: `conv1d.go`, `nn/conv1d.go` (offline), `elu.go`, `reduce_axis.go` + unit tests | — | 2–3 d |
| **P2** | `audio/mimi`: config, seanet, load (SEANet portion); golden post-SEANet test | P0, P1 | 2 d |
| **P3** | transformer.go (offline, sliding window), downsample, encoder.go; golden post-transformer/latent; offline benchmark | P2 | 2–3 d |
| **P4** | `audio/wav.go` + `audio/resample.go` + tests (parallel with P2–P3) | P0 | 1.5–2 d |
| **P5** | Streaming: `Conv1dStream`, `WindowKV`, `stream.go`; streaming≡offline test; chunk benchmark | P3 | 2–3 d |
| **P6** | E2E FSDD parity + README/report updates | P3, P4 (P5 for latency claims) | 1–1.5 d |
| **P7 (opt)** | `quantizer.go`: semantic VQ + acoustic RVQ encode (`codes = argmin‖proj(x)−embed‖²`, residual loop; embed = `embed_sum/clamp(cluster_usage, ε)` — verify ε in `MimiEuclideanCodebook`, believed 1e-5); golden codes test | P3 | 1–2 d |

Total: **~11–15 working days**. Critical path P0→P1→P2→P3→P5; P4 runs
in parallel.

## 7. Risks and mitigations

1. **Causal pad / `extra_padding` off-by-one** (highest likelihood).
   The ceil-based ideal-length formula must match HF exactly or SEANet
   output length/alignment drifts. Mitigation: dedicated unit test of
   the pad computation against a (L, k, stride) → (padL, padR) table
   generated by the fixture script; FSDD clips have "awkward" lengths
   that exercise `extra_padding > 0`.
2. **Downsample replicate-pad + no-bias** is easy to miss (every other
   conv is constant-pad with bias). It's the last stage, so errors
   show only in the final latent — the post-transformer golden fixture
   isolates it.
3. **GELU flavor**: tanh-approx `GELU` gives ~3e-4/layer drift ×8
   layers. Use `GELUErf`; keep a test asserting the two differ
   (guards against accidental substitution).
4. **RoPE convention**: HF `rotate_half` = gorch `RopeLlama` (verified
   in `nn/rope.go`); the `layer0` fixture catches a mixup immediately.
5. **Sliding window silently unimplemented**: all short-clip tests
   pass without it. The 12 s fixture is mandatory.
6. **Streaming replicate-cache seeding**: for the downsample conv, the
   first chunk's left context must replicate the first frame of that
   chunk (mirroring offline left-replicate), not zeros. Covered by the
   streaming-vs-offline exactness test.
7. **Weight-norm fusion**: *not needed* for kyutai/mimi (plain
   `conv.weight` keys, verified). Implement the fallback but don't let
   it complicate the primary loader; fail loudly if both key styles
   are absent.
8. **Numerical drift through 14 convs + 8 layers**: different f32
   summation order vs PyTorch. LayerNorms re-normalize per layer, so
   drift doesn't compound geometrically; per-stage budgets
   (1e-4 → 5e-4 → 1e-3) leave margin. First suspects if exceeded: conv
   accumulation order, softmax (gorch subtracts row max — same as
   PyTorch, fine).
9. **Performance**: offline 10 s = one 250-token 8-layer d=512
   transformer + convs — well under Python's 334 ms with Accelerate.
   Streaming per-chunk allocation churn is the main p95 risk; if
   needed, reuse scratch buffers in `Stream` (repo has
   `AcquireFloat32`/`ReleaseFloat32` pooling used in `ops.go`). Keep
   everything CPU: per-op shapes are far below the Metal crossover
   (~512M FMAs, `ops.go:568`).
10. **Resampler mismatch vs scipy** shifts FSDD features slightly.
    Matching `resample_poly`'s exact Kaiser design (§3.2) plus the
    ≤1e-4 fixture test bounds this; classifier tolerance has slack.

## Critical files for implementation

- `conv.go` — the im2col+Sgemm+autograd pattern `conv1d.go` must mirror
- `nn/gqa.go` — the RoPE-integrated batched attention pattern for `audio/mimi/transformer.go`
- `nn/rope.go` — reused directly (RopeLlama, θ=10000, startPos offsets for streaming)
- `model/safetensors.go` — checkpoint ingestion used by `audio/mimi/load.go`
- `e2e/mimi_fsdd_test.go` — the `trainHead` harness and accuracy baseline the parity test must reproduce
