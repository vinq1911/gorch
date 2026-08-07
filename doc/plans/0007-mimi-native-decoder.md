# Plan 0007: native Mimi decoder inference in gorch

**Status:** COMPLETE — D0–D4 all done (2026-08-07). Native decoder
ships in `audio/mimi` (offline ~120 dB SNR vs HF, streaming
9.5 ms/chunk); the real-world round-trip evidence is regenerated with
the NATIVE decoder (30/30 whisper-verified,
`audio/testdata/realworld/native_roundtrip_transcripts.tsv`, native
WAVs committed) and `roundtrip_decode.py` is demoted to
cross-check-only.
**Predecessor:** `doc/plans/0006-mimi-native-encoder.md` (P0–P7 complete: encoder, streaming, RVQ quantizer all golden-verified). This plan is the second half of that project: it removes the last Python step in the production path — `audio/realworld/roundtrip_decode.py`, the reference Mimi decoder used only to prove Go-produced tokens decode to intelligible speech.
**Goal:** run the Mimi (kyutai/mimi) audio-codec DECODER natively in Go — frozen-weight inference only — so speech tokens produced in Go can be turned into audible 24 kHz waveforms in Go. Targets: offline waveform parity vs the transformers reference (gates in §5.3), streaming 80 ms of audio out per token, <10 ms/chunk on M4 CPU (encoder streaming achieved 8.8 ms), the real-world round-trip evidence (30/30 whisper-verified reconstructions) regenerated with the NATIVE decoder, and `roundtrip_decode.py` demoted to cross-check-only.
**Last updated:** 2026-08-07

## 0. Ground truth established during exploration

Everything below was verified against the LIVE transformers source
(`.../site-packages/transformers/models/mimi/modeling_mimi.py`)
and the actual checkpoint header
(`~/.cache/huggingface/hub/models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7/model.safetensors`,
350 tensors, all F32) plus its `config.json`. Do not re-derive from
memory; layer indices and shapes below are copied from the header.

### 0.1 What gorch has today (verified in source)

| Capability | Where | Status for this project |
|---|---|---|
| `Conv1dForward` — im2col+Sgemm, dilation, asymmetric pad, `PadConstant`/`PadReplicate`, autograd, pooled scratch (`AcquireFloat32`) | `conv1d.go` | Reused as-is for the decoder's plain convs |
| `col2im1d(col, C, L, k, stride, dilation, dx)` — strided scatter-add | `conv1d.go:63` | **The exact primitive ConvTranspose1d forward needs** (see §2.1) |
| `nn.CausalConv1d` + `CausalPad1d` + `Conv1dStream` (left-context streaming cache, replicate seeding) | `nn/conv1d.go` | Reused as-is for decoder SEANet plain convs (all stride 1) |
| Mimi transformer `Layer` — RoPE θ=10000 (`RopeLlama`), bias-free projections, exact-erf GELU, layer scale, eps 1e-5 LayerNorm w/ bias, `Forward(x, rope, window)` + `ForwardCached(x, rope, cache, startPos)` + `WindowKV` | `audio/mimi/transformer.go` | **Reused as-is** for the decoder transformer (§0.2.3) — only the loader prefix changes |
| `Quantizer.Decode(codes) → (T, 512)` latent — sum of semantic+acoustic output projections, exactly HF `quantizer.decode` | `audio/mimi/quantizer.go:171` | **The decoder's input stage already exists and is golden-verified** (`{sig}_quantized` fixtures) |
| `ELU`, `GELUErf`, `MulB`, `Permute`, `BatchedMatMul[TransB]`, `MaskFill`, `Softmax` | root pkg | Reused |
| Encoder `Stream` pattern (conv states + WindowKV + pos), `transposeCT`/`transposeTC` | `audio/mimi/stream.go`, `encoder.go` | Pattern to mirror for `DecodeStream` |
| safetensors load, fail-loudly loader pattern (`take`, consumed-key sweep, `convWeight` weight-norm fallback) | `audio/mimi/load.go` | Extended, not changed |
| `audio.ReadWAV`, `Resample` | `audio/wav.go`, `audio/resample.go` | **`WriteWAV` is missing** — needed for artifacts & whisper (§3) |
| ConvTranspose1d (any form), grouped convolutions | — | **Missing** — the only genuinely new ops |
| Everything `//go:build darwin` | all files | New files must carry the same tag |

### 0.2 Mimi decoder facts — verified

The decode path (`MimiModel._decode_frame`, modeling_mimi.py:1595) is,
in order:

```
codes (B, nq, T12.5)
  → quantizer.decode                → (B, 512, T12.5)   [EXISTS in Go]
  → upsample (ConvTranspose1d)      → (B, 512, T25)
  → decoder_transformer (T-major)   → (B, T25, 512) → transpose
  → decoder (SEANet)                → (B, 1, L24k)
```

Note the order: **upsample comes BEFORE the decoder transformer** — the
transformer runs at 25 Hz, exactly mirroring the encoder where the
transformer runs at 25 Hz before the 25→12.5 Hz downsample.

**0.2.1 upsample** (`MimiModel.__init__`, line 1417):
`MimiConvTranspose1d(512→512, kernel=2·int(25/12.5)=4, stride=2,
bias=False, groups=config.upsample_groups=512)` — i.e. **depthwise**
(one 4-tap transposed filter per channel). Checkpoint:
`upsample.conv.weight [512, 1, 4]` — PyTorch ConvTranspose1d weight
layout is **(inC, outC/groups, k)**, NOT the Conv1d (outC, inC, k).
**No bias key exists** (verified). 12.5 Hz → 25 Hz, L→2L exactly.

**0.2.2 MimiConvTranspose1d semantics** (lines 344–399) — the
highest-risk item, exact math:

- `padding_total = kernel_size - stride`
- causal (`use_causal_conv: true`) → `padding_right =
  ceil(padding_total * trim_right_ratio)`; with **`trim_right_ratio:
  1.0`** (config, verified) → `padding_right = padding_total`,
  `padding_left = 0`.
- `forward`: `y = conv_transpose1d(x)` (padding=0, so `len(y) =
  (L-1)·stride + k`), then trim: `y[..., 0 : L·stride]`.

So **every causal ConvTranspose maps L → exactly L·stride** — no
`extra_padding` analogue (HF trims final excess only via
`padding_mask` in `decode()`). Simpler than the encoder's
`CausalPad1d` ceil formula. No pad_mode — transposed convs never pad
input.

**0.2.3 decoder_transformer**: `MimiTransformerModel(config)` — the
**same class and same config object** as `encoder_transformer` (line
1427 vs 1401). Checkpoint confirms bit-for-bit structural identity:
`decoder_transformer.layers.{0..7}` carry exactly the same 12-key set
per layer with identical shapes as the encoder and **zero non-layer
`decoder_transformer.*` keys** — no final norm, no biases, plain MHA,
RoPE θ=10000, sliding window 250, max positions 8000.
**`audio/mimi/transformer.go`'s `Layer` is reusable unchanged**,
including `ForwardCached`/`WindowKV`. The 0006 sliding-window caveat
carries over identically: HF's offline path applies a **plain causal**
mask; only the streaming KV path evicts at 250. Go must ship `Decode`
(plain causal, HF-offline parity) and `DecodeWindowed` (strict window,
the streaming reference), mirroring `Encode`/`EncodeWindowed`.

**0.2.4 SEANet decoder** (`MimiDecoder`, lines 1150–1180).
Checkpoint-verified indices/shapes (ratios iterated in config order
`[8, 6, 5, 4]` — NOT reversed like the encoder):

| Idx | Layer | Checkpoint keys (shapes verified) |
|---|---|---|
| 0 | CausalConv1d 512→1024, k7, s1 | `decoder.layers.0.conv.weight [1024,512,7]`, `.bias [1024]` |
| 1 | ELU | — |
| 2 | ConvTranspose 1024→512, k16, s8 | `decoder.layers.2.conv.weight [1024,512,16]` (inC,outC,k!), `.bias [512]` |
| 3 | ResnetBlock(512): ELU→conv 512→256 k3 d1→ELU→conv 256→512 k1; identity shortcut | `decoder.layers.3.block.{1,3}.conv.*` |
| 4 | ELU | — |
| 5 | ConvTranspose 512→256, k12, s6 | `decoder.layers.5.conv.weight [512,256,12]`, `.bias [256]` |
| 6 | ResnetBlock(256) | `decoder.layers.6.block.{1,3}.conv.*` |
| 7 | ELU | — |
| 8 | ConvTranspose 256→128, k10, s5 | `decoder.layers.8.conv.weight [256,128,10]`, `.bias [128]` |
| 9 | ResnetBlock(128) | `decoder.layers.9.block.{1,3}.conv.*` |
| 10 | ELU | — |
| 11 | ConvTranspose 128→64, k8, s4 | `decoder.layers.11.conv.weight [128,64,8]`, `.bias [64]` |
| 12 | ResnetBlock(64) | `decoder.layers.12.block.{1,3}.conv.*` |
| 13 | ELU | — |
| 14 | CausalConv1d 64→1, k3, s1 | `decoder.layers.14.conv.weight [1,64,3]`, `.bias [1]` |

125 decoder-side keys total. Details that matter:

- All SEANet ConvTranspose layers have **bias=true**; the upsample is
  the only bias-free transposed conv.
- Plain convs (idx 0, resnets, idx 14) are ordinary causal `MimiConv1d`
  stride 1 (`padding_total = k−1`, `extra_padding = 0` always) — exactly
  `nn.CausalConv1d`. Dilation always 1. Identity shortcuts.
- Resnet blocks come **after** the ConvTranspose in each stage
  (encoder had resnet before the strided conv).
- No weight-norm keys anywhere — plain fused weights; keep the
  `convWeight` g/v fallback for plain convs only.

**0.2.5 Output length semantics**: T tokens → upsample 2T → SEANet
strides 8·6·5·4 = 960 → **exactly T·1920 samples**. Decode returns
1920·T ≥ original L; HF trims to L via `padding_mask`. Go's `Decode`
returns full 1920·T; callers trim. Tests assert the exact-1920·T
property.

**0.2.6 Streaming**: HF has **no streaming decode for the convs**
(only `decoder_past_key_values` for the transformer). Conv/ConvT
streaming state is new territory designed in §4; the Go streaming
reference is `DecodeWindowed` (Go-vs-Go bit-parity) plus the offline
windowed fixtures.

**0.2.7 Streaming chunk math**: 1 token (12.5 Hz) → 2 transformer
frames (25 Hz) → **1920 samples = 80 ms of audio per token**. All
decoder plain convs are stride 1 (any chunk aligned); ConvTranspose
needs no alignment. Session cap: `pos < 8000` (320 s).

## 1. Package layout

```
gorch/
├── convtranspose1d.go             # NEW  core ConvTranspose1d op (root package)
├── convtranspose1d_test.go        # NEW
├── nn/
│   ├── convtranspose1d.go         # NEW  CausalConvTranspose1d module + ConvT1dStream
│   └── convtranspose1d_test.go    # NEW
├── audio/
│   ├── wav.go                     # EXTEND  add WriteWAV (PCM16 mono)
│   ├── wav_test.go                # EXTEND  round-trip Write→Read test
│   ├── export_mimi_fixtures.py    # EXTEND  decoder golden stages (§5.1)
│   ├── testdata/
│   │   ├── mimi_decoder_fixtures.safetensors  # NEW (separate file; §5.1 budget)
│   │   └── mimi_decoder_keys.txt              # NEW loader manifest
│   └── mimi/
│       ├── decoder.go             # NEW  Decoder: upsample → transformer → SEANetDecoder
│       ├── decoder_test.go        # NEW  golden stage tests
│       ├── decode_stream.go       # NEW  DecodeStream (token → 80 ms PCM)
│       ├── decode_stream_test.go  # NEW
│       └── load.go                # EXTEND  LoadDecoder / LoadFull + key map
├── audio/realworld/
│   ├── README.md                  # UPDATE  native decode is the production path
│   └── roundtrip_decode.py        # UPDATE docstring: cross-check only
└── e2e/
    └── mimi_realworld_test.go     # EXTEND  native round-trip evidence (§7)
```

All new `.go` files carry `//go:build darwin`.

## 2. New ops — signatures, placement, LOC

### 2.1 `convtranspose1d.go` (root package) — ~180 LOC + ~200 test

```go
// ConvTranspose1dForward computes 1-D transposed convolution
// (matching PyTorch ConvTranspose1d with padding=0, output_padding=0,
// dilation=1):
//
//	input:  (batch, inC, L)
//	weight: (inC, outC/groups, k)   — PyTorch ConvTranspose layout,
//	                                  NOT Conv1d's (outC, inC, k)
//	bias:   (outC,) or nil
//
// Returns (batch, outC, (L-1)*stride + k). No padding or trimming —
// callers trim. inC and outC must be divisible by groups.
// Inference-only: no autograd graph; panics if grad enabled and any
// input requires grad (decision documented: frozen-weight decoder,
// ConvTranspose autograd out of scope).
func ConvTranspose1dForward(input, weight, bias *Tensor, stride, groups int) *Tensor
```

Per batch, per group g: `col(outCg·k, L) = Wg^T @ Xg` via
`accelerate.SgemmTransA`, then scatter-add with the **existing
`col2im1d`** — transposed-conv forward IS col2im, which conv1d.go
ships for its backward. Bias fused afterward. Depthwise fast path
(`groups == inC == outC`, the upsample): direct per-channel loop, no
GEMM. Decision: implement general groups + depthwise special case;
`Conv1dForward` keeps its no-groups stance. Scratch via
`AcquireFloat32`/`ReleaseFloat32`.

Unit tests: naive triple-loop reference across (k, stride, groups) ∈
{(4,2,1), (4,2,C), (16,8,1), (12,6,1), (3,1,1), (8,4,2)};
PyTorch-generated small-case fixtures from D0 pin the weight layout;
depthwise ≡ general equivalence.

### 2.2 `nn/convtranspose1d.go` — ~170 LOC + ~150 test

```go
// CausalConvTranspose1d reproduces MimiConvTranspose1d with
// use_causal_conv=true, trim_right_ratio=1.0: convolve, trim k-stride
// from the right, 0 from the left: (1, inC, L) → (1, outC, L*stride).
type CausalConvTranspose1d struct {
    Weight *g.Tensor // (inC, outC/groups, k)
    Bias   *g.Tensor // (outC,) or nil
    Stride, Groups int
}
func NewCausalConvTranspose1d(inC, outC, k, stride, groups int, bias bool) *CausalConvTranspose1d
func (c *CausalConvTranspose1d) Forward(x *g.Tensor) *g.Tensor

// ConvT1dStream holds the streaming overlap-add tail: the last
// k-stride output columns of the previous chunk's raw (untrimmed,
// bias-free) convolution.
type ConvT1dStream struct {
    tail []float32 // (outC, k-stride) pending partial sums
    ext  []float32 // reused output scratch
}
func (s *ConvT1dStream) Reset()
func (c *CausalConvTranspose1d) ForwardStream(x *g.Tensor, st *ConvT1dStream) *g.Tensor
```

`ForwardStream` contract: for S input frames, compute raw transposed
conv **without bias** → `(outC, (S-1)·stride + k)`; add stored tail
into the first k-stride columns; **emit the first S·stride columns
with bias added at emission**; store the last k-stride columns
(bias-free) as the new tail. First chunk: zero tail (correct because
padding_left = 0). Bias-at-emission prevents double-counting.
Streaming ≡ offline enforced over random chunk splits.

### 2.3 `audio/wav.go` extension — ~60 LOC

```go
// WriteWAV writes mono float32 as 16-bit PCM, clipping exactly like
// roundtrip_decode.py (np.clip(wav*32768,...).astype(np.int16)) so
// whisper hears bit-identical audio from Go and Python paths.
func WriteWAV(path string, sampleRate int, samples []float32) error
```

## 3. `audio/mimi` — the decoder

### 3.1 `decoder.go` — ~260 LOC

```go
type SEANetDecoder struct {
    Init  *nn.CausalConv1d               // 512→1024 k7 s1
    Ups   [4]*nn.CausalConvTranspose1d   // k16s8, k12s6, k10s5, k8s4
    Res   [4][2]*nn.CausalConv1d         // per-stage resnet convs (k3, k1)
    Final *nn.CausalConv1d               // 64→1 k3 s1
}
// Forward: (1, 512, T25) → (1, 1, T25*960). Per stage: ELU → ConvT →
// resnet(x + conv2(ELU(conv1(ELU(x))))); then ELU → Final.

type Decoder struct {
    Upsample *nn.CausalConvTranspose1d // 512→512 k4 s2 groups=512 no bias
    Layers   [8]*Layer                 // decoder_transformer — same Layer type
    Rope     *nn.RoPE
    SEANet   *SEANetDecoder
    Cfg      Config
}
func NewDecoder(cfg Config) *Decoder
func (d *Decoder) DecodeLatent(latent *g.Tensor) []float32          // HF-offline parity (plain causal)
func (d *Decoder) DecodeLatentWindowed(latent *g.Tensor) []float32  // strict 250 window (streaming ref)
func (d *Decoder) Decode(q *Quantizer, codes [][]int) []float32     // codes→PCM convenience
```

All under `g.NoGrad`. `Config` needs no new fields (upsample geometry
hardcoded k4 s2 groups=HiddenSize with a comment, like the encoder's
downsample).

### 3.2 `load.go` extension — ~180 LOC

```go
func LoadDecoder(path string) (*Decoder, error)
func LoadFull(path string) (*Encoder, *Quantizer, *Decoder, error) // one parse
```

| Go destination | HF key(s) | Shape |
|---|---|---|
| `Decoder.Upsample.Weight` | `upsample.conv.weight` (no bias) | [512, 1, 4] |
| `Layers[i].*` | `decoder_transformer.layers.{i}.*` (same 12-key set as encoder) | as encoder |
| `SEANet.Init.*` | `decoder.layers.0.conv.{weight,bias}` | [1024,512,7]/[1024] |
| `SEANet.Ups[s].*` | `decoder.layers.{2,5,8,11}.conv.{weight,bias}` | (inC,outC,k) layout |
| `SEANet.Res[s][j].*` | `decoder.layers.{3,6,9,12}.block.{1,3}.conv.{weight,bias}` | as table §0.2.4 |
| `SEANet.Final.*` | `decoder.layers.14.conv.{weight,bias}` | [1,64,3]/[1] |

Rules: refactor the transformer-layer loop into a shared
`loadTransformerLayers(take, prefix, layers)` helper (behavior-
preserving, encoder tests guard it); every decoder-side key consumed
or fail loudly; g/v fallback for plain convs only (transposed
weight-norm fusion out of scope — fail loudly); negative tests; fix
stale "decoder never" comments; `LoadFull` asserts consumed-key union
covers all 350 tensors.

## 4. Streaming decode — `decode_stream.go` (~180 LOC)

```go
type DecodeStream struct {
    dec *Decoder; q *Quantizer
    upState   *nn.ConvT1dStream
    kv        [8]*WindowKV
    pos       int
    initState *nn.Conv1dStream
    upStates  [4]*nn.ConvT1dStream
    resStates [4][2]*nn.Conv1dStream
    finState  *nn.Conv1dStream
}
func (d *Decoder) NewStream(q *Quantizer) *DecodeStream
func (s *DecodeStream) Push(codes []int) []float32       // 1 token → 1920 samples
func (s *DecodeStream) PushLatent(latent *g.Tensor) []float32
func (s *DecodeStream) Reset()
```

Per push: q.Decode column → (1,512) → Upsample.ForwardStream (2
frames; tail 2) → 8 × ForwardCached at pos (S=2) → SEANet stream
(Init ctx 6; ConvT tails 8/6/5/4; resnet ctx 2/0; Final ctx 2) → 1920
samples. Acceptance: concatenated Push ≡ DecodeLatentWindowed
(plain-causal DecodeLatent diverges past 250 frames — same caveat as
encoder). PushN free generalization; Push(1) is the live-voice
contract. Cap pos < 8000.

## 5. Fixtures and test strategy

### 5.1 Fixture-generator extension (+~120 LOC)

Extend `audio/export_mimi_fixtures.py` (same pinned versions;
encoder file untouched) to write
**`audio/testdata/mimi_decoder_fixtures.safetensors`** (separate
file). Inputs: existing signals' `{sig}_codes8` + 3 representative
real-world clips' committed tokens (`zero_alloy`, `five_echo`,
`nine_shimmer`). Stages via forward hooks:

| Fixture tensor | Stage | Shape (chirp) |
|---|---|---|
| `{sig}_dec_upsampled` | `model.upsample(quantized)` | (512, 50) |
| `chirp_dec_layer0` only | after decoder_transformer layer 0 | (50, 512) |
| `{sig}_dec_transformer` | decoder_transformer output | (50, 512) |
| `{sig}_dec_seanet0` | after `decoder.layers.0` | (1024, 50) |
| `chirp_dec_stage1` only | after `decoder.layers.3` | (512, 400) |
| `chirp_dec_stage2` only | after `decoder.layers.6` | (256, 2400) |
| `{sig}_dec_wav` | final waveform | (48000,) |
| `long_dec_transformer_win` | offline with explicit window mask | (300, 512) |
| `long_dec_wav_win` | full decode under windowed transformer | (288000,) |
| `rw_{clip}_dec_wav` ×3 | reference decode of committed rw tokens | (1920·T,) |

Deeper SEANet stages deliberately excluded (size; same code path
covered). **Budget ≤10 MB** — actual D0 file: **9.61 MB, 40 tensors**.
D0 size trim: `dec_layer0`/`dec_stage1` are chirp-only like
`dec_stage2` (the full `{sig}_` set would be ~15.2 MB); the long
signal's unique value — >250-frame window behavior — is fully covered
by `long_dec_transformer{,_win}`/`long_dec_wav{,_win}`, and SEANet
stages are time-invariant convs whose code path chirp covers. Also
dump `mimi_decoder_keys.txt` (done: 125 keys), small random
ConvTranspose1d layout-pinning cases `ct_{i}_{in,w,b,out}` +
`ct_manifest` (input/weight/output triples), and **measure the Python
decode baseline** (offline 12 s clip wall time + per-token latency) —
recorded into §7 and the generator's D0 section comment.

### 5.2 Test pyramid

1. **Op unit tests**: naive-reference sweep; PyTorch layout fixtures;
   depthwise≡general; trim L→L·stride for all five Mimi geometries;
   streaming≡offline over random chunkings; first-chunk zero-tail;
   bias-once property (constant input ⇒ constant interior).
2. **Golden stage tests** (`decoder_test.go`): real checkpoint,
   `stageCheck` + `runGolden` recompute-once discipline verbatim from
   encoder_test.go. Stage input = Go `Quantizer.Decode` of fixture
   codes (already exact) so errors attribute to decoder code alone.
3. **Streaming tests**: Push-concat ≡ DecodeLatentWindowed ≤1e-5 on
   chirp; the long signal (300 frames > 250) mandatory; vs
   `long_dec_wav_win` at the §5.3 gate; Reset ≡ fresh.
4. **Length semantics**: `len == 1920·T` for T ∈ {1, 25, 150};
   encode→decode of non-aligned clip returns 1920·ceil ≥ L.
5. **Round-trip smoke**: random latent → finite bounded output;
   out-of-range codes panic.
6. **E2E** and **benchmarks** (§7).

### 5.3 Tolerance policy

Per-stage: 0006 precedent (plan metric restricted to |b|≥1e-2 +
mixed tolerance + recompute-once):

| Stage | relBig gate | mixed absTol + relTol·\|b\| |
|---|---|---|
| dec_upsampled | 1e-4 | 5e-6 + 1e-4 |
| dec_layer0 | 2.5e-4 | 5e-6 + 1e-4 |
| dec_transformer | 1e-4 | 5e-6 + 1e-4 |
| dec_seanet0 | 1e-4 | 5e-6 + 1e-4 |
| dec_stage1/2 | 2.5e-4 | 1e-5 + 2.5e-4 |
| dec_wav | — | two-part gate below |

**Waveform gate** (no LayerNorm near the output → error grows through
~19 unnormalized conv layers; pure relative brittle on near-zero
samples, pure absolute hides loud-sample blowups). Two-part, both
enforced:

1. **Primary: per-clip SNR ≥ 60 dB**
   (`10·log10(Σref²/Σ(go−ref)²)`). f32-parity implementations land
   80–100 dB; 60 dB = rms error ≤0.1% of signal — inaudible, while
   masked bugs (wrong trim, swapped taps, double bias) sit at
   −20…−40 dB and fail immediately. Scale-aware, length-normalized,
   audio-meaningful → authoritative.
2. **Secondary: sample-level mixed `|a−b| ≤ 5e-5 + 1e-3·|b|`**
   (provisional) — catches isolated spikes SNR averages away. D2 must
   measure actual headroom and reset both constants to ~10× measured
   maxima, documenting measurements in the test comment (0006's
   calibrate-and-document step).

`rw_*_dec_wav` decodes use the same two-part gate and cross-check the
exact tensors `roundtrip_decode.py` fed to whisper.

## 6. Phasing, dependencies, effort

| Phase | Content | Depends on | Effort |
|---|---|---|---|
| **D0** | Fixture-generator extension (decoder stages, rw decodes, ConvT layout cases, key manifest); run once; record Python decode baselines into this plan + benchmark comments | — | 0.5–1 d |
| **D1** | `convtranspose1d.go` (general groups + depthwise fast path), `nn/convtranspose1d.go` (Forward + ConvT1dStream), `audio.WriteWAV`; full op unit tests | D0 | 2–3 d |
| **D2** | `decoder.go`, `load.go` extension (+ shared transformer-layer loader refactor), golden stage tests, waveform-gate calibration, `BenchmarkMimiDecode10s` | D0, D1 | 2–3 d |
| **D3** | `decode_stream.go`, streaming≡DecodeWindowed tests (incl. >250 frames), `BenchmarkMimiDecodeStreamChunk` | D2 | 1.5–2 d |
| **D4** | E2E swap: native round-trip regeneration → whisper → `native_roundtrip_transcripts.tsv`; audible artifacts committed; README/plan updates; `roundtrip_decode.py` demoted to cross-check | D2 (D3 for latency claims) | 1–1.5 d |

Total **~7–10 working days**. Critical path D0→D1→D2→D3→D4.

## 7. End-to-end acceptance and performance

- New `TestMimiRealWorldNativeRoundtrip` (darwin && e2e): decode the
  committed 30-clip `tokens.safetensors` natively; under
  `GORCH_MIMI_WRITE_DECODED=1` write WAVs to
  `audio/testdata/realworld/native_roundtrip/` and skip with
  instructions (mirroring the token-writing flow); normal run asserts
  the committed `native_roundtrip_transcripts.tsv` covers 30 clips
  (30/30 expected, gate ≥80%) and per-clip SNR ≥ 40 dB vs the 3
  committed `rw_*_dec_wav` reference waveforms.
- `TestMimiRealWorldTokenProduction` keeps its exact-token assertion;
  intelligibility evidence switches to the native TSV. Python TSV +
  `roundtrip_decode.py` stay as the independent cross-check.
- **Audible artifact**: commit 3 native reconstructions (one per
  voice) under `audio/testdata/realworld/native_roundtrip/`.
- `BenchmarkMimiDecode10s`: 125 tokens → 240 000 samples; Python
  baseline (measured in D0, 2026-08-07, transformers 4.57.1 /
  torch 2.9.1, CPU, Apple M4): offline `model.decode` of the long
  signal's 150 tokens (12 s audio) = **466.9 ms best-of-3**
  (runs 535.2 / 466.9 / 490.2 ms; ≈3.1 ms/token, ≈389 ms per 10 s);
  target ≥5× faster.
- `BenchmarkMimiDecodeStreamChunk`: one Push (1 token → 80 ms);
  target **<10 ms** (encoder: 8.8 ms → full-duplex stays <20 ms per
  80 ms, ~4× real-time headroom). Python per-token baselines
  (measured in D0, same setup, one token per call over all 150
  tokens): KV-cache incremental decode (`decoder_past_key_values` =
  `DynamicCache`; HF streams only the transformer KV, not conv
  state) = **mean 33.95 ms / median 31.88 ms / p95 44.73 ms per
  token**; full-prefix re-decode fallback = **mean 260.96 ms /
  median 240.91 ms per token** (433.3 ms at T=150).

## 8. Risks, ranked, with mitigations

1. **ConvTranspose weight layout + trim math** (highest). Mitigation:
   D0 PyTorch layout fixtures; four non-square SEANet transposes make
   the loader shape check catch transposition; dec_stage1 golden
   localizes the first ConvT.
2. **Streaming overlap-add state**: bias double-count, tail k−stride
   off-by-one, emission misalignment. Mitigation: bias-at-emission
   design; random-chunking streaming≡offline test; Push-concat ≡
   DecodeWindowed ≤1e-5.
3. **No-final-norm error accumulation**. Mitigation: two-part SNR+
   mixed gate with D2 calibration; per-stage fixtures bisect drift.
4. **Depthwise groups fast path diverging**. Mitigation: equivalence
   unit test; dec_upsampled is the first golden stage.
5. **Window/offline divergence** (encoder caveat mirrored).
   Mitigation: long_dec_*_win fixtures mandatory; test comments state
   the reference explicitly.
6. **Length off-by-ones**. Mitigation: exact-length assertions; e2e
   trims to min(len) before SNR.
7. **Loader stale assumptions / double-claimed keys**. Mitigation:
   LoadFull asserts consumed-key union covers all 350 tensors.
8. **Performance (memory-bound scatter-add at 24 kHz)**. Mitigation:
   pooled scratch; stride-1 col2im memmove variant if profiling
   demands; CPU only.
9. **Fixture size creep**. Budget ≤10 MB, deep stages excluded.

## Critical files for implementation

- `conv1d.go` — col2im1d (the ConvTranspose forward primitive), scratch/GEMM patterns
- `nn/conv1d.go` — CausalConv1d/Conv1dStream reused; module+stream pattern to mirror
- `audio/mimi/transformer.go` — Layer/ForwardCached/WindowKV reused unchanged
- `audio/mimi/load.go` — loader to extend (LoadDecoder/LoadFull)
- `audio/export_mimi_fixtures.py` — D0 fixture extension
- `e2e/mimi_realworld_test.go` — D4 acceptance surface
