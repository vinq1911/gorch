# Audio classification with Mimi + gorch

Feasibility study (2026-08): can we train audio classifiers locally with
gorch, using [Mimi](https://huggingface.co/kyutai/mimi) — Kyutai's
streaming neural audio codec — as the feature extractor? **Yes.** This
directory holds the working proof of concept.

## Architecture

```
mic / wav (any rate; audio.ReadWAV + audio.Resample → 24 kHz mono)
   │
   ▼
Mimi encoder (frozen, 39M params, native Go: audio/mimi)
   │   512-dim continuous latent @ 12.5 Hz
   │   (or 16×2048-way discrete codes for token models)
   ▼
pooling (mean+std over time, g.MeanAxis/g.VarAxis → 1024-dim per window)
   │
   ▼
gorch classifier (trained + run natively in Go)
```

**The pipeline is now fully native** (plan `doc/plans/0006-mimi-native-encoder.md`):
WAV read, resampling, Mimi encoding, and pooling all run in Go with zero
Python. Python is only needed to regenerate the golden fixtures and the
reference feature file (`export_mimi_fixtures.py`, `export_fsdd_mimi.py`).

Mimi is the right front-end for this because it is **causal and
streaming by design** (it powers Moshi's full-duplex dialogue): 80 ms
frames, no lookahead. Its first codebook is distilled from WavLM, so the
latent carries *semantic/phonetic* content, while the remaining
capacity preserves *acoustic/paralinguistic* detail (speaker, prosody,
emotion) — both of which our probes confirm are linearly recoverable.

## Proof of concept: FSDD (spoken digits)

- `export_fsdd_mimi.py` — encodes the 3000-clip
  [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
  with frozen Mimi, mean+std-pools the pre-quantizer latent, writes
  train/test splits + digit and speaker labels to one safetensors file
  (gorch's `model.LoadSafetensors` reads it directly).
- `../e2e/mimi_fsdd_test.go` — trains three 265k-param MLP heads in
  gorch (AdamW, dropout, grad clipping) over the same frozen features:
  - **digit head** (10-way): what was said — a tiny transcription task
  - **speaker head** (6-way): who said it — a paralinguistic task,
    the same signal family as emotion/sentiment and language ID
  - **speaker-independent digit head**: trained on 5 voices, tested on
    a fully held-out 6th voice

- `../e2e/mimi_native_fsdd_test.go` — the same three heads, but with
  the embeddings computed **entirely in Go** (ReadWAV → Resample 8k→24k
  → `mimi.Encode` → mean+std pool), plus per-element feature parity vs
  the Python-exported reference.

Reproduce (fully native — no Python):

```bash
FSDD_DIR=/path/to/fsdd/recordings go test -tags e2e ./e2e/ -run TestMimiNativeFSDD -v
```

or with Python-exported features:

```bash
python audio/export_fsdd_mimi.py /path/to/fsdd/recordings audio/fsdd_mimi.safetensors
go test -tags e2e ./e2e/ -run TestMimiFSDD -v
```

Results on Apple M4, CPU only (full report: `../doc/mimi-audio-report.md`):

| Head | Python features | Native Go features |
|---|---|---|
| Digit (standard split) | **100.0%** (300/300) | **100.0%** (300/300) |
| Speaker | **100.0%** (300/300) | **100.0%** (300/300) |
| Digit, unseen speaker | **97.0%** (485/500) | **96.4%** (482/500) |

Classifier inference: **~2 µs per clip** (265k-param MLP, Accelerate BLAS).

Native feature parity (TestMimiNativeFSDD, all 3000 clips × 1024 dims):
Go-vs-Python pooled features agree to a **max abs diff of 8.2e-5** on an
idle machine (gate `|Δ| ≤ 1e-3 + 2e-3·|ref|`; the residual is scipy's
float32 resampler fast path vs Go's float64-exact polyphase, amplified
through the encoder). Native extraction of the whole dataset (read +
resample + encode + pool, ~22 min of audio): **52.9 s, 17.6 ms/clip,
~25× realtime**.

## Measured performance (M4, CPU)

| Stage | Latency |
|---|---|
| Mimi encode (native Go), batch 10 s clip | 283 ms (35× realtime) |
| Mimi encode (Python baseline), batch 10 s clip | 334 ms (30× realtime) |
| Mimi encode (native Go), streaming 80 ms chunk | **8.8 ms/chunk** — was ~43 ms in Python |
| Mimi decode (native Go), batch 10 s clip | 286 ms (35× realtime) |
| Mimi decode (native Go), streaming 80 ms chunk | **9.5 ms/chunk** — was 33.95 ms in Python (KV-cache incremental) |
| gorch MLP head, per window | tens of µs |

The encode streaming figure is the native Go encoder (`audio/mimi`,
plan 0006 Phase 5): `Encoder.NewStream().Push` with SEANet conv caches
and a 250-frame windowed KV cache, steady-state mean via
`BenchmarkMimiStreamChunk`. Streamed output matches the offline
windowed encoder to ~1e-6 max abs. The decode streaming figure is the
native Go decoder (plan 0007 Phase D3): `Decoder.NewStream(q).Push` —
one 8-level token column in, 1920 samples (80 ms) out — with
overlap-add ConvTranspose tails, conv left contexts and a windowed KV
cache, steady-state mean via `BenchmarkMimiDecodeStreamChunk`;
streamed output matches the offline windowed decoder to ~2e-6 max abs
(chirp) and the HF reference at ~118 dB SNR. A live pipeline (mic →
streaming encode → pooled window → gorch head) has an end-to-end
decision latency of roughly **frame (80 ms) + encode (~9 ms) + head
(~0 ms) ≈ 90 ms**, and full-duplex voice (encode 8.8 ms + decode
9.5 ms per 80 ms frame ≈ 18 ms of compute per frame) keeps ~4×
real-time headroom — on CPU alone, leaving the GPU/ANE free.

## What this enables

- **Language detection**: same pipeline, sliding 1–3 s window, trained
  on e.g. CommonVoice/FLEURS clips. 12.5 Hz × 512-d features make even
  a few hundred hours of audio a small training problem for gorch.
- **Sentiment / emotion**: identical head over the same features
  (CREMA-D / RAVDESS scale datasets train in seconds on M4). The
  speaker probe demonstrates the paralinguistic signal survives pooling.
- **Keyword spotting / small-vocab transcription**: the digit head *is*
  a 10-word vocabulary transcriber. Scaling to open-vocabulary ASR
  means a CTC or seq2seq decoder over the 12.5 Hz frame sequence —
  Kyutai's own streaming STT models (kyutai/stt-1b-en_fr) prove the
  Mimi-tokens-to-text route works, but at 1B+ params that is a weight
  *port* to gorch, not a local training run.

## Division of labor, and gaps to close in gorch

The gaps identified during the PoC are now closed (plan 0006): Conv1d
(causal, dilated, replicate pad), ELU/exact GELU, axis-wise
Mean/Var/Max, WAV reader and polyphase resampler, and the full Mimi
encoder (`audio/mimi`) all run natively. Remaining future work:

1. ~~RVQ quantizer~~ (plan 0006 P7) — **done**: discrete Mimi tokens
   are available natively via `mimi.Quantizer` (`Encode`: latent → RVQ
   codes, exact match vs HF `model.encode`; `Decode`: codes → quantized
   latent for token-LM/decoder work). The classifier pipeline still
   uses the continuous pre-quantizer latent.
2. Sequence classification over variable-length clips needs padding /
   masked pooling in `DataLoader`.
