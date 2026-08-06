# Audio classification with Mimi + gorch

Feasibility study (2026-08): can we train audio classifiers locally with
gorch, using [Mimi](https://huggingface.co/kyutai/mimi) — Kyutai's
streaming neural audio codec — as the feature extractor? **Yes.** This
directory holds the working proof of concept.

## Architecture

```
mic / wav (24 kHz mono)
   │
   ▼
Mimi encoder (frozen, 39M params, Python/PyTorch today)
   │   512-dim continuous latent @ 12.5 Hz
   │   (or 16×2048-way discrete codes for token models)
   ▼
pooling (mean+std over time → 1024-dim per window)
   │
   ▼
gorch classifier (trained + run natively in Go)
```

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

Reproduce:

```bash
python audio/export_fsdd_mimi.py /path/to/fsdd/recordings audio/fsdd_mimi.safetensors
go test -tags e2e ./e2e/ -run TestMimiFSDD -v
```

Results on Apple M4, CPU only (full report: `../doc/mimi-audio-report.md`):

| Head | Test accuracy | Train time |
|---|---|---|
| Digit (standard split) | **100.0%** (300/300) | 2.0 s |
| Speaker | **100.0%** (300/300) | 1.9 s |
| Digit, unseen speaker | **97.0%** (485/500) | 2.1 s |

Classifier inference: **~2 µs per clip** (265k-param MLP, Accelerate BLAS).

## Measured performance (M4, CPU)

| Stage | Latency |
|---|---|
| Mimi encode, batch 10 s clip | 334 ms (Python baseline) |
| Mimi encode, streaming 80 ms chunk | **8.8 ms/chunk (native Go)** — was ~43 ms in Python |
| gorch MLP head, per window | tens of µs |

The streaming figure is the native Go encoder (`audio/mimi`, plan
0006 Phase 5): `Encoder.NewStream().Push` with SEANet conv caches and
a 250-frame windowed KV cache, steady-state mean via
`BenchmarkMimiStreamChunk`. Streamed output matches the offline
windowed encoder to ~1e-6 max abs. A live pipeline (mic → streaming
encode → pooled window → gorch head) now has an end-to-end decision
latency of roughly **frame (80 ms) + encode (~9 ms) + head (~0 ms) ≈
90 ms** — comfortably real-time, on CPU alone, leaving the GPU/ANE
free.

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

Mimi stays a frozen extractor outside gorch for now. Porting its
encoder into gorch would need Conv1d (+ causal/dilated variants),
weight norm, ELU, audio I/O, and RVQ — a multi-week effort tracked as
future work. The classifier side works today; the sharp edges hit
during this PoC, in priority order:

1. **axis-wise Mean/Max** — pooling over time had to be done in the
   exporter; a `Mean(axis)` op would let gorch consume raw (T, 512)
   latents and learn attention pooling.
2. **Conv1d** — cheap to add on the existing im2col+BLAS machinery;
   unlocks small temporal CNNs over the 12.5 Hz frame sequence and is
   the first prerequisite for a native Mimi port.
3. **WAV reader + resampler** in `data/` — removes Python from the
   *inference* path entirely once an exported/ported encoder exists.
4. Sequence classification over variable-length clips needs padding /
   masked pooling in `DataLoader`.
