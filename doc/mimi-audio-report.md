# Mimi + gorch: local audio classification feasibility report

**Date:** 2026-08-06
**Hardware:** Apple M4, 24 GB, CPU only (no Metal/ANE used anywhere)
**Question:** can we train and run audio classifiers locally with gorch,
using Kyutai's Mimi neural audio codec as a frozen feature extractor —
and is the pipeline fast enough for real-time language/sentiment
detection and transcription-adjacent tasks?

**Answer: yes.** Full pipeline proven end to end; every stage is
comfortably real-time on CPU.

## 1. Mimi as a feature extractor

Mimi (`kyutai/mimi`, 79M params total, 39M encoder-side) encodes 24 kHz
mono audio into a 512-dim continuous latent at 12.5 Hz (80 ms frames),
optionally quantized to 16–32 codebooks of 2048 entries (~1.1 kbps).
It is causal and streaming by design — it is the audio front-end of
Moshi's full-duplex dialogue system — and its first codebook is
distilled from WavLM, so the latent carries phonetic/semantic content
alongside acoustic detail.

Measured on M4 CPU (PyTorch 2.9, transformers 4.57):

| Stage | Latency | Real-time factor |
|---|---|---|
| Batch encode, 10 s clip | 334 ms | 30× realtime |
| Streaming encode, 80 ms chunk | 43 ms mean, 55 ms p95 | 1.9× per-frame headroom |

## 2. Proof of concept: FSDD spoken digits

3000 clips (10 digits × 6 speakers × 50 takes) encoded once by frozen
Mimi, mean+std-pooled over time to one 1024-dim vector per clip
(`audio/export_fsdd_mimi.py`), then classified by a 265k-param MLP
(1024→256→relu→dropout→classes) trained entirely in gorch with AdamW +
grad clipping (`e2e/mimi_fsdd_test.go`).

| Head | Task type | Test accuracy | Train time (30 epochs) |
|---|---|---|---|
| Digit, standard split | spoken content ("what was said") | **100.0%** (300/300) | 2.0 s |
| Speaker | paralinguistic ("who said it") | **100.0%** (300/300) | 1.9 s |
| Digit, held-out speaker | generalization to an unseen voice | **97.0%** (485/500) | 2.1 s |

Classifier inference is ~2 µs/clip. For reference, published FSDD
baselines with spectrogram CNNs sit around 97–99% on the standard
split; a linear-ish probe on frozen codec features matching that —
including 97% speaker-independent — confirms Mimi latents are highly
separable for both content and speaker identity.

## 3. What this means for the target applications

**Real-time pipeline budget** (mic → decision): 80 ms frame + ~45 ms
streaming encode + ~0 ms gorch head ≈ **130 ms end-to-end**, CPU only.
Decisions can be refreshed every 80 ms frame with a sliding pooled
window.

- **Language detection** — same architecture, sliding 1–3 s window,
  trained on CommonVoice/FLEURS-style clips. At 12.5 Hz × 512-d,
  100 h of audio is only ~4.5M feature frames (~9 GB f32, less pooled)
  — well within local training scale for a gorch MLP/small transformer.
- **Sentiment / emotion** — the 100% speaker probe shows paralinguistic
  signal survives Mimi compression and pooling. CREMA-D (~7k clips) or
  RAVDESS (~1.4k) would train in seconds; expect accuracy in line with
  published frozen-codec probes (~60–75% on 6–8 emotions).
- **Transcription** — the digit head is already a 10-word closed-vocab
  transcriber. Keyword spotting (dozens–hundreds of words) is the same
  recipe. Open-vocabulary streaming ASR over Mimi tokens is proven
  territory (Kyutai's stt-1b-en_fr), but at 1B+ params that is a
  weight-port to gorch, not a local training project.

## 4. Division of labor and gorch gaps

Mimi stays outside gorch as a frozen extractor (Python today; ONNX/
CoreML export is the deployment path). Training and inference of the
classifier is pure gorch: `model.LoadSafetensors` ingests the features
directly, `data.DataLoader` + `nn.Sequential` + `optim.AdamW` cover the
rest with zero new framework code.

Gaps hit during the PoC, in priority order:

1. **Axis-wise Mean/Max ops** — time-pooling had to happen in the
   exporter; `Mean(axis)` would let gorch consume (T, 512) sequences
   and learn attention pooling.
2. **Conv1d** — natural next kernel on the im2col+BLAS machinery;
   unlocks temporal CNNs over frame sequences and is the first
   prerequisite for ever porting Mimi's SEANet encoder natively.
3. **WAV read + resample in `data/`** — removes Python from the
   inference path once an exported encoder exists.
4. **Variable-length batching** (padding/masking in DataLoader) for
   sequence heads instead of pooled vectors.

Porting the full Mimi encoder into gorch (Conv1d + weight norm + ELU +
RVQ + streaming conv state) is estimated at multi-week effort and is
not needed for any of the applications above.
