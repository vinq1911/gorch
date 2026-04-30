# Plan 0005: Quantization for serving (int8 / int4)

**Status:** proposed (placeholder; activate only if/when LLM serving on
gorch becomes a goal)
**Tracks:** `0003-gemini-review.md` line "Quantized weight serving (int4
/ Q4_K) for fitting big models on consumer hardware — Right for serving
/ inference. Out of v1 scope."
**Last updated:** 2026-04-30

## Goal

Run open-weight LLMs (Llama 3 8B, Mistral, Qwen 2.5) on Mac mini-class
hardware by loading int8 or int4-quantized weights instead of fp32/bf16.
Inference-only; this is the serving track, not training.

A 16 GB unified-memory Mac cannot hold Llama 3 8B at fp32 (32 GB), is
tight at bf16 (16 GB before activations), and is comfortable at int4
(~4–5 GB).

## Why this is *not* on the v1 critical path

`mythos_tiny` (Plan 0001 v1 target) is ~5–10 M params. fp32 fits in MB.
bf16 (Plan 0002) is the next-most-important precision step and covers
all training scenarios. Quantization helps **only** for serving large
pretrained models, which is a separate project from "train a recurrent-
depth transformer end-to-end."

## Scope

Two tracks, only the first is realistic short-term:

### Track A — int8 weight loading + dequantize-on-the-fly

| Component | Work |
| --- | --- |
| Safetensors int8 read | Already partial — gorch reads dtype but only fp32/fp16/bf16. Add int8 + per-tensor / per-channel scale tensors. |
| `nn.LinearInt8` module | Stores int8 weights + fp16/fp32 scale; forward = dequantise weight tile → fp32 matmul → output. About 4× memory savings, throughput same as fp32. |
| Quantization-aware loader | Run an offline calibration pass (HF `optimum-quanto` style) on a sample to set per-channel scales. Or just consume someone else's pre-quantized checkpoint. |

Estimated effort: **1–2 weeks**. Ships value the moment it lands —
"Llama 3 8B inference on a 16 GB Mac" becomes possible. Numerical
accuracy is well within the noise budget for greedy/top-p decode.

### Track B — int4 + Q4_K-style block quantization (llama.cpp interop)

This is the bigger lift. llama.cpp's Q4_K format is block-quantized
(typically 32-element blocks with per-block fp16 scale + zero-point) and
needs a custom dequant + matmul kernel. Done well it gets ~3.5× memory
savings *and* improved memory-bandwidth utilisation on Apple GPUs.

| Component | Work |
| --- | --- |
| Q4_K format reader | Read GGUF (or a gorch-native Q4_K-in-safetensors variant). |
| Custom Metal kernel | Block-dequant + accumulate inside threadgroup memory. ~1 wk shader work. |
| Validation | Bit-equivalent output (greedy decode) vs llama.cpp on the same prompt. |

Estimated effort: **3–4 weeks**.

## When to activate this plan

Trigger: "we want to serve Llama 3 8B / Mistral 7B on a Mac with 16 GB."
Until that's a stated goal, this plan stays at status `proposed` and
the v1 OpenMythos port (Plan 0001) gets the engineering time.

If the trigger fires, start with Track A. Track B only if the project
takes on llama.cpp-compatible serving as a target.

## Cross-checks against 0003

- 0003 was right that quantization is real and useful. This plan
  formalises it as a separate track rather than letting it bleed into
  Plan 0001 (OpenMythos training) or Plan 0002 (bf16 native precision).
- 0003 explicitly punts GGUF format support out of v1; same here. Track
  A consumes safetensors with int8 dtype. Track B activates GGUF only
  if llama.cpp interop is needed.

## What we are explicitly not doing

- AWQ / GPTQ / SmoothQuant calibration in gorch. These are research
  projects of their own; consume pre-calibrated weights from HF.
- Quantization-aware training. Different track entirely (fine-tune the
  full-precision model first, then post-quantise for serving).
- bfloat16 inference under this plan. That's Plan 0002.

## First PR if/when activated

`gorch/feature/int8-linear`: extends the safetensors loader to read
int8 + scale, adds `nn.LinearInt8`, validates greedy-decode output of a
known small int8-quantized model against an fp32 baseline at 1e-1
relative error.
