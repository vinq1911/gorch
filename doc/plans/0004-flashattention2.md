# Plan 0004: Native Metal FlashAttention-2 + non-MatMul GPU autograd

**Status:** in progress (part A kernel 1 of 6 shipped)
**Tracks:** README roadmap items "GPU autograd for non-MatMul ops" and the
FlashAttention claim called out in `0003-gemini-review.md`.
**Last updated:** 2026-04-30

## Status

- **Part A — RMSNorm Metal forward + dx backward** shipped 2026-04-30
  as `feature/rmsnorm-metal-backward`. New kernels `rmsnorm_forward`
  and `rmsnorm_dx` in `metal/kernels.go`; threadgroup-controlled
  dispatch via the new `Dispatch1DThreadgroups` shim;
  `RMSNormForwardMetal` / `RMSNormBackwardDXMetal` in
  `gorch/rmsnorm_metal.go`; `nn.RMSNorm.Forward` routes to the GPU
  path when both x and gamma live on Metal. dW remains a host loop
  over `(M, N)` reading through unified memory — folding it into a
  per-column kernel is part of the next non-MatMul item. Tests:
  `TestRMSNormMetalForwardMatchesCPU` (1e-4 abs),
  `TestRMSNormMetalBackwardMatchesCPU` (1e-3),
  `TestRMSNormMetalBackwardMatchesNumerical` (2e-2 — finite-difference
  gold standard against the GPU forward).

## Why these are grouped

Plan 0003 identifies these as a single Phase 5 "Metal kernel performance
pass" because they share the same hard parts: writing custom `.metal`
kernels, threadgroup memory management, dispatch/fence orchestration, and
numerical-equivalence test rigging. Doing them serially with the same
shader-author toolchain is much faster than as separate one-off pushes.

## Goal

End-to-end transformer training on Metal at large shapes, with no
forward+backward op falling back to CPU. Today's matmul-only GPU autograd
loses to CPU above the size threshold because every grad propagation
through LayerNorm/Softmax/GELU forces a CPU sgemm reading from
Metal-backed memory (ADR-009-fix). Closing this gap is what makes
`gpt.ToMetal()` a real training accelerator instead of "at parity with
CPU" at typical shapes.

## Two deliverables

### A) Custom Metal backward kernels for non-matmul ops

| Op | Forward Metal status | Backward Metal status | New work |
| --- | --- | --- | --- |
| LayerNorm / RMSNorm | CPU only (uses Accelerate path) | CPU only | Forward + backward custom kernels |
| Softmax (last dim) | CPU only | CPU only | Forward + backward (output·grad − Σ(output·grad)) |
| GELU | `vec_gelu` exists for forward (PR #12) | CPU only | Backward kernel |
| Embedding lookup (Gather) | CPU only | CPU only | Forward + scatter-add backward |
| Add / Mul (residual paths) | `vec_add`, `vec_mul` forward | CPU only | Just teach Add backward to dispatch when grad is on Metal |

**Why this is a clean batch.** Every kernel is element-wise or
reduction-along-last-dim, threadgroup size is `dim`, dispatch grid is
`batch * seq`. Shader template is the same; only the per-element math
varies. Testing template is the same too: forward agrees with CPU at
1e-5, analytical-backward agrees with numerical-grad at 1e-2.

Implementation order (each ~1–2 days):

1. RMSNorm forward + backward (smallest math; first since Plan 0001 ships
   the RMSNorm Go implementation, gives an exact reference).
2. LayerNorm backward (Layer norm forward already exists in CPU only,
   port to Metal forward + backward).
3. GELU backward (forward kernel exists, just write the matching
   backward).
4. Softmax forward + backward (the per-row last-dim reduction).
5. Embedding scatter-add backward (one-shot atomic-add into weight rows).
6. Add backward + Mul backward Metal dispatch (trivial; just route the
   existing element-wise kernel from forward).

Total: ~7–10 days of shader work + testing.

### B) FlashAttention-2 fused attention kernel

**Why fused, not chained primitives.** Composed attention computes
`scores = Q @ K^T`, materialises `(seq, seq)` scores, applies mask,
softmax, then `attn @ V`. The (seq, seq) intermediate is the memory
bottleneck — at 4096 sequence length it's 64 MB per head per layer, for
12 heads × 12 layers that's 9 GB just for scores. FlashAttention
eliminates the materialisation by tiling Q in threadgroup memory and
streaming through K/V tiles, computing softmax incrementally with the
online-softmax trick (Milakov & Gimelshein 2018, refined in FA-2 by Dao
2023).

**Scope of the kernel:**

- `metal/flash_attention.metal` — single Metal compute kernel
- Threadgroup memory holds one Q tile (e.g. 128 rows × headDim)
- Outer loop over K/V tiles (each headDim wide, BLOCK_KV rows)
- Inner: compute `Qi @ Kj^T`, online-softmax update of running max +
  running sum + accumulator, multiply by `Vj` and accumulate into output
- Causal mask handled by exit-early on j > i within the tile
- Output: `(seq, headDim)` per (batch, head)

Scope deliberately **excludes** in v1:
- bf16 / fp16 variants (Plan 0002 lands first)
- RoPE inside the kernel (apply via element-wise pre-multiply, same as
  llama.cpp's split kernel approach)
- Sparse / windowed / sliding-window variants (one variant at a time)

**Validation:** Bit-equivalent (within 1e-3) output to gorch's existing
batched-MHA forward at causal=true on randomised Q/K/V at shapes
matching GPT-2 small (12 heads × 64 headDim × 1024 seq) and Llama 3 8B
(32 heads × 128 headDim × 4096 seq).

## Cross-checks against 0003

- 0003 advisory was right that FA-2 is real and valuable; gorch lacks it
  today. Add as Phase 5.
- Real-valued RoPE goes outside the FA kernel as element-wise multiply,
  matching llama.cpp / nanoGPT / vLLM. Do **not** put complex-typed
  arithmetic inside the FA shader.
- 0003's "advisory missed: non-MatMul GPU autograd is the bigger
  bottleneck than missing FlashAttention" is the reason A) ships before
  B) inside this plan.

## Effort

- A) Non-MatMul backward kernels: ~7–10 days
- B) FlashAttention-2: ~2–3 weeks (kernel design + correctness + test
  matrix at multiple seq lengths)

Total: **3–4 weeks** of focused shader work. Run after Plan 0002 (bf16)
so the kernels can ship in both fp32 and bf16 from day one.

## Decision points

- Does Plan 0002 land first, or do we ship A) in fp32 only and bf16 it
  later? Recommend Plan 0002 first to avoid double-writing every kernel.
- Does FlashAttention-2 wait for non-MatMul GPU autograd to land, or do
  the work concurrently? Concurrent is fine; they don't share code.

## What we are explicitly not doing

- A "general fused-op compiler" like XLA. v1 = one hand-written FA kernel
  and one each per non-matmul op.
- Triton-style kernel DSL. Apple Metal Shading Language is the target;
  the existing `metal/kernels.go` template is the convention.
- Backwards-compatible fallback toggles. If the kernel produces wrong
  output, the bug fix is the only path; we don't keep the CPU path as a
  runtime switch.

## First PR after kickoff

`gorch/feature/rmsnorm-metal-backward`: forward + backward Metal kernel
for the RMSNorm op shipped in feature/rmsnorm. Smallest, tightest math,
exercises the whole "ship a custom Metal kernel + autograd hook + test"
path. Sets the template for the next five non-matmul kernels.
