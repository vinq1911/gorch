# Plan 0001: OpenMythos port to gorch

**Status:** proposed (planning, no code yet)
**Branch:** `claude/gorch-microcontroller-inference-Uw9vI` (planning notes only;
implementation work would land on dedicated feature branches per phase).
**Last updated:** 2026-04-29

## Goal

Port [OpenMythos](https://github.com/kyegomez/OpenMythos) from PyTorch to Go,
running on gorch instead of torch. v1 success = a downscaled `mythos_tiny`
config trains end-to-end on TinyStories on a single Apple Silicon Mac and
demonstrates the recurrent-depth claim (more loop iterations → lower
perplexity).

## What OpenMythos is

Speculative reconstruction of a "Recurrent-Depth Transformer," ~1050 LOC in
`open_mythos/main.py`. Three-stage flow:

- **Prelude** (N standard transformer blocks, run once)
- **Recurrent block** (looped up to `max_loop_iters` times with LTI-stable
  injection: `h_{t+1} = A·h_t + B·e + Block(h_t, e)`)
- **Coda** (N standard blocks, run once)

Components:

- Two attention variants (config-switched): GQA and MLA (DeepSeek-V2 style,
  separate nope/rope head dims, KV LoRA compression)
- Sparse MoE FFN with routed + shared experts (top-k routing)
- RoPE with precomputed `(cos, sin)` tables
- Depth-wise LoRA adapters across recurrent iterations
- ACT halting (Adaptive Computation Time)
- Optional Flash-Attention-2 with SDPA fallback
- Variants from `mythos_1b` (2048 dim, 64 experts, 4k context) up to
  `mythos_1t` (16384 dim, 512 experts, 1M context)

## v1 scope decision

**Target only `mythos_tiny`** (a downscaled config defined below). Do not
attempt `mythos_1b` or larger in v1. The 30B-token FineWeb-Edu training run
the OpenMythos paper targets is a multi-week-on-a-cluster project that
requires distributed training gorch does not have today.

`mythos_tiny` configuration:

| Parameter | Value |
| --- | --- |
| `dim` | 128 |
| `n_heads` / `n_kv_heads` | 4 / 2 |
| Prelude / Coda layers | 2 / 2 |
| Recurrent iterations | 4 |
| `n_experts` / `n_experts_per_tok` | 4 / 2 |
| `expert_dim` | 256 |
| `max_seq_len` | 512 |
| Total params | ~5–10 M |

## Capability gap

Inventory of what gorch has versus what OpenMythos uses (taken at commit
`267226f` on `main`).

### Already in gorch (verified)

- Linear, Embedding, LayerNorm, ReLU, GELU, Softmax, Dropout, BatchNorm
- MultiHeadAttention with KV cache integrated into GPT forward (`33d97c3`)
- Cross-entropy loss, `model.CausalLMLoss` for fine-tuning (`6444a27`)
- Adam, SGD-momentum, StepLR, CosineAnnealingLR, WarmupCosineScheduler
- MatMul / BatchedMatMul, ScaledMatMul, broadcasting (AddB/SubB/MulB/DivB),
  Transpose2D
- Safetensors load/save (F32, F16, BF16 read), BPE tokenizer
- KV cache, top-k/top-p/temperature sampling, greedy decode
- NoGrad gating that actually skips graph construction (`7780b33`)
- Tied LM-head/embedding weights HF-style (`bd169cb`)
- GPU autograd for MatMul + Linear backward (`556c1f0`, `eb5a8e3`)
- Streaming safetensors loader, halved peak RSS (`c99ccce`)

### Missing — must implement upstream in gorch

Each is intended to land as its own PR against gorch `main`.

1. **RMSNorm** (~80 LOC). Mirrors `nn/layernorm.go`.
2. **SiLU / SwiGLU** (~30 LOC). Same shape as the existing GELU op.
3. **N-D permute / general transpose** (touches `tensor.go`, autograd, Metal
   path). The blocker: every multi-head reshape needs it. Currently only
   2D `Transpose2D` exists.
4. **`gather`** (~80 LOC). Needed for MoE expert dispatch.
5. **`topk`** (~80 LOC). MoE router top-k selection.
6. **`multinomial`** (~40 LOC). Already exists for sampling but not as a
   public tensor op; expose it.
7. **`repeat_interleave`** (~50 LOC). GQA needs it to expand K/V across
   query groups.
8. **RoPE** (`nn/rope.go`, ~150 LOC). Real-valued precomputed
   `(cos, sin)` tables, applied as element-wise multiplies. **Do not
   implement complex/polar arithmetic on GPU** — production code (nanoGPT,
   llama.cpp, vLLM) uses real-valued sin/cos pairs.
9. **GQA module** (`nn/gqa.go`, ~250 LOC). Reuses the existing MHA KV cache
   pattern.
10. **MLA module** (`nn/mla.go`, ~400 LOC). The heaviest single piece.
    Separate nope/rope head dims, KV LoRA compression, separate KV cache
    shape from GQA.
11. **MoE FFN** (`nn/moe.go`, ~300 LOC). Top-k router + batched expert
    dispatch (gather → grouped matmul → scatter; **not** multi-Metal-queue
    parallelism — that's I/O overlap, not expert parallelism) + auxiliary
    load-balancing loss.
12. **AdamW** (~80 LOC). Copy `optim/optim.go` Adam; add weight decay term.
13. **Gradient clipping** (`optim/clip.go`, ~40 LOC). `clip_grad_norm`-style
    global norm clipping. Needed for transformer training stability.

Each primitive ships with a numerical-gradient-vs-analytical autograd test,
plus (for the larger modules) bit-for-bit forward-pass agreement with a
tiny pinned PyTorch reference at ~1e-5 tolerance.

## Phases and effort estimate

Effort estimates are calendar weeks of focused work for one engineer, not
person-weeks adjusted for context-switching.

| Phase | Deliverable | Effort | Cumulative |
| --- | --- | --- | --- |
| 1 | Eleven primitive PRs in gorch main + AdamW + grad clip | 3–4 wks | 4 wks |
| 2 | `model/mythos/` with verified forward pass | 1 wk | 5 wks |
| 3 | Training infra (TinyStories streaming loader + checkpoint helper) | 3–4 days | ~5.5 wks |
| 4 | Trained `mythos_tiny` on TinyStories with recurrent-depth ablation | 1–2 wks | **6–7 wks** |
| 5 | Scale-up to `mythos_1b` (DDP, bf16, ZeRO, activation checkpointing) | 2–3 mo | deferred |

### Phase 1 — primitives in gorch (3–4 weeks)

Order chosen so each PR unblocks the next. Stop the count if any item
fails its numerical-agreement test before starting the next.

### Phase 2 — `model/mythos/` assembly (1 week)

```
model/mythos/
  config.go     # Go equivalent of MythosConfig dataclass
  block.go      # TransformerBlock (RMSNorm + GQA-or-MLA + MoE)
  lti.go        # LTI-stable injection (matrix exp of -log eigenvalues)
  act.go        # ACTHalting
  lora.go       # Depth-wise LoRA adapter
  recurrent.go  # Looped block driver
  mythos.go     # Top-level OpenMythos: prelude → recurrent → coda
```

Success criterion: a single forward pass on `mythos_tiny` agrees with a
PyTorch reference to ~1e-5.

### Phase 3 — training infra (3–4 days)

Most of what was originally a week is already in gorch:

- ✅ Cosine + WarmupCosine LR schedulers exist
- ✅ Fine-tuning loop pattern established (`TestFinetuneShortCorpusConverges`,
  `791b711` — GPT-2 124M fine-tunes from loss 4.9 → 7.6e-6 in 60 steps)
- ✅ NoGrad gating works for eval/sampling during training

What still needs writing:

- TinyStories streaming text dataset loader (`data/text_loader.go`,
  ~150 LOC). Reads a tokenized `.bin`, returns randomly sampled
  `(batch, seq_len)` int32 tensors. nanoGPT-style.
- Checkpoint helper (`model/checkpoint.go`). Periodic save of
  `(weights, optimizer state, step, RNG state)` to safetensors.
  Reuses existing safetensors writer.

### Phase 4 — convergence run (1–2 weeks elapsed; ~24 h M-series GPU time)

Dataset: **TinyStories** (~500 MB synthetic short stories with a small
vocabulary — designed exactly for sub-100M-param models). Use Shakespeare
(~1 MB, character-level) for a 1-hour smoke test before committing to the
multi-day TinyStories run.

Success criteria:

1. Training loss decreases monotonically over the first 1k steps.
2. Validation perplexity drops to <8 on TinyStories within ~24 h of M-series
   GPU time (nanoGPT-equivalent baseline).
3. Generation produces coherent short stories at temperature 0.8.
4. **Recurrent-depth ablation**: increasing `max_loop_iters` at inference
   improves perplexity. This is the architecture's central claim and the
   main reason for porting OpenMythos rather than a vanilla GPT.

### Phase 5 — scale-up (deferred, captured for backlog hygiene)

Out of scope for v1 but listed so it doesn't get lost.

| Item | Why we'd need it | Estimated effort |
| --- | --- | --- |
| DDP across multiple Macs (TCP allreduce) | Single M4 GPU is ~10 TFLOPS fp32; 1B × 30B tokens needs ~1e21 FLOPS → ~3 years on one M4 | 3–4 wks |
| bf16 training | 2× throughput, 2× memory headroom (already on gorch's public roadmap) | 2–3 wks |
| ZeRO-1 optimizer sharding | AdamW state for 1B params = 8 GB; tight on 16 GB Macs | 1 wk after DDP |
| Activation checkpointing | Recurrent block re-running M times balloons activation memory | 3–4 days |
| Streaming dataset at scale | FineWeb-Edu's 1.3T tokens needs on-the-fly tokenization | 1 wk |
| Distributed evaluation | Periodic eval at 1B scale needs more than one box | 2–3 days |

Realistic Phase 5 effort if pursued: **2–3 months**, before training compute
is purchased.

## Risks

1. **MLA is the hardest single piece.** Separate rope/nope head dims and
   KV LoRA compression are unusual; high chance of subtle bugs. Plan for
   the most thorough numerical-agreement test suite of any module.
2. **Autograd correctness under composition.** Each new op must pass a
   numerical-gradient-vs-analytical test in isolation, but the real risk
   is composed gradients. Plan for end-to-end gradient checks on
   `mythos_tiny` before any training run.
3. **Non-MatMul GPU autograd is still CPU-only** (ADR-009-fix). For
   `mythos_tiny` at <1G FMAs this is unlikely to matter; for any larger
   config it will. Plan B: train on CPU through Accelerate if Metal training
   is too slow.
4. **Darwin-only.** gorch is `//go:build darwin`. The OpenMythos port
   inherits that. CI on Linux can run pure-Go reference paths but not the
   actual model.

## Open questions

- Should the gradient clipping op be in `optim/` or `nn/`? PyTorch puts it
  in `nn.utils`; gorch's pattern would suggest `optim/`.
- Do we need a deterministic RNG state checkpoint for the dataset loader?
  TinyStories convergence might depend on data order in subtle ways.
- Will the LTI parameterization need a separate Metal kernel for the matrix
  exponent? For a 4-iteration recurrent block at `dim=128`, probably not.

## What we are explicitly not doing

- Implementing complex-number arithmetic in Metal shaders for RoPE. Use real-
  valued sin/cos pairs. (Standard production technique.)
- Implementing MoE expert parallelism via multiple Metal command queues.
  That's I/O overlap, not expert parallelism. Use batched expert dispatch.
- Targeting "GPT-4 support." GPT-4's architecture isn't published; the
  honest framing is "GPT-4-class open LLMs" — Llama 3, DeepSeek-V2/V3,
  Qwen 2.5. Same primitives unlock all of them.
- Porting GGUF format support in v1. Safetensors covers Llama 3, Mistral,
  Qwen, DeepSeek already. GGUF is a separate track if we ever serve via
  llama.cpp-compatible tooling.

## Decision points before kicking off

1. Confirm `mythos_tiny` on TinyStories is the v1 target, not a smaller
   slice of FineWeb-Edu. **Answered yes.**
2. Confirm fp32 throughout for v1, bf16 deferred. **Answered yes.** bf16
   support tracked separately in `0002-bf16-support.md`.

## First PR after kickoff

`gorch/feature/rmsnorm`: implement RMSNorm with autograd + Metal hook +
test. Smallest, self-contained, exercises the whole "new module" path,
sets the template for the next ten primitives.
