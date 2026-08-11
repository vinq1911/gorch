# Plan 0009: Training acceleration execution — GPU-resident autograd + bf16 for the LoRA/Mimi fine-tune workload

**Status:** proposed (execution plan; operationalizes plans 0002 and 0004)
**Tracks:** plan 0002 (bf16), plan 0004 (Metal non-matmul autograd + FA2), and the upcoming plan 0008 workload (LoRA-adapting a Qwen3-0.6B-class LLM to Mimi speech tokens)
**Supersedes-in-part:** 0002 (remaining work re-scoped here), 0004 (kernel list re-ranked, FA2 deferred here)
**Hardware target:** Apple M4, 10-core GPU, 24 GB unified memory
**Last updated:** 2026-08-11

## 0. Why this plan exists

Plans 0002 and 0004 were written 2026-04-30, before the Mimi work (0006/0007) grew the op surface, before `MatMulMetalThreshold`/ADR-009-fix, and before the concrete training consumer existed. The consumer is now defined: LoRA fine-tuning a Qwen3-0.6B-class model (28 layers, hidden 1024, GQA 16Q/8KV heads, head_dim 128, SwiGLU FFN ≈3072, RoPE, RMSNorm, vocab ≈151,936 + ≈16,384 appended Mimi tokens ≈ 168k, tied embeddings) on sequences of 500–1500 tokens. Frozen base, trainable LoRA adapters + embedding/head rows. On today's CPU f32 path this is a multi-day run; the goal is hours.

This plan replaces the old plans' orderings with a workload-ranked execution sequence, defines the benchmark harness first, and specifies the Azure-assisted implementation protocol for the kernel work.

## 1. Reconciliation with plans 0002 and 0004 — what is done, stale, obsolete

### 1.1 Plan 0002 (bf16)

| Item in 0002 | Status today | Action |
| --- | --- | --- |
| PR 1: `BFloat16` dtype, `data16 []uint16`, `NewTensorBF16`, `ToF32`/`ToBF16` | **DONE** (`tensor.go`) | none |
| PR 2: per-op promote-to-f32 dispatch with autograd-aware `upcastBF16`/`downcastToBF16` | **DONE** (`bf16_dispatch.go`; parity tests in `bf16_ops_test.go` at 5e-2 fwd / 8e-2 grad rel) | none |
| PR 3: AdamW master-f32 moments for bf16 weights | **NOT DONE** — `optim/adamw.go` moments are `[][]float32` keyed to `p.Data()`, which is nil for bf16 tensors; `Step()` would index nil | X3 item B3 |
| Safetensors native bf16 load (stop promoting) | **NOT DONE** — `model/safetensors.go` `decodeBF16` still widens to f32 on load | X3 item B1 |
| Native bf16 Metal kernels for *all* custom kernels | **RE-SCOPED** — only matmul + the X2 kernel set matter for this workload; the rest stays promote-to-f32 | X3 item B4, narrowed |
| "GPT-2 inference bf16 vs f32 e2e" | **OBSOLETE** as the acceptance test — replaced by the LoRA-step golden test at 0.6B geometry (X4) | mark in 0002 |
| fp16 as near-free add-on, default-dtype setter API | **DEFERRED OUT** — not needed by the workload; bf16 needs no loss scaling (same exponent range as f32), fp16 would; skip both fp16 and loss-scaling infrastructure entirely | mark in 0002 |
| "Does the MPS shim expose bf16?" open question | **STILL OPEN** — `metal/shim.m` is 100% `MPSDataTypeFloat32`; must be answered by probe (X3 gate task B0) | this plan |

### 1.2 Plan 0004 (Metal backward + FA2)

| Item in 0004 | Status today | Action |
| --- | --- | --- |
| Part A kernel 1: RMSNorm Metal forward + dx (`rmsnorm_forward`, `rmsnorm_dx`, `Dispatch1DThreadgroups`) | **DONE** (`metal/kernels.go`, `rmsnorm_metal.go`, `nn/rmsnorm.go` routes to GPU) | none |
| RMSNorm **dgamma** on GPU | **NOT DONE** — `RMSNormBackwardDXMetal` computes dW as a host loop over (M, N) | X2 item K5 |
| LayerNorm Metal fwd+bwd | **DEPRIORITIZED** — target model is RMSNorm-only; LayerNorm matters only for GPT-2-family. Do after X4, if at all | mark in 0004 |
| GELU backward kernel | kept, but **SiLU/SwiGLU fwd+bwd now higher priority** (SwiGLU is the workload's FFN; `silu.go` shipped after 0004 was written) | X2 items K4 |
| Softmax fwd+bwd kernel | **NOT DONE**, now ranked #1 non-matmul kernel; upgraded to a *causal-mask-and-scale-fused* softmax (see §3.3) | X2 item K1 |
| Embedding scatter-add backward | **NOT DONE**, now critical: trainable 168k×1024 table means `EmbeddingLookup` backward allocates a dense 172M-float (≈690 MB) grad per step (`attention_ops.go`) | X2 item K3 |
| Add/Mul backward Metal dispatch | **PARTLY MOOT** — Add/Mul backwards are pass-through/elementwise; the real problem is residency propagation (outputs of CPU-path ops are CPU tensors, breaking GPU dispatch downstream). Solved structurally in X1 | X1 |
| Part B: FlashAttention-2 fused kernel | **DEFERRED with arithmetic** (see §3.5) — at seq ≤1500, head_dim 128, memory is handled by fusion + bf16; compute gain over MPS batched matmul + fused softmax is small. Revisit at seq ≥4k | mark in 0004 |
| "Run after 0002 so kernels ship in fp32 and bf16 from day one" | **REVERSED** — kernels ship f32-storage-first with f32 accumulation; bf16 arrives as storage type at the matmul boundary (X3). Elementwise kernels read/write bf16 via widen/narrow in-register later, only if profiling justifies it | this plan |

### 1.3 New since both plans (must be accounted for)

- `MatMulMetalThreshold` (default 512M FMAs) + ADR-009-fix: below-threshold matmuls run Accelerate even when operands are Metal-backed. Measured crossover on M4: 768³ GPU=0.45× CPU, 1024³ = 1.22×, 2048³ = 2.10× (`doc/metal_crossover_results.json`; CPU AMX ≈1.55–1.67 TFLOPS f32, GPU ≈3.0–3.3 TFLOPS f32).
- `BatchedMatMul`/`BatchedMatMulTransB` forward can dispatch MPS, but **their backwards are unconditional CPU loops** ("no batched-MPS-transA today", `ops.go`). Attention backward is therefore always CPU.
- `nn.Linear` has a full GPU backward (dx/dW via MPS) but only when grad, x, W are *all* Metal-resident (`nn/module.go`, `nn/gpu_backward.go`) — which today never happens in a real training loop because the loss produces a CPU grad (ADR-009-update's diagnosis, still true).
- `accelerate/shim.c` grew `acc_velu`, `acc_vgelu_erf` during Mimi P3 — the convert-at-boundary Accelerate pattern is established.
- `conv1d.go`/`convtranspose1d.go` are CPU-only and **out of scope** for 0009 (audio codec inference, not LLM training).
- Every Metal dispatch is synchronous (`waitUntilCompleted` per op in `metal/shim.m`) — per-op latency ~0.2–1 ms; fine for the ≥1G-FMA matmuls this plan targets, but a known ceiling (§6 risk R6).
- GQA/RoPE/MLA/MoE modules exist; `nn/gqa.go` builds a full autograd graph despite a stale "inference-only" comment (worth fixing in passing during X1).

## 2. X0 first: benchmark harness + profile (the ranking evidence)

The Mimi-plan discipline applied to perf: **no kernel is written before the baseline exists**, and every phase reports speedup against the same fixture.

### 2.1 Harness spec

New file `e2e/train_accel_bench_test.go` + `bench` helpers:

- **Synthetic block step**: one transformer block at the 0.6B geometry (hidden 1024, 16Q/8KV × head_dim 128, SwiGLU inter 3072, RMSNorm ×2, RoPE), forward + `Sum` loss + `Backward` + AdamW step, at seq ∈ {512, 1024, 1500}. Deterministic seeded weights.
- **Full-step estimate**: block time × 28 + measured embedding fwd/bwd + measured lm_head (1500×1024 @ 1024×168320) fwd/bwd + measured `CrossEntropyLoss` over (1500, 168320) + AdamW step over a 172M-param table. These tails are measured separately because they are *not* per-layer and are suspected hotspots.
- **Per-op wall-clock breakdown**: instrument with simple timers around op classes (matmul, softmax, mask/scale, permute/reshape copies, RMSNorm, activations, RoPE, embedding, loss, optimizer), reported as a table. `runtime/pprof` CPU profile captured once for the record.
- **Output**: `doc/training_accel_results.json` (same convention as `metal_crossover_results.json`), keyed by phase label, so X1–X4 append comparable rows.
- **Thermal discipline**: median of 5 runs, 2 discarded warmups, machine on AC, no concurrent load; record `sysctl machdep.cpu.brand_string` and check clock sanity by re-running the X0 baseline row at the end of every phase's bench (a >10% drift on the *baseline* invalidates the session's numbers — rerun after cooldown).

### 2.2 Paper profile (to be confirmed by X0, but it drives the plan's ordering)

FLOPs per step at seq 1500: per layer ≈ 65.6 GF forward (projections 18.9 + FFN 28.3 + attention matmuls 18.4), ×28 layers ≈ 1.84 TF; lm_head ≈ 0.52 TF; forward total ≈ 2.4 TF; fwd+bwd ≈ 7 TF/step.

- **Matmul lower bound on CPU** (Accelerate ≈1.6 TFLOPS): ≈4.4 s/step *if everything else were free*.
- **It is not free.** Known pure-Go single-threaded costs per step at seq 1500: Softmax over 28 × (16·1500 rows × 1500) ≈ 1.0 B exp calls fwd (+ backward pass of same size); `CrossEntropyLoss` does LogSoftmax over 1500×168k ≈ 252 M exp fwd *and recomputes* `Softmax` in backward (double work, `loss.go`); `EmbeddingLookup` backward allocates a dense 690 MB zero tensor and scatters into it; GQA forward allocates per layer a 144 MB `Full` scale tensor, a 36 MB bool mask, and 4–5 seq² (144 MB) intermediates; AdamW is a scalar Go loop with `math.Sqrt` per element (a trainable 172M-param table alone ≈1–2 s/step). Realistic current step time: **tens of seconds**, i.e. multi-day for a 5–10k-step run — consistent with the motivation.
- **Memory wall at seq 1500 f32**: ~5 live seq² tensors/layer × 144 MB × 28 layers ≈ **20 GB of attention intermediates** — the current graph does not fit 24 GB at seq 1500 regardless of speed. Fusion (K1) + bf16 activations (X3) are needed for memory, not just speed.

**Gate X0:** harness merged; JSON baseline recorded at seq 512/1024/1500 (1500 expected to OOM or swap — record that fact); per-op table published; ranking below confirmed or corrected. Effort: 2–3 days.

## 3. The two tracks, updated

### 3.1 Ranked leverage (replaces both old orderings)

1. **X1 — GPU-resident autograd wiring for matmul-class ops.** ~85% of step FLOPs are matmuls that MPS already executes at 2.0–2.5× Accelerate at these shapes; today they run on CPU during backward (batched backwards are CPU-only; Linear GPU backward never triggers because grads arrive CPU-resident). This is wiring, not kernels.
2. **X2-K1 — causal-fused Softmax fwd+bwd** (biggest pure-Go loop + biggest memory reduction: 5 seq² intermediates → 1).
3. **X2-K2 — fused CrossEntropy/LogSoftmax fwd+bwd** over 168k vocab (second-biggest loop; removes the backward double-softmax).
4. **X2-K3 — embedding gather + scatter-add backward** (kills the 690 MB/step dense alloc; enables trainable Mimi rows cheaply).
5. **X2-K4 — SiLU/SwiGLU + GELU fwd+bwd elementwise kernels**; **K5 — RMSNorm dgamma**; **K6 — RoPE fwd+bwd** (small FLOPs; value is keeping the chain resident).
6. **X2-K7 — AdamW vectorization** (Accelerate vDSP first; masked-row updates for the embedding table).
7. **X3 — bf16 storage + GPU bf16 matmul for the frozen path** (2× memory on weights and activations; up to ~2× matmul throughput *if* the MPS bf16 probe passes).
8. **FA2 — deferred** (§3.5).

### 3.2 X1: GPU-resident autograd (wiring, not kernels)

Concrete gaps, in order:

1. **Residency propagation rule.** Ops whose inputs are Metal-backed must allocate outputs with `ZerosOnMetal`/`NewBuffer` even when the compute runs on CPU through unified memory (Accelerate below threshold, Go loops). Today `out := Zeros(...)` produces `buf == nil` and every downstream op loses GPU dispatch permanently. On unified memory this is nearly free — same physical pages, just tagged with a buffer. Touch points: `binaryOp`/`unaryOp`, `Softmax`, `LogSoftmax`, `MaskFill`, `Permute`, broadcast ops, `nn.Linear` CPU branch, RoPE, activations. Add a helper `zerosLike(a *Tensor, shape ...int)` that inherits residency.
2. **Batched matmul backward on MPS.** Add `metal_mps_batched_matmul_transA` (and the missing transB-grad variants) to `metal/shim.m` + `metal/metal.go`, and give `BatchedMatMul`/`BatchedMatMulTransB` backward the same residency-and-threshold-gated GPU path `MatMul` already has. This alone moves attention backward (≈1 TF/step) to GPU.
3. **Loss-side grad seeding on GPU.** `Backward()` seeds grad on CPU; after K2 the CE backward produces a Metal-resident dLogits so the whole backward chain stays resident. Until K2 lands, an interim `SeedGradOnMetal` in the harness is acceptable for measurement.
4. **Threshold audit at workload shapes.** Per-layer matmuls at seq 1500: 1500·1024·2048 ≈ 3.1G FMAs (dispatches), 1500·1024·1024 ≈ 1.6G (dispatches), attention batched 16·1500·1500·128 ≈ 4.6G (dispatches), lm_head 258G (dispatches). At seq 512 several fall below 512M — measure whether the threshold needs a per-callsite override; do not silently lower the global default.
5. Fix the stale "inference-only" comment in `nn/gqa.go`; replace the `Full`+`Mul` scaling (144 MB alloc) with grad-aware scalar scaling, and hoist the per-forward causal-mask construction into a cached mask per (heads, seq).

**Gate X1:** block-step bench with weights+activations resident shows every matmul (fwd and bwd) dispatching MPS above threshold (assert via a dispatch counter in tests); block step ≥1.8× vs X0 baseline at seq 1024; `TestGPUMatMulBackwardMatchesCPU`-style parity tests extended to the batched backwards (1e-3 abs). Effort: 4–6 days.

### 3.3 X2: the ranked kernel set (each = golden test first, then kernel; all f32 storage, f32 accumulation)

All follow the `rmsnorm_forward`/`rmsnorm_dx` template: per-row threadgroup of 256, strided loops, tree reduction in threadgroup memory, `Dispatch1DThreadgroups`, compiled from `metal/kernels.go` source strings; Go-side driver in a `*_metal.go` file mirroring `rmsnorm_metal.go`; `nn`-layer routing mirroring `nn/rmsnorm.go` `forwardMetal`.

- **K1 `softmax_causal_forward` / `softmax_backward`.** Forward fuses: scale by 1/√d, causal mask (compare column>row+offset — no bool mask tensor, no −1e9 fill tensor), row max, exp, sum, normalize. One kernel, one output, replaces 4–5 intermediates. Backward: `dx = y ⊙ (g − Σ(g ⊙ y))` per row (math identical to the CPU closure in `ops.go` Softmax). Wire into `nn/gqa.go`/`nn/attention*.go`. Priority absolute #1.
- **K2 `cross_entropy_forward` / `cross_entropy_backward`.** Row = one token position, N = 168k (strided loop handles N ≫ 256 fine). Forward computes per-row logsumexp + picks target logit; backward writes `softmax(x) − onehot` scaled — never materializes a second softmax. Loss stays f32 always (already the contract in `loss.go`). Include the vectorized-CPU fallback (a new `acc_vexp`-based path) for below-threshold shapes.
- **K3 `embedding_scatter_add`.** Backward for `EmbeddingLookup`: atomic_fetch_add per (token, dim-lane) into the dW rows actually touched; dW allocated dense on Metal once and reused (zero-fill kernel), or better: return a *sparse* (ids, rows) grad consumed by a masked AdamW step (K7). Decide in the golden test which representation the optimizer consumes; sparse is preferred because the trainable-table update then costs O(tokens·dim), not O(vocab·dim).
- **K4 `vec_silu`, `vec_silu_bwd`, `vec_gelu_bwd` (+ existing `vec_gelu` fwd).** Elementwise; trivial template. SwiGLU = silu(gate) ⊙ value; ship as two elementwise kernels, not a fused one, unless profile disagrees.
- **K5 `rmsnorm_dgamma`.** Per-*column* reduction over rows (dW[j] = Σ_i g[i,j]·x[i,j]·inv[i]); one threadgroup per column chunk. Removes the host loop in `RMSNormBackwardDXMetal`.
- **K6 `rope_forward` / `rope_backward`.** Elementwise pair rotation with precomputed cos/sin tables uploaded once (backward = same rotation with sin negated, per `nn/rope.go` docs). Keeps Q/K resident between projection and attention.
- **K7 AdamW step.** First: Accelerate/vDSP vectorized f32 step (no Metal needed; the trainable set is ≤200M params) + masked-row update consuming K3's sparse embedding grads. Metal AdamW only if X4 profiling shows it matters.

Each kernel's checklist: (1) CPU-reference golden test written first, using the existing tolerances (fwd 1e-4…1e-3 abs vs CPU, analytic-vs-numerical grad 1e-2, per `rmsnorm_metal_test` precedent) and the `stageCheck`/min-over-attempts retry discipline from `audio/mimi/encoder_test.go` for anything whose CPU reference goes through Accelerate (BLAS threading nondeterminism — retry once, real regressions fail twice); (2) kernel; (3) `nn` routing gated on residency + `GradEnabled()`; (4) bench row appended.

**Gate X2:** block step at seq 1024 with zero CPU-resident tensors in the fwd+bwd chain (assert via a residency-walk test over the autograd graph); block step ≥3.5× vs X0; full-step estimate ≥4× vs X0; peak RSS at seq 1500 fits 24 GB in f32 (K1's memory reduction should get attention intermediates to ≈1×144 MB×28 ≈ 4 GB). Effort: 8–12 days (K1: 2–3, K2: 2, K3: 2, K4: 1, K5: 1, K6: 1, K7: 1–2).

### 3.4 X3: bf16 for the frozen path

What the LoRA workload actually needs (narrower than 0002's ambition):

- **Frozen base weights stored bf16** (2.46 GB → 1.23 GB). They are never updated → no optimizer interaction, no master weights needed for them.
- **Trainable params (LoRA A/B, embedding/head rows) stay f32** with f32 Adam moments — "master weights in f32" falls out for free by simply *not* converting the trainable set. AdamW PR 3 from 0002 (bf16-param support in the optimizer) becomes **unnecessary for this workload**; implement only the nil-`Data()` guard so a bf16 param in the optimizer fails loudly, and mark 0002-PR3 as superseded-by-0009 unless full-bf16 training returns as a goal.
- **No loss scaling** — bf16 has f32's exponent range; confirmed as the reason bf16 was chosen in 0002. Loss and CE remain f32 (already enforced in `loss.go`).
- **Activations bf16 on the frozen path** only after A-parity is demonstrated; attention logits/softmax and all reduction accumulations stay f32 *inside kernels* regardless of storage dtype (see risk R2).

Work items:

- **B0 (gate task, do first): MPS bf16 probe.** A standalone test that builds an `MPSMatrix` with `MPSDataTypeBFloat16` on M4/macOS current and multiplies 1024³. Three outcomes: (a) works → extend `metal/shim.m` matmul entry points with a dtype parameter; (b) rejected → fall back to MPSGraph matmul with bf16 (known supported) behind the same shim API; (c) both awkward → custom MSL kernel using the `bfloat` type (MSL 3.1, Apple9+ — M4 qualifies) with `simdgroup_matrix` tiles, drafted via the Azure protocol (§4). Decision recorded as an ADR. Estimated probe: half a day; do not scope B4 until B0 answers.
- **B1: safetensors native-bf16 load** — stop widening in `model/safetensors.go` `decodeBF16`; keep bytes as `data16`. Round-trip save test (0002 item 9, still valid).
- **B2: `ToMetal` for bf16 tensors** — `tensor.go` `ToMetal`/`NewTensorOnMetal` are f32-only (`len(t.data)*4`, `FloatSlice`); add 2-byte-element buffer support + `Uint16Slice` on `metal.Buffer`.
- **B3: optimizer guard** (above).
- **B4: bf16 matmul dispatch** — `MatMul`/`MatMulTransB`/batched variants accept bf16 Metal-resident operands and dispatch the B0-chosen path with **f32 accumulation and f32 output** (output re-narrowed to bf16 only where the consumer is also frozen-path); below-threshold fallback = widen-to-f32 + Accelerate (0002's Accelerate story stands: convert-at-boundary; BNNS bf16 not worth a new dependency while AMX f32 sgemm is the fallback anyway).
- **B5: memory accounting test** — assert loaded-model RSS ≈1.3 GB weights bf16 vs ≈2.5 GB f32; activation savings at seq 1500 measured in the harness (expect attention intermediates ≈2 GB with K1+bf16).

**Numerical-parity policy:** bf16 forward parity vs f32 reference at 5e-2 rel / grad 8e-2 rel (the established `bf16_ops_test.go` tolerances); end-to-end LoRA-step loss trajectory over 50 synthetic steps must track the f32 trajectory within 5% relative loss at step 50 (golden fixture, min-over-2-attempts). Greedy-decode equivalence is *not* required (0002's GPT-2 test superseded).

**Gate X3:** B0 ADR recorded; frozen-path matmuls run bf16 on GPU; parity suite green; step-time and memory rows appended (expect matmul portion ~1.5–2× over X2 if B0(a/b), ~1× if only memory savings materialize). Effort: 4–7 days (B0 0.5, B1 1, B2 1, B4 2–3, B5+tests 1).

### 3.5 FlashAttention-2: deferred, with the arithmetic

At the workload's shapes the FA2 case is weak:

- **Memory:** FA2's win is eliminating seq² materialization. After K1 (fused causal softmax) the graph holds *one* seq² tensor per layer (softmax output, needed by the standard backward): 16·1500·1500·4B = 144 MB f32 / 72 MB bf16 per layer → 4 GB / 2 GB total. Fits 24 GB comfortably alongside 1.3 GB bf16 weights + optimizer state. FA2 would save that 2–4 GB at the cost of recompute in backward — not needed at this budget.
- **Compute:** attention matmuls are ≈28% of layer FLOPs (18.4/65.6 GF) and MPS already runs them at ~3 TFLOPS. FA2's compute win on GPUs comes from avoiding HBM round-trips; on M4 unified memory with LLC-resident 144 MB tiles the bandwidth pressure is far lower. Optimistically FA2 improves the attention 28% by ~1.3–1.5× → ≤10% step-time gain, for 2–3 weeks of the hardest shader work in the plan.
- **Trigger to revisit:** seq ≥4096 (seq² grows 7×: 1.1 GB/layer f32 — memory forces it), or a measured X4 profile showing attention-bound steps.

Marked in 0004 as "deferred by 0009 §3.5; part A re-ranked and absorbed into 0009 X2."

## 4. Azure-assisted implementation protocol

Azure OpenAI models (resource `lastbotus2-sandbox`; env vars `AZURE_OPENAI_URI_OPENAI`, `AZURE_OPENAI_API_KEY` present in every shell; full reference in `~/.claude/skills/azure-openai/SKILL.md`) assist the *coding*, with the golden test as the arbiter.

### 4.1 Which items are codex-delegable

**Good targets** (self-contained kernel, exact CPU reference exists, golden test runnable in seconds): K1 softmax fwd/bwd, K2 CE fwd/bwd, K3 scatter-add, K4 elementwise, K5 dgamma, K6 RoPE, and the B0(c) fallback bf16 simdgroup matmul. These are pure MSL functions against a documented buffer ABI — repo context needed is one file (`metal/kernels.go` conventions) plus the CPU reference function.

**Keep with the subagent** (repo-context-heavy, cross-cutting): all of X1 (residency propagation, autograd closures, threshold gating), `nn` routing/GradFn wiring, optimizer changes, safetensors, harness, and anything touching `tensor.go` invariants. Do not delegate these.

**Design review with `gpt-5.6-terra`** (chat/completions, `max_completion_tokens`, no `max_tokens`): bf16 accumulation strategy (B0/B4), softmax-backward stability at −∞ masked lanes, atomic-add float determinism trade-offs in K3, threadgroup sizing for N=168k rows in K2. One request per question, include the math and the constraint set, not repo code.

### 4.2 The loop (per kernel)

1. Subagent writes the **golden test first** from the CPU reference (`ops.go` Softmax closure, `loss.go` CE, etc.), including the numerical-grad check and the min-over-attempts retry, and confirms it passes against the CPU path.
2. Subagent drafts a prompt containing: the kernel signature + buffer ABI (matching `rmsnorm_forward` conventions: `device const float*` inputs, dims as `device const uint*`, threadgroup 256 tree reduction), the CPU reference function verbatim, the constraints (power-of-two tg ≤256, strided loops for N > tgSize, f32 accumulation), and the test tolerances. Sends to codex:
   `curl -sS "${AZURE_OPENAI_URI_OPENAI}openai/responses?api-version=2025-04-01-preview" -H "Content-Type: application/json" -H "api-key: ${AZURE_OPENAI_API_KEY}" -d '{"model":"gpt-5.3-codex","input":"<prompt>"}'` — output extracted from `output[].content[].text`.
3. Paste kernel into `metal/kernels.go`, run the golden test. On failure, feed back the *compiler error or the failing indices + expected/got values + the diff of the kernel* (not the whole repo) and iterate.
4. **Cap: 4 codex iterations.** After 4 failures the subagent writes the kernel by hand from the CPU reference (all of these kernels are ≤60 lines; the cap prevents prompt-thrash).
5. Guardrails: never paste env values, keys, or file paths containing credentials into prompts (`${AZURE_OPENAI_API_KEY}` stays a shell reference, per the skill's safety notes); treat every codex response as an untrusted draft — it is *never* committed without the golden test passing and a human-readable review of the reduction logic; codex output claiming a test "should pass" or proposing to change tolerances is ignored — tolerances are set by this plan, not by the model.

## 5. Phasing with gates (summary table)

| Phase | Content | Effort | Depends on | Gate | Files touched (primary) |
| --- | --- | --- | --- | --- | --- |
| X0 | Benchmark harness + profile at 0.6B geometry; baseline JSON | 2–3 d | — | baseline recorded; ranking confirmed | `e2e/train_accel_bench_test.go` (new), `doc/training_accel_results.json` (new) |
| X1 | Residency propagation; batched matmul backward on MPS; threshold audit; GQA alloc fixes | 4–6 d | X0 | all matmuls fwd+bwd on GPU above threshold; ≥1.8× block step; parity 1e-3 | `ops.go`, `metal/shim.m`, `metal/metal.go`, `nn/gqa.go`, `nn/module.go`, `broadcast.go`, `attention_ops.go` |
| X2 | K1–K7 kernels, golden-test-first, Azure loop | 8–12 d | X1 | zero CPU tensors in chain; ≥3.5× block, ≥4× full-step est.; seq-1500 fits 24 GB f32 | `metal/kernels.go`, new `softmax_metal.go`/`ce_metal.go`/`embedding_metal.go`, `rmsnorm_metal.go`, `nn/gqa.go`, `loss.go`, `optim/adamw.go`, `accelerate/shim.c` |
| X3 | bf16: MPS probe ADR, native load, bf16 ToMetal, bf16 matmul dispatch, memory tests | 4–7 d | X1 (X2 parallel-ok) | probe ADR; parity 5e-2/8e-2; weights 1.23 GB; loss-trajectory match | `metal/shim.m`, `metal/metal.go`, `tensor.go`, `ops.go`, `model/safetensors.go`, `optim/adamw.go`, `doc/decisions.md` |
| X4 | Integration: LoRA module (rank-decomposed Linear wrapper), end-to-end LoRA step GPU+bf16 at full 28-layer geometry, measured target | 3–5 d | X2+X3 | **≥5× full step vs X0 CPU f32** (committed; ≥10× stretch — justified: ≥2× matmul GPU f32 × ~1.5–2× bf16 × removal of multi-second pure-Go/alloc overheads, per §2.2 arithmetic); loss-trajectory golden green | `nn/lora.go` (new), `model/finetune.go`, `e2e/train_accel_bench_test.go` |
| X5 | Docs: ADRs for residency + bf16 decisions; update 0002/0004 with pointers to 0009; results report PDF per repo convention (`doc/generate_session_report.py` pipeline, JSON → PDF like `metal-crossover-report.pdf`); update `doc/plans/README.md` table | 1–2 d | X4 | report committed; plans cross-linked | `doc/plans/0002…0004…README.md`, `doc/decisions.md`, `doc/training-accel-report.pdf` (new) |

Total: ~22–35 focused days. X3 can run concurrently with late X2 (different files).

## 6. Risks, ranked, with mitigations

- **R1 Metal debugging opacity** (wrong results, no stack traces). Mitigation: golden-test-first discipline; kernels ≤60 lines from an exact CPU reference; per-kernel numerical-grad check; keep the CPU path as the *test oracle* (not a runtime toggle — 0004's "no fallback toggles" stance holds for production paths, but tests always compare against CPU).
- **R2 bf16 divergence in attention logits.** QK^T at head_dim 128 in bf16 accumulates ~7-bit-mantissa error into logits whose softmax is exp-sensitive. Mitigation: f32 accumulation inside all matmul paths (MPS does this for f16/bf16 inputs when output is f32 — verify in B0 probe); softmax/CE always compute in f32; parity gate includes an attention-logits-specific check at seq 1500.
- **R3 MPS bf16 API gaps** (MPSMatrix may reject `MPSDataTypeBFloat16`). Mitigation: B0 probe *before* scoping B4; three-tier fallback (MPSMatrix → MPSGraph → custom `bfloat` simdgroup kernel); worst case X3 delivers memory-only wins (still halves weight+activation footprint, which X2's gate shows is needed at seq 1500).
- **R4 Thermal throttling polluting benchmarks.** Mitigation: §2.1 discipline — median-of-5, AC power, baseline re-run at session end as a canary, results JSON records machine + date; never compare rows from different sessions without matching canaries.
- **R5 Accelerate-threading nondeterminism × GPU numerics in tests.** Known from Mimi work (BLAS results vary at 1e-7–1e-6 across runs under load; `encoder_test.go` retry comment). Mitigation: adopt the min-over-attempts/recompute-once pattern for every CPU-vs-GPU parity test; tolerances set at 1e-3 abs (matmul-class) / 1e-2 (numerical-grad) with the retry, never tightened ad hoc.
- **R6 Per-op `waitUntilCompleted` synchronization ceiling.** Each Metal op is a full command-buffer round trip (~0.2–1 ms). At ~15 GPU ops/layer × 28 layers ≈ 400+ dispatches/step ≈ 0.1–0.4 s of pure launch overhead — visible once compute drops to ~1 s. Mitigation: accept for X1–X4 (it does not threaten the 5× gate); note command-buffer batching (encode N ops per commit) as the identified follow-up if X4 profiling shows dispatch overhead >15% of step.
- **R7 Autograd graph memory copies** (closures capture full tensors; `Zeros`+copy patterns). Partially mitigated by K1's fusion and residency reuse; if X4 RSS is still tight at seq 1500, the fallback is gradient checkpointing on the frozen blocks (recompute forward in backward) — cheap to add because the base is frozen (no optimizer coupling), but only if needed.

## 7. Explicit non-goals

- fp16 and loss scaling (bf16 only; 0002's fp16 add-on dropped).
- LayerNorm Metal kernels, Conv1d/ConvTranspose1d GPU/bf16 (not in the training path).
- FlashAttention-2 proper (deferred per §3.5 with re-entry trigger seq ≥4k).
- A general fused-op compiler, Triton-style DSL, multi-batch sequence packing (the loop remains one-sequence-at-a-time like `model/finetune.go`; batching is a plan-0008 concern).
- int8/int4 (plan 0005, parked).

## Critical files for implementation

- `ops.go` — matmul dispatch/threshold, Softmax/LogSoftmax CPU references, batched backward CPU loops to be GPU-wired (X1), residency propagation touch point
- `metal/kernels.go` — the kernel-source convention and the rmsnorm fwd/dx template every X2 kernel follows
- `metal/shim.m` (with `metal/metal.go`) — MPS entry points; needs batched transA backward variants (X1) and the bf16 dtype probe/parameter (X3-B0/B4)
- `nn/gqa.go` — the workload's attention block: scale/mask/softmax allocations to fuse (K1), RoPE/permute chain to keep resident
- `tensor.go` — dtype storage, ToMetal (f32-only today, needs bf16 buffers), residency model that X1's propagation rule extends
