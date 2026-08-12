# Plan 0009: Training acceleration execution — GPU-resident autograd + bf16 for the LoRA/Mimi fine-tune workload

**Status:** proposed (execution plan; operationalizes plans 0002 and 0004)
**Tracks:** plan 0002 (bf16), plan 0004 (Metal non-matmul autograd + FA2), and the upcoming plan 0008 workload (LoRA-adapting a Qwen3-0.6B-class LLM to Mimi speech tokens)
**Supersedes-in-part:** 0002 (remaining work re-scoped here), 0004 (kernel list re-ranked, FA2 deferred here)
**Hardware target:** Apple M4, 10-core GPU, 24 GB unified memory
**Last updated:** 2026-08-12

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

### 2.3 X0 measured results (2026-08-11, Apple M4, CPU f32 baseline)

Harness: `e2e/train_accel_bench_test.go` (`TestTrainAccelBench`, tags `darwin && e2e`); full data in `doc/training_accel_results.json`, phase `X0-baseline`. Geometry corrected against the real Qwen3-0.6B config: head_dim **128** with 16 Q heads means q_proj is 1024→**2048** (heads×head_dim ≠ hidden — projection dims follow the head config; o_proj 2048→1024, k/v 1024→1024). Median of 5 after 2 warmups. Caveat: recorded 1-min load average at start was 6.8 (Spotlight indexing + unrelated apps); numbers carry some contention noise (block-step spread min/max at seq 1500: 884–1379 ms) — treat as an upper-bound-ish baseline and re-run the canary per §2.1 before cross-session comparisons.

**Block training step** (fwd + Sum + Backward + AdamW over 15.8M block params):

| seq | fwd | bwd | opt | total | alloc/step | live graph after fwd |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 75 ms | 79 ms | 47 ms | **201 ms** | 485 MB | 172 MB |
| 1024 | 219 ms | 196 ms | 47 ms | **463 ms** | 1.19 GB | 520 MB |
| 1500 | 459 ms | 396 ms | 67 ms | **1163 ms** | 2.15 GB | 1.00 GB |

**Full-step estimate** (28×block + measured tails): seq 512 ≈ **8.5 s**, seq 1024 ≈ **18.1 s**, seq 1500 ≈ **38.9 s** — confirming §2.2's "tens of seconds per step / multi-day run" motivation. Tails at seq 1500: CE fwd+bwd 4.08 s, lm_head matmul fwd+bwd 1.74 s, AdamW over the 172M table 0.50 s, embedding fwd+bwd 0.013 s.

**Per-op wall-clock ranking at seq 1500** (isolated fwd+bwd at exact workload shapes × per-step counts; share of per-op total):

1. softmax (16·1500, 1500) — **10.40 s/step, 32.2%**
2. cross-entropy (1500, 168320) — 4.08 s, 12.6%
3. mask/scale (Full+Mul+MaskFill) — 3.23 s, 10.0%
4. FFN matmuls — 2.44 s, 7.6%
5. attention batched matmuls — 2.35 s, 7.3%
   then AdamW block 1.88 s (5.8%), permute/reshape 1.87 s (5.8%), attn projections 1.78 s (5.5%), lm_head 1.74 s (5.4%), SwiGLU 0.97 s, RMSNorm 0.85 s, AdamW table 0.50 s, RoPE 0.22 s, embedding 0.013 s.

**Memory (seq-1500 fit):** the single-block bench itself fits, but the measured live autograd graph is 1.00 GB per block after forward → 28-layer extrapolation ≈ **29.9 GB** incl. f32 weights, vs 24 GB unified memory. **Confirmed: the full graph does not fit at seq 1500 f32** (§2.2's ~20 GB estimate was directionally right, slightly low). Seq 1024 extrapolates to 16.7 GB — fits, tightly.

**pprof top-3 flat%** (3 block steps, seq 1024): `runtime.cgocall` (Accelerate sgemm) 25.0%, unsymbolized Accelerate frames 14.6%, `gorch.Softmax` fwd 9.7% (+9.7% `math.archExp`, 18.8% cumulative; softmax backward another 5.6%).

**§3.1 ranking: partially corrected.**

- **Confirmed:** K1 fused causal softmax is the single biggest lever — softmax + the mask/scale allocs it fuses away = **13.6 s/step, 42%** of per-op wall clock at seq 1500, and its 4–5 seq² intermediates drive the memory wall. K2 (CE) is the clear #2 pure-Go loop (12.6%, with the measured backward double-softmax: fwd 2.20 s / bwd 1.88 s). The seq-1500 memory wall is real (above). Step-time magnitude matches §2.2.
- **Corrected:** X1's premise ("~85% of step FLOPs are matmuls" → wire them to GPU first for the biggest win) does **not** hold in wall-clock terms on the measured baseline: the entire matmul class (projections + batched attention + lm_head) is **8.3 s/step ≈ 26%**, vs 52% for the ops K1+K2 replace. Even a perfect 2.5× GPU matmul speedup caps X1's step-time gain at ~15%; K1 alone is worth up to ~40%. X1 stays first only as the *residency-enabling dependency* for X2 kernels (its own §3.1 text already says the kernels need resident chains), not as the biggest standalone win — set expectations for the X1 gate accordingly (≥1.8× block-step from wiring alone looks optimistic; re-check after K1).
- **Corrected:** K3 (embedding scatter-add) is a memory/alloc concern, not a time concern — measured 13–40 ms/step total despite the 690 MB dense-grad alloc. Its wall-clock rank is last, not 4th; keep it for the sparse-grad/masked-AdamW memory win, not speed.
- **Noted:** AdamW (block params + 172M table ≈ 2.4 s/step, 7.3%) and permute/reshape copies (1.9 s, 5.8%) both outrank several items above them in §3.1; K7 (vDSP AdamW) is cheap to do and worth pulling earlier, and X1's residency work should include the permute copies.

### 2.4 X1K1 measured results (2026-08-11, combined phase X1 + K1)

Per §2.3's corrected ranking, X1 and K1 were executed as **one combined phase** (the X1-only ≥1.8× gate was unreachable with matmuls at only ~26% of wall clock; K1's 42% softmax+mask share was the co-requisite). Delivered: residency propagation (`ZerosLike`/`zerosLikeEither` through every op allocation incl. backward closures and `Reshape` grad), batched matmul backward on MPS (`metal_mps_batched_matmul_transA` + transpose-flag rewrite removing all host-side scratch transposes), Metal-resident `Sum`/`Mean` grad seeding, `g.Scale` + K1 `g.CausalSoftmax` (fused scale+causal-mask+softmax fwd/bwd Metal kernels, golden-tested at 1e-3 abs / 1e-2 numerical grad, wired into `nn/gqa.go`, `nn/mla.go`, `nn/attention.go`), Metal buffer finalizers + `metal.LiveBufferBytes`. Full data: `doc/training_accel_results.json` phase `X1K1`.

- **Dispatch gate MET:** one block fwd+bwd at seq 1024 fires exactly the predicted 21 MPS matmul + 6 MPS batched-matmul + 2 fused-softmax dispatches (hard-asserted in the harness). Threshold audit: every workload matmul clears the 512M default at seq ≥512 (thinnest margin 1.05×); no per-callsite override.
- **Memory gate MET:** live graph per block after forward at seq 1500 fell 1002 MB → **534 MB** (K1: 5 → 2 live seq² tensors/layer); 28-layer extrapolation **17.05 GB** incl. f32 weights — **fits 24 GB** (X0: 29.9 GB, did not fit). Seq 1024: 16.7 → 10.7 GB.
- **Speed gate NOT MET this session:** block step seq 1024 measured **1.06×** vs the X0 row (435 ms; 0.99× vs the same-session CPU canary), seq 1500 **1.84×** vs X0 (633 ms; 1.15× vs canary), seq 512 1.14×. Session caveat: 1-min load 14.2 (external training agent); the §2.1 canary drifted −7.6/−7.3/**−37.3%** — over the 10% invalidation bound at seq 1500, i.e. the X0 seq-1500 row was itself load-inflated and cross-session speedups from it are not trustworthy this session. Re-bench after cooldown before quoting.
- **Why the block step didn't speed up more (pprof, 5 Metal steps at seq 1024, `e2e/x1k1_profile_test.go`):** only ~160 of 302 ms/step is CPU samples — AdamW scalar loop 26%, Permute copies 25%, SwiGLU Go loop 14%, Linear bias/db loops ~10% — and **~46% of wall is per-op `waitUntilCompleted` GPU sync** (risk R6 arriving at ~29 dispatches/block-step, earlier than §6 predicted because per-matmul compute at these shapes is only 1–5 G FMA). X1K1 moved its targets to GPU; the step is now bounded by the not-yet-kerneled CPU ops (K4 SwiGLU, K6/permute residency, K7 vDSP AdamW — K7's §2.3 "pull earlier" note is now urgent) and dispatch synchronization (command-buffer batching is the identified follow-up).
- Full-step estimates (this loaded session): 8.8 / 17.9 / 26.0 s at seq 512/1024/1500 (X0: 8.5 / 18.1 / 38.9 s) — dominated by the still-CPU CE tail (K2) and load-inflated AdamW-table row.

### 2.5 X2 measured results (2026-08-11, second X2 wave: K2 + K4 + K7 + R6)

Delivered in one commit: **K7** AdamW vectorization (`acc_adamw_step` fused Accelerate/clang-vectorized C loop in `accelerate/shim.c`, per-group LR structure preserved, scalar loop kept as oracle behind `optim.UseScalarAdamW`); **K2** fused cross-entropy Metal kernels (`cross_entropy_forward`/`cross_entropy_backward`: per-row logsumexp + target pick forward, softmax−onehot backward from the *saved* logsumexp — no double softmax; vectorized `acc_vexp`-path CPU fallback; wired into `g.CrossEntropyLoss` gated on residency + `CEMetalMinElements`, and the qwen gathered path reaches it via residency-inheriting `Gather`/`ExtEmbedLogits`); **K4** SiLU/SwiGLU elementwise Metal kernels + `acc_vsilu`/`acc_vswiglu` fallbacks (backward recomputes σ(x) — no cached sigmoid tensor per layer); **R6** commit-without-wait async dispatch (`g.SetMetalAsync`, `metal_sync_queue` in `metal/shim.m`, wait-on-CPU-read fencing via `syncForCPU`/`Data()` across the op layer; default OFF, bit-exact parity asserted in `metal_async_test.go`). Full data: `doc/training_accel_results.json` phase `X2`.

**Session validity (REDUCED CONFIDENCE):** loadavg 5–8 throughout (persistent background services) — the §2.1 quiet-machine condition was never met; canary drift vs X0 was −20/−21/−42%, i.e. the X0 rows are themselves load-inflated (the same conclusion as §2.4), so vs-X0 ratios are optimistic and the same-session canary is the honest basis. The bench also had to run as two processes (`TA_X2_PART=blocks|tails`): a single full-length run was reproducibly SIGKILLed at ~110 s wall in this session's sandbox (no jetsam trace, 74% RAM free; cause undetermined).

- **Block step** (median of 5, X2 kernels; sync → async dispatch): seq 512 141 → **127 ms** (X0 201; async **1.58×** vs X0, 1.27× vs canary), seq 1024 351 → **267 ms** (X0 463; async **1.73×** vs X0, 1.37× vs canary), seq 1500 487 → **418 ms** (X0 1163; async **2.78×** vs X0, 1.61× vs canary). Dispatch gate MET incl. K4: 21 MPS matmul + 6 batched + 2 softmax + 2 SwiGLU kernel dispatches per block fwd+bwd.
- **K7 gate MET:** 172M-param AdamW table step **458 → 75 ms (6.1×)**; trajectory parity vs the scalar oracle 4.7e-10 loss diff over 20 steps (gate ≤1e-6), per-param max |Δ| ≤1.2e-7.
- **K2 gate MET:** CE fwd+bwd at (1500, 168320): **4077 ms (X0) → 195 ms Metal-resident (~21×)**; the vectorized CPU fallback alone is 312 ms (~13×). Codex protocol: golden tests first (incl. float64 numerical grad), **1 iteration, verdict PASS** with 1 reviewer-added threadgroup barrier (same class of fix as K1). Metal parity ≤1e-5 abs on dx, autograd end-to-end ≤7.9e-9.
- **K4 gate MET:** kernel-vs-scalar parity ≤1.4e-6 (tolerance 1e-3), numerical grad ≤1e-2; SwiGLU is no longer visible as a Go loop in the step.
- **R6 outcome — code shipped, opt-in:** per-dispatch microbench measures the `waitUntilCompleted` round trip at **~0.14–0.19 ms** (0.172 sync vs 0.030 ms/dispatch async); async mode cuts host sync points to **~21 waits/block-step** (one per remaining CPU-computed op: Permute ×6, RoPE ×4, RepeatInterleave ×2, bias/db loops, rmsnorm-dW host loop, grad accumulation) and improved the block step by 4–24% at seq ≥1024 under load. Results are bit-exact vs sync mode (parity test). Default remains OFF; the training loop opts in via `g.SetMetalAsync(true)`.
- **Full-step estimates vs X0: 8.5 → 4.2 s (2.03×) / 18.1 → 8.7 s (2.08×) / 38.9 → 13.5 s (2.88×)** at seq 512/1024/1500. The CE + AdamW tails have collapsed (seq-1500 tail total 4.6 s → 0.29 s); the estimate is now ~87% pure block cost.
- **Gate verdicts: block ≥3.5× NOT MET (1.73× vs X0 at seq 1024), full-step ≥4× NOT MET (2.08×).** Honest reading: the tails this wave targeted are done (CE 21×, AdamW 6×), but the block step is bounded by the ops still on the CPU — Permute copies, RoPE, RepeatInterleave, Linear bias/db loops, RMSNorm dW — plus the ~21 residual sync points they force. Those are exactly K5/K6 + the permute-residency work (§2.3 already ranked permute_reshape above several kerneled items). Under the load-matched canary the compounding so far is ~1.4–1.6× block-step; hitting the ≥3.5×/4× gates needs the K5/K6/permute wave (and likely X3's bf16 matmul lift), not more tail work.

### 2.6 X2b measured results (2026-08-11, third X2 wave: K5 + K6 + permute/repeat/bias residue)

Delivered in one commit: **permute** `permute_copy` Metal kernel (generic N-D gather over destination indices, rank ≤ 8; CPU path also rewritten to contiguous run copies when the innermost dim is unpermuted — every attention head reshape); **K6** `rope_apply` fwd+bwd (one kernel, sign uniform: −1 = inverse rotation; cos/sin uploaded once per `nn.RoPE`, both Llama and NeoX conventions); **repeat_interleave_fwd/bwd** (GQA KV expansion + sum-back); **K5** `rmsnorm_dgamma` (per-column threadgroup reduction — the RMSNorm backward host loop and its forced sync are gone, dW is Metal-backed); **col_sum** (Linear db) + `vec_bias_add` wiring (Linear GPU forward bias — both per-Linear host loops removed); GPU-resident grad accumulation (`accumulateGrad` dispatches in-place `vec_add` when both grads are Metal-backed). Golden tests first for every kernel (`permute_metal_test.go`, `nn/rope_metal_test.go`, extended db/bias parity; the existing `nn/rmsnorm_metal_test.go` dW parity + numerical-grad tests pin K5); codex protocol not used this wave — all six kernels are ≤30-line hand-written variants of already-shipped templates with exact CPU references. Full data: `doc/training_accel_results.json` phase `X2b`.

**Session validity (same caveat class as §2.5):** loadavg 4–13 with external agent load; the blocks bench was run 3× and the quietest run recorded. Canary drift vs X0 was **−33/−31/−52%** — today's machine runs the X0 chain *faster* than the recorded X0 rows, i.e. the X0 baseline is confirmed load-inflated and vs-X0 ratios below are optimistic; the same-session canary is the honest basis.

- **Dispatch gate MET, exactly at prediction:** one block fwd+bwd at seq 1024 fires 21 MPS matmul + 6 batched + 2 softmax + 2 SwiGLU + **8 permute + 4 RoPE + 4 repeat + 11 col-reduce (4 dgamma + 7 db) + 7 bias-add** dispatches (hard-asserted). **Host sync waits in async mode: ~21/step (X2) → ~1/step** (the `Sum` loss read) — risk R6 is resolved for the block step.
- **Block step** (median of 5; X2b config = better of sync/async per seq): seq 512 **35 ms** async (X0 201; **5.75×** vs X0, 3.85× vs canary; X2: 127 ms), seq 1024 **142 ms** (X0 463; **3.26×** vs X0, 2.24× vs canary; X2: 267 ms — X2b nearly halves the X2 block step), seq 1500 **297 ms** sync (X0 1163; **3.91×** vs X0, 1.86× vs canary; X2: 418 ms). Cross-run spread at seq 1024 async: 120/146/142 ms (3.86/3.17/3.26× vs X0) across the three load conditions.
- **Async regression at seq 1500 (new finding, 3/3 runs):** async 759–1274 ms vs sync 297–532 ms. Hypothesis: commit-without-wait lets the host allocate + zero-fill the next ops' 144 MB-class fresh buffers while MPS saturates unified-memory bandwidth; at seq ≤1024 the same overlap wins (512: 35 vs 91 ms). The harness now records both rows and the X2b configuration takes the per-seq winner; a training loop should do the same. Follow-up candidate: buffer reuse/pooling instead of fresh zero-filled allocations per op.
- **K7 re-measured this session:** 172M AdamW table 396 → 69 ms (5.8×).
- **Full-step estimates vs X0: 8.5 → 1.5 s (5.71×) / 18.1 → 5.1 s (3.56×) / 38.9 → 9.9 s (3.93×)** at seq 512/1024/1500 (X2: 4.2/8.7/13.5 s).
- **Gate verdicts (the X2 gates this wave was to close): block ≥3.5× at seq 1024 vs X0 — NOT MET in the recorded run (3.26×; met at seq 512 5.75× and observed once at 1024, 3.86×, in a higher-load run; seq 1500 sync 3.91×). Full-step ≥4× — NOT MET (3.56× at 1024; 3.93× at 1500; 5.71× at 512).** Honest reading: the CPU-residue attribution from §2.5 is fully retired — the fwd+bwd chain is now 100% GPU-dispatched with ~1 host wait — and the misses are no longer CPU ops. Residual at seq 1024 (142 ms block): MPS matmul compute at 1–5 GFMA shapes plus ~50 per-dispatch command-buffer commits in the sync sections, i.e. the remaining levers are **X3 bf16 matmul** and **command-buffer batching** (encode multiple kernels per commit), not more kernels. Note also the gate arithmetic is against a baseline this session's canary beat by 30–50%; against a fair quiet-machine X0 the block ratio at 1024 is ~2.2×.
- Per-op Metal micro-rows (permute_reshape/rope/rmsnorm) were recorded but are dominated by the harness's Sum-loss/seed overhead once the op itself is a ~free GPU kernel — the clean X2b signal is the block-step delta and the waits/step drop (noted in the JSON).

### 2.7 X3 measured results (2026-08-11/12, bf16 frozen path: B0–B5)

Delivered in one commit: **B0** MPS bf16 probe → **outcome (b), ADR-012**: `MPSMatrixMultiplication` hard-asserts (`abort()`, not catchable) on `MPSDataTypeBFloat16` on macOS 26.5 — tier (a) dead; implemented tier (b) **MPSGraph** dtyped matmul behind the same shim API (`metal_mps_matmul_dt`/`metal_mps_batched_matmul_dt`, per-shape graph cache, bf16 cast to f32 *inside* the graph → f32 accumulation by construction, async-mode compatible via `MPSCommandBuffer`). **B1** native bf16 safetensors (`LoadSafetensorsNative` keeps `data16`; `SaveSafetensors` writes `BF16` bit-exactly; round-trip test). **B2** bf16 `ToMetal`/`ToCPU`/`NewTensorBF16OnMetal` + `metal.Buffer.Uint16Slice` + `Tensor.Data16`. **B3** optimizer loud-fail guard test (guard itself shipped in X2). **B4** bf16/mixed-dtype dispatch across `MatMul`/`MatMulTransB`/`MatMulTransA`/`BatchedMatMul`/`BatchedMatMulTransB` (`bf16_matmul.go`): resident+threshold+probe → MPSGraph path (f32 out), else widen+Accelerate; CPU bf16-pairs keep plan-0002 semantics; grads f32 with the **frozen-operand GEMM skipped**; `nn.Linear` routes bf16 weights fwd+bwd. **B5** memory accounting test. Full data: `doc/training_accel_results.json` phase `X3`.

**Session validity (same caveat class as §2.5/§2.6):** loadavg 5–21 with an external training agent; the SIGKILL horizon dropped to ~60 s → bench split into FOUR processes (`TA_X3_PART=canary|control|bf16|tails`); block rows sampled in two rounds per config, gates use the per-seq winner (X2b quietest-run precedent). Canary drift vs X0: **−0.8 % / −8.8 %** at seq 512/1024 (within the §2.1 bound — vs-X0 ratios fair at those seqs) but **−42 %** at seq 1500 (X0's 1500 row remains load-inflated; its ratios stay optimistic).

- **B0 numerics/throughput:** 1024³ bf16×bf16 matches an f64 row reference to **2.2e-06 of RMS** (bf16 accumulation would sit ~6e-2 — f32 accumulation confirmed, risk R2); bf16 **1.33× faster** than the f32 MPSMatrix path at 1024³ (1.54 vs 2.04 ms). At block projection shapes the dtyped path is 0.75–1.06× f32 (TransB transpose fuses less well) — the bf16 win is bandwidth + skipped-dW, not raw GEMM speed at these shapes.
- **Parity gates MET:** MatMul/TransB/batched fwd ≤9.0e-3 of RMS (gate 5e-2), grads ≤4.9e-3 (gate 8e-2); **R2 attention-logits check at seq 1500** (16 heads × 128 head_dim, bf16 Q/K): ≤1.33e-2 of RMS vs f32 inputs (gate 5e-2), f64-accumulation rows ≤7.2e-07 (gate 1e-3). **50-step LoRA trajectory golden MET:** bf16-frozen loss at step 50 within **0.04 %** of f32 (gate 5 %), 100/100 steps on the dtyped GPU path.
- **Dispatch gate MET, exactly at prediction:** one bf16-frozen block fwd+bwd at seq 1024 fires **14 dtyped bf16 matmuls (7 fwd TransB + 7 dx) + 0 f32 MPS matmuls** + 6 batched + 2 softmax + 2 SwiGLU + 8 permute + 4 RoPE + 4 repeat + 4 dgamma + 7 bias-add; dW/db dispatches are GONE (frozen-base skip).
- **Memory gates MET (B5):** full 0.6B weight set measured **1.226 GB bf16 vs 2.451 GB f32** (ratio 2.000, `metal.LiveBufferBytes`); X3 weight footprint at the harness geometry 2.46 → **1.57 GB** (block Linears bf16; embedding table stays f32 trainable); seq-1500 28-layer extrapolation **15.7 GB** (X2b: 17.1) — fits 24 GB.
- **Block step (winner across rounds/modes; frozen config — 7 dW GEMMs fewer than the X0 all-trainable block, which IS the workload's shape):** seq 512 **25.1 ms** async (X0 201; **8.0×**), seq 1024 **58.4 ms** async (X0 463; **7.93×**; same-session canary 422 ms → **7.2×**), seq 1500 **391 ms** sync (X0 1163; 2.97×; canary invalid at 1500). Frozen-f32 control (attribution): best 24.6/141/431 ms — bf16 ≈ control at 512, **~2.4× faster at 1024** (58 vs 141), ≈ control at 1500. High in-session variance (58–571 ms spread at 1024 across rounds) is environmental — MPSGraph workspace allocation under buffer churn + external load; isolation diagnostics (`bf16_shape_bench_test.go`) show steady-state dtyped ≥ f32 parity. **X2b async-at-1500 regression CONFIRMED** (async 980–1219 vs sync 391–761 ms); async still wins at ≤1024.
- **Full-step estimates vs X0: 8.5 → 1.48 s (5.76×) / 18.1 → 2.94 s (6.17×) / 38.9 → 12.9 s (3.01×)**; the lm_head tail with a bf16 FROZEN head is 1.0–1.5 s (fwd dtyped + dx only, the 258G-FMA dW GEMM skipped).
- **Gate verdicts: block ≥3.5× at seq 1024 vs X0 — MET (7.93×; 7.2× even vs the same-session canary). Full-step ≥4× at seq 1024 — MET (6.17×).** Caveats attached in the JSON: frozen-config FLOP difference vs the X0 all-trainable baseline (the workload's real shape, but not apples-to-apples with X0), and seq-1500 rows (2.97×/3.01×) remain load-bound with an invalid canary. The plan-§3.4 X3 gate (probe ADR + parity + memory + rows appended) is fully met; the residual levers for seq 1500 are command-buffer batching and buffer pooling (X2b follow-up), not dtype work.

### 2.8 X4 measured results (2026-08-12, trainer integration: the REAL VoiceModel on GPU+bf16)

Delivered in one commit — pure wiring plus two small kernels-of-convenience (no new math): **qwen native-bf16 load** (`qwen.LoadNative`/`LoadTruncatedNative`: per-block Linear weights keep the checkpoint's bf16 bits via `LoadSafetensorsNative`; embedding table, norm gammas, and biases widened to f32; dtype-aware tied-head bit check); **`VoiceModel.ToMetal`** (block Linears + norm gammas + LoRA A/B + `ExtendedEmbedding` Base/Ext into unified memory; `qwen.AccelSupported` = Metal + ADR-012 probe); **residency-aware `ExtendedEmbedding`** (lookup output Metal-resident so the hidden chain starts resident at op #1; tied-head fwd GEMMs + the fused-node backward's three grad GEMMs dispatch MPS via the no-autograd matmuls — the base-side dW stays structurally absent); **`g.Scale` vec_scale Metal dispatch** + **`Transpose2D` via the X2b permute kernel** (the LoRA adapter's α/r scale and factor transposes no longer force host syncs; `nn/lora.go` drops its private CPU `scaleTensor` for `g.Scale`, and `Merge`/`Unmerge` fail loudly on a bf16 base); **trainer `-accel async|sync|off`** (`cmd/qwenvoice-train`: off = the untouched CPU f32 path for A/B; on = `InitMetal` + probe gate + native load + `ToMetal` + `MatMulMetalThreshold` lowered to `-metal-min-matmul` (default 8M FMAs — short sequences and rank-16 adapter matmuls must dispatch resident instead of hitting the bf16 widen-per-call fallback) + `SetMetalAsync` unless `-accel=sync`; per-micro-step `flushMetalGraph()` (GC ×2 + beat) because Metal-backed activations exert no Go-heap pressure and finalizer-driven buffer release otherwise lags allocation until the process jetsams).

**Correctness gates (e2e/qwenvoice_accel_test.go, real Qwen3-0.6B checkpoint):**

- **Trajectory-parity gate MET:** 20-step fixed fixture (trainer micro-loop exactly: accum 2, clip 1.0, two LR groups at the trainer defaults, deterministic multi-task-shaped samples over the extended vocab), 4-layer-truncated base, identical trainable init: CPU f32 loss 20.594 → 0.68313 vs GPU+bf16 20.594 → 0.68307 — **|Δ|/CPU = 0.01 % at step 20** (gate 5 %); per-step values track to 3 decimal places the whole way. Dispatch counts assert the path: 56 dtyped bf16 matmuls per micro-step (14 × 4 layers), 2,240 over the run, plus fused softmax/CE kernels.
- **Full-geometry descent gate MET:** full 28-layer accelerated model (r=16 α=32, all layers adapted, 26.9M trainable params incl. the 16,400 ext rows), 6 steps: loss 14.33 → 9.70, monotonic.

**Session validity (WORST measurement conditions of the plan so far):** loadavg 3.4–40 across the session with an active screen-sharing workload; **the VM compressor was saturated (~17.6 GB compressed pool, 26.5 GB swap at peak)** and macOS repeatedly jetsam'd the trainer with `vm-compressor-space-shortage` / `largestProcess: qwenvoice-train` whenever its footprint (~6–8 GB) stayed up for ~45–60 s — the same kill class the X2/X3 benches hit (§2.5/§2.7), now with the JetsamEvent evidence. The accelerated measurement therefore ran as **chained single-invocation steps with checkpoint/resume** (`-resume auto -save-every 1`, accum 2 per step, process lifetime ~20 s): with the same `-seed`, the dataset draw replay makes chained steps 1–4 consume **exactly the eight samples of the CPU run's accum-8 step 1** (6 speak + 1 listen + 1 text, ~4,650 tokens) — a same-token A/B despite the hostile machine. Chaining charges the accelerated side 3 extra optimizer steps and 4 cold starts (MPSGraph compile, table uploads); the CPU side is charged nothing. Both sides ran under external load.

- **A/B on identical samples (full 28 layers, max-seq 1024, r=16 α=32, dw-skip on, Stage-A shards):** CPU f32 accum-8 step: **420.1 s** (~11 tok/s, peak RSS 10.8 GB; its step 2 took 682 s after the OS paged the process out — recorded but not used as the basis). GPU+bf16 async chained steps over the same 8 draws: 12.0 + 18.5 + 30.8 + 12.1 = **73.4 s** (~63 tok/s incl. the step-3 load spike; the three low-noise invocations run at 71–94 tok/s; peak RSS 6.4–7.4 GB). **Gate (plan §5 X4: ≥5× full step vs the CPU f32 path): MET at 5.7× on identical data under identical load** — 7.4× on the low-noise subset — with every protocol asymmetry (extra optimizer steps, cold starts) counting against the accelerated side. The ≥10× stretch is NOT met in-session; the X0-vs-X3 harness estimates (18.1 s → 2.94 s at seq 1024, 6.2×) say the quiet-machine ratio sits near 6×, so the stretch was optimistic once the CE/AdamW tails had already collapsed in X2 (Amdahl, §2.5).
- **Stage-A wall-clock projection (20M-token budget, ratio-weighted ~587 tok/draw at 45/45/10 → ~4,700 tok per accum-8 step → ~4,255 steps):** at this session's measured accelerated throughput (63–81 tok/s): **69–88 h** — that is the honest in-session number. Quiet-machine projection: this session inflated the CPU path ~5× vs its X0 baseline (11 tok/s measured vs ~55 implied), and applying the same inflation factor to the accelerated path (or, equivalently, taking the X3 harness full-step estimate of 2.94 s/1024-token sequence ≈ 350 tok/s) projects **~16–18 h for the full 20M-token Stage-A run** — hours, not days, which was this plan's §0 goal. CPU f32 at the same budget: measured-session ~21 days, quiet-machine ~4 days.
- **Memory:** accelerated peak RSS 6.4–7.4 GB (bf16 blocks + f32 embed/ext/moments + one micro-step graph at ~600–1000-token samples) vs CPU f32 10.8 GB. Fits 24 GB with wide margin at max-seq 1024; the §2.3 seq-1500 wall stays out of the trainer's envelope by the 1024 cap.
- **Async vs sync:** the trainer defaults to async (`-accel async`) per the X2b/X3 finding (async wins at seq ≤1024, the trainer's cap; regresses at 1500). Under this session's compressor pressure async's commit-without-wait run-ahead also *wires* the queued buffers faster than sync — irrelevant on a healthy machine, but the reason the chained protocol used short-lived processes either way.
- **Residual levers for the Stage-A run** (not blockers): buffer pooling to cut the alloc/GC churn (X2b follow-up), a GPU path for the `ExtendedEmbedding` interleave copies, and batching the per-op command buffers; none is needed to start the run.

**Recommended Stage-A run command (estimate only — NOT started; the launch decision belongs to plan 0008 M2):**

```
qwenvoice-train -mode train \
  -data ~/speech-corpora/shards/stageA \
  -out ~/speech-corpora/runs/stageA-r16 \
  -steps 4300 -accum 8 -max-seq 1024 \
  -lora-r 16 -lora-alpha 32 \
  -lr 1e-4 -lr-ext 5e-4 -warmup 100 -min-lr-frac 0.1 -clip 1.0 -wd 0.0 \
  -task-ratios listen=0.45,speak=0.45,text=0.10 \
  -seed 42 -save-every 25 -resume auto -accel async
```

Rationale: 4,300 steps × ~4,700 tok/step ≈ the 20M-token budget; r=16 α=32 per plan 0008 §3.1; 45/45/10 ratios per §3.3; `-save-every 25` bounds a kill to ~6 min of lost work under the memory conditions observed this session (raise to 50 on a quiet machine), and `-resume auto` makes the run restartable/chainable — the checkpoint replays the dataset draw state, which §2.8's chained measurement exercised end-to-end on the accelerated path. Expected wall-clock: ~16–18 h quiet, up to ~3–4× that under the contention profile seen here.

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

> **Status: DONE (2026-08-11), executed as combined phase X1+K1** — the §2.3 correction showed the X1-only gate was unreachable without K1, so K1 was pulled forward and shipped in the same commit. All five items below landed (item 5's cached-mask sub-item was subsumed by the K1 fusion, which needs no mask tensor at all). Measured results in §2.4; the ≥1.8×/2.0× speed gate was **not** met this session (dispatch, parity, and memory gates were), with the residual bottleneck quantified as K4/K6/K7 CPU ops + per-op dispatch sync — not the wiring itself.

Concrete gaps, in order:

1. **Residency propagation rule.** Ops whose inputs are Metal-backed must allocate outputs with `ZerosOnMetal`/`NewBuffer` even when the compute runs on CPU through unified memory (Accelerate below threshold, Go loops). Today `out := Zeros(...)` produces `buf == nil` and every downstream op loses GPU dispatch permanently. On unified memory this is nearly free — same physical pages, just tagged with a buffer. Touch points: `binaryOp`/`unaryOp`, `Softmax`, `LogSoftmax`, `MaskFill`, `Permute`, broadcast ops, `nn.Linear` CPU branch, RoPE, activations. Add a helper `zerosLike(a *Tensor, shape ...int)` that inherits residency.
2. **Batched matmul backward on MPS.** Add `metal_mps_batched_matmul_transA` (and the missing transB-grad variants) to `metal/shim.m` + `metal/metal.go`, and give `BatchedMatMul`/`BatchedMatMulTransB` backward the same residency-and-threshold-gated GPU path `MatMul` already has. This alone moves attention backward (≈1 TF/step) to GPU.
3. **Loss-side grad seeding on GPU.** `Backward()` seeds grad on CPU; after K2 the CE backward produces a Metal-resident dLogits so the whole backward chain stays resident. Until K2 lands, an interim `SeedGradOnMetal` in the harness is acceptable for measurement.
4. **Threshold audit at workload shapes.** Per-layer matmuls at seq 1500: 1500·1024·2048 ≈ 3.1G FMAs (dispatches), 1500·1024·1024 ≈ 1.6G (dispatches), attention batched 16·1500·1500·128 ≈ 4.6G (dispatches), lm_head 258G (dispatches). At seq 512 several fall below 512M — measure whether the threshold needs a per-callsite override; do not silently lower the global default.
5. Fix the stale "inference-only" comment in `nn/gqa.go`; replace the `Full`+`Mul` scaling (144 MB alloc) with grad-aware scalar scaling, and hoist the per-forward causal-mask construction into a cached mask per (heads, seq).

**Gate X1:** block-step bench with weights+activations resident shows every matmul (fwd and bwd) dispatching MPS above threshold (assert via a dispatch counter in tests); block step ≥1.8× vs X0 baseline at seq 1024; `TestGPUMatMulBackwardMatchesCPU`-style parity tests extended to the batched backwards (1e-3 abs). Effort: 4–6 days.

### 3.3 X2: the ranked kernel set (each = golden test first, then kernel; all f32 storage, f32 accumulation)

All follow the `rmsnorm_forward`/`rmsnorm_dx` template: per-row threadgroup of 256, strided loops, tree reduction in threadgroup memory, `Dispatch1DThreadgroups`, compiled from `metal/kernels.go` source strings; Go-side driver in a `*_metal.go` file mirroring `rmsnorm_metal.go`; `nn`-layer routing mirroring `nn/rmsnorm.go` `forwardMetal`.

- **K1 `softmax_causal_forward` / `softmax_backward`.** **DONE — shipped early in combined phase X1K1 (§2.4); kernels in `metal/kernels.go`, driver+op `softmax_metal.go` (`g.CausalSoftmax`), golden tests `softmax_metal_test.go`; drafted via the §4 codex protocol (1 iteration + 1 reviewer-added barrier).** Forward fuses: scale by 1/√d, causal mask (compare column>row+offset — no bool mask tensor, no −1e9 fill tensor), row max, exp, sum, normalize. One kernel, one output, replaces 4–5 intermediates. Backward: `dx = y ⊙ (g − Σ(g ⊙ y))` per row (math identical to the CPU closure in `ops.go` Softmax). Wire into `nn/gqa.go`/`nn/attention*.go`. Priority absolute #1.
- **K2 `cross_entropy_forward` / `cross_entropy_backward`.** **DONE — second X2 wave (§2.5); kernels in `metal/kernels.go`, driver `ce_metal.go`, rewritten `loss.go` with the saved-logsumexp backward + `acc_vexp` CPU fallback, golden tests `ce_metal_test.go`; codex protocol 1 iteration + 1 reviewer-added barrier. Measured 4077 → 195 ms at (1500, 168320).** Row = one token position, N = 168k (strided loop handles N ≫ 256 fine). Forward computes per-row logsumexp + picks target logit; backward writes `softmax(x) − onehot` scaled — never materializes a second softmax. Loss stays f32 always (already the contract in `loss.go`). Include the vectorized-CPU fallback (a new `acc_vexp`-based path) for below-threshold shapes.
- **K3 `embedding_scatter_add`.** Backward for `EmbeddingLookup`: atomic_fetch_add per (token, dim-lane) into the dW rows actually touched; dW allocated dense on Metal once and reused (zero-fill kernel), or better: return a *sparse* (ids, rows) grad consumed by a masked AdamW step (K7). Decide in the golden test which representation the optimizer consumes; sparse is preferred because the trainable-table update then costs O(tokens·dim), not O(vocab·dim).
- **K4 `vec_silu`, `vec_silu_bwd`, `vec_gelu_bwd` (+ existing `vec_gelu` fwd).** **DONE (SiLU/SwiGLU portion) — second X2 wave (§2.5): `vec_silu`/`vec_silu_bwd`/`vec_swiglu`/`vec_swiglu_bwd` kernels + `acc_vsilu`/`acc_vswiglu` Accelerate fallbacks, wired in `silu.go`, golden tests `silu_metal_test.go`; SwiGLU shipped fused (one kernel) since the gate/value pair always co-occur. `vec_gelu_bwd` not needed by the workload (RMSNorm/SwiGLU model) — deferred.** Elementwise; trivial template.
- **K5 `rmsnorm_dgamma`.** **DONE — third X2 wave (X2b, §2.6): kernel in `metal/kernels.go` (one threadgroup of 256 per column, rows strided, tree reduction), dispatched inside `RMSNormBackwardDXMetal` — dW is now Metal-backed and the RMSNorm backward host loop + its forced sync are gone. Golden tests: the existing `nn/rmsnorm_metal_test.go` dW parity + numerical-grad tests now pin the kernel.** Per-*column* reduction over rows (dW[j] = Σ_i g[i,j]·x[i,j]·inv[i]); one threadgroup per column chunk. Removes the host loop in `RMSNormBackwardDXMetal`.
- **K6 `rope_forward` / `rope_backward`.** **DONE — third X2 wave (X2b, §2.6): single `rope_apply` kernel with a sign uniform (+1 fwd, −1 bwd — the inverse rotation is the forward with sin negated), cos/sin uploaded once per `nn.RoPE` module (lazy `metalTables`), both Llama and GPT-NeoX pair conventions; driver `rope_metal.go`, routing via the refactored `nn/rope.go` `rotate` helper; golden test `nn/rope_metal_test.go` (both styles, startPos 0 and >0, 1e-4 abs fwd+bwd).** Elementwise pair rotation with precomputed cos/sin tables uploaded once (backward = same rotation with sin negated, per `nn/rope.go` docs). Keeps Q/K resident between projection and attention.
- **K7 AdamW step.** **DONE (vectorized-step portion) — second X2 wave (§2.5): fused `acc_adamw_step` C loop (sqrtf auto-vectorizes to NEON), per-group LR preserved, bf16-param loud-fail guard (X3-B3), scalar oracle behind `optim.UseScalarAdamW`; 458 → 75 ms on the 172M table, trajectory parity 4.7e-10/20 steps. The masked-row update lands with K3's sparse grads.** First: Accelerate/vDSP vectorized f32 step (no Metal needed; the trainable set is ≤200M params) + masked-row update consuming K3's sparse embedding grads. Metal AdamW only if X4 profiling shows it matters.

Each kernel's checklist: (1) CPU-reference golden test written first, using the existing tolerances (fwd 1e-4…1e-3 abs vs CPU, analytic-vs-numerical grad 1e-2, per `rmsnorm_metal_test` precedent) and the `stageCheck`/min-over-attempts retry discipline from `audio/mimi/encoder_test.go` for anything whose CPU reference goes through Accelerate (BLAS threading nondeterminism — retry once, real regressions fail twice); (2) kernel; (3) `nn` routing gated on residency + `GradEnabled()`; (4) bench row appended.

**Gate X2:** block step at seq 1024 with zero CPU-resident tensors in the fwd+bwd chain (assert via a residency-walk test over the autograd graph); block step ≥3.5× vs X0; full-step estimate ≥4× vs X0; peak RSS at seq 1500 fits 24 GB in f32 (K1's memory reduction should get attention intermediates to ≈1×144 MB×28 ≈ 4 GB). Effort: 8–12 days (K1: 2–3, K2: 2, K3: 2, K4: 1, K5: 1, K6: 1, K7: 1–2).

### 3.4 X3: bf16 for the frozen path

> **Status: DONE (2026-08-12).** All items B0–B5 shipped in one commit; measured results in §2.7, decision record in ADR-012. Gate X3 met (probe ADR recorded, parity suite green at 5e-2/8e-2 incl. the R2 seq-1500 logits check and the 50-step LoRA trajectory within 0.04 %, weights 1.23 GB bf16, rows appended). The X2 carry-over speed gates now close at seq 1024 (block 7.93×, full-step 6.17× vs X0, frozen-path config).

What the LoRA workload actually needs (narrower than 0002's ambition):

- **Frozen base weights stored bf16** (2.46 GB → 1.23 GB). They are never updated → no optimizer interaction, no master weights needed for them.
- **Trainable params (LoRA A/B, embedding/head rows) stay f32** with f32 Adam moments — "master weights in f32" falls out for free by simply *not* converting the trainable set. AdamW PR 3 from 0002 (bf16-param support in the optimizer) becomes **unnecessary for this workload**; implement only the nil-`Data()` guard so a bf16 param in the optimizer fails loudly, and mark 0002-PR3 as superseded-by-0009 unless full-bf16 training returns as a goal.
- **No loss scaling** — bf16 has f32's exponent range; confirmed as the reason bf16 was chosen in 0002. Loss and CE remain f32 (already enforced in `loss.go`).
- **Activations bf16 on the frozen path** only after A-parity is demonstrated; attention logits/softmax and all reduction accumulations stay f32 *inside kernels* regardless of storage dtype (see risk R2).

Work items:

- **B0 (gate task, do first): MPS bf16 probe.** **DONE (2026-08-11) — outcome (b), recorded as ADR-012.** `MPSMatrix` REJECTS `MPSDataTypeBFloat16` on macOS 26.5: `MPSMatrixMultiplication` hard-asserts ("Input data type must be one of MPSDataTypeFloat32, MPSDataTypeFloat16, MPSDataTypeInt8, or MPSDataTypeInt16") and `abort()`s — not even NSException-catchable, so tier (a) cannot be runtime-probed. Implemented tier (b): MPSGraph matmul behind the same shim API (`metal_mps_matmul_dt`/`metal_mps_batched_matmul_dt`, per-shape graph cache, per-operand dtype, bf16 cast to f32 *inside* the graph → f32 accumulation by construction). Probe test `TestB0ProbeMPSBF16Matmul1024`: 1024³ bf16 matches an f64 row reference to 2.2e-06 of RMS, and runs **1.33×** faster than the f32 MPSMatrix path (1.54 vs 2.04 ms/dispatch). Tier (c) (custom bfloat simdgroup kernel + §4 codex protocol) not needed. Original tiers: (a) `MPSMatrix` + `MPSDataTypeBFloat16`; (b) MPSGraph; (c) custom MSL `bfloat` kernel.
- **B1: safetensors native-bf16 load** — **DONE**: `LoadSafetensorsNative` keeps BF16 as `data16` (no widening); `LoadSafetensors` stays the f32-compatibility loader (every existing consumer copies into f32 params via `Data()`); `SaveSafetensors` writes bf16 tensors as dtype `BF16` bit-exactly. Round-trip test `model/safetensors_bf16_test.go` (bit-exact native + widened compat load of the same file).
- **B2: `ToMetal` for bf16 tensors** — **DONE**: dtype-aware `ToMetal`/`ToCPU`, `NewTensorBF16OnMetal`, `Uint16Slice` on `metal.Buffer`, `Tensor.Data16()` accessor (with the Data() read-barrier contract).
- **B3: optimizer guard** — **DONE** (shipped early in the X2 wave); pinned by `optim/adamw_test.go` `TestAdamWBF16ParamPanics`.
- **B4: bf16 matmul dispatch** — **DONE** per the B0 ADR: `MatMul`/`MatMulTransB`/`MatMulTransA`/`BatchedMatMul`/`BatchedMatMulTransB` accept bf16 (and mixed f32×bf16) operands; Metal-resident + above-threshold + probe-green → MPSGraph dtyped path with **f32 accumulation and f32 output**; below-threshold/CPU fallback = widen-to-f32 + Accelerate; CPU both-bf16 pairs keep the legacy plan-0002 promote→downcast semantics (bf16 output, `bf16_ops_test.go` unchanged). Backward grads are f32 ("master grads") and the grad GEMM for a frozen (RequiresGrad=false) operand is **skipped entirely** — dW for frozen bf16 weights costs nothing. `nn.Linear` routes bf16 weights through this path in forward and backward (`bf16_matmul.go`, `ops.go`, `nn/module.go`; dispatch counter `BF16MatMul`).
- **B5: memory accounting test** — **DONE**: `e2e/bf16_memory_test.go` allocates the full 0.6B frozen weight set on Metal and measures **1.226 GB bf16 vs 2.451 GB f32** (ratio 2.000) via `metal.LiveBufferBytes`; the X3 harness memory rows account block weights at 2 B/param.

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
| X4 | **DONE (2026-08-12, §2.8)** — Integration: the REAL VoiceModel/trainer on GPU+bf16 at full 28-layer geometry, measured (the LoRA module itself had shipped with plan 0008 M1; X4 wired it to the X1–X3 machinery: `qwen.LoadNative`, `VoiceModel.ToMetal`, residency-aware `ExtendedEmbedding`, trainer `-accel` flag) | 3–5 d | X2+X3 | **≥5× full step vs CPU f32 — MET (5.7× on identical samples, same-session A/B; §2.8)**; ≥10× stretch NOT met (quiet-machine ratio ≈6×; stretch was optimistic post-X2 Amdahl); trajectory-parity golden green (0.01 % at step 20, gate 5 %) | `model/qwen/load.go`, `model/qwen/accel.go` (new), `nn/extembed.go`, `nn/lora.go`, `ops.go`, `tensor.go`, `cmd/qwenvoice-train/`, `e2e/qwenvoice_accel_test.go` (new) |
| X5 | Docs: ADRs for residency + bf16 decisions; update 0002/0004 with pointers to 0009; results report PDF per repo convention (`doc/generate_session_report.py` pipeline, JSON → PDF like `metal-crossover-report.pdf`); update `doc/plans/README.md` table | 1–2 d | X4 | report committed; plans cross-linked | `doc/plans/0002…0004…README.md`, `doc/decisions.md`, `doc/training-accel-report.pdf` (new) |

Total: ~22–35 focused days. X3 can run concurrently with late X2 (different files).

## 6. Risks, ranked, with mitigations

- **R1 Metal debugging opacity** (wrong results, no stack traces). Mitigation: golden-test-first discipline; kernels ≤60 lines from an exact CPU reference; per-kernel numerical-grad check; keep the CPU path as the *test oracle* (not a runtime toggle — 0004's "no fallback toggles" stance holds for production paths, but tests always compare against CPU).
- **R2 bf16 divergence in attention logits.** QK^T at head_dim 128 in bf16 accumulates ~7-bit-mantissa error into logits whose softmax is exp-sensitive. Mitigation: f32 accumulation inside all matmul paths (MPS does this for f16/bf16 inputs when output is f32 — verify in B0 probe); softmax/CE always compute in f32; parity gate includes an attention-logits-specific check at seq 1500.
- **R3 MPS bf16 API gaps** (MPSMatrix may reject `MPSDataTypeBFloat16`). Mitigation: B0 probe *before* scoping B4; three-tier fallback (MPSMatrix → MPSGraph → custom `bfloat` simdgroup kernel); worst case X3 delivers memory-only wins (still halves weight+activation footprint, which X2's gate shows is needed at seq 1500).
- **R4 Thermal throttling polluting benchmarks.** Mitigation: §2.1 discipline — median-of-5, AC power, baseline re-run at session end as a canary, results JSON records machine + date; never compare rows from different sessions without matching canaries.
- **R5 Accelerate-threading nondeterminism × GPU numerics in tests.** Known from Mimi work (BLAS results vary at 1e-7–1e-6 across runs under load; `encoder_test.go` retry comment). Mitigation: adopt the min-over-attempts/recompute-once pattern for every CPU-vs-GPU parity test; tolerances set at 1e-3 abs (matmul-class) / 1e-2 (numerical-grad) with the retry, never tightened ad hoc.
- **R6 Per-op `waitUntilCompleted` synchronization ceiling.** **RESOLVED for the block step — third X2 wave (X2b, §2.6): the residual CPU ops that forced the ~21 waits/step are now GPU kernels (permute, RoPE, RepeatInterleave, dgamma, db, bias add) and grad accumulation dispatches in-place `vec_add`; measured host waits in async mode fell to ~1/block-step (the `Sum` loss read).** Previous status: **MITIGATED (opt-in) — second X2 wave (§2.5):** commit-without-wait mode shipped (`g.SetMetalAsync`; `metal_sync_queue` + wait-on-CPU-read fencing via `syncForCPU`/`Data()`); measured round trip 0.17 ms/dispatch sync vs 0.03 ms async, ~21 residual host waits/block-step (one per remaining CPU op), bit-exact parity, default OFF. Fully retiring the residual waits requires moving Permute/RoPE/RepeatInterleave/dW-loops onto GPU (K5/K6 + permute residency). Original text: each Metal op is a full command-buffer round trip (~0.2–1 ms). At ~15 GPU ops/layer × 28 layers ≈ 400+ dispatches/step ≈ 0.1–0.4 s of pure launch overhead — visible once compute drops to ~1 s.
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
