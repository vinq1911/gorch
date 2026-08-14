# Architecture Decisions

## ADR-001: Raw Metal + MPS over MLX bindings

**Date:** 2026-04-10
**Status:** Accepted

We chose to use raw Metal API + MPS kernels via an Objective-C CGo shim rather than binding to Apple's MLX framework.

**Rationale:** MLX is a complete ML framework — binding to it from Go would make gorch a thin wrapper, not a framework. We own every design decision this way. The ObjC bridging is straightforward (C-compatible), whereas MLX's C++ API would require painful name-mangling workarounds.

**Trade-off:** More work to implement autograd and ops ourselves, but we learn everything and can optimize for our specific needs.

## ADR-002: Apple Accelerate for CPU backend

**Date:** 2026-04-13
**Status:** Accepted

All CPU tensor operations use Apple's Accelerate framework (BLAS, vDSP, vForce) instead of naive Go loops.

**Result:** 628x speedup on matmul (512x512), 30x speedup on MNIST training (30s to 1s).

**Dispatch order:** Metal GPU → Accelerate CPU → Go fallback.

## ADR-003: Conv2d implementation strategy

**Date:** 2026-04-13
**Status:** Accepted

### Phase 1 (current build)

- **CPU:** im2col + `cblas_sgemm` (Accelerate BLAS)
- **GPU:** `MPSCNNConvolution` for Metal
- **1x1 special case:** Skip im2col entirely, treat as pure GEMM
- **Fuse bias + ReLU** into conv output loop (single memory pass)

### Data duplication mitigation

im2col expands input data by kernel_size^2x (a 3x3 conv with 64 channels turns 64 values into 576 per output pixel). Mitigations:

1. **Tiled im2col:** Don't materialize the full expanded matrix. Process in tiles — expand one tile of output rows into a fixed-size scratch buffer, call sgemm on the tile, repeat. Buffer size bounded regardless of input size.
2. **Scratch buffer reuse:** Pre-allocate the im2col scratch buffer once per Conv2d layer, reuse across forward calls. No allocation in the hot loop.
3. **1x1 bypass:** 1x1 convolutions skip im2col entirely — input data is already in GEMM-ready shape after a reshape. Zero duplication.
4. **Inference buffer pooling (future):** Under `NoGrad`, intermediate tensors could be recycled from a pool instead of freshly allocated.

### Phase 2 (future, if profiling warrants)

- Weight prepacking for inference
- Direct 3x3 kernel with NEON
- Separate border/interior handling (branchless hot path)
- Winograd for 3x3 if compute-bound

### What we deliberately skip

- Hand-written NEON/SIMD assembly
- Codegen for specialized kernel shapes
- Winograd transforms
- Implicit im2col in Metal threadgroups (MPS handles this)
- Depthwise conv specialization

These are real techniques but premature until we have profiling data showing conv is the bottleneck.

## ADR-004: Memory allocation strategy

**Date:** 2026-04-13
**Status:** Proposed

Current ops allocate a new tensor per call. This is correct but wasteful for hot loops. Future mitigations:

1. **Output tensor reuse:** Allow ops to write into a pre-existing tensor (`AddOut(a, b, out)` pattern)
2. **Scratch buffers:** Conv2d, matmul backward, and other ops that need temporary workspace should pre-allocate and reuse
3. **NoGrad buffer pool:** During inference, intermediate tensors can be recycled since the autograd tape isn't recording
4. **Unified memory awareness:** Metal buffers on Apple Silicon share physical memory with CPU — avoid redundant CPU copies of GPU results

## ADR-005: Pretrained model loading strategy

**Date:** 2026-04-14
**Status:** Accepted

GPT-2 weights are loaded from HuggingFace safetensors format with these transformations:

1. **Conv1D → Linear transposition:** GPT-2 stores weights as (in, out), gorch Linear expects (out, in). Transpose during loading.
2. **Fused QKV split:** GPT-2's `c_attn` concatenates Q, K, V into one (dim, 3*dim) matrix. Split into separate Wq, Wk, Wv during loading.
3. **Tied LM head:** GPT-2 shares token embedding weights with the output projection. Copy wte.weight into LMHead.Weight.
4. **GELU activation:** GPT-2 uses GELU, not ReLU. Added GELU op with tanh approximation.

## ADR-006: Fragmind pipeline parallelism

**Date:** 2026-04-14
**Status:** Accepted

Models are split into "fragments" — contiguous slices of transformer blocks. Fragment 0 handles embeddings + first N blocks, last fragment handles remaining blocks + LM head.

**Transport:** TCP with binary tensor serialization (ndim + shape + float32 data). Simple, portable, works across machines.

**Result:** Local pipeline has <3% overhead. TCP pipeline is serialization-bound (~821ms per token for 768-dim activations). Production optimization: shared memory, RDMA, or batched token transfer.

**Output consistency:** All split configurations produce bit-identical output to unsplit model, verified in e2e tests.

## ADR-007: Broadcasting implementation

**Date:** 2026-04-14
**Status:** Accepted

NumPy-compatible broadcasting via separate `AddB`/`SubB`/`MulB`/`DivB` functions (not replacing the original same-shape `Add`/`Sub`/`Mul`/`Div`).

**Rationale:** Keeping both avoids the overhead of broadcast shape checking on the hot path where shapes are known to match. The `B` suffix makes broadcast intent explicit.

**Autograd:** Backward pass uses `reduceBroadcastGrad` to sum gradients along broadcast dimensions back to the original shape.

## ADR-008: Text generation sampling strategy

**Date:** 2026-04-14
**Status:** Accepted

Generation supports: greedy (argmax), temperature scaling, top-K filtering, and top-P nucleus sampling. KV cache struct exists for future incremental decoding but is not yet integrated into the GPT forward pass (full sequence recomputation per token).

**Current throughput:** ~40 tok/s on GPT-2 small (124M params) without KV cache. With KV cache, expect 3-5x improvement for long sequences.

## ADR-009: GPU autograd is matmul-first, not all-or-nothing

**Date:** 2026-04-29
**Status:** Accepted

Backward passes are wired to dispatch to MPS only for MatMul (and Linear, which composes MatMul). Other ops (LayerNorm, Softmax, GELU, etc.) keep their CPU backwards. Gradients flowing through a chain therefore land on Metal whenever the surrounding ops are MatMul-shaped, and on CPU otherwise.

**Rationale:** MatMul is the dominant cost in transformer training (typically >80% of FLOPs) and the math maps cleanly onto two transposed MPS calls (`MatMulTransA` and `MatMulTransB`, both already exposed for forward use). The remaining ops require either custom Metal kernels or significant per-op work, and shipping them piecemeal would clutter the codebase faster than it helps. Apple Silicon's unified memory makes the mixed-device chain cheap — Metal-backed slices are still float32 slices that CPU loops can iterate.

**What works today:** Weights on `ToMetal(dev)`, run forward + Backward, dW/db match CPU within fp32 noise, training converges. Verified by `TestLinearBackwardMatchesCPUOnGPU` and `TestTrainTinyMLPOnGPU`.

**What's deferred:** Custom Metal kernels for LayerNorm/Softmax/GELU backward, which would close the remaining gap for transformer training throughput.

## ADR-009-update: measured wall-clock on Apple M5 — GPU autograd is currently a regression for transformer-shaped workloads

**Date:** 2026-04-29
**Status:** Findings

Empirical Linear training-step benchmarks (single Linear layer, forward + Sum loss + Backward, full step) on Apple M5:

| Shape | CPU (Accelerate) | Metal (MPS) | Ratio |
| ------------------ | ---------------- | ------------- | ------- |
| (64, 768, 768)     | 0.50 ms          | 2.27 ms       | 4.6× SLOWER on GPU |
| (256, 2048, 2048)  | 5.48 ms          | 26.8 ms       | 4.9× SLOWER on GPU |

These shapes bracket what GPT-2 small ((seq, 768, 768) for QKV/Wo, FFN expansion to 3072) and bigger transformer architectures use. **At every shape gorch is likely to encounter in a transformer, the matmul-only Metal backward path loses to Accelerate.**

Likely cause: the loss in these benches is `g.Sum`, which produces a CPU-resident grad. MatMul backward checks every operand's residency at backward time and falls back to CPU when grad is on CPU — but the operand weights are still Metal-allocated, so the CPU sgemm reads/writes through unified-memory slices. That works numerically but costs L2/L3 coherence traffic over a pure-CPU baseline.

This ADR-009 update therefore deprecates the recommendation to call `gpt.ToMetal()` for training. Inference-on-Metal still wins (forward MatMul without the cross-device grad flow). For training, stay on CPU until either:
1. The whole loss path lands on Metal (so grads stay on GPU), OR
2. Custom Metal backward kernels exist for the activation ops.

Both are bigger structural changes than the matmul-first slice.

## ADR-009-fix: matmul size-threshold for Metal dispatch

**Date:** 2026-04-29
**Status:** Accepted

The regression in ADR-009-update wasn't about autograd specifically — it was about Metal dispatching at all for shapes too small to amortise MPS launch overhead. The existing `doc/metal_crossover_results.json` shows the actual crossover: 768³ matmul GPU is 0.45× CPU; 1024³ is 1.22× CPU; the inflection lives between 768³ and 1024³ on M-series.

`MatMulMetalThreshold` is a package-level int (default 512_000_000 FMAs) that all matmul dispatch sites consult. Below it, the CPU Accelerate path runs even when both operands are on Metal — Accelerate sgemm reads through unified-memory slices fine, no actual transfer cost. Above it, MPS dispatches as before.

This applies to `MatMul`, `MatMulTransA`, `MatMulTransB`, `BatchedMatMul`, `BatchedMatMulTransB` for both forward and backward.

**Wall-clock impact, Apple M5, single-Linear training step (forward + Sum + Backward):**

| Shape | pre-threshold Metal | post-threshold Metal | CPU |
| ------------------ | ------------------- | -------------------- | --- |
| (64, 768, 768)     | 2.27 ms             | **0.32 ms**          | 0.35 ms |

At GPT-2 small dims with weights on Metal, the train step is now **at parity with pure CPU** rather than 4.6× slower. `gpt.ToMetal()` is no longer actively harmful at small shapes; it falls back to CPU Accelerate transparently. At shapes above the threshold (≥1G FMAs, e.g., 2048×2048×256+ batched workloads) MPS dispatches as before.

Tests that exercise the GPU code path (`TestGPUMatMulBackwardMatchesCPU`, `TestMatMulTransAPublicOp`) lower the threshold to 0 via `setMatMulMetalThresholdForTest` so they keep verifying numerical equivalence.

**What this fixes vs. doesn't:**
- ✓ `gpt.ToMetal()` no longer causes a 4.6× training regression at GPT-2 small dims
- ✓ Forward inference at small shapes is at parity with CPU (no penalty for ToMetal)
- ✗ Above the threshold, large-shape training (2048+) still has the cross-device grad-coherence cost — addressing it needs the full Metal loss path or custom backward kernels (still ADR-009 deferred work)

## ADR-010: NoGrad gating + transient scratch pooling

**Date:** 2026-04-29
**Status:** Accepted

`g.NoGrad` now actually does something. Until this change, `NoGrad` only manipulated a depth counter; no op anywhere checked `GradEnabled()`. Every op built a full autograd graph regardless. PR #15 wires `GradEnabled()` into all 31 direct field-setter sites in `ops.go` / `attention_ops.go` / `broadcast.go` / `conv.go` / `pool.go` / `loss.go`, plus into `Tensor.SetGradFn` / `SetRequiresGrad`. Inside `NoGrad`, no graph is built and activations are GC-eligible immediately after their consuming op.

`AcquireFloat32` / `ReleaseFloat32` is a sync.Pool of float32 slices for *within-op transient scratch* — buffers that don't escape the op (GELU's `inner`, LayerNorm's `xNorm` and `invStd`). The pool is goroutine-safe; lifetime is bounded by the op call.

Allocation pooling for *escaping* tensors (Linear.Forward output, attention reshape outputs) needs explicit Tensor.Release semantics — separate change tied to ADR-004 and not done yet.

**Wall-clock impact, GPT-2 small, seq=64 (batched encode 16 at the bottom):**

| Bench | original main | post-NoGrad+pool |
| ------------------------ | ------------- | ---------------- |
| `Encode`                 | 55.7 ms       | 25.3 ms (2.2×)   |
| `EncodeBatch16`          | 652 ms        | 274 ms (2.4×)    |

## ADR-011: KV cache delivers as advertised — measured

**Date:** 2026-04-29
**Status:** Findings

Tiny GPT (vocab=256, dim=64, 4 heads, 4 layers, prompt=8, generate 64 tokens) on Apple M5:

| Path | ns/op |
| ------------------------- | --------- |
| `BenchmarkGenerateUncached` | 35.5 ms   |
| `BenchmarkGenerateCached`   |  4.4 ms   |

**8.1× speedup at 72 tokens generated** — and the gap widens with sequence length because uncached is O(N²) per token and cached is O(N). Validates ADR-008's "expect 3-5× improvement for long sequences" claim with a concrete number on a small model. Real-world GPT-2-small numbers should be similar or better.

## ADR-012: bf16 matmul goes through MPSGraph (plan 0009 X3-B0 probe outcome)

**Date:** 2026-08-11
**Status:** Accepted

Plan 0009 §3.4 B0 required probing whether the MPS shim can run bf16 matmuls before scoping B4, with a three-tier fallback: (a) `MPSMatrix` + `MPSDataTypeBFloat16`, (b) MPSGraph matmul with bf16, (c) custom `bfloat` simdgroup MSL kernel via the §4 codex protocol.

**Probe result (Apple M4, macOS 26.5, 2026-08-11): tier (a) is dead.** `MPSMatrixMultiplication` hard-asserts on encode:

> `MPSMatrixMultiplication.mm:3260: failed assertion 'Input data type must be one of MPSDataTypeFloat32, MPSDataTypeFloat16, MPSDataTypeInt8, or MPSDataTypeInt16.'`

The assertion calls `abort()` — it is not an `NSException`, so it cannot even be probed safely at runtime from the shim (`@try/@catch` does not fire). `MPSDataTypeBFloat16` exists in the SDK (macOS 14+) and works elsewhere in MPS, but the classic `MPSMatrix` kernels reject it outright.

**Decision: tier (b) — MPSGraph.** `metal_mps_matmul_dt` / `metal_mps_batched_matmul_dt` in `metal/shim.m` build an `MPSGraph` per (M, N, K, batch, transA, transB, dtypes) signature, cached in a dictionary for the life of the process (shapes are static per model layer, so the cache stays small and steady-state cost is one encode on a cached graph). Key properties:

- **Per-operand dtype:** A and B are independently bf16 or f32 placeholders; C is always f32. Mixed f32-activation × bf16-frozen-weight — the LoRA workload's shape — is a first-class case (verified in the probe).
- **f32 accumulation by construction (risk R2):** bf16 placeholders are `castTensor`-ed to f32 *inside the graph* before the `matrixMultiplication` node. The MPSGraph compiler fuses the cast, so memory traffic stays 2 bytes/element while the matmul accumulates in f32. Measured: 1024³ bf16×bf16 matches an f64 row reference to 2.2e-06 of RMS (bf16 accumulation would sit near 6e-2); attention-logits shape (16 heads, seq 1500, head_dim 128) matches f64 to 7.2e-07 of RMS.
- **Async-mode compatible:** the graph is encoded onto an `MPSCommandBuffer` wrapping the shared queue; the shim commits without waiting in async mode exactly like every other dispatch (`metal_finish` on the root command buffer).
- **Throughput:** 1024³ bf16 measured **1.33× faster** than the f32 `MPSMatrix` path (1.54 ms vs 2.04 ms per sync dispatch) — half the operand bandwidth, same f32 math.
- **Runtime guard:** `gorch.MetalBF16MatMulSupported()` runs a once-per-process 16³ numeric probe (bf16×bf16 and f32×bf16 vs a CPU reference) before any real dispatch routes to the bf16 path; on failure (older OS, wrong numerics) every bf16 matmul silently takes the widen-to-f32 + Accelerate fallback. MPS silently producing garbage instead of erroring is a known failure mode, so support is defined by *numerics*, not by a non-error return.

Tier (c) — the custom bfloat simdgroup kernel and its §4 Azure/codex protocol — is **not needed**: tier (b) closes B0 with a measured compute win, not just the memory win.

## ADR-013: Activation checkpointing for the LoRA trainer

**Date:** 2026-08-14
**Status:** Accepted

After the R1 fix (`d5cc131`, released Metal buffers discard their CPU-faulted pages) gradient accumulation stopped compounding, and what remained was legitimate: a single micro-step at 28 layers retains every layer's forward activations, and there is no leak to remove. Activation checkpointing is the standard trade, and it is unusually clean here because **the base model is frozen and the block path has no RNG** — no dropout, no stochastic routing, nothing to reproduce (verified by grep over `nn/gqa.go`, `nn/moe.go`, `nn/rmsnorm.go`, `nn/lora.go`, `nn/rope.go`). Unlike `torch.utils.checkpoint`, gorch's `Checkpoint` needs no RNG fork/restore.

**Mechanism** (`checkpoint.go`). `Checkpoint(name, x, fn)` runs `fn` on a *detached* handle inside `NoGrad`, so the segment builds no graph and allocates no backward closures, then installs one `GradFn` whose only input is `x`. Its backward re-runs `fn` under `enableGrad` (Backward itself runs inside `NoGrad` — the recompute has to force tracking back on) on a *fresh detached leaf* `xl`, backprops the local graph with the same `backpropFrom` engine `Backward` uses, and returns `xl.grad` as `dL/dx`. LoRA A/B are leaves *inside* the recomputed subgraph, so the local backward accumulates into their real `.grad` exactly once and they are deliberately absent from the outer node's `inputs`.

Three details are load-bearing and all three were review findings (`gpt-5.6-terra`, once on the design and once on the shipped code):

- `xl` must be a detached handle, not `x` itself. Otherwise the local backward writes `x.grad` *and* the outer engine accumulates the returned `dx` into it — measured as a 1.5× gradient on a two-segment toy.
- The degenerate identity closure (`fn` returns its argument) must short-circuit, or `y` and `xl` are the same tensor and the grad-restore zeroes `dx`.
- A closure that returns a captured *leaf* produces no local graph, so restoring the root's saved gradient would discard that segment's entire contribution to the parameter. It has to be accumulated by hand.

Each has a named regression test in `checkpoint_test.go`, and each test was verified to fail against a deliberately broken implementation.

**The Metal half of the trade is not optional.** "Not retained" and "freed" are different things when the memory lives outside the Go heap: dropping the last reference to an MTLBuffer exerts no heap pressure, so no GC runs, so the release finalizer never fires, and live bytes track the pass's *cumulative* allocation volume whether or not a graph was retained. Checkpointing without a flush hook measured a 30% dent (peak live 7687 → 5391 MB) while *raising* the physical footprint, because the recompute nearly doubles allocation volume. `CheckpointSegmentDone` fires at both points where a segment's intermediates die — after the no-grad forward and after the recompute — and the trainer installs a `SyncMetal` + GC + settle + GC flush there. That is what turns the mechanism into memory.

The settle beat matters because a buffer released while GPU work is in flight *skips its page purge* (`shim.m`'s `g_maybeInFlight` gate) and its pages are reclaimed only much later. At 28 layers / seq 1024 / accum 1: a 2 ms beat left peak footprint at 12.9 GB (ceiling abort), 10 ms brought it to 7.4 GB. Default `-checkpoint-flush-ms 10`. Corroboration: `-accel=sync`, which drains the queue after every dispatch, drives unpurged releases from 6284 to 845.

**Correctness gates.** CPU f32 gradients are **bit-identical** with checkpointing on and off — worst relative error exactly 0.0 across every LoRA A/B tensor and the ext rows, at segment lengths 1–4, at 4 and 28 layers, and across three accumulated micro-steps without an intervening `ZeroGrad`. On the GPU bf16 path, 20-step runs resumed from one shared checkpoint: two identical checkpointing-OFF runs differ by max |Δloss| 5e-6, and ON-vs-OFF differs by 6e-6 — the same floor. (Comparing *fresh* runs instead measures 1.8%, but that is unseeded weight init, not the GPU path: `g.RandN` draws from `math/rand`'s global source.)

**Measured, 28 layers, `-rss-limit-mb 9000`.** Peak live Metal bytes per micro-step; "abort" = hit the ceiling.

| accum | max-seq | OFF peak | ON (every=1) peak | ON step time |
| ----- | ------- | -------- | ----------------- | ------------ |
| 1     | 512     | abort    | 3255 MB           |  8.3 s |
| 2     | 512     | abort    | 3338 MB           | 13.5 s |
| 1     | 1024    | abort    | 4320 MB           |  9.6 s |
| 2     | 1024    | abort    | 4404 MB           | 15.8 s |
| 4     | 1024    | abort    | 4404 MB           | 27.5 s |

Every one of these configurations fits with checkpointing and none of them fits without it: OFF peaks at 7687 MB on a 312-token micro-step and trips the ceiling before finishing a 512-token one.

Segment length at seq 1024, accum 1 (peak live / peak footprint): every=1 → 4320 / 8397 MB, every=2 → 4861 / 7885 MB, every=4 → 6010 / 10035 MB (abort), every=7 → 7756 / 14336 MB (abort). **Use 1 or 2.**

**Compute cost**, full depth at max-seq 256 (the largest configuration OFF survives), 6 steps: OFF 30.3 s, every=1 55.8 s (+84%), every=2 46.3 s (+53%), every=4 41.7 s (+38%). The recompute itself is the expected ~+33%; the rest is the flush's sleeping (28 segments × 2 hooks × 2 × 10 ms ≈ 2.2 s per step at accum 2), which is why the overhead *falls* as segments get longer. Removing it needs a deterministic-release path in the Metal shim — the purge-skip gate is the real remaining cost, not the recompute.

**Also added:** `metal.SetLiveBufferLimit` — a hard ceiling on live Metal bytes, checked on every buffer allocation. The trainer's `-rss-limit-mb` guard samples `vmmap` at micro-step boundaries, which cannot see the peak: measured at 28 layers / seq 512 / accum 1, live buffers peaked at 12.3 GB mid-micro-step while the boundary sample read 1.9 GB. One atomic compare per allocation turns a jetsam event that takes the desktop down into a stack trace.
