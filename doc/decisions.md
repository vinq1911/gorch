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
