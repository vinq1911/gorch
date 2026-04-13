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
