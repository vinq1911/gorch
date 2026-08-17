//go:build darwin

#ifndef GORCH_METAL_SHIM_H
#define GORCH_METAL_SHIM_H

#include <stdint.h>

// Opaque handles to Metal objects.
// Go holds these as unsafe.Pointer; the ObjC side casts back.
typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;
typedef void* MTLBufferRef;
typedef void* MTLComputePipelineRef;

// Device and command queue lifecycle.
MTLDeviceRef metal_create_device(void);
MTLCommandQueueRef metal_create_command_queue(MTLDeviceRef dev);

// Shared-memory buffer management.
// Shared mode = unified memory: Go and GPU read/write the same bytes.
MTLBufferRef metal_create_shared_buffer(MTLDeviceRef dev, uint64_t length);
void*        metal_buffer_contents(MTLBufferRef buf);
uint64_t     metal_buffer_length(MTLBufferRef buf);
// metal_release_buffer destroys a buffer, discarding its physical
// pages first when that is provably safe. Returns 1 if the pages were
// discarded (immediately or by handing the buffer to the deferred
// list), 0 if they were abandoned unreclaimed. See shim.m.
int          metal_release_buffer(MTLBufferRef buf);

// Compile a Metal kernel from source at runtime.
// Returns NULL on failure; errOut (if non-NULL) receives a message.
MTLComputePipelineRef metal_compile_kernel(MTLDeviceRef dev,
                                           const char* source,
                                           const char* funcName,
                                           char** errOut);
void metal_release_pipeline(MTLComputePipelineRef pipe);

// Dispatch a 1-D compute kernel.
// bufs/bufCount: array of buffers bound at indices 0..N-1.
// threadCount:   total number of threads to launch.
void metal_dispatch_1d(MTLCommandQueueRef queue,
                       MTLComputePipelineRef pipeline,
                       MTLBufferRef* bufs, uint32_t bufCount,
                       uint32_t threadCount);

// Dispatch a 1-D compute kernel as a fixed grid of threadgroups, each
// of a fixed size. Used by reduction kernels (RMSNorm, Softmax, …) that
// need per-threadgroup shared memory and barriers — those depend on
// knowing exactly how many lanes participate.
//
// groupCount:     number of threadgroups to launch
// groupThreads:   threads per threadgroup (≤ pipeline's max)
void metal_dispatch_threadgroups_1d(MTLCommandQueueRef queue,
                                    MTLComputePipelineRef pipeline,
                                    MTLBufferRef* bufs, uint32_t bufCount,
                                    uint32_t groupCount,
                                    uint32_t groupThreads);

// MPS matrix multiply: C = A @ B.
// A is MxK, B is KxN, C is MxN. All row-major float32.
void metal_mps_matmul(MTLCommandQueueRef queue,
                      MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                      uint32_t M, uint32_t N, uint32_t K);

// MPS matrix multiply with transpose: C = A @ B^T.
// A is MxK, B is NxK (stored row-major), C is MxN.
void metal_mps_matmul_transB(MTLCommandQueueRef queue,
                             MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                             uint32_t M, uint32_t N, uint32_t K);

// MPS matrix multiply with transpose: C = A^T @ B.
// A is KxM (stored row-major), B is KxN, C is MxN.
void metal_mps_matmul_transA(MTLCommandQueueRef queue,
                             MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                             uint32_t M, uint32_t N, uint32_t K);

// Batched MPS matrix multiply: C[i] = A[i] @ B[i] for i in 0..batchSize-1.
// All matrices packed contiguously: A is (batchSize*M*K), B is (batchSize*K*N), C is (batchSize*M*N).
void metal_mps_batched_matmul(MTLCommandQueueRef queue,
                              MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                              uint32_t M, uint32_t N, uint32_t K,
                              uint32_t batchSize);

// Batched MPS: C[i] = A[i] @ B[i]^T for i in 0..batchSize-1.
// A is (batchSize*M*K), B is (batchSize*N*K), C is (batchSize*M*N).
void metal_mps_batched_matmul_transB(MTLCommandQueueRef queue,
                                     MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t batchSize);

// Batched MPS: C[i] = A[i]^T @ B[i] for i in 0..batchSize-1.
// Per batch, A is stored (K, M) row-major, B is (K, N), C is (M, N).
// A is (batchSize*K*M), B is (batchSize*K*N), C is (batchSize*M*N).
void metal_mps_batched_matmul_transA(MTLCommandQueueRef queue,
                                     MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t batchSize);

// ---------------------------------------------------------------------------
// Dtype-parameterized MPS matmul (plan 0009 X3, B0/B4).
//
// C = opA(A) @ opB(B) where opX is transpose iff transX != 0. A and B
// may independently be bfloat16 (aBF16/bBF16 != 0, 2 bytes/element,
// MPSDataTypeBFloat16) or float32. C is ALWAYS float32 with f32
// accumulation — the risk-R2 contract (bf16 storage, f32 math).
//
// Implementation is MPSGraph, NOT MPSMatrix: the B0 probe (2026-08-11,
// M4, macOS 26.5) showed MPSMatrixMultiplication hard-asserts (abort)
// on MPSDataTypeBFloat16 inputs — plan 0009 §3.4 B0 outcome (b), see
// ADR-012. bf16 operands are cast to f32 inside the graph before the
// matmul node, making f32 accumulation structural; compiled graphs are
// cached per shape/dtype signature.
//
// Logical (post-transpose) shapes: opA(A) is (M, K), opB(B) is (K, N),
// C is (M, N). Stored shapes: A is (K, M) when transA else (M, K);
// B is (N, K) when transB else (K, N). All row-major, contiguous.
//
// Returns 0 on success; 1 when MPSDataTypeBFloat16 is unavailable
// (pre-macOS 14) or MPS rejects the configuration (NSException
// caught). Callers must verify numerics once per process (the B0
// probe) — MPS can silently produce garbage rather than throw.
int metal_mps_matmul_dt(MTLCommandQueueRef queue,
                        MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                        uint32_t M, uint32_t N, uint32_t K,
                        int transA, int transB,
                        int aBF16, int bBF16);

// Batched variant: C[i] = opA(A[i]) @ opB(B[i]) for i in
// 0..batchSize-1, matrices packed contiguously per operand. Same
// dtype/transpose semantics and return convention as
// metal_mps_matmul_dt. Single command buffer, one commit.
int metal_mps_batched_matmul_dt(MTLCommandQueueRef queue,
                                MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                                uint32_t M, uint32_t N, uint32_t K,
                                uint32_t batchSize,
                                int transA, int transB,
                                int aBF16, int bBF16);

// Enable/disable discarding a buffer's physical pages immediately
// before releasing it. ON by default — see metal_release_buffer for
// why it is required and why it is guarded. Turning it off reproduces
// the unbounded-footprint behaviour and is for tests only.
void metal_set_purge_on_release(int on);

// Counts of buffer releases by disposition:
//   purged    — pages discarded on the spot (queue was drained)
//   deferred  — queued for discard at the next drain (work in flight)
//   unpurged  — pages abandoned; the leak the deferred list eliminates.
// With the deferred list in place, unpurged should only ever advance
// when the purge is switched off or the pending list cannot grow.
uint64_t metal_purged_releases(void);
uint64_t metal_unpurged_releases(void);
uint64_t metal_deferred_releases(void);

// Bytes currently sitting on the deferred-release list (released by
// their Go owner, purge pending on the next drain), and the high-water
// mark of that figure. These allocations are still live, so the caller
// must count them against any live-memory ceiling.
uint64_t metal_pending_bytes(void);
uint64_t metal_pending_peak_bytes(void);

// Purge and destroy anything on the deferred-release list, if the queue
// currently reads drained; a no-op otherwise. Never clears the
// in-flight gate — that is only sound where a wait has returned.
void metal_drain_pending_releases(void);

// Number of distinct MPSGraph objects currently held by the dtyped
// matmul cache (see gorch_dt_graph in shim.m). Each entry owns a
// compiled MPSGraph executable plus its MPS workspace allocations —
// memory that is invisible to both the Go heap and gorch's MTLBuffer
// accounting, but fully visible in the process's physical footprint.
// Exposed so the trainer and its regression tests can assert the cache
// stays bounded.
uint64_t metal_dt_graph_cache_count(void);

// Drop every cached dtyped-matmul graph. Subsequent calls recompile on
// demand. Used to bound the cache (see metal_set_dt_graph_cache_limit).
void metal_clear_dt_graph_cache(void);

// Cap the dtyped-matmul graph cache at `limit` entries; the cache is
// cleared wholesale when it would exceed the cap. 0 = unbounded (the
// pre-fix behaviour).
void metal_set_dt_graph_cache_limit(uint64_t limit);

// ---------------------------------------------------------------------------
// Async dispatch mode (plan 0009 X2, risk R6).
//
// By default every dispatch/matmul entry point commits its command
// buffer and blocks in waitUntilCompleted — one full GPU round trip
// (~0.2–1 ms) per op, measured at ~46% of the X1K1 block-step wall.
// With async mode ON, entry points commit WITHOUT waiting and remember
// the last committed command buffer; metal_sync_queue() blocks until
// it (and, by the command queue's in-order execution guarantee, every
// earlier buffer) has completed. Callers must sync before any CPU read
// of GPU-written unified memory. Buffers released while work is in
// flight are safe: command buffers retain their encoded resources
// until completion (default retained-references mode).
//
// Single-queue, single-threaded by design — matches gorch's one global
// command queue and single-threaded training loop.
// ---------------------------------------------------------------------------

void metal_set_async(int on);   // turning off also syncs
void metal_sync_queue(void);    // wait for all committed GPU work

// Release a device or command queue.
void metal_release(void* obj);

// Free an error string returned by metal_compile_kernel.
void metal_free_string(char* s);

#endif // GORCH_METAL_SHIM_H
