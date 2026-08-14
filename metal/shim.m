//go:build darwin

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "shim.h"
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Async dispatch mode (plan 0009 X2, risk R6) — see shim.h.
// ---------------------------------------------------------------------------

static int g_async = 0;
static id<MTLCommandBuffer> g_lastCmdBuf = nil; // strong ref under ARC

// g_maybeInFlight is 1 whenever GPU work may be outstanding. It is
// raised by metal_begin_work() BEFORE any command buffer is created or
// committed, and lowered only where the queue is provably drained
// (after waitUntilCompleted in sync mode, or in metal_sync_queue).
//
// It exists so metal_release_buffer's page purge can never race a
// commit: g_lastCmdBuf alone is assigned AFTER [cmdBuf commit], so
// there is a window in which work is in flight but the pointer is
// still nil. Nothing may be purged in that window.
//
// Atomic because it is written by the goroutine driving training and
// read by whichever thread runs a Buffer finalizer. Only one direction
// of a stale read is dangerous — seeing 0 while work is in flight
// would allow an unsafe purge — so this is sequentially consistent
// rather than relaxed.
static _Atomic int g_maybeInFlight = 0;

// metal_begin_work marks the start of an encode+commit sequence. Every
// entry point that builds a command buffer must call it first.
static inline void metal_begin_work(void) { g_maybeInFlight = 1; }

// metal_finish: called after [cmdBuf commit] by every dispatch entry
// point. Sync mode blocks; async mode records the buffer so
// metal_sync_queue can block later. Command buffers on one queue
// execute in commit order, so waiting on the last suffices for all.
static inline void metal_finish(id<MTLCommandBuffer> cmdBuf) {
    if (g_async) {
        g_lastCmdBuf = cmdBuf;
    } else {
        [cmdBuf waitUntilCompleted];
        g_maybeInFlight = 0; // queue drained: this buffer was the only work
    }
}

void metal_sync_queue(void) {
    if (g_lastCmdBuf) {
        [g_lastCmdBuf waitUntilCompleted];
        g_lastCmdBuf = nil;
    }
    g_maybeInFlight = 0;
}

void metal_set_async(int on) {
    if (!on) {
        metal_sync_queue();
    }
    g_async = on;
}

// ---------------------------------------------------------------------------
// Device / queue
// ---------------------------------------------------------------------------

MTLDeviceRef metal_create_device(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return (__bridge_retained void*)dev;
}

MTLCommandQueueRef metal_create_command_queue(MTLDeviceRef dev) {
    id<MTLDevice> d = (__bridge id<MTLDevice>)dev;
    return (__bridge_retained void*)[d newCommandQueue];
}

// ---------------------------------------------------------------------------
// Shared-memory buffers
// ---------------------------------------------------------------------------

MTLBufferRef metal_create_shared_buffer(MTLDeviceRef dev, uint64_t length) {
    id<MTLDevice> d = (__bridge id<MTLDevice>)dev;
    id<MTLBuffer> buf = [d newBufferWithLength:length
                                       options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

void* metal_buffer_contents(MTLBufferRef buf) {
    return [(__bridge id<MTLBuffer>)buf contents];
}

uint64_t metal_buffer_length(MTLBufferRef buf) {
    return [(__bridge id<MTLBuffer>)buf length];
}

static int g_purgeOnRelease = 1;
// Atomic: finalizers releasing buffers can run on any thread.
static _Atomic uint64_t g_purgedReleases = 0;
static _Atomic uint64_t g_unpurgedReleases = 0;

void metal_set_purge_on_release(int on) { g_purgeOnRelease = on; }

uint64_t metal_purged_releases(void) { return g_purgedReleases; }
uint64_t metal_unpurged_releases(void) { return g_unpurgedReleases; }

// metal_release_buffer destroys a buffer, first discarding its
// physical pages when that is provably safe.
//
// WHY THE PURGE (2026-08-13 measurement). Releasing an
// MTLResourceStorageModeShared buffer does NOT return the physical
// pages the CPU has faulted in through the buffer's `contents`
// mapping, and the driver does not recycle them for later allocations
// of the same size. Pages only ever touched by the GPU come back.
//
// Measured over 200 iterations of allocate-4MB / fill / release, all
// variants matched on size, pages touched (all), byte pattern (a
// non-compressible integer hash, bit-identical whichever side produced
// it), alignment, sync, and release timing; every iteration verified
// by a GPU-side checksum reduction so an unmaterialized write cannot
// masquerade as a reclaim:
//
//   CPU access   GPU access        end footprint (800 MB churned)
//   ----------   ---------------   ------------------------------
//   write        none                810 MB   (== everything churned)
//   none         write + read         50 MB   (flat)
//   write        read (encoded)      850 MB
//   READ only    write + read        850 MB
//
// So the trigger is ANY CPU access to the mapping — a read is as bad
// as a write — and whether the buffer was encoded into a command
// buffer is irrelevant. Purging before release makes the CPU-touched
// case flat (10 MB).
//
// gorch maps every tensor's storage into Go (FloatSlice/Uint16Slice at
// construction) and reads and writes it from the CPU throughout, so
// without the purge a training step's footprint equals its CUMULATIVE
// allocation volume rather than its live set.
//
// WHY THE GUARD. setPurgeableState:Empty discards the contents
// immediately, ignoring the retain count — so a command buffer that
// has encoded this buffer and is still in flight would read garbage.
// That failure mode is silent wrong numbers, not a crash, which makes
// the gate more important than the purge.
//
// Purging is therefore allowed ONLY when the queue is drained by
// COMPLETION, not merely by commit or end-of-encode:
// g_maybeInFlight is raised by metal_begin_work() BEFORE any command
// buffer is created, and lowered only after a waitUntilCompleted has
// actually returned (sync mode) or in metal_sync_queue, which waits on
// the last committed buffer — and a single queue completes in commit
// order, so that one wait drains every earlier buffer too. g_lastCmdBuf
// could not serve as the gate: it is assigned AFTER [cmdBuf commit],
// leaving a window in which work is in flight but the pointer reads
// nil.
//
// The trainer's micro-step flush is sync-then-GC, so the entire
// release wave lands inside the purgeable window. A release that
// arrives with work in flight simply skips the purge: still correct,
// just not reclaimed until a later allocation reuses the region.
void metal_release_buffer(MTLBufferRef buf) {
    if (!buf) return;
    if (g_purgeOnRelease && !g_maybeInFlight) {
        [(__bridge id<MTLBuffer>)buf setPurgeableState:MTLPurgeableStateEmpty];
        g_purgedReleases++;
    } else {
        g_unpurgedReleases++;
    }
    CFRelease(buf);
}

// ---------------------------------------------------------------------------
// Kernel compilation
// ---------------------------------------------------------------------------

MTLComputePipelineRef metal_compile_kernel(MTLDeviceRef dev,
                                            const char* source,
                                            const char* funcName,
                                            char** errOut) {
    @autoreleasepool {
        id<MTLDevice> d = (__bridge id<MTLDevice>)dev;
        NSError* err = nil;

        NSString* src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            if (errOut) {
                const char* msg = [[err localizedDescription] UTF8String];
                *errOut = strdup(msg);
            }
            return NULL;
        }

        NSString* name = [NSString stringWithUTF8String:funcName];
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) {
            if (errOut) {
                char buf[256];
                snprintf(buf, sizeof(buf), "function '%s' not found in shader source", funcName);
                *errOut = strdup(buf);
            }
            return NULL;
        }

        id<MTLComputePipelineState> pso = [d newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            if (errOut) {
                const char* msg = [[err localizedDescription] UTF8String];
                *errOut = strdup(msg);
            }
            return NULL;
        }

        return (__bridge_retained void*)pso;
    }
}

void metal_release_pipeline(MTLComputePipelineRef pipe) {
    if (pipe) {
        CFRelease(pipe);
    }
}

// ---------------------------------------------------------------------------
// 1-D compute dispatch
// ---------------------------------------------------------------------------

void metal_dispatch_1d(MTLCommandQueueRef queue,
                       MTLComputePipelineRef pipeline,
                       MTLBufferRef* bufs, uint32_t bufCount,
                       uint32_t threadCount) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline;

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];

        for (uint32_t i = 0; i < bufCount; i++) {
            [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:i];
        }

        NSUInteger maxThreads = [pso maxTotalThreadsPerThreadgroup];
        MTLSize grid = MTLSizeMake(threadCount, 1, 1);
        MTLSize group = MTLSizeMake(maxThreads < threadCount ? maxThreads : threadCount, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];

        [enc endEncoding];
        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

void metal_dispatch_threadgroups_1d(MTLCommandQueueRef queue,
                                    MTLComputePipelineRef pipeline,
                                    MTLBufferRef* bufs, uint32_t bufCount,
                                    uint32_t groupCount,
                                    uint32_t groupThreads) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline;

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];

        for (uint32_t i = 0; i < bufCount; i++) {
            [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:i];
        }

        // Cap requested threadgroup size to the pipeline's max — silent
        // truncation is fine for our uses (always 256 today, well under
        // the 1024 default cap on Apple Silicon).
        NSUInteger cap = [pso maxTotalThreadsPerThreadgroup];
        NSUInteger threads = groupThreads;
        if (threads > cap) threads = cap;

        MTLSize grid = MTLSizeMake(groupCount, 1, 1);
        MTLSize group = MTLSizeMake(threads, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];

        [enc endEncoding];
        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

// ---------------------------------------------------------------------------
// MPS matrix multiply: C = A @ B  (row-major float32)
// ---------------------------------------------------------------------------

void metal_mps_matmul(MTLCommandQueueRef queue,
                      MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                      uint32_t M, uint32_t N, uint32_t K) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
            rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                 descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)B
                                                 descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                 descriptor:descC];

        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
            initWithDevice:dev resultRows:M resultColumns:N interiorColumns:K];

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

// ---------------------------------------------------------------------------
// Batched MPS matrix multiply: C[i] = A[i] @ B[i]
// All matrices packed contiguously. Single command buffer, one commit.
// ---------------------------------------------------------------------------

void metal_mps_batched_matmul(MTLCommandQueueRef queue,
                              MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                              uint32_t M, uint32_t N, uint32_t K,
                              uint32_t batchSize) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;
        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];

        uint32_t aStride = M * K;
        uint32_t bStride = K * N;
        uint32_t cStride = M * N;

        for (uint32_t i = 0; i < batchSize; i++) {
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:K
                rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
                matrixDescriptorWithRows:K columns:N
                rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:N
                rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                         offset:i * aStride * sizeof(float)
                                                     descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)B
                                                         offset:i * bStride * sizeof(float)
                                                     descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                         offset:i * cStride * sizeof(float)
                                                     descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev resultRows:M resultColumns:N interiorColumns:K];

            [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        }

        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

// ---------------------------------------------------------------------------
// Batched MPS: C[i] = A[i] @ B[i]^T
// A is (batch*M*K), B is (batch*N*K), C is (batch*M*N)
// ---------------------------------------------------------------------------

void metal_mps_batched_matmul_transB(MTLCommandQueueRef queue,
                                     MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t batchSize) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;
        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];

        uint32_t aStride = M * K;
        uint32_t bStride = N * K;
        uint32_t cStride = M * N;

        // MPS transposes B on the fly (transposeRight:YES) — describe B
        // in its stored (N, K) layout. Replaces the previous host-side
        // transpose into a scratch buffer (plan 0009 X1: that transpose
        // was O(batch·N·K) single-threaded CPU work per dispatch).
        for (uint32_t i = 0; i < batchSize; i++) {
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:K
                rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
                matrixDescriptorWithRows:N columns:K
                rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:N
                rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                         offset:i * aStride * sizeof(float)
                                                     descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)B
                                                         offset:i * bStride * sizeof(float)
                                                     descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                         offset:i * cStride * sizeof(float)
                                                     descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev transposeLeft:NO transposeRight:YES
                    resultRows:M resultColumns:N interiorColumns:K
                         alpha:1.0 beta:0.0];

            [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        }

        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

// ---------------------------------------------------------------------------
// Batched MPS: C[i] = A[i]^T @ B[i]
// Per batch, A is stored (K, M) row-major, B is (K, N), C is (M, N).
// Plan 0009 X1 — needed by BatchedMatMul/BatchedMatMulTransB backward.
// ---------------------------------------------------------------------------

void metal_mps_batched_matmul_transA(MTLCommandQueueRef queue,
                                     MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t batchSize) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;
        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];

        uint32_t aStride = K * M;
        uint32_t bStride = K * N;
        uint32_t cStride = M * N;

        for (uint32_t i = 0; i < batchSize; i++) {
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
                matrixDescriptorWithRows:K columns:M
                rowBytes:M * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
                matrixDescriptorWithRows:K columns:N
                rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:N
                rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                         offset:i * aStride * sizeof(float)
                                                     descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)B
                                                         offset:i * bStride * sizeof(float)
                                                     descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                         offset:i * cStride * sizeof(float)
                                                     descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev transposeLeft:YES transposeRight:NO
                    resultRows:M resultColumns:N interiorColumns:K
                         alpha:1.0 beta:0.0];

            [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        }

        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

// ---------------------------------------------------------------------------
// Dtype-parameterized matmul (plan 0009 X3, B0/B4) — see shim.h.
// A/B may independently be bf16; C is always f32 (f32 accumulation).
//
// B0 probe result (2026-08-11, Apple M4, macOS 26.5): MPSMatrix
// REJECTS MPSDataTypeBFloat16 — MPSMatrixMultiplication.mm hard-asserts
// "Input data type must be one of MPSDataTypeFloat32, MPSDataTypeFloat16,
// MPSDataTypeInt8, or MPSDataTypeInt16" and abort()s (not catchable as
// an NSException). Tier (a) of the plan's three-tier fallback is
// therefore DEAD ON THIS OS; this is the tier-(b) implementation:
// MPSGraph matmul with bf16 placeholders (ADR-012).
//
// Numerics: bf16 inputs are cast to f32 INSIDE the graph before the
// matrixMultiplication node, so accumulation is f32 by construction
// (risk R2 contract) — the cast is fused by the MPSGraph compiler, so
// memory traffic stays 2 bytes/element for bf16 operands.
//
// Graphs are cached per (M, N, K, batch, trans, dtype) signature —
// building + compiling an MPSGraph per call would dominate; with the
// cache, steady-state cost is one encode per call on a cached
// executable (shapes are static per model layer).
// ---------------------------------------------------------------------------

API_AVAILABLE(macos(14.0))
@interface GorchDTGraphEntry : NSObject
@property(strong) MPSGraph* graph;
@property(strong) MPSGraphTensor* phA;
@property(strong) MPSGraphTensor* phB;
@property(strong) MPSGraphTensor* out;
@end
@implementation GorchDTGraphEntry
@end

static NSMutableDictionary* g_dtGraphCache = nil; // NSString -> GorchDTGraphEntry
static uint64_t g_dtGraphCacheLimit = 0;          // 0 = unbounded

uint64_t metal_dt_graph_cache_count(void) {
    return g_dtGraphCache ? (uint64_t)[g_dtGraphCache count] : 0;
}

void metal_clear_dt_graph_cache(void) {
    if (g_dtGraphCache) [g_dtGraphCache removeAllObjects];
}

void metal_set_dt_graph_cache_limit(uint64_t limit) {
    g_dtGraphCacheLimit = limit;
}

API_AVAILABLE(macos(14.0))
static GorchDTGraphEntry* gorch_dt_graph(uint32_t M, uint32_t N, uint32_t K,
                                         uint32_t batch, int transA, int transB,
                                         int aBF16, int bBF16) {
    if (!g_dtGraphCache) g_dtGraphCache = [NSMutableDictionary new];
    NSString* key = [NSString stringWithFormat:@"%u_%u_%u_%u_%d%d%d%d",
                     M, N, K, batch, transA, transB, aBF16, bBF16];
    GorchDTGraphEntry* e = g_dtGraphCache[key];
    if (e) return e;

    // Bound the cache. Keys embed M/N/K, and M/K track the SEQUENCE
    // LENGTH of the sample being trained — which varies per sample — so
    // an unbounded cache grows by a fresh compiled MPSGraph per matmul
    // site per unique sequence length, forever. Each graph retains its
    // compiled executable and MPS workspace, neither of which is a
    // gorch MTLBuffer, so the growth is invisible to LiveBufferBytes()
    // and to the Go heap while being fully charged to the process's
    // physical footprint (the 2026-08-13 cross-micro-step compounding).
    if (g_dtGraphCacheLimit > 0 && (uint64_t)[g_dtGraphCache count] >= g_dtGraphCacheLimit) {
        [g_dtGraphCache removeAllObjects];
    }

    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSDataType aType = aBF16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
    MPSDataType bType = bBF16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;

    NSArray<NSNumber*>* aShape = transA ? @[ @(batch), @(K), @(M) ] : @[ @(batch), @(M), @(K) ];
    NSArray<NSNumber*>* bShape = transB ? @[ @(batch), @(N), @(K) ] : @[ @(batch), @(K), @(N) ];

    MPSGraphTensor* phA = [graph placeholderWithShape:aShape dataType:aType name:nil];
    MPSGraphTensor* phB = [graph placeholderWithShape:bShape dataType:bType name:nil];

    // Cast to f32 FIRST: the matmul node then runs (and accumulates)
    // in f32 regardless of storage dtype.
    MPSGraphTensor* aT = aBF16 ? [graph castTensor:phA toType:MPSDataTypeFloat32 name:nil] : phA;
    MPSGraphTensor* bT = bBF16 ? [graph castTensor:phB toType:MPSDataTypeFloat32 name:nil] : phB;
    if (transA) aT = [graph transposeTensor:aT dimension:1 withDimension:2 name:nil];
    if (transB) bT = [graph transposeTensor:bT dimension:1 withDimension:2 name:nil];

    MPSGraphTensor* out = [graph matrixMultiplicationWithPrimaryTensor:aT
                                                       secondaryTensor:bT
                                                                  name:nil];

    e = [GorchDTGraphEntry new];
    e.graph = graph;
    e.phA = phA;
    e.phB = phB;
    e.out = out;
    g_dtGraphCache[key] = e;
    return e;
}

static int gorch_dt_run(MTLCommandQueueRef queue,
                        MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t batch,
                        int transA, int transB, int aBF16, int bBF16) {
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            @try {
                metal_begin_work();
                id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
                GorchDTGraphEntry* e = gorch_dt_graph(M, N, K, batch, transA, transB, aBF16, bBF16);

                MPSDataType aType = aBF16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
                MPSDataType bType = bBF16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
                NSArray<NSNumber*>* aShape = transA ? @[ @(batch), @(K), @(M) ] : @[ @(batch), @(M), @(K) ];
                NSArray<NSNumber*>* bShape = transB ? @[ @(batch), @(N), @(K) ] : @[ @(batch), @(K), @(N) ];
                NSArray<NSNumber*>* cShape = @[ @(batch), @(M), @(N) ];

                MPSGraphTensorData* tdA = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:(__bridge id<MTLBuffer>)A shape:aShape dataType:aType];
                MPSGraphTensorData* tdB = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:(__bridge id<MTLBuffer>)B shape:bShape dataType:bType];
                MPSGraphTensorData* tdC = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:(__bridge id<MTLBuffer>)C shape:cShape dataType:MPSDataTypeFloat32];

                MPSCommandBuffer* cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:q];
                [e.graph encodeToCommandBuffer:cmdBuf
                                         feeds:@{e.phA : tdA, e.phB : tdB}
                              targetOperations:nil
                             resultsDictionary:@{e.out : tdC}
                           executionDescriptor:nil];
                [cmdBuf commit];
                metal_finish(cmdBuf.rootCommandBuffer);
                return 0;
            } @catch (NSException* ex) {
                return 1;
            }
        }
    }
    return 1;
}

int metal_mps_matmul_dt(MTLCommandQueueRef queue,
                        MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                        uint32_t M, uint32_t N, uint32_t K,
                        int transA, int transB,
                        int aBF16, int bBF16) {
    return gorch_dt_run(queue, A, B, C, M, N, K, 1, transA, transB, aBF16, bBF16);
}

int metal_mps_batched_matmul_dt(MTLCommandQueueRef queue,
                                MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                                uint32_t M, uint32_t N, uint32_t K,
                                uint32_t batchSize,
                                int transA, int transB,
                                int aBF16, int bBF16) {
    return gorch_dt_run(queue, A, B, C, M, N, K, batchSize, transA, transB, aBF16, bBF16);
}

// ---------------------------------------------------------------------------
// MPS matrix multiply: C = A @ B^T
// A is MxK, B is NxK (stored row-major), C is MxN.
// MPS transposes B on the fly via transposeRight:YES — B is described
// in its stored (N, K) layout. (Previously this did a host-side
// transpose into a scratch buffer; removed in plan 0009 X1 because the
// O(N·K) single-threaded copy dominated at lm_head-scale shapes.)
// ---------------------------------------------------------------------------

void metal_mps_matmul_transB(MTLCommandQueueRef queue,
                             MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                             uint32_t M, uint32_t N, uint32_t K) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
            rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:N columns:K
            rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                 descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)B
                                                 descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                 descriptor:descC];

        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
            initWithDevice:dev transposeLeft:NO transposeRight:YES
                resultRows:M resultColumns:N interiorColumns:K
                     alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

// ---------------------------------------------------------------------------
// MPS matrix multiply: C = A^T @ B
// A is KxM (stored row-major), B is KxN, C is MxN.
// ---------------------------------------------------------------------------

void metal_mps_matmul_transA(MTLCommandQueueRef queue,
                             MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                             uint32_t M, uint32_t N, uint32_t K) {
    @autoreleasepool {
        metal_begin_work();
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;

        // A is stored (K, M) row-major; MPS transposes it on the fly
        // via transposeLeft:YES (host-side scratch transpose removed in
        // plan 0009 X1 — it was O(K·M) single-threaded CPU work).
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:M
            rowBytes:M * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                 descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)B
                                                 descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                 descriptor:descC];

        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
            initWithDevice:dev transposeLeft:YES transposeRight:NO
                resultRows:M resultColumns:N interiorColumns:K
                     alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        metal_finish(cmdBuf);
    }
}

// ---------------------------------------------------------------------------
// Generic release / string free
// ---------------------------------------------------------------------------

void metal_release(void* obj) {
    if (obj) {
        CFRelease(obj);
    }
}

void metal_free_string(char* s) {
    free(s);
}
