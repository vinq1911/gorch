//go:build darwin

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "shim.h"
#include <os/lock.h>
#include <stdatomic.h>
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

// ---------------------------------------------------------------------------
// Deferred release list
// ---------------------------------------------------------------------------
//
// A buffer released while g_maybeInFlight is raised cannot have its
// pages discarded (see metal_release_buffer). Dropping the purge in
// that case is safe but LOSSY: shared-buffer pages the CPU has touched
// are not returned by release alone, so those bytes stay resident for
// the life of the process. Per micro-step it is noise; across
// optimizer steps it is a monotone ramp (measured 2026-08-17 at
// accum 2 / seq 1024 / 28 layers: a step that abandoned 2672 MB grew
// the physical footprint by 2560 MB, and steps that abandoned nothing
// grew by nothing).
//
// So instead of abandoning those buffers, we keep them — holding the
// +1 retain that Go's handle carried — on a pending list, and purge
// plus release them at the next moment the queue is provably drained.
// Nothing is reclaimed later than it would have been; the buffers that
// used to leak now get discarded one fence later.
//
// CONCURRENCY. The list is pushed by whichever thread runs a Buffer
// finalizer (the Go runtime's finalizer goroutine, concurrent with the
// training goroutine) and drained by the training goroutine. The lock
// covers more than the array: the DECISION to defer and the transition
// out of the in-flight state must be one atomic step with respect to
// each other. Otherwise:
//
//	finalizer  reads inFlight == 1, decides to defer
//	training   waitUntilCompleted returns, clears inFlight, drains — empty
//	finalizer  pushes
//
// and that buffer is stranded on the list forever, retaining its whole
// allocation rather than just its pages: strictly worse than the bug
// being fixed. gorch_leave_inflight therefore clears the flag AND
// detaches the batch inside the same critical section that
// metal_release_buffer uses to read the flag and push.
//
// The purge itself runs OUTSIDE the lock — it is a driver call, and
// holding a lock across it would serialize finalizers against the
// training thread for no benefit.

// Release dispositions. Atomic: finalizers releasing buffers can run on
// any thread. g_purgedReleases counts page discards wherever they
// happen; g_deferredReleases counts how many of those had to wait for a
// drain; g_unpurgedReleases counts pages abandoned unreclaimed.
static _Atomic uint64_t g_purgedReleases = 0;
static _Atomic uint64_t g_unpurgedReleases = 0;

static os_unfair_lock g_pendingLock = OS_UNFAIR_LOCK_INIT;
static MTLBufferRef* g_pending = NULL; // each element carries a +1 retain
static size_t g_pendingLen = 0;
static size_t g_pendingCap = 0;
static _Atomic uint64_t g_pendingBytes = 0;
static _Atomic uint64_t g_pendingPeakBytes = 0;
static _Atomic uint64_t g_deferredReleases = 0;

// g_drainGen counts transitions from "work may be outstanding" to
// "provably drained" — it is bumped in gorch_leave_inflight, in the
// same critical section that clears g_maybeInFlight, and NOWHERE else.
//
// It exists for the R1b buffer reuse cache (ADR-015). Recycling a
// buffer's CONTENTS is a strictly stronger act than releasing it: a
// released buffer's bytes are merely abandoned, whereas a recycled
// buffer's bytes get overwritten by its next owner. So the cache
// cannot lean on the retained-references guarantee that makes a plain
// release safe in flight; it needs to know when the work that could
// still be READING those bytes has completed.
//
// The generation answers exactly that. A buffer entering the cache
// while the gate reads 1 records g_drainGen+1 and may not be handed
// out until g_drainGen has reached it, i.e. until one full
// waitUntilCompleted has returned — and because a single queue
// completes in commit order, that one wait covers every command
// buffer committed before it, which is all of the work that could
// have encoded this buffer.
//
// A buffer entering while the gate reads 0 records g_drainGen and is
// immediately reusable, by the same argument metal_release_buffer
// uses to purge on the spot: its Go owner has already dropped the
// handle, so no FUTURE encode can name it, and gate==0 means every
// PAST commit has completed.
static uint64_t g_drainGen = 0; // guarded by g_pendingLock

uint64_t metal_pending_bytes(void) { return g_pendingBytes; }
uint64_t metal_pending_peak_bytes(void) { return g_pendingPeakBytes; }
uint64_t metal_deferred_releases(void) { return g_deferredReleases; }

// metal_reuse_epoch returns the generation a buffer released RIGHT NOW
// must see before its bytes may be handed to a new owner. Read under
// the pending lock so the gate and the generation are one observation:
// sampling them separately would admit "gate==1, then the drain runs,
// then we read the post-drain generation" — which would wait for a
// drain that had already happened and hand the buffer out early.
uint64_t metal_reuse_epoch(void) {
    os_unfair_lock_lock(&g_pendingLock);
    uint64_t e = g_drainGen + (g_maybeInFlight ? 1 : 0);
    os_unfair_lock_unlock(&g_pendingLock);
    return e;
}

// metal_drain_epoch returns the current drained generation. A cached
// buffer is safe to reuse iff metal_drain_epoch() >= its reuse epoch.
uint64_t metal_drain_epoch(void) {
    os_unfair_lock_lock(&g_pendingLock);
    uint64_t e = g_drainGen;
    os_unfair_lock_unlock(&g_pendingLock);
    return e;
}

// pending_push_locked takes ownership of buf. Returns 0 if the list
// could not grow, in which case ownership stays with the caller — the
// token must never be dropped on the floor.
static int pending_push_locked(MTLBufferRef buf) {
    if (g_pendingLen == g_pendingCap) {
        size_t cap = g_pendingCap ? g_pendingCap * 2 : 256;
        MTLBufferRef* grown = realloc(g_pending, cap * sizeof(MTLBufferRef));
        if (!grown) return 0; // old array intact; caller keeps the token
        g_pending = grown;
        g_pendingCap = cap;
    }
    g_pending[g_pendingLen++] = buf;
    return 1;
}

// gorch_leave_inflight moves the queue from "work may be outstanding"
// to "provably drained" and hands back the batch of buffers that were
// released during the flight. Call ONLY where a waitUntilCompleted has
// actually returned.
//
// The caller purges and releases the batch. Splitting it this way keeps
// the driver calls out of the critical section while keeping the flag
// clear and the detach indivisible.
static void gorch_leave_inflight(MTLBufferRef** batch, size_t* count) {
    os_unfair_lock_lock(&g_pendingLock);
    g_maybeInFlight = 0;
    g_drainGen++; // see g_drainGen: same critical section, by contract
    *batch = g_pending;
    *count = g_pendingLen;
    g_pending = NULL;
    g_pendingLen = 0;
    g_pendingCap = 0;
    g_pendingBytes = 0;
    os_unfair_lock_unlock(&g_pendingLock);
}

// gorch_purge_batch discards and destroys a detached batch.
//
// Safe outside the lock: every buffer in it has already been released
// by its Go owner, so no FUTURE command buffer can encode it, and the
// only work that could have read it was committed before the drain we
// just waited on. The training thread raising g_maybeInFlight again
// while this loop runs therefore cannot make these buffers live.
static void gorch_purge_batch(MTLBufferRef* batch, size_t count) {
    for (size_t i = 0; i < count; i++) {
        [(__bridge id<MTLBuffer>)batch[i] setPurgeableState:MTLPurgeableStateEmpty];
        // g_purgedReleases counts page DISCARDS, wherever they happen;
        // g_deferredReleases counts how many of them had to wait for a
        // drain. So a deferred release advances both, one at a time.
        g_purgedReleases++;
        CFRelease(batch[i]);
    }
    free(batch);
}

// gorch_drain_inflight is gorch_leave_inflight + gorch_purge_batch.
static void gorch_drain_inflight(void) {
    MTLBufferRef* batch = NULL;
    size_t count = 0;
    gorch_leave_inflight(&batch, &count);
    gorch_purge_batch(batch, count);
}

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
        // Queue drained: this buffer was the only work. Sync mode never
        // reaches metal_sync_queue, so this is the ONLY drain point for
        // deferred releases when async is off — without it they would
        // accumulate unbounded.
        gorch_drain_inflight();
    }
}

void metal_sync_queue(void) {
    if (g_lastCmdBuf) {
        [g_lastCmdBuf waitUntilCompleted];
        g_lastCmdBuf = nil;
    }
    gorch_drain_inflight();
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

static _Atomic int g_purgeOnRelease = 1;

// Turning the purge off must not strand buffers already sitting on the
// deferred list: they are unreachable from Go, so nothing else will
// ever free them. Take the batch and destroy it under the semantics
// being selected — "off" means abandon the pages, so no purge here.
void metal_set_purge_on_release(int on) {
    g_purgeOnRelease = on;
    if (on) return;
    MTLBufferRef* batch = NULL;
    size_t count = 0;
    os_unfair_lock_lock(&g_pendingLock);
    batch = g_pending;
    count = g_pendingLen;
    g_pending = NULL;
    g_pendingLen = 0;
    g_pendingCap = 0;
    g_pendingBytes = 0;
    os_unfair_lock_unlock(&g_pendingLock);
    for (size_t i = 0; i < count; i++) {
        g_unpurgedReleases++;
        CFRelease(batch[i]);
    }
    free(batch);
}

uint64_t metal_purged_releases(void) { return g_purgedReleases; }
uint64_t metal_unpurged_releases(void) { return g_unpurgedReleases; }

// metal_drain_pending_releases purges whatever is on the deferred list
// IF the queue currently reads drained. Unlike the drain performed by
// metal_finish / metal_sync_queue it never CLEARS the in-flight flag —
// clearing it is only sound where a waitUntilCompleted has actually
// returned, and this entry point makes no such promise. Safe to call
// from anywhere at any time; a no-op while work is outstanding.
void metal_drain_pending_releases(void) {
    MTLBufferRef* batch = NULL;
    size_t count = 0;
    os_unfair_lock_lock(&g_pendingLock);
    if (!g_maybeInFlight) {
        batch = g_pending;
        count = g_pendingLen;
        g_pending = NULL;
        g_pendingLen = 0;
        g_pendingCap = 0;
        g_pendingBytes = 0;
    }
    os_unfair_lock_unlock(&g_pendingLock);
    gorch_purge_batch(batch, count);
}

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
// The trainer's micro-step flush is sync-then-GC, so most of the
// release wave lands inside the purgeable window. A release that
// arrives with work in flight is DEFERRED to the pending list rather
// than abandoned — see the deferred-release block above for why, and
// for the locking that makes the defer/drain handoff race-free.
//
// WHY THE TOCTOU ON THE GATE IS NOT A BUG. A buffer reaching this
// function has had its last Go reference dropped (Buffer.Release nils
// its handle; a finalizer runs only once the *Buffer is unreachable,
// and gorch pins buffers across every encode with runtime.KeepAlive).
// So no FUTURE command buffer can encode it, and the only work that
// could still read it was committed before this call. Observing the
// gate at 0 is therefore enough to purge even if the training thread
// raises it again immediately afterwards: the new work provably cannot
// touch this buffer. What the gate protects against is work already
// submitted or being encoded, and for that a stale-1 read is the safe
// direction.
int metal_release_buffer(MTLBufferRef buf) {
    if (!buf) return 0;
    if (!g_purgeOnRelease) {
        g_unpurgedReleases++;
        CFRelease(buf);
        return 0;
    }

    // Outside the lock: no message sends in the critical section.
    uint64_t n = [(__bridge id<MTLBuffer>)buf length];

    os_unfair_lock_lock(&g_pendingLock);
    if (g_maybeInFlight) {
        // The push must happen under the same lock that reads the flag:
        // gorch_leave_inflight clears the flag and takes the batch in
        // one critical section, so a buffer can never be enqueued
        // behind the drain that was supposed to collect it.
        if (pending_push_locked(buf)) {
            uint64_t tot = (g_pendingBytes += n);
            os_unfair_lock_unlock(&g_pendingLock);
            for (;;) { // peak, for the pending-list high-water report
                uint64_t old = g_pendingPeakBytes;
                if (tot <= old) break;
                if (atomic_compare_exchange_weak(&g_pendingPeakBytes, &old, tot)) break;
            }
            g_deferredReleases++;
            return 2;
        }
        // The list could not grow. Ownership is still ours, so fall
        // back to the old lossy behaviour rather than leak the object:
        // abandoning pages beats abandoning the whole allocation.
        os_unfair_lock_unlock(&g_pendingLock);
        g_unpurgedReleases++;
        CFRelease(buf);
        return 0;
    }
    os_unfair_lock_unlock(&g_pendingLock);

    [(__bridge id<MTLBuffer>)buf setPurgeableState:MTLPurgeableStateEmpty];
    g_purgedReleases++;
    CFRelease(buf);
    return 1;
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
