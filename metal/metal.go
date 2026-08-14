//go:build darwin

// Package metal provides low-level Go bindings to Apple Metal GPU compute.
// It wraps a thin Objective-C shim (shim.m) via CGo, exposing device management,
// shared-memory buffers, kernel compilation, compute dispatch, and MPS matrix ops.
package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
#include "shim.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// Device wraps a Metal GPU device.
type Device struct{ ptr C.MTLDeviceRef }

// CommandQueue wraps a Metal command queue for submitting GPU work.
type CommandQueue struct{ ptr C.MTLCommandQueueRef }

// Buffer wraps a Metal shared-memory buffer.
// Shared mode means Go and the GPU access the same physical memory (zero copy).
//
// Lifecycle: buffers are released either explicitly via Release or by a
// GC finalizer set at allocation time (plan 0009 X1 — residency
// propagation makes every autograd intermediate Metal-backed, so
// leaving release to explicit calls would leak the whole training
// graph every step). Holders must keep the *Buffer reachable for as
// long as any unsafe.Slice view of its contents is in use; gorch
// Tensors do this by retaining both the slice and the *Buffer.
type Buffer struct {
	ptr   C.MTLBufferRef
	bytes int64
}

// liveBufferBytes tracks the total bytes of all live (not-yet-released)
// Metal buffers. Used by benchmarks to measure GPU-resident graph
// memory, which Go's runtime.MemStats cannot see.
var liveBufferBytes atomic.Int64

// LiveBufferBytes returns the total bytes of Metal buffers currently
// allocated and not released. Buffers pending finalization still
// count — call runtime.GC() (twice, finalizers run async) to flush.
func LiveBufferBytes() int64 { return liveBufferBytes.Load() }

// peakLiveBufferBytes is the high-water mark of liveBufferBytes since
// the last ResetBufferStats; totalAllocBytes/totalAllocCount are the
// cumulative allocation volume over the same window.
//
// LiveBufferBytes alone cannot distinguish "nothing was retained" from
// "a huge transient came and went": both read the same at a micro-step
// boundary. The peak is what the allocator (and the physical
// footprint) actually had to accommodate, so it is the number that
// explains footprint.
var (
	peakLiveBufferBytes atomic.Int64
	totalAllocBytes     atomic.Int64
	totalAllocCount     atomic.Int64
)

// BufferStats returns the peak live bytes, cumulative allocated bytes,
// and allocation count since the last ResetBufferStats.
func BufferStats() (peakLive, totalAlloc, allocCount int64) {
	return peakLiveBufferBytes.Load(), totalAllocBytes.Load(), totalAllocCount.Load()
}

// ResetBufferStats restarts the peak/cumulative window, seeding the
// peak with the currently live bytes.
func ResetBufferStats() {
	peakLiveBufferBytes.Store(liveBufferBytes.Load())
	totalAllocBytes.Store(0)
	totalAllocCount.Store(0)
}

// notePeak raises the high-water mark to cur if cur exceeds it.
func notePeak(cur int64) {
	for {
		old := peakLiveBufferBytes.Load()
		if cur <= old || peakLiveBufferBytes.CompareAndSwap(old, cur) {
			return
		}
	}
}

// ---------- async dispatch mode (plan 0009 X2, risk R6) ----------
//
// Default (sync) mode blocks in waitUntilCompleted after every
// dispatch — one full GPU round trip per op, measured at ~46% of the
// X1K1 block-step wall clock. Async mode commits without waiting;
// SyncQueue blocks until all committed work completes (command buffers
// on one queue run in commit order, so waiting on the last suffices).
// Callers must SyncQueue before any CPU read of GPU-written memory —
// gorch's op layer does this via its syncForCPU helper and
// Tensor.Data(). Single global queue, single-threaded by design.

var (
	asyncMode    atomic.Bool
	asyncPending atomic.Bool
	// SyncWaits counts SyncQueue calls that actually had pending GPU
	// work to wait for — the R6 measurement of how many host-visible
	// sync points remain per step in async mode.
	SyncWaits atomic.Int64
)

// SetAsync switches commit-without-wait mode on or off. Turning it off
// synchronizes first, so pending work never outlives the mode.
func SetAsync(on bool) {
	if on {
		asyncMode.Store(true)
		C.metal_set_async(1)
		return
	}
	C.metal_set_async(0) // also waits for pending work
	asyncPending.Store(false)
	asyncMode.Store(false)
}

// AsyncEnabled reports whether commit-without-wait mode is on.
func AsyncEnabled() bool { return asyncMode.Load() }

// SyncQueue blocks until every committed command buffer has completed.
// Cheap no-op (one atomic load) when nothing is pending.
func SyncQueue() {
	if !asyncPending.Load() {
		return
	}
	SyncWaits.Add(1)
	C.metal_sync_queue()
	asyncPending.Store(false)
}

// notePending marks that a dispatch was committed without waiting.
func notePending() {
	if asyncMode.Load() {
		asyncPending.Store(true)
	}
}

// Pipeline wraps a compiled Metal compute pipeline (one kernel function).
type Pipeline struct{ ptr C.MTLComputePipelineRef }

// NewDevice returns the system default Metal device.
func NewDevice() (*Device, error) {
	ptr := C.metal_create_device()
	if ptr == nil {
		return nil, fmt.Errorf("metal: no GPU device found")
	}
	return &Device{ptr: ptr}, nil
}

// NewCommandQueue creates a command queue on this device.
func (d *Device) NewCommandQueue() *CommandQueue {
	return &CommandQueue{ptr: C.metal_create_command_queue(d.ptr)}
}

// NewBuffer allocates a shared-memory GPU buffer of the given size in
// bytes. Contents are zero-filled (Metal's newBufferWithLength
// contract). The buffer is released automatically when the *Buffer
// becomes unreachable; call Release for deterministic freeing.
func (d *Device) NewBuffer(sizeBytes int) *Buffer {
	b := &Buffer{ptr: C.metal_create_shared_buffer(d.ptr, C.uint64_t(sizeBytes)), bytes: int64(sizeBytes)}
	notePeak(liveBufferBytes.Add(b.bytes))
	totalAllocBytes.Add(b.bytes)
	totalAllocCount.Add(1)
	runtime.SetFinalizer(b, (*Buffer).Release)
	return b
}

// FloatSlice returns a Go float32 slice backed by the buffer's unified memory.
// The slice length is buffer size / 4. Writes to the slice are visible to the GPU
// and vice versa — no copies needed.
func (b *Buffer) FloatSlice() []float32 {
	ptr := C.metal_buffer_contents(b.ptr)
	n := int(C.metal_buffer_length(b.ptr)) / 4
	return unsafe.Slice((*float32)(ptr), n)
}

// Uint16Slice returns the buffer's contents as a Go []uint16 slice
// (length = buffer size / 2). Used for bfloat16 tensor storage (plan
// 0009 X3-B2): gorch bf16 tensors keep their bits as uint16, and this
// view lets them live in unified memory exactly like FloatSlice does
// for f32.
func (b *Buffer) Uint16Slice() []uint16 {
	ptr := C.metal_buffer_contents(b.ptr)
	n := int(C.metal_buffer_length(b.ptr)) / 2
	return unsafe.Slice((*uint16)(ptr), n)
}

// Uint32Slice returns the buffer's contents as a Go []uint32 slice.
// Used to fill small uniform buffers (dims, counts) for kernels that
// expect `device const uint*` arguments.
func (b *Buffer) Uint32Slice() []uint32 {
	ptr := C.metal_buffer_contents(b.ptr)
	n := int(C.metal_buffer_length(b.ptr)) / 4
	return unsafe.Slice((*uint32)(ptr), n)
}

// Len returns the buffer size in bytes.
func (b *Buffer) Len() int {
	return int(C.metal_buffer_length(b.ptr))
}

// Release frees the Metal buffer. The Go slice from FloatSlice becomes
// invalid. Idempotent — safe to call before the GC finalizer runs.
func (b *Buffer) Release() {
	if b.ptr == nil {
		return
	}
	runtime.SetFinalizer(b, nil)
	liveBufferBytes.Add(-b.bytes)
	C.metal_release_buffer(b.ptr)
	b.ptr = nil
}

// CompileKernel compiles a Metal shader source string and returns a pipeline
// for the named kernel function.
func (d *Device) CompileKernel(source, funcName string) (*Pipeline, error) {
	csrc := C.CString(source)
	cfn := C.CString(funcName)
	defer C.free(unsafe.Pointer(csrc))
	defer C.free(unsafe.Pointer(cfn))

	var errMsg *C.char
	ptr := C.metal_compile_kernel(d.ptr, csrc, cfn, &errMsg)
	if ptr == nil {
		msg := C.GoString(errMsg)
		C.metal_free_string(errMsg)
		return nil, fmt.Errorf("metal: compile kernel %q: %s", funcName, msg)
	}
	return &Pipeline{ptr: ptr}, nil
}

// Dispatch1D launches a 1-D compute kernel with the given buffers bound at
// sequential indices and the specified total thread count.
func (q *CommandQueue) Dispatch1D(pipe *Pipeline, bufs []*Buffer, threadCount int) {
	cbufs := make([]C.MTLBufferRef, len(bufs))
	for i, b := range bufs {
		cbufs[i] = b.ptr
	}
	C.metal_dispatch_1d(q.ptr, pipe.ptr,
		&cbufs[0], C.uint32_t(len(cbufs)),
		C.uint32_t(threadCount))
	notePending()
	runtime.KeepAlive(bufs)
}

// Dispatch1DThreadgroups launches groupCount threadgroups of exactly
// groupThreads lanes each. Use for kernels that depend on a known
// threadgroup size — typically reduction kernels with shared memory
// (RMSNorm, Softmax, …). For simple element-wise kernels prefer
// Dispatch1D, which lets Metal pick the threadgroup shape.
func (q *CommandQueue) Dispatch1DThreadgroups(pipe *Pipeline, bufs []*Buffer, groupCount, groupThreads int) {
	cbufs := make([]C.MTLBufferRef, len(bufs))
	for i, b := range bufs {
		cbufs[i] = b.ptr
	}
	C.metal_dispatch_threadgroups_1d(q.ptr, pipe.ptr,
		&cbufs[0], C.uint32_t(len(cbufs)),
		C.uint32_t(groupCount), C.uint32_t(groupThreads))
	notePending()
	runtime.KeepAlive(bufs)
}

// MatMul computes C = A @ B using MPS (Metal Performance Shaders).
// A is MxK, B is KxN, C is MxN. All row-major float32.
func (q *CommandQueue) MatMul(a, b, c *Buffer, M, N, K int) {
	C.metal_mps_matmul(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K))
	notePending()
	keepAlive3(a, b, c)
}

// keepAlive3 pins three buffers past the preceding CGo call so the GC
// finalizer cannot release one mid-dispatch.
func keepAlive3(a, b, c *Buffer) {
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	runtime.KeepAlive(c)
}

// MatMulTransB computes C = A @ B^T using MPS.
// A is MxK, B is NxK (row-major), C is MxN.
func (q *CommandQueue) MatMulTransB(a, b, c *Buffer, M, N, K int) {
	C.metal_mps_matmul_transB(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K))
	notePending()
	keepAlive3(a, b, c)
}

// MatMulTransA computes C = A^T @ B using MPS.
// A is KxM (row-major), B is KxN, C is MxN.
func (q *CommandQueue) MatMulTransA(a, b, c *Buffer, M, N, K int) {
	C.metal_mps_matmul_transA(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K))
	notePending()
	keepAlive3(a, b, c)
}

// BatchedMatMul computes C[i] = A[i] @ B[i] for i in 0..batchSize-1 using MPS.
// All matrices packed contiguously. Single GPU command buffer submission.
// A: (batchSize*M*K), B: (batchSize*K*N), C: (batchSize*M*N).
func (q *CommandQueue) BatchedMatMul(a, b, c *Buffer, M, N, K, batchSize int) {
	C.metal_mps_batched_matmul(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K), C.uint32_t(batchSize))
	notePending()
	keepAlive3(a, b, c)
}

// BatchedMatMulTransB computes C[i] = A[i] @ B[i]^T for i in 0..batchSize-1.
// A: (batchSize*M*K), B: (batchSize*N*K), C: (batchSize*M*N).
func (q *CommandQueue) BatchedMatMulTransB(a, b, c *Buffer, M, N, K, batchSize int) {
	C.metal_mps_batched_matmul_transB(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K), C.uint32_t(batchSize))
	notePending()
	keepAlive3(a, b, c)
}

// BatchedMatMulTransA computes C[i] = A[i]^T @ B[i] for i in
// 0..batchSize-1 using MPS. Per batch, A is stored (K, M) row-major,
// B is (K, N), C is (M, N) — the contraction runs over the leading
// (row) dimension of both operands, matching the unbatched
// MatMulTransA convention. Needed by the batched-matmul backward
// passes (plan 0009 X1): dB = A^T @ grad and dB = grad^T @ A.
// A: (batchSize*K*M), B: (batchSize*K*N), C: (batchSize*M*N).
func (q *CommandQueue) BatchedMatMulTransA(a, b, c *Buffer, M, N, K, batchSize int) {
	C.metal_mps_batched_matmul_transA(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K), C.uint32_t(batchSize))
	notePending()
	keepAlive3(a, b, c)
}

// ErrBF16Unsupported is returned by the dtyped matmul entry points
// when MPSDataTypeBFloat16 is unavailable on this OS/device or MPS
// rejects the configuration (plan 0009 X3-B0 outcome b/c territory).
var ErrBF16Unsupported = fmt.Errorf("metal: MPS bf16 matmul unsupported on this device/OS")

// MatMulDT computes C = opA(A) @ opB(B) with per-operand dtypes (plan
// 0009 X3-B4). aBF16/bBF16 mark the corresponding operand buffer as
// bfloat16 (2 bytes/element); C is always float32 — MPS accumulates in
// f32 when the result matrix is f32 (risk R2 contract, verified by the
// B0 probe). Logical shapes after transposes: (M,K) @ (K,N) → (M,N);
// stored A is (K,M) when transA, stored B is (N,K) when transB.
//
// Returns ErrBF16Unsupported when the shim reports failure. A nil
// error does NOT guarantee correct numerics on untested OS versions —
// gorch verifies once per process via its bf16 probe before routing
// real work here.
func (q *CommandQueue) MatMulDT(a, b, c *Buffer, M, N, K int, transA, transB, aBF16, bBF16 bool) error {
	rc := C.metal_mps_matmul_dt(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K),
		cbool(transA), cbool(transB), cbool(aBF16), cbool(bBF16))
	keepAlive3(a, b, c)
	if rc != 0 {
		return ErrBF16Unsupported
	}
	notePending()
	return nil
}

// BatchedMatMulDT is the batched variant of MatMulDT: C[i] =
// opA(A[i]) @ opB(B[i]) for i in 0..batchSize-1, matrices packed
// contiguously per operand. Same dtype/transpose semantics.
func (q *CommandQueue) BatchedMatMulDT(a, b, c *Buffer, M, N, K, batchSize int, transA, transB, aBF16, bBF16 bool) error {
	rc := C.metal_mps_batched_matmul_dt(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K), C.uint32_t(batchSize),
		cbool(transA), cbool(transB), cbool(aBF16), cbool(bBF16))
	keepAlive3(a, b, c)
	if rc != 0 {
		return ErrBF16Unsupported
	}
	notePending()
	return nil
}

// ---------- shared-buffer page reclaim ----------
//
// Releasing an MTLResourceStorageModeShared buffer does not return the
// physical pages the CPU has faulted in through its contents mapping —
// a read is as bad as a write — and the driver does not recycle them
// for later same-size allocations. Pages only ever touched by the GPU
// do come back. Release therefore discards a buffer's pages explicitly
// before destroying it, but only once GPU work is known to have
// COMPLETED (see metal_release_buffer in shim.m for the matched
// measurements and the safety argument).
//
// This buys bounded footprint without an allocator. Recycling buffers
// in a size-classed cache would be the better long-term fix: it avoids
// the churn entirely rather than paying to undo it.

// SetPurgeOnRelease enables or disables the page discard performed by
// Release. It is ON by default; turning it off restores the unbounded
// footprint growth and exists only so tests can demonstrate the
// difference.
func SetPurgeOnRelease(on bool) { C.metal_set_purge_on_release(cbool(on)) }

// PurgeStats returns how many buffer releases were able to discard
// their pages immediately, and how many could not because GPU work was
// still in flight. Unpurged releases are correct but reclaim late.
func PurgeStats() (purged, unpurged int64) {
	return int64(C.metal_purged_releases()), int64(C.metal_unpurged_releases())
}

// ---------- dtyped-matmul MPSGraph cache (plan 0009 X3 / ADR-012) ----------
//
// The shim caches one compiled MPSGraph per
// (M, N, K, batch, transA, transB, aBF16, bBF16) signature. M and K
// carry the SEQUENCE LENGTH of the sample being trained, so on
// variable-length data every new sequence length mints a fresh graph at
// every matmul site. Each graph retains a compiled MPSGraph executable
// and its MPS workspace; those are neither Go-heap objects nor gorch
// MTLBuffers, so they are invisible to runtime.MemStats AND to
// LiveBufferBytes, while being fully charged to the process's physical
// footprint. Left unbounded, this is a monotone footprint leak across
// gradient-accumulation micro-steps.

// DTGraphCacheLen returns the number of compiled dtyped-matmul graphs
// currently cached. Exact and instant — no GC or sampling involved.
func DTGraphCacheLen() int { return int(C.metal_dt_graph_cache_count()) }

// ClearDTGraphCache drops every cached dtyped-matmul graph, releasing
// the compiled executables and their MPS workspaces. Subsequent
// matmuls recompile on demand.
func ClearDTGraphCache() { C.metal_clear_dt_graph_cache() }

// SetDTGraphCacheLimit caps the dtyped-matmul graph cache at n entries;
// the cache is dropped wholesale when a miss would exceed the cap.
// n <= 0 restores unbounded growth (the pre-fix behaviour) and is only
// useful for reproducing the leak in tests.
func SetDTGraphCacheLimit(n int) {
	if n < 0 {
		n = 0
	}
	C.metal_set_dt_graph_cache_limit(C.uint64_t(n))
}

func cbool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// Release frees a device.
func (d *Device) Release() {
	C.metal_release(unsafe.Pointer(d.ptr))
	d.ptr = nil
}

// Release frees the command queue.
func (q *CommandQueue) Release() {
	C.metal_release(unsafe.Pointer(q.ptr))
	q.ptr = nil
}

// Release frees the pipeline.
func (p *Pipeline) Release() {
	C.metal_release_pipeline(p.ptr)
	p.ptr = nil
}
