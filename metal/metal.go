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
type Device struct {
	ptr C.MTLDeviceRef
	// cache is this device's size-classed buffer reuse pool (R1b,
	// ADR-015). See buffercache.go — recycling is what keeps a long
	// run from stranding one IOAccelerator VM map entry per allocation.
	cache *bufferCache
}

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
	ptr C.MTLBufferRef
	// bytes is the CALLER-VISIBLE length: exactly what was requested.
	// cap is what was actually allocated — the size class, which the
	// reuse cache keys on and which is the real memory cost. A recycled
	// 64 KB buffer handed out for a 40 KB request presents as 40 KB to
	// Len, to every slice view, and therefore to MPS; nothing outside
	// this package ever sees cap. Keeping them apart is what makes the
	// cache invisible to shape-sensitive callers.
	bytes int64
	cap   int64
	// cache is the pool this buffer returns to on Release; nil for
	// buffers too large to be worth pooling.
	cache *bufferCache
	// released makes Release exactly-once under the race between an
	// explicit call and the GC finalizer.
	released atomic.Bool
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

// liveBufferLimit, when > 0, is a hard ceiling on live Metal bytes:
// NewBuffer panics rather than allocating past it.
//
// WHY A PANIC IS THE KIND OPTION. Metal shared buffers are the bulk of
// this process's physical footprint, and the footprint is what macOS
// jetsam acts on. On a saturated 24 GB machine jetsam does not politely
// kill the trainer — it takes the desktop with it (2026-08-12
// post-mortem: three SIGKILLs, one hard reboot). The trainer's own
// -rss-limit-mb guard cannot catch this because it samples vmmap at
// MICRO-STEP boundaries, and the dangerous peak is transient: measured
// at 28 layers / seq 512 / accum 1, live buffers peaked at 12.3 GB
// mid-micro-step while the boundary sample read 1.9 GB. The peak is
// only visible where the allocation happens.
//
// So: one atomic compare per buffer allocation, on a path that already
// does a cgo call into the Metal allocator. Free, and it turns a
// machine-killing event into a stack trace.
var liveBufferLimit atomic.Int64

// SetLiveBufferLimit sets the hard ceiling on total live Metal buffer
// bytes; 0 (the default) disables it. See liveBufferLimit.
func SetLiveBufferLimit(bytes int64) { liveBufferLimit.Store(bytes) }

// LiveBufferLimit returns the current ceiling (0 = none).
func LiveBufferLimit() int64 { return liveBufferLimit.Load() }

// checkLiveLimit panics if live bytes have passed the ceiling.
//
// Buffers on the deferred-release list count. Their Go owners are gone,
// so liveBufferBytes has stopped tracking them, but their allocations
// are still resident until the next drain — leaving them out would let
// the ceiling under-report exactly the memory it exists to bound.
func checkLiveLimit(live int64) {
	lim := liveBufferLimit.Load()
	if lim <= 0 {
		return
	}
	pend, cache := int64(C.metal_pending_bytes()), cachedBytes.Load()
	if live+pend+cache <= lim {
		return
	}
	// The reuse cache holds real resident memory, so it counts — but it
	// is also the one component we can give back on the spot, and doing
	// so is strictly kinder than aborting a training run to protect
	// memory we were only holding speculatively.
	DrainBufferCache()
	pend, cache = int64(C.metal_pending_bytes()), cachedBytes.Load()
	total := live + pend + cache
	if total > lim {
		panic(fmt.Sprintf("metal: live buffer ceiling exceeded: %.0f MB (%.0f MB live + "+
			"%.0f MB awaiting purge + %.0f MB cached) > limit %.0f MB "+
			"— aborting before the OS does (lower -accum / -max-seq, or enable -checkpoint-every)",
			float64(total)/(1<<20), float64(live)/(1<<20),
			float64(pend)/(1<<20), float64(cache)/(1<<20), float64(lim)/(1<<20)))
	}
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
	return &Device{ptr: ptr, cache: newBufferCache()}, nil
}

// NewCommandQueue creates a command queue on this device.
func (d *Device) NewCommandQueue() *CommandQueue {
	return &CommandQueue{ptr: C.metal_create_command_queue(d.ptr)}
}

// NewBuffer allocates a shared-memory GPU buffer of the given size in
// bytes. Contents are zero-filled — by Metal's newBufferWithLength
// contract for a fresh allocation, and explicitly by this function for
// one recycled out of the reuse cache. The buffer is released
// automatically when the *Buffer becomes unreachable; call Release for
// deterministic freeing.
//
// Len and every slice view report exactly sizeBytes even when the
// underlying allocation is a larger size class, so callers — and MPS —
// cannot tell a recycled buffer from a fresh one.
func (d *Device) NewBuffer(sizeBytes int) *Buffer {
	if sizeBytes < 0 {
		panic(fmt.Sprintf("gorch/metal: negative buffer size %d", sizeBytes))
	}
	class := int64(SizeClassFor(sizeBytes))
	cache := d.cache
	if cache == nil || class > maxCacheable() {
		// Too big to pool: allocate exactly what was asked for rather
		// than round up. These are the model's frozen weight tensors,
		// which are allocated once and held for the run, so they neither
		// benefit from recycling nor contribute to the region leak — and
		// rounding a 51 MB weight up would cost real memory 28 times over.
		cache, class = nil, int64(sizeBytes)
	} else if p, ok := cache.get(int(class)); ok {
		cacheHits.Add(1)
		return d.adoptBuffer(p, int64(sizeBytes), class, cache, true)
	}
	cacheMisses.Add(1)

	ptr := C.metal_create_shared_buffer(d.ptr, C.uint64_t(class))
	if ptr == nil {
		// newBufferWithLength: returned nil — the driver refused the
		// allocation. Until 2026-08-17 this was passed straight through
		// to MPS, which aborts the PROCESS on an internal assertion
		// ("buffer may not be nil") with no Go stack and no chance to
		// recover. That killed a Stage-A training run at step ~250
		// eight times in a row.
		//
		// The most likely cause is transient pressure, and we are
		// holding memory we can give back. Two pools, in the order that
		// frees the most for the least disruption:
		//
		//  1. the reuse cache — speculative by definition, and after
		//     ADR-015 the largest thing we hold that nobody is using;
		//  2. the deferred-release list (buffers released while GPU work
		//     was in flight, purged at the next drained window).
		//
		// Drain both and retry once. The cache goes first because its
		// releases land ON the deferred list when work is in flight, so
		// the sync that follows sweeps them too.
		DrainBufferCache()
		C.metal_sync_queue()
		ptr = C.metal_create_shared_buffer(d.ptr, C.uint64_t(class))
	}
	if ptr == nil {
		cs := BufferCacheStatsSnapshot()
		panic(fmt.Sprintf("gorch/metal: allocation of %d bytes (%.1f MB, class %.1f MB) failed even "+
			"after draining the reuse cache and pending releases: live %.0f MB, pending %.0f MB, "+
			"cached %.0f MB, %d buffers allocated this process (cache %d hits / %d misses). "+
			"The driver is out of resources — see ADR-014 (every CPU-touched buffer strands a VM map "+
			"region for the life of the process) and ADR-015 (the reuse cache that stops it)",
			sizeBytes, float64(sizeBytes)/(1<<20), float64(class)/(1<<20),
			float64(liveBufferBytes.Load())/(1<<20),
			float64(C.metal_pending_bytes())/(1<<20),
			float64(cs.CachedBytes)/(1<<20),
			totalAllocCount.Load(), cs.Hits, cs.Misses))
	}
	return d.adoptBuffer(ptr, int64(sizeBytes), class, cache, false)
}

// adoptBuffer wraps a raw MTLBuffer — fresh or recycled — in a *Buffer.
//
// ZEROING. A fresh newBufferWithLength: is zero-filled; a recycled one
// carries its previous owner's bytes, and handing those to a caller
// that writes only part of its allocation is a silent wrong answer —
// far worse than the leak this cache exists to fix. gorch has ~30
// NewBuffer call sites and several of them fill their storage
// partially, so "provably fully overwritten" is not a property this
// package can assert on their behalf. We zero.
//
// Only the REQUESTED length is cleared, not the whole class. That is
// exactly the range a fresh buffer would have had, and therefore
// exactly the range any correct consumer may touch: Len and every slice
// view stop there, MPS matrix descriptors and MPSGraph shapes are built
// from the caller's own dimensions, and the compute kernels dispatch
// exact thread counts. GORCH_METAL_CACHE_POISON=1 fills the bytes past
// it with a NaN pattern so the whole test suite can be run as an audit
// of that claim.
//
// The cost is not what it looks like. A fresh buffer's pages are
// zero-fill-on-demand: the caller pays a fault plus a kernel clear on
// first touch of every page. A recycled buffer's pages are already
// resident and already faulted, so this memclr replaces that path
// rather than adding to it.
func (d *Device) adoptBuffer(ptr C.MTLBufferRef,
	requested, class int64, cache *bufferCache, recycled bool) *Buffer {

	if recycled {
		// metal_buffer_contents is deliberately NOT called for a fresh
		// buffer: establishing the CPU mapping is itself what strands
		// the VM map entry (ADR-014 measures a never-mapped buffer at
		// 0.000 regions per allocation, and eagerly caching the pointer
		// took that to 1.004). A recycled buffer has been mapped by a
		// previous owner already, so asking for it here costs nothing
		// new — and zeroing needs it.
		contents := C.metal_buffer_contents(ptr)
		if cachePoison.Load() {
			w := unsafe.Slice((*uint32)(contents), int(class/4))
			for i := range w {
				w[i] = poisonWord
			}
		}
		if cacheZeroing.Load() {
			clear(unsafe.Slice((*byte)(contents), int(requested)))
		}
	}
	b := &Buffer{ptr: ptr, bytes: requested, cap: class, cache: cache}
	live := liveBufferBytes.Add(class)
	notePeak(live)
	checkLiveLimit(live)
	totalAllocBytes.Add(class)
	totalAllocCount.Add(1)
	runtime.SetFinalizer(b, (*Buffer).Release)
	return b
}

// FloatSlice returns a Go float32 slice backed by the buffer's unified memory.
// The slice length is buffer size / 4. Writes to the slice are visible to the GPU
// and vice versa — no copies needed.
func (b *Buffer) FloatSlice() []float32 {
	return unsafe.Slice((*float32)(C.metal_buffer_contents(b.ptr)), int(b.bytes)/4)
}

// Uint16Slice returns the buffer's contents as a Go []uint16 slice
// (length = buffer size / 2). Used for bfloat16 tensor storage (plan
// 0009 X3-B2): gorch bf16 tensors keep their bits as uint16, and this
// view lets them live in unified memory exactly like FloatSlice does
// for f32.
func (b *Buffer) Uint16Slice() []uint16 {
	return unsafe.Slice((*uint16)(C.metal_buffer_contents(b.ptr)), int(b.bytes)/2)
}

// Uint32Slice returns the buffer's contents as a Go []uint32 slice.
// Used to fill small uniform buffers (dims, counts) for kernels that
// expect `device const uint*` arguments.
func (b *Buffer) Uint32Slice() []uint32 {
	return unsafe.Slice((*uint32)(C.metal_buffer_contents(b.ptr)), int(b.bytes)/4)
}

// Len returns the buffer size in bytes as the CALLER asked for it, not
// as the driver allocated it. A recycled buffer taken from a larger
// size class still reports the requested length — that is the contract
// that lets the reuse cache exist without corrupting shapes.
func (b *Buffer) Len() int { return int(b.bytes) }

// Release hands the Metal buffer back — to the reuse cache when it fits
// there, to the driver otherwise. The Go slice from FloatSlice becomes
// invalid EITHER WAY, and after ADR-015 that is sharper than it used
// to be: a stale slice no longer reads abandoned memory, it reads
// whatever tensor got the recycled allocation next. Holders must not
// keep a slice past the *Buffer it came from.
//
// Idempotent, and exactly-once even if an explicit call races the GC
// finalizer.
func (b *Buffer) Release() {
	if !b.released.CompareAndSwap(false, true) {
		return
	}
	runtime.SetFinalizer(b, nil)
	ptr, n := b.ptr, b.cap
	b.ptr = nil
	liveBufferBytes.Add(-n)
	// Into the cache if it will take it. A buffer entering the cache
	// must NOT be purged: setPurgeableState:Empty discards its contents,
	// which is the right thing for a buffer being destroyed and the
	// wrong thing for one being kept. Eviction, which IS a destruction,
	// goes through the ordinary purge + in-flight-gate path.
	if b.cache != nil && b.cache.put(ptr, n) {
		return
	}
	releaseRawBuffer(ptr, n)
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
// A release that lands while work IS in flight cannot purge on the
// spot. It used to give up, and that residue was the trainer's
// ~1 GB-per-optimizer-step footprint ramp: measured over 8 steps at
// the Stage-A geometry, 3316 MB of releases abandoned their pages
// while the process footprint grew 3379 MB, step for step. Such
// releases are now DEFERRED — the buffer keeps its retain on a pending
// list and is purged at the next fence — so the reclaim no longer
// depends on GC timing landing inside the drained window.
//
// This buys bounded footprint without an allocator. Recycling buffers
// in a size-classed cache would be the better long-term fix: it avoids
// the churn entirely rather than paying to undo it.

// SetPurgeOnRelease enables or disables the page discard performed by
// Release. It is ON by default; turning it off restores the unbounded
// footprint growth and exists only so tests can demonstrate the
// difference. Turning it off also disposes of anything already on the
// deferred list — under the semantics being selected, so those pages
// are abandoned rather than purged.
//
// Not a linearization point: a release already past its branch on
// another thread finishes under the old setting. Call it at startup,
// not mid-run.
func SetPurgeOnRelease(on bool) { C.metal_set_purge_on_release(cbool(on)) }

// PurgeStats returns how many buffer releases discarded their pages
// immediately, how many were deferred to the next drain because GPU
// work was in flight, and how many abandoned their pages entirely.
// The last is the one that costs footprint; it should stay at zero
// unless the purge is switched off.
func PurgeStats() (purged, deferred, unpurged int64) {
	return int64(C.metal_purged_releases()),
		int64(C.metal_deferred_releases()),
		int64(C.metal_unpurged_releases())
}

// purgedBytes/unpurgedBytes are PurgeStats in bytes rather than
// releases. The counts alone cannot be compared against a footprint:
// a micro-step's buffers span four orders of magnitude in size, so
// "30% of releases could not purge" says nothing about how many
// megabytes were abandoned. The byte totals are what a footprint
// ramp has to be reconciled against.
var (
	purgedBytes   atomic.Int64
	deferredBytes atomic.Int64
	unpurgedBytes atomic.Int64
)

// PurgeByteStats returns the cumulative bytes of buffers whose pages
// were reclaimed on release, those deferred to the next drain, and
// those abandoned unreclaimed. The last number is the leak; with the
// deferred list wired up it should stay at zero.
func PurgeByteStats() (purged, deferred, unpurged int64) {
	return purgedBytes.Load(), deferredBytes.Load(), unpurgedBytes.Load()
}

// PendingReleaseBytes returns the bytes of buffers released by their Go
// owner but not yet destroyed, because they were released while GPU
// work was in flight and their pages cannot be discarded until the
// queue drains. Their allocations are still live — LiveBufferBytes has
// already stopped counting them, so a memory ceiling must add this in.
func PendingReleaseBytes() int64 { return int64(C.metal_pending_bytes()) }

// PendingReleasePeakBytes is the high-water mark of PendingReleaseBytes
// since process start: how much extra live memory deferring the
// destruction of in-flight releases actually costs.
func PendingReleasePeakBytes() int64 { return int64(C.metal_pending_peak_bytes()) }

// DrainPendingReleases discards and destroys everything on the deferred
// list if the GPU queue currently reads drained; a no-op otherwise. The
// normal drain points (every fence) make this unnecessary — it exists
// so a caller that has just done something unusual can force the issue.
func DrainPendingReleases() { C.metal_drain_pending_releases() }

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

// Release frees a device. Its reuse cache is drained first — the
// buffers in it are unreachable from Go, so nothing else would ever
// free them.
func (d *Device) Release() {
	if d.cache != nil {
		d.cache.drain()
		d.cache.unregister()
		d.cache = nil
	}
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
