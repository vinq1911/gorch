//go:build darwin

// Package metal provides low-level Go bindings to Apple Metal GPU compute.
// It wraps a thin Objective-C shim (shim.m) via CGo, exposing device management,
// shared-memory buffers, kernel compilation, compute dispatch, and MPS matrix ops.
package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders
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
	liveBufferBytes.Add(b.bytes)
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
	runtime.KeepAlive(bufs)
}

// MatMul computes C = A @ B using MPS (Metal Performance Shaders).
// A is MxK, B is KxN, C is MxN. All row-major float32.
func (q *CommandQueue) MatMul(a, b, c *Buffer, M, N, K int) {
	C.metal_mps_matmul(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K))
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
	keepAlive3(a, b, c)
}

// MatMulTransA computes C = A^T @ B using MPS.
// A is KxM (row-major), B is KxN, C is MxN.
func (q *CommandQueue) MatMulTransA(a, b, c *Buffer, M, N, K int) {
	C.metal_mps_matmul_transA(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K))
	keepAlive3(a, b, c)
}

// BatchedMatMul computes C[i] = A[i] @ B[i] for i in 0..batchSize-1 using MPS.
// All matrices packed contiguously. Single GPU command buffer submission.
// A: (batchSize*M*K), B: (batchSize*K*N), C: (batchSize*M*N).
func (q *CommandQueue) BatchedMatMul(a, b, c *Buffer, M, N, K, batchSize int) {
	C.metal_mps_batched_matmul(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K), C.uint32_t(batchSize))
	keepAlive3(a, b, c)
}

// BatchedMatMulTransB computes C[i] = A[i] @ B[i]^T for i in 0..batchSize-1.
// A: (batchSize*M*K), B: (batchSize*N*K), C: (batchSize*M*N).
func (q *CommandQueue) BatchedMatMulTransB(a, b, c *Buffer, M, N, K, batchSize int) {
	C.metal_mps_batched_matmul_transB(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K), C.uint32_t(batchSize))
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
	keepAlive3(a, b, c)
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
