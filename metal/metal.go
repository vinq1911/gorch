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
	"unsafe"
)

// Device wraps a Metal GPU device.
type Device struct{ ptr C.MTLDeviceRef }

// CommandQueue wraps a Metal command queue for submitting GPU work.
type CommandQueue struct{ ptr C.MTLCommandQueueRef }

// Buffer wraps a Metal shared-memory buffer.
// Shared mode means Go and the GPU access the same physical memory (zero copy).
type Buffer struct{ ptr C.MTLBufferRef }

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

// NewBuffer allocates a shared-memory GPU buffer of the given size in bytes.
func (d *Device) NewBuffer(sizeBytes int) *Buffer {
	return &Buffer{ptr: C.metal_create_shared_buffer(d.ptr, C.uint64_t(sizeBytes))}
}

// FloatSlice returns a Go float32 slice backed by the buffer's unified memory.
// The slice length is buffer size / 4. Writes to the slice are visible to the GPU
// and vice versa — no copies needed.
func (b *Buffer) FloatSlice() []float32 {
	ptr := C.metal_buffer_contents(b.ptr)
	n := int(C.metal_buffer_length(b.ptr)) / 4
	return unsafe.Slice((*float32)(ptr), n)
}

// Len returns the buffer size in bytes.
func (b *Buffer) Len() int {
	return int(C.metal_buffer_length(b.ptr))
}

// Release frees the Metal buffer. The Go slice from FloatSlice becomes invalid.
func (b *Buffer) Release() {
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
}

// MatMul computes C = A @ B using MPS (Metal Performance Shaders).
// A is MxK, B is KxN, C is MxN. All row-major float32.
func (q *CommandQueue) MatMul(a, b, c *Buffer, M, N, K int) {
	C.metal_mps_matmul(q.ptr, a.ptr, b.ptr, c.ptr,
		C.uint32_t(M), C.uint32_t(N), C.uint32_t(K))
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
