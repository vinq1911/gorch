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
void         metal_release_buffer(MTLBufferRef buf);

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

// Release a device or command queue.
void metal_release(void* obj);

// Free an error string returned by metal_compile_kernel.
void metal_free_string(char* s);

#endif // GORCH_METAL_SHIM_H
