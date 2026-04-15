//go:build darwin

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "shim.h"
#include <stdlib.h>
#include <string.h>

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

void metal_release_buffer(MTLBufferRef buf) {
    if (buf) {
        CFRelease(buf);
    }
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
        [cmdBuf waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// MPS matrix multiply: C = A @ B  (row-major float32)
// ---------------------------------------------------------------------------

void metal_mps_matmul(MTLCommandQueueRef queue,
                      MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                      uint32_t M, uint32_t N, uint32_t K) {
    @autoreleasepool {
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
        [cmdBuf waitUntilCompleted];
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
        [cmdBuf waitUntilCompleted];
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
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;

        // Transpose all B[i] into a temp buffer, then do batched matmul
        uint32_t bSrcStride = N * K;
        uint32_t bDstStride = K * N;
        uint32_t totalBT = batchSize * K * N;
        id<MTLBuffer> btBuf = [dev newBufferWithLength:totalBT * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        float* bData = (float*)[(__bridge id<MTLBuffer>)B contents];
        float* btData = (float*)[btBuf contents];

        for (uint32_t batch = 0; batch < batchSize; batch++) {
            for (uint32_t i = 0; i < N; i++) {
                for (uint32_t j = 0; j < K; j++) {
                    btData[batch * bDstStride + j * N + i] = bData[batch * bSrcStride + i * K + j];
                }
            }
        }

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];

        uint32_t aStride = M * K;
        uint32_t cStride = M * N;

        for (uint32_t i = 0; i < batchSize; i++) {
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:K
                rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descBT = [MPSMatrixDescriptor
                matrixDescriptorWithRows:K columns:N
                rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:N
                rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                         offset:i * aStride * sizeof(float)
                                                     descriptor:descA];
            MPSMatrix* matBT = [[MPSMatrix alloc] initWithBuffer:btBuf
                                                          offset:i * bDstStride * sizeof(float)
                                                      descriptor:descBT];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                         offset:i * cStride * sizeof(float)
                                                     descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev resultRows:M resultColumns:N interiorColumns:K];

            [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matBT resultMatrix:matC];
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// MPS matrix multiply: C = A @ B^T
// A is MxK, B is NxK (stored row-major), C is MxN.
// We describe B as having rows=K, columns=N but with rowBytes=N*sizeof(float)
// so MPS reads it transposed. Actually we use the alpha/interiorColumns trick:
// A(MxK) @ B^T(KxN) = treating B(NxK) as if B^T is (K,N).
// MPS reads B row-major, so we need B^T contiguous.
// Simplest correct approach: describe B as (N,K) and swap interpretation.
// ---------------------------------------------------------------------------

void metal_mps_matmul_transB(MTLCommandQueueRef queue,
                             MTLBufferRef A, MTLBufferRef B, MTLBufferRef C,
                             uint32_t M, uint32_t N, uint32_t K) {
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;

        // A is (M, K) row-major
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
            rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];

        // B is stored as (N, K) row-major. We want B^T = (K, N).
        // Trick: describe B as having K rows and N columns, but with
        // rowBytes = K * sizeof(float) (the original row stride of NxK matrix).
        // This makes MPS read column-by-column through B's original rows.
        // Actually this doesn't work directly — MPS rowBytes must be >= columns*sizeof.

        // Correct approach: we need to actually transpose into a temp buffer.
        // On unified memory this is still fast.
        uint32_t btSize = K * N * sizeof(float);
        id<MTLBuffer> btBuf = [dev newBufferWithLength:btSize
                                               options:MTLResourceStorageModeShared];
        float* bData = (float*)[(__bridge id<MTLBuffer>)B contents];
        float* btData = (float*)[btBuf contents];
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < K; j++) {
                btData[j * N + i] = bData[i * K + j];
            }
        }

        MPSMatrixDescriptor* descBT = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A
                                                 descriptor:descA];
        MPSMatrix* matBT = [[MPSMatrix alloc] initWithBuffer:btBuf descriptor:descBT];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                 descriptor:descC];

        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
            initWithDevice:dev resultRows:M resultColumns:N interiorColumns:K];

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        [mul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matBT resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
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
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q.device;

        // A is stored as (K, M) row-major. We want A^T = (M, K).
        uint32_t atSize = M * K * sizeof(float);
        id<MTLBuffer> atBuf = [dev newBufferWithLength:atSize
                                               options:MTLResourceStorageModeShared];
        float* aData = (float*)[(__bridge id<MTLBuffer>)A contents];
        float* atData = (float*)[atBuf contents];
        for (uint32_t i = 0; i < K; i++) {
            for (uint32_t j = 0; j < M; j++) {
                atData[j * K + i] = aData[i * M + j];
            }
        }

        MPSMatrixDescriptor* descAT = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
            rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
            rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* matAT = [[MPSMatrix alloc] initWithBuffer:atBuf descriptor:descAT];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)B
                                                 descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)C
                                                 descriptor:descC];

        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
            initWithDevice:dev resultRows:M resultColumns:N interiorColumns:K];

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        [mul encodeToCommandBuffer:cmdBuf leftMatrix:matAT rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
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
