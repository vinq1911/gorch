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
