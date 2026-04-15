//go:build darwin

package metal

// KernelSource contains Metal shader source for element-wise tensor operations.
// These are compiled at runtime via CompileKernel.
const KernelSource = `
#include <metal_stdlib>
using namespace metal;

kernel void vec_add(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    C[id] = A[id] + B[id];
}

kernel void vec_sub(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    C[id] = A[id] - B[id];
}

kernel void vec_mul(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    C[id] = A[id] * B[id];
}

kernel void vec_div(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    C[id] = A[id] / B[id];
}

kernel void vec_relu(device const float* A [[buffer(0)]],
                     device float*       B [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    B[id] = max(0.0f, A[id]);
}

kernel void vec_sigmoid(device const float* A [[buffer(0)]],
                        device float*       B [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    B[id] = 1.0f / (1.0f + exp(-A[id]));
}

kernel void vec_tanh_act(device const float* A [[buffer(0)]],
                         device float*       B [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    B[id] = tanh(A[id]);
}

kernel void vec_scale(device const float* A    [[buffer(0)]],
                      device float*       B    [[buffer(1)]],
                      device const float* s    [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    B[id] = A[id] * s[0];
}

kernel void vec_sum(device const float* A      [[buffer(0)]],
                    device atomic_float* out    [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(out, A[id], memory_order_relaxed);
}

kernel void vec_gelu(device const float* A [[buffer(0)]],
                     device float*       B [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    float x = A[id];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    B[id] = 0.5f * x * (1.0f + tanh(inner));
}

kernel void vec_bias_add(device const float* A    [[buffer(0)]],
                         device const float* bias  [[buffer(1)]],
                         device float*       C     [[buffer(2)]],
                         device const uint*  ncols [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    uint n = ncols[0];
    C[id] = A[id] + bias[id % n];
}
`
