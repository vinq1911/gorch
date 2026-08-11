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

// vec_silu / vec_silu_bwd / vec_swiglu / vec_swiglu_bwd
//
// Plan 0009 K4 — SiLU (x·σ(x)) and the SwiGLU gate (silu(gate)·value)
// with their backwards, as trivial element-wise kernels. The backward
// kernels recompute σ(x) instead of reading a cached sigmoid tensor —
// one exp per lane is cheaper than materializing (and keeping alive) a
// seq×3072 cache per layer. CPU references: silu.go (SiLU/SwiGLU) and
// accelerate acc_vsilu/acc_vswiglu*; golden tests silu_metal_test.go.
//
//	silu(x)  = x·σ(x)
//	silu'(x) = σ(x)·(1 + x·(1−σ(x)))
kernel void vec_silu(device const float* A [[buffer(0)]],
                     device float*       B [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    float x = A[id];
    float s = 1.0f / (1.0f + exp(-x));
    B[id] = x * s;
}

kernel void vec_silu_bwd(device const float* X  [[buffer(0)]],
                         device const float* G  [[buffer(1)]],
                         device float*       DX [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    float x = X[id];
    float s = 1.0f / (1.0f + exp(-x));
    DX[id] = G[id] * s * (1.0f + x * (1.0f - s));
}

kernel void vec_swiglu(device const float* GATE [[buffer(0)]],
                       device const float* VAL  [[buffer(1)]],
                       device float*       OUT  [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    float gx = GATE[id];
    float s = 1.0f / (1.0f + exp(-gx));
    OUT[id] = gx * s * VAL[id];
}

kernel void vec_swiglu_bwd(device const float* GATE  [[buffer(0)]],
                           device const float* VAL   [[buffer(1)]],
                           device const float* G     [[buffer(2)]],
                           device float*       DGATE [[buffer(3)]],
                           device float*       DVAL  [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    float gx = GATE[id];
    float s = 1.0f / (1.0f + exp(-gx));
    float gi = G[id];
    DVAL[id]  = gi * gx * s;
    DGATE[id] = gi * VAL[id] * s * (1.0f + gx * (1.0f - s));
}

kernel void vec_bias_add(device const float* A    [[buffer(0)]],
                         device const float* bias  [[buffer(1)]],
                         device float*       C     [[buffer(2)]],
                         device const uint*  ncols [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    uint n = ncols[0];
    C[id] = A[id] + bias[id % n];
}

// rmsnorm_forward
//
// Plan 0004 part A, first kernel. One threadgroup per row of (M, N).
// Lanes do a strided pass over the row's N elements to compute the
// sum-of-squares, reduce across the threadgroup via threadgroup memory,
// invert+rsqrt, then write y[i,j] = x[i,j] * inv * weight[j] in a second
// strided pass. Per-row inv is also written to invRMS[row] so the
// backward kernel can read it without recomputing.
//
// Constraints: threadsPerThreadgroup must be a power of two ≤ 256
// (the tree reduction unrolls for powers of two and the shared array
// is sized 256). Driver always dispatches with 256 threads — the
// strided loops handle N < 256 (lanes whose j ≥ N contribute zero
// to the sum).
kernel void rmsnorm_forward(device const float* x        [[buffer(0)]],
                            device const float* weight   [[buffer(1)]],
                            device const uint*  dims     [[buffer(2)]],
                            device const float* eps      [[buffer(3)]],
                            device float*       y        [[buffer(4)]],
                            device float*       invRMS   [[buffer(5)]],
                            uint tid       [[thread_index_in_threadgroup]],
                            uint group     [[threadgroup_position_in_grid]],
                            uint tgSize    [[threads_per_threadgroup]]) {
    uint M = dims[0];
    uint N = dims[1];
    if (group >= M) return;

    threadgroup float scratch[256];

    float partial = 0.0f;
    for (uint j = tid; j < N; j += tgSize) {
        float v = x[group * N + j];
        partial += v * v;
    }
    scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = rsqrt(scratch[0] / float(N) + eps[0]);
    if (tid == 0) {
        invRMS[group] = inv;
    }

    for (uint j = tid; j < N; j += tgSize) {
        y[group * N + j] = x[group * N + j] * inv * weight[j];
    }
}

// rmsnorm_dx
//
// Plan 0004 part A. Backward with respect to the input x. Companion
// to rmsnorm_forward — assumes invRMS[row] was already written by the
// forward pass. dW is left to the host (CPU loop over (M, N) with
// reads through unified memory) since it is a per-column reduction
// over rows that does not fit the per-row-threadgroup template; the
// follow-up plan-0004 work will fold dW into a second per-column
// kernel.
//
// Math (from the existing CPU backward in nn/rmsnorm.go):
//
//   normalised[j] = x[j] * inv
//   sumDot         = (1/N) * sum_j (weight[j] * grad[j] * normalised[j])
//   dx[j]          = inv * (weight[j] * grad[j] - normalised[j] * sumDot)
kernel void rmsnorm_dx(device const float* x        [[buffer(0)]],
                       device const float* weight   [[buffer(1)]],
                       device const float* grad     [[buffer(2)]],
                       device const float* invRMS   [[buffer(3)]],
                       device const uint*  dims     [[buffer(4)]],
                       device float*       dx       [[buffer(5)]],
                       uint tid    [[thread_index_in_threadgroup]],
                       uint group  [[threadgroup_position_in_grid]],
                       uint tgSize [[threads_per_threadgroup]]) {
    uint M = dims[0];
    uint N = dims[1];
    if (group >= M) return;

    threadgroup float scratch[256];

    float inv = invRMS[group];
    float partial = 0.0f;
    for (uint j = tid; j < N; j += tgSize) {
        float nrm = x[group * N + j] * inv;
        partial += weight[j] * grad[group * N + j] * nrm;
    }
    scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sumDot = scratch[0] / float(N);

    for (uint j = tid; j < N; j += tgSize) {
        float nrm = x[group * N + j] * inv;
        dx[group * N + j] = inv * (weight[j] * grad[group * N + j] - nrm * sumDot);
    }
}

// softmax_causal_forward / softmax_backward
//
// Plan 0009 K1 — fused causal softmax for the attention chain. One
// threadgroup of 256 per row of the (heads·qSeq, kSeq) score view.
// Forward fuses the 1/√d scale, the causal compare-mask (column j
// allowed iff j <= (row % qSeq) + (cols − qSeq) — no mask tensor, no
// −1e9 fill tensor), the row max, exp, sum, and normalize; masked
// columns are written as exact 0. Backward is the standard softmax
// backward dx = y ⊙ (g − Σ(g⊙y)) times the fused scale factor.
// CPU reference: causalSoftmaxForwardCPU / softmaxBackwardCPU in
// softmax_metal.go; golden test softmax_metal_test.go.
//
// Drafted via the plan §4.2 Azure gpt-5.3-codex protocol (see commit
// message for iteration record).
kernel void softmax_causal_forward(device const float* x      [[buffer(0)]],
                                   device const uint*  dims   [[buffer(1)]],
                                   device const float* scale  [[buffer(2)]],
                                   device float*       y      [[buffer(3)]],
                                   uint tid    [[thread_index_in_threadgroup]],
                                   uint group  [[threadgroup_position_in_grid]],
                                   uint tgSize [[threads_per_threadgroup]])
{
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint qSeq = dims[2];
    if (group >= rows) return;

    threadgroup float scratch[256];

    const uint r = group;
    const uint base = r * cols;
    const uint offset = cols - qSeq;
    const uint i = r % qSeq;
    const uint limit = i + offset;
    const float s = scale[0];

    float localMax = -INFINITY;
    for (uint j = tid; j < cols; j += tgSize) {
        if (j <= limit) {
            float v = x[base + j] * s;
            localMax = max(localMax, v);
        }
    }

    scratch[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rowMax = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localSum = 0.0f;
    for (uint j = tid; j < cols; j += tgSize) {
        if (j <= limit) {
            float e = exp((x[base + j] * s) - rowMax);
            y[base + j] = e;
            localSum += e;
        } else {
            y[base + j] = 0.0f;
        }
    }

    scratch[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float invSum = 1.0f / scratch[0];
    for (uint j = tid; j < cols; j += tgSize) {
        if (j <= limit) {
            y[base + j] *= invSum;
        } else {
            y[base + j] = 0.0f;
        }
    }
}

// cross_entropy_forward / cross_entropy_backward
//
// Plan 0009 K2 — fused cross-entropy over vocab-scale rows (N up to
// ~168k, strided loops). Forward: per-row logsumexp + target pick in
// one pass — rowLoss[r] = lse − x[r, target], rowLse[r] kept for
// backward. Backward: dx = (softmax(x) − onehot)·scale reconstructed
// from the saved logsumexp — the pre-K2 CPU path recomputed a full
// Softmax tensor here (the "double softmax" of plan §2.2). f32
// accumulation, rmsnorm threadgroup template. CPU reference:
// ceForwardCPU/ceBackwardCPU in loss.go; golden test ce_metal_test.go.
//
// Drafted via the plan §4.2 Azure gpt-5.3-codex protocol (see commit
// message for iteration record; one reviewer-added barrier between
// the max reduction read and the sum reduction write, as in K1).
kernel void cross_entropy_forward(device const float* x       [[buffer(0)]],
                                  device const uint*  targets [[buffer(1)]],
                                  device const uint*  dims    [[buffer(2)]],
                                  device float*       rowLoss [[buffer(3)]],
                                  device float*       rowLse  [[buffer(4)]],
                                  uint tid    [[thread_index_in_threadgroup]],
                                  uint group  [[threadgroup_position_in_grid]],
                                  uint tgSize [[threads_per_threadgroup]])
{
    const uint rows = dims[0];
    const uint cols = dims[1];
    if (group >= rows) return;

    threadgroup float scratch[256];

    const uint rowBase = group * cols;

    // 1) rowMax = max_j x[row, j]
    float localMax = -INFINITY;
    for (uint j = tid; j < cols; j += tgSize) {
        localMax = max(localMax, x[rowBase + j]);
    }
    scratch[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rowMax = scratch[0];
    // Reviewer-added: every lane must read rowMax before scratch is
    // reused by the sum reduction below.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) sumExp = sum_j exp(x[row, j] - rowMax)
    float localSum = 0.0f;
    for (uint j = tid; j < cols; j += tgSize) {
        localSum += exp(x[rowBase + j] - rowMax);
    }
    scratch[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float sumExp = scratch[0];
        const float lse = rowMax + log(sumExp);
        const uint t = targets[group];
        rowLse[group] = lse;
        rowLoss[group] = lse - x[rowBase + t];
    }
}

kernel void cross_entropy_backward(device const float* x       [[buffer(0)]],
                                   device const uint*  targets [[buffer(1)]],
                                   device const float* rowLse  [[buffer(2)]],
                                   device const uint*  dims    [[buffer(3)]],
                                   device const float* scale   [[buffer(4)]],
                                   device float*       dx      [[buffer(5)]],
                                   uint tid    [[thread_index_in_threadgroup]],
                                   uint group  [[threadgroup_position_in_grid]],
                                   uint tgSize [[threads_per_threadgroup]])
{
    const uint rows = dims[0];
    const uint cols = dims[1];
    if (group >= rows) return;

    const uint rowBase = group * cols;
    const uint t = targets[group];
    const float lse = rowLse[group];
    const float s = scale[0];

    for (uint j = tid; j < cols; j += tgSize) {
        float v = exp(x[rowBase + j] - lse) * s;
        if (j == t) v -= s;
        dx[rowBase + j] = v;
    }
}

kernel void softmax_backward(device const float* y      [[buffer(0)]],
                             device const float* grad   [[buffer(1)]],
                             device const uint*  dims   [[buffer(2)]],
                             device const float* scale  [[buffer(3)]],
                             device float*       dx     [[buffer(4)]],
                             uint tid    [[thread_index_in_threadgroup]],
                             uint group  [[threadgroup_position_in_grid]],
                             uint tgSize [[threads_per_threadgroup]])
{
    const uint rows = dims[0];
    const uint cols = dims[1];
    if (group >= rows) return;

    threadgroup float scratch[256];

    const uint r = group;
    const uint base = r * cols;
    const float s = scale[0];

    float localDot = 0.0f;
    for (uint j = tid; j < cols; j += tgSize) {
        localDot += grad[base + j] * y[base + j];
    }

    scratch[tid] = localDot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float dot = scratch[0];
    for (uint j = tid; j < cols; j += tgSize) {
        float yv = y[base + j];
        float gv = grad[base + j];
        dx[base + j] = s * yv * (gv - dot);
    }
}
`
