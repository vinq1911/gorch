//go:build darwin

#ifndef GORCH_ACCELERATE_SHIM_H
#define GORCH_ACCELERATE_SHIM_H

#include <stdint.h>

// ---------- BLAS ----------

// C = alpha * A @ B + beta * C.
// A is MxK, B is KxN, C is MxN. All row-major float32.
void acc_sgemm(int M, int N, int K,
               float alpha, const float* A, const float* B,
               float beta, float* C);

// C = alpha * A @ B^T + beta * C.
void acc_sgemm_transB(int M, int N, int K,
                      float alpha, const float* A, const float* B,
                      float beta, float* C);

// C = alpha * A^T @ B + beta * C.
void acc_sgemm_transA(int M, int N, int K,
                      float alpha, const float* A, const float* B,
                      float beta, float* C);

// ---------- vDSP vector ops ----------

void acc_vadd(const float* A, const float* B, float* C, int64_t n);
void acc_vsub(const float* A, const float* B, float* C, int64_t n); // C = A - B
void acc_vmul(const float* A, const float* B, float* C, int64_t n);
void acc_vdiv(const float* A, const float* B, float* C, int64_t n); // C = A / B
void acc_vscale(const float* A, float scalar, float* C, int64_t n); // C = A * scalar
void acc_sve(const float* A, float* out, int64_t n);                // sum
void acc_meanv(const float* A, float* out, int64_t n);              // mean
void acc_maxv(const float* A, float* out, int64_t n);               // max

// C = max(0, A) — ReLU via vDSP threshold
void acc_vrelu(const float* A, float* C, int64_t n);

// C = A + scalar
void acc_vsadd(const float* A, float scalar, float* C, int64_t n);

// ---------- fused optimizer step (plan 0009 K7) ----------

// One AdamW update over n contiguous f32 elements. In-place on p/m/v:
//
//   m = beta1*m + (1-beta1)*g
//   v = beta2*v + (1-beta2)*g*g
//   p -= lr * ((m/bc1) / (sqrt(v/bc2) + eps) + wd*p)
//
// bc1/bc2 are the bias-correction denominators (1 - beta^t) computed by
// the caller — exact same math as the scalar Go loop in optim/adamw.go.
// The loop body is pure float32 arithmetic + sqrtf, which clang
// auto-vectorizes to NEON (vector fsqrt is IEEE-exact, so
// vectorization changes nothing numerically vs the scalar C tail).
void acc_adamw_step(float* restrict p, const float* restrict g,
                    float* restrict m, float* restrict v, int64_t n,
                    float lr, float beta1, float beta2, float eps,
                    float wd, float bc1, float bc2);

// ---------- vForce transcendentals ----------

void acc_vexp(const float* A, float* C, int n);
// ELU alpha=1: C = A > 0 ? A : exp(A) - 1. A and C must not overlap.
void acc_velu(const float* restrict A, float* restrict C, int n);
// Exact-erf GELU: C = 0.5*A*(1+erf(A/sqrt(2))). A and C must not overlap.
void acc_vgelu_erf(const float* restrict A, float* restrict C, int n);
void acc_vlog(const float* A, float* C, int n);
void acc_vtanh(const float* A, float* C, int n);

// Sigmoid: 1 / (1 + exp(-x)) composed from vForce + vDSP
void acc_vsigmoid(const float* A, float* C, int n);

// ---------- SiLU / SwiGLU (plan 0009 K4 CPU fallback) ----------

// SiLU: C = A * sigmoid(A). A and C must not overlap.
void acc_vsilu(const float* restrict A, float* restrict C, int n);
// SiLU backward: DX = G * s * (1 + X*(1-s)), s = sigmoid(X).
// DX may not overlap X or G.
void acc_vsilu_bwd(const float* restrict X, const float* restrict G,
                   float* restrict DX, int n);
// SwiGLU: C = gate * sigmoid(gate) * val. C may not overlap gate/val.
void acc_vswiglu(const float* restrict gate, const float* restrict val,
                 float* restrict C, int n);
// SwiGLU backward, s = sigmoid(gate):
//   dGate = G * val * s * (1 + gate*(1-s))
//   dVal  = G * gate * s
// dGate/dVal may not overlap the inputs or each other.
void acc_vswiglu_bwd(const float* restrict gate, const float* restrict val,
                     const float* restrict G,
                     float* restrict dGate, float* restrict dVal, int n);

#endif // GORCH_ACCELERATE_SHIM_H
