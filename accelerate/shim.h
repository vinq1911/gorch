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

// ---------- vForce transcendentals ----------

void acc_vexp(const float* A, float* C, int n);
void acc_vlog(const float* A, float* C, int n);
void acc_vtanh(const float* A, float* C, int n);

// Sigmoid: 1 / (1 + exp(-x)) composed from vForce + vDSP
void acc_vsigmoid(const float* A, float* C, int n);

#endif // GORCH_ACCELERATE_SHIM_H
