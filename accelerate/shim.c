//go:build darwin

#include <Accelerate/Accelerate.h>
#include "shim.h"

// ---------- BLAS ----------

void acc_sgemm(int M, int N, int K,
               float alpha, const float* A, const float* B,
               float beta, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

void acc_sgemm_transB(int M, int N, int K,
                      float alpha, const float* A, const float* B,
                      float beta, float* C) {
    // B is KxN stored, but we want A @ B^T => result is MxK... no.
    // A is MxK, B is NxK (stored row-major), B^T is KxN.
    // C = A(MxK) @ B^T(KxN) = MxN
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, alpha, A, K, B, K, beta, C, N);
}

void acc_sgemm_transA(int M, int N, int K,
                      float alpha, const float* A, const float* B,
                      float beta, float* C) {
    // A is KxM stored row-major, A^T is MxK. B is KxN.
    // C = A^T(MxK) @ B(KxN) = MxN
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, alpha, A, M, B, N, beta, C, N);
}

// ---------- vDSP ----------

void acc_vadd(const float* A, const float* B, float* C, int64_t n) {
    vDSP_vadd(A, 1, B, 1, C, 1, (vDSP_Length)n);
}

void acc_vsub(const float* A, const float* B, float* C, int64_t n) {
    // vDSP_vsub computes C = A - B, but arguments are (B, strideB, A, strideA, ...)
    // i.e. C[i] = A[i] - B[i]  where A is the SECOND pointer argument.
    vDSP_vsub(B, 1, A, 1, C, 1, (vDSP_Length)n);
}

void acc_vmul(const float* A, const float* B, float* C, int64_t n) {
    vDSP_vmul(A, 1, B, 1, C, 1, (vDSP_Length)n);
}

void acc_vdiv(const float* A, const float* B, float* C, int64_t n) {
    // vDSP_vdiv: C = A / B, but arguments are (B, strideB, A, strideA, ...)
    // i.e. C[i] = A[i] / B[i]  where A is the SECOND pointer argument.
    vDSP_vdiv(B, 1, A, 1, C, 1, (vDSP_Length)n);
}

void acc_vscale(const float* A, float scalar, float* C, int64_t n) {
    vDSP_vsmul(A, 1, &scalar, C, 1, (vDSP_Length)n);
}

void acc_sve(const float* A, float* out, int64_t n) {
    vDSP_sve(A, 1, out, (vDSP_Length)n);
}

void acc_meanv(const float* A, float* out, int64_t n) {
    vDSP_meanv(A, 1, out, (vDSP_Length)n);
}

void acc_maxv(const float* A, float* out, int64_t n) {
    vDSP_maxv(A, 1, out, (vDSP_Length)n);
}

void acc_vrelu(const float* A, float* C, int64_t n) {
    // ReLU: C = max(0, A) using vDSP_vthres (threshold to lower bound)
    float zero = 0.0f;
    vDSP_vthres(A, 1, &zero, C, 1, (vDSP_Length)n);
}

// ---------- vForce ----------

void acc_vexp(const float* A, float* C, int n) {
    vvexpf(C, A, &n);
}

void acc_vlog(const float* A, float* C, int n) {
    vvlogf(C, A, &n);
}

void acc_vtanh(const float* A, float* C, int n) {
    vvtanhf(C, A, &n);
}

void acc_vsigmoid(const float* A, float* C, int n) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    // Step 1: C = -A
    float neg = -1.0f;
    vDSP_vsmul(A, 1, &neg, C, 1, (vDSP_Length)n);
    // Step 2: C = exp(C)
    vvexpf(C, C, &n);
    // Step 3: C = C + 1
    float one = 1.0f;
    vDSP_vsadd(C, 1, &one, C, 1, (vDSP_Length)n);
    // Step 4: C = 1 / C
    vDSP_svdiv(&one, C, 1, C, 1, (vDSP_Length)n);
}
