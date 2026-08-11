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

void acc_vsadd(const float* A, float scalar, float* C, int64_t n) {
    vDSP_vsadd(A, 1, &scalar, C, 1, (vDSP_Length)n);
}

// ---------- fused optimizer step (plan 0009 K7) ----------

void acc_adamw_step(float* restrict p, const float* restrict g,
                    float* restrict m, float* restrict v, int64_t n,
                    float lr, float beta1, float beta2, float eps,
                    float wd, float bc1, float bc2) {
    const float omb1 = 1.0f - beta1;
    const float omb2 = 1.0f - beta2;
    for (int64_t i = 0; i < n; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + omb1 * gi;
        float vi = beta2 * v[i] + omb2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        // Same expression structure as the Go scalar loop (divisions,
        // not reciprocal-multiplies) so trajectories stay comparable to
        // well under the 1e-6-over-20-steps gate.
        float mHat = mi / bc1;
        float vHat = vi / bc2;
        p[i] -= lr * (mHat / (sqrtf(vHat) + eps) + wd * p[i]);
    }
}

// ---------- vForce ----------

void acc_vexp(const float* A, float* C, int n) {
    vvexpf(C, A, &n);
}

void acc_velu(const float* restrict A, float* restrict C, int n) {
    // Single pass with an inlined Cephes-style expf polynomial instead
    // of vvexpf: vForce's expf rounds differently depending on the
    // call's array length (large-array vs small-array code paths), so
    // the same input value run offline (one big call) and streamed
    // (many small calls) came out ulp-different — breaking the
    // bit-exact streaming == offline property the Mimi encoder tests
    // assert. This loop is strictly element-wise: every operation is
    // an fmaf/rint/convert with identical fused semantics in clang's
    // auto-vectorized body (fmla/frintn) and the scalar tail
    // (fmadd/frintx under round-to-nearest-even), so results do not
    // depend on position or call size. Accuracy ~2 ulp on exp(x) for
    // x in [-87, 0], which only feeds the x <= 0 arm.
    for (int i = 0; i < n; i++) {
        float x = A[i];
        float xn = x < 0.0f ? x : 0.0f;      // exp arg: min(x, 0)
        xn = xn < -87.0f ? -87.0f : xn;      // stay in normal float range
        float k = rintf(xn * 1.44269504088896341f); // x/ln2
        float r = fmaf(k, -0.693359375f, xn);       // x - k*ln2_hi
        r = fmaf(k, 2.12194440e-4f, r);             // - k*ln2_lo
        float y = 1.9875691500e-4f;
        y = fmaf(y, r, 1.3981999507e-3f);
        y = fmaf(y, r, 8.3334519073e-3f);
        y = fmaf(y, r, 4.1665795894e-2f);
        y = fmaf(y, r, 1.6666665459e-1f);
        y = fmaf(y, r, 5.0000001201e-1f);
        y = fmaf(y * r, r, r);               // r + r^2*P(r)
        union { int32_t i; float f; } s;
        s.i = ((int32_t)k + 127) << 23;      // 2^k
        float e = fmaf(y, s.f, s.f);         // exp(xn) = 2^k * (1 + y)
        C[i] = x > 0.0f ? x : e - 1.0f;
    }
}

void acc_vgelu_erf(const float* restrict A, float* restrict C, int n) {
    // Exact-erf GELU: C = 0.5*x*(1+erf(x/sqrt(2))). erff has no vForce
    // equivalent; the float32 libm call is still ~2x faster than Go's
    // float64 math.Erf per element.
    const float invSqrt2 = 0.70710678118654752440f;
    for (int i = 0; i < n; i++) {
        float x = A[i];
        C[i] = 0.5f * x * (1.0f + erff(x * invSqrt2));
    }
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

// ---------- SiLU / SwiGLU (plan 0009 K4 CPU fallback) ----------
//
// All four use the same recipe: one vForce-vectorized sigmoid into the
// output (or a designated output used as scratch), then a pure-
// arithmetic combine loop that clang auto-vectorizes. The combine reads
// the sigmoid value from the scratch slot before overwriting it —
// element-wise same-index access, so no ordering hazard.

void acc_vsilu(const float* restrict A, float* restrict C, int n) {
    acc_vsigmoid(A, C, n);       // C = sigmoid(A)
    vDSP_vmul(A, 1, C, 1, C, 1, (vDSP_Length)n); // C = A * sigmoid(A)
}

void acc_vsilu_bwd(const float* restrict X, const float* restrict G,
                   float* restrict DX, int n) {
    acc_vsigmoid(X, DX, n);      // DX = s
    for (int i = 0; i < n; i++) {
        float s = DX[i];
        DX[i] = G[i] * s * (1.0f + X[i] * (1.0f - s));
    }
}

void acc_vswiglu(const float* restrict gate, const float* restrict val,
                 float* restrict C, int n) {
    acc_vsigmoid(gate, C, n);    // C = s
    for (int i = 0; i < n; i++) {
        C[i] = gate[i] * C[i] * val[i];
    }
}

void acc_vswiglu_bwd(const float* restrict gate, const float* restrict val,
                     const float* restrict G,
                     float* restrict dGate, float* restrict dVal, int n) {
    acc_vsigmoid(gate, dGate, n); // dGate = s (scratch)
    for (int i = 0; i < n; i++) {
        float s = dGate[i];
        float gx = gate[i];
        float gi = G[i];
        dVal[i]  = gi * gx * s;
        dGate[i] = gi * val[i] * s * (1.0f + gx * (1.0f - s));
    }
}
