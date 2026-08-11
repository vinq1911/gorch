//go:build darwin

// Package accelerate provides Go bindings to Apple's Accelerate framework
// for high-performance CPU vector/matrix operations using BLAS, vDSP, and vForce.
package accelerate

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include "shim.h"
*/
import "C"

import "unsafe"

func ptr(s []float32) *C.float {
	return (*C.float)(unsafe.Pointer(&s[0]))
}

// ---------- BLAS matmul ----------

// Sgemm computes out = alpha * A @ B + beta * out.
// A is MxK, B is KxN, out is MxN. All row-major float32.
func Sgemm(M, N, K int, alpha float32, A, B []float32, beta float32, out []float32) {
	C.acc_sgemm(C.int(M), C.int(N), C.int(K),
		C.float(alpha), ptr(A), ptr(B),
		C.float(beta), ptr(out))
}

// SgemmTransB computes out = alpha * A @ B^T + beta * out.
// A is MxK, B is NxK (row-major), out is MxN.
func SgemmTransB(M, N, K int, alpha float32, A, B []float32, beta float32, out []float32) {
	C.acc_sgemm_transB(C.int(M), C.int(N), C.int(K),
		C.float(alpha), ptr(A), ptr(B),
		C.float(beta), ptr(out))
}

// SgemmTransA computes out = alpha * A^T @ B + beta * out.
// A is KxM (row-major), B is KxN, out is MxN.
func SgemmTransA(M, N, K int, alpha float32, A, B []float32, beta float32, out []float32) {
	C.acc_sgemm_transA(C.int(M), C.int(N), C.int(K),
		C.float(alpha), ptr(A), ptr(B),
		C.float(beta), ptr(out))
}

// ---------- vDSP vector ops ----------

// VAdd computes out = A + B element-wise.
func VAdd(A, B, out []float32) {
	C.acc_vadd(ptr(A), ptr(B), ptr(out), C.int64_t(len(A)))
}

// VSub computes out = A - B element-wise.
func VSub(A, B, out []float32) {
	C.acc_vsub(ptr(A), ptr(B), ptr(out), C.int64_t(len(A)))
}

// VMul computes out = A * B element-wise.
func VMul(A, B, out []float32) {
	C.acc_vmul(ptr(A), ptr(B), ptr(out), C.int64_t(len(A)))
}

// VDiv computes out = A / B element-wise.
func VDiv(A, B, out []float32) {
	C.acc_vdiv(ptr(A), ptr(B), ptr(out), C.int64_t(len(A)))
}

// VScale computes out = A * scalar.
func VScale(A []float32, scalar float32, out []float32) {
	C.acc_vscale(ptr(A), C.float(scalar), ptr(out), C.int64_t(len(A)))
}

// VSAdd computes out = A + scalar element-wise.
func VSAdd(A []float32, scalar float32, out []float32) {
	C.acc_vsadd(ptr(A), C.float(scalar), ptr(out), C.int64_t(len(A)))
}

// Sum returns the sum of all elements.
func Sum(A []float32) float32 {
	var out C.float
	C.acc_sve(ptr(A), &out, C.int64_t(len(A)))
	return float32(out)
}

// Mean returns the mean of all elements.
func Mean(A []float32) float32 {
	var out C.float
	C.acc_meanv(ptr(A), &out, C.int64_t(len(A)))
	return float32(out)
}

// Max returns the maximum element.
func Max(A []float32) float32 {
	var out C.float
	C.acc_maxv(ptr(A), &out, C.int64_t(len(A)))
	return float32(out)
}

// ReLU computes out = max(0, A) element-wise.
func ReLU(A, out []float32) {
	C.acc_vrelu(ptr(A), ptr(out), C.int64_t(len(A)))
}

// ---------- vForce transcendentals ----------

// Exp computes out = exp(A) element-wise.
func Exp(A, out []float32) {
	C.acc_vexp(ptr(A), ptr(out), C.int(len(A)))
}

// ELU computes out = A > 0 ? A : exp(A)-1 element-wise (alpha = 1).
func ELU(A, out []float32) {
	C.acc_velu(ptr(A), ptr(out), C.int(len(A)))
}

// GELUErf computes out = 0.5*A*(1+erf(A/sqrt(2))) element-wise — the
// exact (approximate='none') GELU.
func GELUErf(A, out []float32) {
	C.acc_vgelu_erf(ptr(A), ptr(out), C.int(len(A)))
}

// Log computes out = log(A) element-wise.
func Log(A, out []float32) {
	C.acc_vlog(ptr(A), ptr(out), C.int(len(A)))
}

// Tanh computes out = tanh(A) element-wise.
func Tanh(A, out []float32) {
	C.acc_vtanh(ptr(A), ptr(out), C.int(len(A)))
}

// Sigmoid computes out = 1/(1+exp(-A)) element-wise.
func Sigmoid(A, out []float32) {
	C.acc_vsigmoid(ptr(A), ptr(out), C.int(len(A)))
}

// ---------- SiLU / SwiGLU (plan 0009 K4) ----------

// SiLU computes out = A*sigmoid(A) element-wise. out must not alias A.
func SiLU(A, out []float32) {
	C.acc_vsilu(ptr(A), ptr(out), C.int(len(A)))
}

// SiLUBwd computes dx = G*s*(1+X*(1-s)) with s = sigmoid(X).
// dx must not alias X or G.
func SiLUBwd(X, G, dx []float32) {
	C.acc_vsilu_bwd(ptr(X), ptr(G), ptr(dx), C.int(len(X)))
}

// SwiGLU computes out = gate*sigmoid(gate)*val element-wise.
// out must not alias gate or val.
func SwiGLU(gate, val, out []float32) {
	C.acc_vswiglu(ptr(gate), ptr(val), ptr(out), C.int(len(gate)))
}

// SwiGLUBwd computes, with s = sigmoid(gate):
//
//	dGate = G*val*s*(1+gate*(1-s))
//	dVal  = G*gate*s
//
// dGate/dVal must not alias the inputs or each other.
func SwiGLUBwd(gate, val, G, dGate, dVal []float32) {
	C.acc_vswiglu_bwd(ptr(gate), ptr(val), ptr(G), ptr(dGate), ptr(dVal), C.int(len(gate)))
}

// ---------- fused optimizer step (plan 0009 K7) ----------

// AdamWStep applies one AdamW update over n contiguous f32 elements,
// in-place on p, m, v. bc1/bc2 are the bias-correction denominators
// (1 − βᵗ). Exact same math as the scalar reference in optim/adamw.go.
func AdamWStep(p, g, m, v []float32, lr, beta1, beta2, eps, wd, bc1, bc2 float32) {
	if len(p) == 0 {
		return
	}
	C.acc_adamw_step(ptr(p), ptr(g), ptr(m), ptr(v), C.int64_t(len(p)),
		C.float(lr), C.float(beta1), C.float(beta2), C.float(eps),
		C.float(wd), C.float(bc1), C.float(bc2))
}
