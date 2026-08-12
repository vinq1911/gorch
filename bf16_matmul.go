//go:build darwin

package gorch

// bf16 matmul dispatch — plan 0009 X3 (B0 probe + B4).
//
// The frozen path of the LoRA workload stores base weights in bf16
// (2.46 GB → 1.23 GB) while activations, trainable params, and every
// gradient stay f32. The matmul family therefore needs to accept a
// bf16 operand next to an f32 one. Per the B0 ADR (doc/decisions.md
// ADR-012), the chosen path is MPSMatrix with MPSDataTypeBFloat16 —
// per-operand dtype descriptors, f32 result matrix, f32 accumulation
// (risk R2 contract, verified by the probe below and the B0 probe
// test).
//
// Dispatch rules:
//   - ≥1 bf16 operand, both Metal-resident, above MatMulMetalThreshold,
//     and the once-per-process numeric probe passed → MPS dtyped path,
//     f32 output.
//   - both operands bf16, both CPU-resident → legacy plan-0002
//     semantics: promote → f32 op → bf16 output (pinned by
//     bf16_ops_test.go; unchanged).
//   - anything else (mixed dtype on CPU, below threshold, probe
//     failed) → widen the bf16 side to f32 and run the ordinary f32
//     path (Accelerate; 0002's convert-at-boundary story). Output f32.
//
// Backward: gradients are always f32 ("master grads"). The grad
// matmuls re-enter the same dtyped dispatch, and — because frozen
// weights have RequiresGrad()==false — the dW GEMM for a frozen bf16
// operand is skipped entirely rather than computed and discarded.

import (
	"fmt"
	"sync"

	"github.com/vinq1911/gorch/accelerate"
)

var (
	bf16ProbeOnce sync.Once
	bf16MatMulOK  bool
)

// MetalBF16MatMulSupported reports whether the MPS bf16 matmul path is
// available AND numerically verified on this device. The first call
// runs a small probe (bf16×bf16 and f32×bf16 against a CPU reference);
// the result is cached for the process. False when Metal isn't
// initialized, the OS predates MPSDataTypeBFloat16 (macOS 14), or the
// probe's numerics were wrong — in which case every bf16 matmul takes
// the widen-to-f32 fallback (B0 outcome would be recorded as tier b/c;
// see ADR-012).
func MetalBF16MatMulSupported() bool {
	if gpu == nil {
		return false
	}
	bf16ProbeOnce.Do(runBF16MatMulProbe)
	return bf16MatMulOK
}

// runBF16MatMulProbe verifies the dtyped MPS path numerically: MPS can
// silently produce garbage instead of erroring on an unsupported
// dtype, so a nil error from the shim is not proof of support.
func runBF16MatMulProbe() {
	const n = 16
	af := make([]float32, n*n)
	bff := make([]float32, n*n)
	for i := range af {
		af[i] = float32(i%7)*0.25 - 0.75
		bff[i] = float32((i*5)%11)*0.125 - 0.625
	}
	a := NewTensorBF16OnMetal(gpu.Dev, F32ToBF16Slice(af), n, n)
	b := NewTensorBF16OnMetal(gpu.Dev, F32ToBF16Slice(bff), n, n)
	bf32 := NewTensorOnMetal(gpu.Dev, bff, n, n)
	c := ZerosOnMetal(gpu.Dev, n, n)
	defer func() {
		a.buf.Release()
		b.buf.Release()
		bf32.buf.Release()
		c.buf.Release()
	}()

	check := func(refA, refB []float32) bool {
		syncForCPU(c)
		ref := make([]float32, n*n)
		accelerate.Sgemm(n, n, n, 1.0, refA, refB, 0.0, ref)
		for i := range ref {
			d := ref[i] - c.data[i]
			if d > 1e-3 || d < -1e-3 {
				return false
			}
		}
		return true
	}
	aw := BF16ToF32Slice(a.data16)
	bw := BF16ToF32Slice(b.data16)

	// bf16 × bf16 → f32.
	if gpu.Queue.MatMulDT(a.buf, b.buf, c.buf, n, n, n, false, false, true, true) != nil {
		return
	}
	if !check(aw, bw) {
		return
	}
	// mixed bf16 × f32 → f32 (the frozen-weight × f32-activation shape).
	if gpu.Queue.MatMulDT(a.buf, bf32.buf, c.buf, n, n, n, false, false, true, false) != nil {
		return
	}
	if !check(aw, bff) {
		return
	}
	bf16MatMulOK = true
}

// bf16MatMulEligible gates the B4 MPS path: both operands
// Metal-resident, logical FMA count above the threshold, probe green.
func bf16MatMulEligible(a, b *Tensor, M, N, K int) bool {
	return gpu != nil && a.buf != nil && b.buf != nil &&
		shouldUseMetalMatMul(M, N, K) && MetalBF16MatMulSupported()
}

// dispatchMatMulDT encodes out = opA(a) @ opB(b) on MPS with each
// operand's own dtype; out must be f32 Metal-resident. Returns false
// on shim failure so callers can fall back to widening.
func dispatchMatMulDT(a, b, out *Tensor, M, N, K int, transA, transB bool) bool {
	err := gpu.Queue.MatMulDT(a.buf, b.buf, out.buf, M, N, K, transA, transB,
		a.dtype == BFloat16, b.dtype == BFloat16)
	if err != nil {
		return false
	}
	mpsBF16MatMulDispatches.Add(1)
	return true
}

func dispatchBatchedMatMulDT(a, b, out *Tensor, M, N, K, batchSize int, transA, transB bool) bool {
	err := gpu.Queue.BatchedMatMulDT(a.buf, b.buf, out.buf, M, N, K, batchSize, transA, transB,
		a.dtype == BFloat16, b.dtype == BFloat16)
	if err != nil {
		return false
	}
	mpsBF16MatMulDispatches.Add(1)
	return true
}

// widenF32 returns t unchanged if f32, else a fresh non-tracking f32
// copy of the bf16 values (no autograd node — used inside backward
// closures and no-autograd ops where the graph must not grow).
func widenF32(t *Tensor) *Tensor {
	if t.dtype != BFloat16 {
		return t
	}
	return &Tensor{dtype: Float32, data: BF16ToF32Slice(t.data16), shape: copyShape(t.shape)}
}

// matMulDTGrad computes a single (M, N) matmul with contraction K,
// per-operand dtype, and NO autograd. Stored shapes: x is (K, M) when
// transA else (M, K); y is (N, K) when transB else (K, N). GPU dtyped
// path when eligible, widen + Accelerate otherwise. Output f32,
// residency inherited from the inputs.
func matMulDTGrad(x, y *Tensor, M, N, K int, transA, transB bool) *Tensor {
	if (x.dtype == BFloat16 || y.dtype == BFloat16) && bf16MatMulEligible(x, y, M, N, K) {
		out := ZerosOnMetal(gpu.Dev, M, N)
		if dispatchMatMulDT(x, y, out, M, N, K, transA, transB) {
			return out
		}
	}
	xf, yf := widenF32(x), widenF32(y)
	out := zerosLikeEither([]int{M, N}, x, y)
	syncForCPU(xf, yf)
	switch {
	case transA && !transB:
		accelerate.SgemmTransA(M, N, K, 1.0, xf.data, yf.data, 0.0, out.data)
	case !transA && transB:
		accelerate.SgemmTransB(M, N, K, 1.0, xf.data, yf.data, 0.0, out.data)
	case !transA && !transB:
		accelerate.Sgemm(M, N, K, 1.0, xf.data, yf.data, 0.0, out.data)
	default:
		panic("gorch: matMulDTGrad does not support transA && transB")
	}
	return out
}

// batchedMatMulDTGrad is the batched analogue of matMulDTGrad.
// x: (batchSize, ·, ·) stored (K, M) per batch when transA else (M, K);
// y: stored (N, K) per batch when transB else (K, N); out (batchSize, M, N).
func batchedMatMulDTGrad(x, y *Tensor, batchSize, M, N, K int, transA, transB bool) *Tensor {
	if (x.dtype == BFloat16 || y.dtype == BFloat16) && bf16MatMulEligible(x, y, batchSize*M, N, K) {
		out := ZerosOnMetal(gpu.Dev, batchSize, M, N)
		if dispatchBatchedMatMulDT(x, y, out, M, N, K, batchSize, transA, transB) {
			return out
		}
	}
	xf, yf := widenF32(x), widenF32(y)
	out := zerosLikeEither([]int{batchSize, M, N}, x, y)
	syncForCPU(xf, yf)
	xStride, yStride, cStride := M*K, K*N, M*N
	for i := 0; i < batchSize; i++ {
		xs := xf.data[i*xStride : (i+1)*xStride]
		ys := yf.data[i*yStride : (i+1)*yStride]
		cs := out.data[i*cStride : (i+1)*cStride]
		switch {
		case transA && !transB:
			accelerate.SgemmTransA(M, N, K, 1.0, xs, ys, 0.0, cs)
		case !transA && transB:
			accelerate.SgemmTransB(M, N, K, 1.0, xs, ys, 0.0, cs)
		case !transA && !transB:
			accelerate.Sgemm(M, N, K, 1.0, xs, ys, 0.0, cs)
		default:
			panic("gorch: batchedMatMulDTGrad does not support transA && transB")
		}
	}
	return out
}

// matMulBF16 is MatMul's forward when ≥1 operand is bf16, with
// autograd. Grads are f32; the grad GEMM for an operand with
// RequiresGrad()==false is skipped entirely (frozen-weight fast path).
func matMulBF16(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	K2, N := b.shape[0], b.shape[1]
	if K != K2 {
		panic(fmt.Sprintf("gorch: MatMul shape mismatch: (%d,%d) @ (%d,%d)", M, K, K2, N))
	}
	if bf16MatMulEligible(a, b, M, N, K) {
		out := ZerosOnMetal(gpu.Dev, M, N)
		if dispatchMatMulDT(a, b, out, M, N, K, false, false) {
			if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
				out.requiresGrad = true
				out.gradFn = &GradFn{
					name:   "MatMulBF16",
					inputs: []*Tensor{a, b},
					backward: func(grad *Tensor) []*Tensor {
						var dA, dB *Tensor
						if a.requiresGrad {
							dA = matMulDTGrad(grad, b, M, K, N, false, true) // grad @ B^T
						}
						if b.requiresGrad {
							dB = matMulDTGrad(a, grad, K, N, M, true, false) // A^T @ grad
						}
						return []*Tensor{dA, dB}
					},
				}
			}
			return out
		}
	}
	if a.dtype == BFloat16 && b.dtype == BFloat16 && a.buf == nil && b.buf == nil {
		// Legacy plan-0002 semantics: CPU bf16 pair keeps a bf16 output.
		return downcastToBF16(MatMul(promoteToF32(a), promoteToF32(b)))
	}
	// Widen fallback: below threshold, mixed-residency, or MPS bf16
	// unavailable. Autograd flows through the upcast wrappers; f32 out.
	return MatMul(promoteToF32(a), promoteToF32(b))
}

// batchedMatMulBF16 covers BatchedMatMul (transB=false) and
// BatchedMatMulTransB (transB=true) when ≥1 operand is bf16.
func batchedMatMulBF16(a, b *Tensor, batchSize, M, N, K int, transB bool) *Tensor {
	if bf16MatMulEligible(a, b, batchSize*M, N, K) {
		out := ZerosOnMetal(gpu.Dev, batchSize, M, N)
		if dispatchBatchedMatMulDT(a, b, out, M, N, K, batchSize, false, transB) {
			if GradEnabled() && (a.requiresGrad || b.requiresGrad) {
				name := "BatchedMatMulBF16"
				if transB {
					name = "BatchedMatMulTransBBF16"
				}
				out.requiresGrad = true
				out.gradFn = &GradFn{
					name:   name,
					inputs: []*Tensor{a, b},
					backward: func(grad *Tensor) []*Tensor {
						var dA, dB *Tensor
						if transB {
							if a.requiresGrad {
								dA = batchedMatMulDTGrad(grad, b, batchSize, M, K, N, false, false) // grad @ B
							}
							if b.requiresGrad {
								dB = batchedMatMulDTGrad(grad, a, batchSize, N, K, M, true, false) // grad^T @ A
							}
						} else {
							if a.requiresGrad {
								dA = batchedMatMulDTGrad(grad, b, batchSize, M, K, N, false, true) // grad @ B^T
							}
							if b.requiresGrad {
								dB = batchedMatMulDTGrad(a, grad, batchSize, K, N, M, true, false) // A^T @ grad
							}
						}
						return []*Tensor{dA, dB}
					},
				}
			}
			return out
		}
	}
	if a.dtype == BFloat16 && b.dtype == BFloat16 && a.buf == nil && b.buf == nil {
		if transB {
			return downcastToBF16(BatchedMatMulTransB(promoteToF32(a), promoteToF32(b), batchSize, M, N, K))
		}
		return downcastToBF16(BatchedMatMul(promoteToF32(a), promoteToF32(b), batchSize, M, N, K))
	}
	if transB {
		return BatchedMatMulTransB(promoteToF32(a), promoteToF32(b), batchSize, M, N, K)
	}
	return BatchedMatMul(promoteToF32(a), promoteToF32(b), batchSize, M, N, K)
}
