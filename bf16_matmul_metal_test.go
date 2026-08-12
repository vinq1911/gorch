//go:build darwin

package gorch

// Plan 0009 X3 tests: the B0 MPS bf16 probe (1024³, f32-accumulation
// verification — the evidence behind ADR-012) and the B4 dispatch
// parity suite at the plan's tolerances (5e-2 fwd / 8e-2 grad rel,
// the bf16_ops_test.go precedent) plus the risk-R2 attention-logits
// check at seq 1500.

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/vinq1911/gorch/accelerate"
)

// bf16RelClose checks |got-want| <= tol*scale where scale is a
// magnitude floor (RMS of the reference) — attention logits and matmul
// outputs cross zero, so raw elementwise rel error is ill-conditioned.
func bf16CheckRel(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch %d vs %d", label, len(got), len(want))
	}
	var ss float64
	for _, v := range want {
		ss += float64(v) * float64(v)
	}
	rms := math.Sqrt(ss / float64(len(want)))
	if rms < 1e-6 {
		rms = 1e-6
	}
	worst := 0.0
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if r := d / rms; r > worst {
			worst = r
		}
		if d > tol*rms {
			t.Fatalf("%s[%d]: got=%g want=%g |diff|=%g > tol %g × rms %g",
				label, i, got[i], want[i], d, tol, rms)
		}
	}
	t.Logf("%s: worst |diff|/rms = %.2e (tol %.0e)", label, worst, tol)
}

// TestB0ProbeMPSBF16Matmul1024 is the plan-0009 X3-B0 gate probe:
// multiply 1024³ with bf16 operands through the shim's dtyped entry
// point and verify BOTH that the result matches an f32-accumulation
// reference of the same bf16-rounded inputs AND that the error is far
// below what bf16 accumulation would produce (~2^-8·√K ≈ 6e-2 rel at
// K=1024; we demand 1e-3 of the f64 row reference).
//
// B0 outcome (recorded in ADR-012): tier (a) — MPSMatrix with
// MPSDataTypeBFloat16 — hard-asserts and abort()s on macOS 26.5
// ("Input data type must be one of MPSDataTypeFloat32,
// MPSDataTypeFloat16, MPSDataTypeInt8, or MPSDataTypeInt16"), so the
// shim implements tier (b): MPSGraph matmul with bf16 placeholders
// cast to f32 inside the graph (f32 accumulation by construction).
// This test pins the tier-(b) numerics and logs its 1024³ timing vs
// the f32 MPSMatrix path.
func TestB0ProbeMPSBF16Matmul1024(t *testing.T) {
	g, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	const n = 1024
	rng := rand.New(rand.NewSource(9))
	af := make([]float32, n*n)
	bf := make([]float32, n*n)
	for i := range af {
		af[i] = float32(rng.NormFloat64())
		bf[i] = float32(rng.NormFloat64())
	}
	a16 := F32ToBF16Slice(af)
	b16 := F32ToBF16Slice(bf)

	a := NewTensorBF16OnMetal(g.Dev, a16, n, n)
	b := NewTensorBF16OnMetal(g.Dev, b16, n, n)
	c := ZerosOnMetal(g.Dev, n, n)

	if err := g.Queue.MatMulDT(a.MetalBuffer(), b.MetalBuffer(), c.MetalBuffer(),
		n, n, n, false, false, true, true); err != nil {
		t.Fatalf("B0 probe: dtyped bf16 matmul path unavailable (tier-b MPSGraph failed too): %v", err)
	}
	SyncMetal()

	// Reference 1: full C against Accelerate f32 sgemm over the SAME
	// bf16-rounded values (both sides accumulate in f32; ordering noise
	// only).
	aw := BF16ToF32Slice(a16)
	bw := BF16ToF32Slice(b16)
	ref := make([]float32, n*n)
	accelerate.Sgemm(n, n, n, 1.0, aw, bw, 0.0, ref)
	bf16CheckRel(t, "B0 probe vs f32-accum reference", c.Data(), ref, 1e-3)

	// Reference 2: 8 rows in float64 — the accumulation-precision
	// check. bf16 accumulation at K=1024 would sit around 1e-2..1e-1
	// of RMS; f32 accumulation sits near 1e-6. Demand 1e-3.
	rows := []int{0, 1, 17, 256, 511, 700, 1000, 1023}
	got := make([]float32, 0, len(rows)*n)
	want := make([]float32, 0, len(rows)*n)
	cData := c.Data()
	for _, r := range rows {
		for j := 0; j < n; j++ {
			var acc float64
			for k := 0; k < n; k++ {
				acc += float64(aw[r*n+k]) * float64(bw[k*n+j])
			}
			want = append(want, float32(acc))
			got = append(got, cData[r*n+j])
		}
	}
	bf16CheckRel(t, "B0 probe vs f64 rows (f32-accumulation gate)", got, want, 1e-3)

	// Timing note for the ADR: bf16 vs f32 MPS at 1024³.
	af32 := NewTensorOnMetal(g.Dev, aw, n, n)
	bf32 := NewTensorOnMetal(g.Dev, bw, n, n)
	cf32 := ZerosOnMetal(g.Dev, n, n)
	time.Sleep(50 * time.Millisecond)
	const iters = 20
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		g.Queue.MatMul(af32.MetalBuffer(), bf32.MetalBuffer(), cf32.MetalBuffer(), n, n, n)
	}
	SyncMetal()
	f32Ms := float64(time.Since(t0).Microseconds()) / 1000 / iters
	t1 := time.Now()
	for i := 0; i < iters; i++ {
		if err := g.Queue.MatMulDT(a.MetalBuffer(), b.MetalBuffer(), c.MetalBuffer(),
			n, n, n, false, false, true, true); err != nil {
			t.Fatalf("bf16 timing dispatch failed: %v", err)
		}
	}
	SyncMetal()
	bf16Ms := float64(time.Since(t1).Microseconds()) / 1000 / iters
	t.Logf("B0 probe timing 1024^3 (median-free, %d iters): f32 %.3f ms, bf16 %.3f ms (%.2fx)",
		iters, f32Ms, bf16Ms, f32Ms/bf16Ms)

	if !MetalBF16MatMulSupported() {
		t.Fatal("B0 probe: runtime support probe (MetalBF16MatMulSupported) disagrees with the direct probe")
	}
}

// TestBF16MatMulParityGPU: B4 forward + gradient parity at the plan's
// tolerances (5e-2 fwd / 8e-2 grad rel) for the frozen-path shapes:
// mixed f32 × bf16 through MatMul and MatMulTransB, on the MPS dtyped
// path (threshold lowered to 0).
func TestBF16MatMulParityGPU(t *testing.T) {
	_, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !MetalBF16MatMulSupported() {
		t.Skip("MPS bf16 matmul unsupported (B0 tier-a failed) — widen fallback covered by CPU tests")
	}
	defer setMatMulMetalThresholdForTest(t, 0)()

	const M, K, N = 33, 128, 65
	rng := rand.New(rand.NewSource(7))
	xf := make([]float32, M*K)
	wf := make([]float32, N*K) // (out, in) layout for TransB
	for i := range xf {
		xf[i] = float32(rng.NormFloat64())
	}
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.1
	}

	// --- f32 reference: y = x @ W^T, loss = Sum(y), dx ---
	xF := NewTensor(xf, M, K).SetRequiresGrad(true)
	wT := make([]float32, K*N)
	for o := 0; o < N; o++ {
		for i := 0; i < K; i++ {
			wT[i*N+o] = wf[o*K+i]
		}
	}
	wFT := NewTensor(wT, K, N)
	yF := MatMul(xF, wFT)
	Sum(yF).Backward()

	// --- bf16 path: W bf16 Metal-resident frozen, x f32 resident ---
	dev := MetalDev()
	xB := NewTensorOnMetal(dev, xf, M, K).SetRequiresGrad(true)
	wB16 := NewTensorBF16OnMetal(dev, F32ToBF16Slice(wT), K, N)
	c0 := ReadMetalDispatchCounts()
	yB := MatMul(xB, wB16)
	if yB.Dtype() != Float32 {
		t.Fatalf("bf16 matmul output dtype = %v, want F32", yB.Dtype())
	}
	Sum(yB).Backward()
	c1 := ReadMetalDispatchCounts()
	if c1.BF16MatMul-c0.BF16MatMul < 2 {
		t.Fatalf("expected >=2 bf16 MPS dispatches (fwd + dx), got %d", c1.BF16MatMul-c0.BF16MatMul)
	}

	bf16CheckRel(t, "MatMul(f32, bf16W) forward", yB.Data(), yF.Data(), 5e-2)
	bf16CheckRel(t, "MatMul(f32, bf16W) dx", xB.Grad().Data(), xF.Grad().Data(), 8e-2)

	// --- MatMulTransB with bf16 W (the nn.Linear forward shape) ---
	wB16tb := NewTensorBF16OnMetal(dev, F32ToBF16Slice(wf), N, K)
	yTB := MatMulTransB(xB, wB16tb)
	refTB := make([]float32, M*N)
	accelerate.SgemmTransB(M, N, K, 1.0, xf, wf, 0.0, refTB)
	bf16CheckRel(t, "MatMulTransB(f32, bf16W) forward", yTB.Data(), refTB, 5e-2)
}

// TestBF16BatchedMatMulParityGPU: B4 batched variants with bf16
// operands, fwd + grad parity.
func TestBF16BatchedMatMulParityGPU(t *testing.T) {
	_, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !MetalBF16MatMulSupported() {
		t.Skip("MPS bf16 matmul unsupported")
	}
	defer setMatMulMetalThresholdForTest(t, 0)()

	const B, M, K, N = 4, 17, 64, 23
	rng := rand.New(rand.NewSource(11))
	af := make([]float32, B*M*K)
	bf := make([]float32, B*K*N)
	for i := range af {
		af[i] = float32(rng.NormFloat64())
	}
	for i := range bf {
		bf[i] = float32(rng.NormFloat64()) * 0.2
	}

	// f32 reference (CPU): both operands require grad.
	aF := NewTensor(af, B, M, K).SetRequiresGrad(true)
	bF := NewTensor(bf, B, K, N).SetRequiresGrad(true)
	yF := BatchedMatMul(aF, bF, B, M, N, K)
	Sum(yF).Backward()

	// bf16 path: a bf16 resident (activations-bf16 shape), b f32 resident.
	dev := MetalDev()
	aB := NewTensorBF16OnMetal(dev, F32ToBF16Slice(af), B, M, K).SetRequiresGrad(true)
	bB := NewTensorOnMetal(dev, bf, B, K, N).SetRequiresGrad(true)
	yB := BatchedMatMul(aB, bB, B, M, N, K)
	if yB.Dtype() != Float32 {
		t.Fatalf("bf16 batched matmul output dtype = %v, want F32", yB.Dtype())
	}
	Sum(yB).Backward()

	bf16CheckRel(t, "BatchedMatMul(bf16, f32) forward", yB.Data(), yF.Data(), 5e-2)
	bf16CheckRel(t, "BatchedMatMul(bf16, f32) dB", bB.Grad().Data(), bF.Grad().Data(), 8e-2)
	bf16CheckRel(t, "BatchedMatMul(bf16, f32) dA (f32 master grad)",
		aB.Grad().Data(), aF.Grad().Data(), 8e-2)

	// TransB variant.
	b2f := make([]float32, B*N*K)
	for i := range b2f {
		b2f[i] = float32(rng.NormFloat64()) * 0.2
	}
	a2F := NewTensor(af, B, M, K).SetRequiresGrad(true)
	b2F := NewTensor(b2f, B, N, K).SetRequiresGrad(true)
	y2F := BatchedMatMulTransB(a2F, b2F, B, M, N, K)
	Sum(y2F).Backward()

	a2B := NewTensorBF16OnMetal(dev, F32ToBF16Slice(af), B, M, K).SetRequiresGrad(true)
	b2B := NewTensorBF16OnMetal(dev, F32ToBF16Slice(b2f), B, N, K).SetRequiresGrad(true)
	y2B := BatchedMatMulTransB(a2B, b2B, B, M, N, K)
	Sum(y2B).Backward()

	bf16CheckRel(t, "BatchedMatMulTransB(bf16, bf16) forward", y2B.Data(), y2F.Data(), 5e-2)
	bf16CheckRel(t, "BatchedMatMulTransB(bf16, bf16) dA", a2B.Grad().Data(), a2F.Grad().Data(), 8e-2)
	bf16CheckRel(t, "BatchedMatMulTransB(bf16, bf16) dB", b2B.Grad().Data(), b2F.Grad().Data(), 8e-2)
}

// TestBF16AttentionLogitsSeq1500 is the risk-R2 gate: QK^T at
// head_dim 128, seq 1500, 16 heads with bf16 Q/K storage. Two checks:
// (1) logits within 5e-2·rms of the f32-input reference (bf16 input
// rounding), and (2) 4 sampled heads' rows within 1e-3·rms of an f64
// reference over the SAME bf16-rounded inputs — proving the MPS path
// accumulates in f32, not bf16 (bf16 accumulation at K=128 would show
// ~1e-2 of rms).
func TestBF16AttentionLogitsSeq1500(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	_, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !MetalBF16MatMulSupported() {
		t.Skip("MPS bf16 matmul unsupported")
	}
	const heads, seq, hd = 16, 1500, 128
	rng := rand.New(rand.NewSource(15))
	qf := make([]float32, heads*seq*hd)
	kf := make([]float32, heads*seq*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64())
		kf[i] = float32(rng.NormFloat64())
	}
	dev := MetalDev()
	q16 := F32ToBF16Slice(qf)
	k16 := F32ToBF16Slice(kf)
	qB := NewTensorBF16OnMetal(dev, q16, heads, seq, hd)
	kB := NewTensorBF16OnMetal(dev, k16, heads, seq, hd)

	scores := BatchedMatMulTransB(qB, kB, heads, seq, seq, hd) // 4.6G FMA — above threshold
	if !scores.IsOnMetal() || scores.Dtype() != Float32 {
		t.Fatalf("scores: onMetal=%v dtype=%v, want Metal f32 (MPS bf16 path)", scores.IsOnMetal(), scores.Dtype())
	}
	sData := scores.Data()

	// Check 1: vs f32-original-input reference (per-head, CPU Accelerate).
	ref := make([]float32, seq*seq)
	for _, h := range []int{0, 7, 15} {
		accelerate.SgemmTransB(seq, seq, hd, 1.0,
			qf[h*seq*hd:(h+1)*seq*hd], kf[h*seq*hd:(h+1)*seq*hd], 0.0, ref)
		bf16CheckRel(t, "attention logits vs f32 inputs (head)", sData[h*seq*seq:(h+1)*seq*seq], ref, 5e-2)
	}

	// Check 2: f64 accumulation gate over bf16-rounded inputs, 4 sampled rows.
	qw := BF16ToF32Slice(q16)
	kw := BF16ToF32Slice(k16)
	type rc struct{ h, r int }
	var got, want []float32
	for _, s := range []rc{{0, 0}, {3, 750}, {9, 1}, {15, 1499}} {
		for j := 0; j < seq; j++ {
			var acc float64
			for d := 0; d < hd; d++ {
				acc += float64(qw[(s.h*seq+s.r)*hd+d]) * float64(kw[(s.h*seq+j)*hd+d])
			}
			want = append(want, float32(acc))
			got = append(got, sData[(s.h*seq+s.r)*seq+j])
		}
	}
	bf16CheckRel(t, "attention logits f32-accumulation gate (f64 rows)", got, want, 1e-3)
}

// TestBF16MatMulWidenFallbackCPU: below threshold / non-resident bf16
// operands take the widen-to-f32 + Accelerate path with f32 output
// (B4's fallback contract), including autograd through the upcast.
func TestBF16MatMulWidenFallbackCPU(t *testing.T) {
	const M, K, N = 5, 16, 7
	rng := rand.New(rand.NewSource(3))
	xf := make([]float32, M*K)
	wf := make([]float32, K*N)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64())
	}
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	// Mixed dtype on CPU (previously a requireSameDtype panic): f32 x, bf16 W.
	x := NewTensor(xf, M, K).SetRequiresGrad(true)
	w := NewTensorBF16(F32ToBF16Slice(wf), K, N)
	y := MatMul(x, w)
	if y.Dtype() != Float32 {
		t.Fatalf("widen-fallback output dtype = %v, want F32", y.Dtype())
	}
	Sum(y).Backward()
	if x.Grad() == nil {
		t.Fatal("no dx through the widen fallback")
	}

	xF := NewTensor(xf, M, K).SetRequiresGrad(true)
	wF := NewTensor(BF16ToF32Slice(F32ToBF16Slice(wf)), K, N)
	yF := MatMul(xF, wF)
	Sum(yF).Backward()
	bf16CheckRel(t, "widen fallback forward", y.Data(), yF.Data(), 1e-4)
	bf16CheckRel(t, "widen fallback dx", x.Grad().Data(), xF.Grad().Data(), 1e-4)

	// Both-bf16 CPU pair keeps the legacy plan-0002 bf16 output.
	a := NewTensorBF16(F32ToBF16Slice(xf), M, K)
	b := NewTensorBF16(F32ToBF16Slice(wf), K, N)
	if got := MatMul(a, b).Dtype(); got != BFloat16 {
		t.Fatalf("legacy CPU bf16-pair MatMul dtype = %v, want BF16", got)
	}
}
