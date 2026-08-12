//go:build darwin

package gorch

// Diagnostics for the X3 dtyped-matmul path (plan 0009 X3-B4): time
// the MPSGraph bf16 matmul vs the f32 MPSMatrix path at the block's
// actual projection shapes (TransB forward, plain backward) and under
// graph-signature alternation, sync dispatch. Recorded 2026-08-11 (M4):
// square 1024^3 bf16 1.33x faster; block shapes 0.75-1.06x; 4-shape
// round-robin 1.06x — i.e. steady-state dtyped throughput is at or
// above f32 parity, so in-block bf16 slowdowns are environmental
// (MPSGraph workspace allocation under buffer churn / external load),
// not kernel throughput. See the X3 phase notes in
// doc/training_accel_results.json.

import (
	"math/rand"
	"testing"
	"time"
)

func TestBF16ShapeBenchDiag(t *testing.T) {
	g, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	rng := rand.New(rand.NewSource(1))
	mk := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	type shape struct {
		name    string
		M, N, K int
		transB  bool
	}
	shapes := []shape{
		{"wq_fwd  (1024,1024)@(2048,1024)^T", 1024, 2048, 1024, true},
		{"wq_dx   (1024,2048)@(2048,1024)", 1024, 1024, 2048, false},
		{"gate_fwd(1024,1024)@(3072,1024)^T", 1024, 3072, 1024, true},
		{"sq_1024 (1024,1024)@(1024,1024)", 1024, 1024, 1024, false},
	}
	const iters = 10
	for _, s := range shapes {
		// stored shapes
		aRows, aCols := s.M, s.K
		bRows, bCols := s.K, s.N
		if s.transB {
			bRows, bCols = s.N, s.K
		}
		af := mk(aRows * aCols)
		bf := mk(bRows * bCols)
		aF := NewTensorOnMetal(g.Dev, af, aRows, aCols)
		bF := NewTensorOnMetal(g.Dev, bf, bRows, bCols)
		bB := NewTensorBF16OnMetal(g.Dev, F32ToBF16Slice(bf), bRows, bCols)
		c := ZerosOnMetal(g.Dev, s.M, s.N)

		runF32 := func() {
			if s.transB {
				g.Queue.MatMulTransB(aF.buf, bF.buf, c.buf, s.M, s.N, s.K)
			} else {
				g.Queue.MatMul(aF.buf, bF.buf, c.buf, s.M, s.N, s.K)
			}
		}
		runBF16 := func() {
			if err := g.Queue.MatMulDT(aF.buf, bB.buf, c.buf, s.M, s.N, s.K, false, s.transB, false, true); err != nil {
				t.Fatal(err)
			}
		}
		// warm
		for i := 0; i < 3; i++ {
			runF32()
			runBF16()
		}
		SyncMetal()
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			runF32()
		}
		SyncMetal()
		f32Ms := float64(time.Since(t0).Microseconds()) / 1000 / iters
		t1 := time.Now()
		for i := 0; i < iters; i++ {
			runBF16()
		}
		SyncMetal()
		bfMs := float64(time.Since(t1).Microseconds()) / 1000 / iters
		t.Logf("%s: f32 %.3f ms, bf16-dtyped %.3f ms (%.2fx)", s.name, f32Ms, bfMs, f32Ms/bfMs)
	}
}

// TestBF16AlternationDiag: same shapes but ROUND-ROBIN across the
// graph signatures each iteration (the block's dispatch pattern), f32
// MPSMatrix vs bf16 MPSGraph.
func TestBF16AlternationDiag(t *testing.T) {
	g, err := InitMetal()
	if err != nil {
		t.Skipf("metal not available: %v", err)
	}
	rng := rand.New(rand.NewSource(2))
	mk := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	type sh struct{ M, N, K int }
	shapes := []sh{{1024, 2048, 1024}, {1024, 1024, 1024}, {1024, 3072, 1024}, {1024, 1024, 3072}}
	var aF, bF, bB, cc []*Tensor
	for _, s := range shapes {
		aF = append(aF, NewTensorOnMetal(g.Dev, mk(s.M*s.K), s.M, s.K))
		bfl := mk(s.N * s.K)
		bF = append(bF, NewTensorOnMetal(g.Dev, bfl, s.N, s.K))
		bB = append(bB, NewTensorBF16OnMetal(g.Dev, F32ToBF16Slice(bfl), s.N, s.K))
		cc = append(cc, ZerosOnMetal(g.Dev, s.M, s.N))
	}
	const iters = 10
	// warm both
	for i, s := range shapes {
		g.Queue.MatMulTransB(aF[i].buf, bF[i].buf, cc[i].buf, s.M, s.N, s.K)
		if err := g.Queue.MatMulDT(aF[i].buf, bB[i].buf, cc[i].buf, s.M, s.N, s.K, false, true, false, true); err != nil {
			t.Fatal(err)
		}
	}
	SyncMetal()
	t0 := time.Now()
	for it := 0; it < iters; it++ {
		for i, s := range shapes {
			g.Queue.MatMulTransB(aF[i].buf, bF[i].buf, cc[i].buf, s.M, s.N, s.K)
		}
	}
	SyncMetal()
	f32Ms := float64(time.Since(t0).Microseconds()) / 1000 / iters
	t1 := time.Now()
	for it := 0; it < iters; it++ {
		for i, s := range shapes {
			if err := g.Queue.MatMulDT(aF[i].buf, bB[i].buf, cc[i].buf, s.M, s.N, s.K, false, true, false, true); err != nil {
				t.Fatal(err)
			}
		}
	}
	SyncMetal()
	bfMs := float64(time.Since(t1).Microseconds()) / 1000 / iters
	t.Logf("round-robin 4 shapes: f32 %.3f ms/iter, bf16-dtyped %.3f ms/iter (%.2fx)", f32Ms, bfMs, f32Ms/bfMs)
}
