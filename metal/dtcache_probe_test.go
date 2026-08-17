//go:build darwin

package metal

import (
	"testing"
	"time"
)

// TestDTGraphCacheSizeDoesNotSlowExecution exonerates the compiled
// MPSGraph cache, which has twice now been the leading suspect for a
// per-step slowdown and twice been innocent.
//
// The cache is keyed on (M, N, K, batch, transposes, dtypes), and M/K
// carry the training sequence length, so on variable-length data it
// mints ~10 fresh entries per accumulation micro-step and never
// evicts. That growth is real and is what the cap
// (SetDTGraphCacheLimit) bounds. What it is NOT is a source of
// slowdown, and this test is the evidence:
//
//   - a FIXED-shape matmul's wall time must not degrade as the cache
//     fills with hundreds of unrelated shapes (if it did, cache size
//     would be a driver-side tax on every op, and bucketing sequence
//     lengths would be mandatory);
//   - compiling a brand-new shape must stay a roughly CONSTANT cost
//     (if it grew with cache size, the per-micro-step compile bill
//     would ramp).
//
// Measured 2026-08-17 (M4, macOS 26.5): hot-shape time was flat within
// noise from 1 to 600 cached graphs, and a fresh compile was ~10 ms
// throughout — 0.1 s against a ~5 s micro-step. The actual cause of the
// ramp was the footprint guard forking `vmmap --summary` on the
// per-micro-step path; see cmd/qwenvoice-train/footprint.go.
//
// The thresholds are deliberately loose. This is a timing test on a
// shared machine and it exists to catch a REGIME change — "cached graph
// count now costs you" — not to police a few percent.
func TestDTGraphCacheSizeDoesNotSlowExecution(t *testing.T) {
	dev, queue, _ := reclaimFixture(t)
	SetAsync(false)
	wasLimit := DTGraphCacheLen()
	_ = wasLimit
	SetDTGraphCacheLimit(0)
	ClearDTGraphCache()
	t.Cleanup(func() {
		SetDTGraphCacheLimit(0)
		ClearDTGraphCache()
	})

	const hotM, hotN, hotK = 512, 1024, 1024
	const maxM = 1200
	bufA := dev.NewBuffer(maxM * hotK * 4)
	bufB := dev.NewBuffer(hotK * hotN * 4)
	bufC := dev.NewBuffer(maxM * hotN * 4)
	defer bufA.Release()
	defer bufB.Release()
	defer bufC.Release()

	// hot times one already-cached shape, taking the BEST of several
	// batches: the machine is shared, so the minimum is the measurement
	// least polluted by whatever else is running.
	hot := func() time.Duration {
		best := time.Duration(1<<62 - 1)
		for b := 0; b < 3; b++ {
			start := time.Now()
			const reps = 10
			for i := 0; i < reps; i++ {
				if err := queue.MatMulDT(bufA, bufB, bufC, hotM, hotN, hotK, false, false, false, false); err != nil {
					t.Skipf("dtyped matmul unavailable: %v", err)
				}
			}
			SyncQueue()
			if d := time.Since(start) / reps; d < best {
				best = d
			}
		}
		return best
	}

	hot() // warm the hot shape's own graph
	baseline := hot()
	t.Logf("cacheLen=%d hot=%v (baseline)", DTGraphCacheLen(), baseline)

	m := 20
	var firstFill, lastFill, last time.Duration
	for round := 0; round < 12; round++ {
		fillStart := time.Now()
		const fresh = 50
		for i := 0; i < fresh; i++ {
			m++ // a shape never seen before, so always a compile
			if err := queue.MatMulDT(bufA, bufB, bufC, m, hotN, hotK, false, false, false, false); err != nil {
				t.Fatal(err)
			}
		}
		SyncQueue()
		fill := time.Since(fillStart) / fresh
		last = hot()
		if round == 0 {
			firstFill = fill
		}
		lastFill = fill
		t.Logf("cacheLen=%d hot=%v newShapeAvg=%v", DTGraphCacheLen(), last, fill)
	}

	if n := DTGraphCacheLen(); n < 600 {
		t.Fatalf("cache holds %d entries after minting 600 fresh shapes — the probe did "+
			"not actually grow the cache, so it proves nothing", n)
	}
	if last > 5*baseline {
		t.Fatalf("a fixed-shape matmul went from %v to %v as the cache grew to %d entries "+
			"(%.1fx) — cached graph count now taxes execution, which changes the whole "+
			"cost model: sequence-length bucketing stops being optional",
			baseline, last, DTGraphCacheLen(), float64(last)/float64(baseline))
	}
	if lastFill > 5*firstFill {
		t.Fatalf("compiling a fresh shape went from %v to %v as the cache grew (%.1fx) — "+
			"compile cost is supposed to be constant in cache size", firstFill, lastFill,
			float64(lastFill)/float64(firstFill))
	}
}
