//go:build darwin

package metal

import "testing"

// Regression tests for the 2026-08-13 shared-buffer reclaim bug.
//
// Releasing an MTLResourceStorageModeShared buffer does not return the
// physical pages the CPU has faulted in through its contents mapping
// (a read is as bad as a write), and the driver does not recycle them
// for later same-size allocations. A training micro-step allocates
// ~1400 Metal-backed tensors and touches essentially all of them from
// the CPU, so the process footprint grew by the micro-step's whole
// ALLOCATION volume (~1.7 GB) every micro-step while the library's own
// liveBufferBytes accounting — and the Go heap — stayed perfectly
// flat. Release now discards a buffer's pages before destroying it.
//
// The discard ignores the retain count, so purging a buffer that
// in-flight GPU work still reads would silently corrupt results —
// wrong numbers, not a crash. That makes the GATE, not the purge, the
// invariant worth pinning:
//
//  1. a release with GPU work in flight must NOT purge (safety)
//  2. a release with the queue drained MUST purge (the fix works)
//  3. purging is on by default (the fix is actually wired up)
//  4. buffers still holding live data are unaffected (numerics)
//
// These are exact counter assertions, not memory-growth measurements:
// the counters are incremented on the two branches of the gate itself,
// so they observe the decision directly rather than inferring it from
// a footprint sample.

const reclaimKernel = `
#include <metal_stdlib>
using namespace metal;
kernel void scale_two(device const float* A [[buffer(0)]],
                      device float*       B [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    B[id] = A[id] * 2.0f;
}
`

// reclaimFixture builds a device, queue and pipeline, restoring the
// async mode and purge setting the process started with.
func reclaimFixture(t *testing.T) (*Device, *CommandQueue, *Pipeline) {
	t.Helper()
	dev, err := NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	queue := dev.NewCommandQueue()
	pipe, err := dev.CompileKernel(reclaimKernel, "scale_two")
	if err != nil {
		t.Fatal(err)
	}
	wasAsync := AsyncEnabled()
	t.Cleanup(func() {
		SyncQueue()
		SetAsync(wasAsync)
		SetPurgeOnRelease(true)
		pipe.Release()
		queue.Release()
		dev.Release()
	})
	return dev, queue, pipe
}

// TestReleaseWithWorkInFlightDoesNotPurge is the SAFETY invariant.
// setPurgeableState:Empty discards a buffer's contents immediately,
// regardless of who still references it — so a release that happens
// while the GPU may still be reading must leave the pages alone. If
// this regresses, training does not crash: it silently computes on
// discarded memory.
func TestReleaseWithWorkInFlightDoesNotPurge(t *testing.T) {
	dev, queue, pipe := reclaimFixture(t)
	SetPurgeOnRelease(true)
	SetAsync(true)

	const n = 4096
	in, out := dev.NewBuffer(n*4), dev.NewBuffer(n*4)
	defer in.Release()
	defer out.Release()

	// Commit work and deliberately do NOT sync: the queue is now
	// possibly-in-flight for as long as we refuse to wait on it.
	queue.Dispatch1D(pipe, []*Buffer{in, out}, n)

	victim := dev.NewBuffer(n * 4)
	beforeP, beforeU := PurgeStats()
	victim.Release()
	afterP, afterU := PurgeStats()

	if afterP != beforeP {
		t.Errorf("release purged %d buffer(s) while GPU work was in flight — "+
			"the pages of a buffer the GPU may still be reading were discarded; "+
			"this corrupts results silently rather than crashing",
			afterP-beforeP)
	}
	if afterU != beforeU+1 {
		t.Errorf("unpurged-release count went %d -> %d, want +1 — the gate did not "+
			"take the in-flight branch, so this test is not observing what it claims",
			beforeU, afterU)
	}
}

// TestReleaseAfterSyncPurges is the FIX invariant: once the queue is
// drained, releasing a buffer must discard its pages. Without this the
// footprint grows by the cumulative allocation volume.
func TestReleaseAfterSyncPurges(t *testing.T) {
	dev, queue, pipe := reclaimFixture(t)
	SetPurgeOnRelease(true)
	SetAsync(true)

	const n = 4096
	in, out := dev.NewBuffer(n*4), dev.NewBuffer(n*4)
	defer in.Release()
	defer out.Release()

	queue.Dispatch1D(pipe, []*Buffer{in, out}, n)
	SyncQueue() // drained — purging is now safe

	victim := dev.NewBuffer(n * 4)
	// Dirty every page from the CPU: these are exactly the pages that
	// release fails to reclaim without the purge.
	s := victim.FloatSlice()
	for i := range s {
		s[i] = 1
	}

	beforeP, beforeU := PurgeStats()
	victim.Release()
	afterP, afterU := PurgeStats()

	if afterP != beforeP+1 {
		t.Errorf("purged-release count went %d -> %d, want +1 — a release with the "+
			"queue drained did not discard its pages, so CPU-dirtied Metal memory "+
			"is never reclaimed and the footprint grows without bound",
			beforeP, afterP)
	}
	if afterU != beforeU {
		t.Errorf("release took the in-flight branch %d time(s) after SyncQueue — "+
			"the drain is not clearing the in-flight flag", afterU-beforeU)
	}
}

// TestPurgeOnByDefault guards the wiring: the shim defaults the purge
// ON, and nothing in normal startup turns it off. A fix that is
// present but disabled is the failure mode this catches.
func TestPurgeOnByDefault(t *testing.T) {
	dev, _, _ := reclaimFixture(t)
	SyncQueue()

	// Do not call SetPurgeOnRelease first — observe the default.
	b := dev.NewBuffer(4096)
	beforeP, _ := PurgeStats()
	b.Release()
	afterP, _ := PurgeStats()

	if afterP == beforeP {
		t.Error("a release with the queue drained did not purge — page discard is " +
			"off by default, so the shared-buffer reclaim fix is inert")
	}
}

// TestPurgeDoesNotDisturbLiveBuffers is the NUMERICS guard. Purging is
// per-buffer, so releasing one buffer must not touch another's
// contents — including a neighbour allocated immediately before it,
// which is the case most likely to share a driver region.
func TestPurgeDoesNotDisturbLiveBuffers(t *testing.T) {
	dev, queue, pipe := reclaimFixture(t)
	SetPurgeOnRelease(true)
	SetAsync(false) // every dispatch completes before it returns

	const n = 4096
	keepIn, keepOut := dev.NewBuffer(n*4), dev.NewBuffer(n*4)
	defer keepIn.Release()
	defer keepOut.Release()

	src := keepIn.FloatSlice()
	for i := range src {
		src[i] = float32(i)
	}

	// Churn released buffers around the live ones, then recompute.
	for i := 0; i < 32; i++ {
		scratch := dev.NewBuffer(n * 4)
		s := scratch.FloatSlice()
		for j := range s {
			s[j] = -1
		}
		scratch.Release()
	}

	queue.Dispatch1D(pipe, []*Buffer{keepIn, keepOut}, n)
	SyncQueue()

	got := keepOut.FloatSlice()
	for i := 0; i < n; i++ {
		if want := float32(i) * 2; got[i] != want {
			t.Fatalf("keepOut[%d] = %v, want %v — purging released buffers corrupted "+
				"memory that was still live", i, got[i], want)
		}
	}
	// The input must survive too: it was CPU-dirtied and never released.
	for i := 0; i < n; i++ {
		if src[i] != float32(i) {
			t.Fatalf("keepIn[%d] = %v, want %v — a live buffer's CPU-written contents "+
				"were discarded", i, src[i], float32(i))
		}
	}
}

// TestDTGraphCacheIsObservable pins the instrument added alongside this
// fix. The dtyped-matmul MPSGraph cache is keyed on (M,N,K,batch,...),
// and M/K carry the SEQUENCE LENGTH, so on variable-length training
// data it mints a graph per matmul site per unique length. That was
// the first suspect for the footprint growth; it turned out not to be
// the cause (footprint compounded identically with the cache pinned at
// 12 entries) but it is a real unbounded-growth risk, so the counter
// and the cap stay.
func TestDTGraphCacheIsObservable(t *testing.T) {
	dev, queue, _ := reclaimFixture(t)
	SetAsync(false)

	base := DTGraphCacheLen()

	// Two DIFFERENT shapes must produce two different cache entries.
	for _, m := range []int{8, 16} {
		a := dev.NewBuffer(m * m * 4)
		b := dev.NewBuffer(m * m * 4)
		c := dev.NewBuffer(m * m * 4)
		if err := queue.MatMulDT(a, b, c, m, m, m, false, false, false, false); err != nil {
			t.Skipf("dtyped matmul unavailable: %v", err)
		}
		SyncQueue()
		a.Release()
		b.Release()
		c.Release()
	}
	grew := DTGraphCacheLen() - base
	if grew < 2 {
		t.Fatalf("cache grew by %d for two distinct shapes, want >= 2 — the counter "+
			"is not observing graph creation, so it cannot guard cache growth", grew)
	}

	// The cap must actually bound it.
	SetDTGraphCacheLimit(1)
	defer SetDTGraphCacheLimit(0)
	for _, m := range []int{24, 32, 40} {
		a := dev.NewBuffer(m * m * 4)
		b := dev.NewBuffer(m * m * 4)
		c := dev.NewBuffer(m * m * 4)
		if err := queue.MatMulDT(a, b, c, m, m, m, false, false, false, false); err != nil {
			t.Fatal(err)
		}
		SyncQueue()
		a.Release()
		b.Release()
		c.Release()
	}
	if n := DTGraphCacheLen(); n > 1 {
		t.Errorf("cache holds %d entries under a limit of 1 — the cap does not bound "+
			"growth, so variable sequence lengths can grow it without end", n)
	}
}
