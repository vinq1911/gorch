//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"
)

// Regression tests for the 2026-08-17 VM map region growth (R3a).
//
// THE SYMPTOM. At the Stage-A geometry the trainer's VM map grew ~9000
// regions per optimizer step, linearly, with no plateau — 2302 at
// startup to 132672 fifteen steps in — while the physical footprint,
// metal.LiveBufferBytes and the Go heap all stayed flat. A micro-step
// allocates and frees ~10000 MTLBuffers, so the growth tracked
// ALLOCATION COUNT, not live bytes.
//
// THE CAUSE, established by the tests below. It is neither the page
// purge nor the deferred-release list, both of which were the prime
// suspects. It is the same driver behaviour the 2026-08-13 reclaim bug
// exposed, seen from the other side:
//
//	A shared MTLBuffer whose `contents` mapping the CPU has TOUCHED
//	leaves its IOAccelerator VM map entry behind when it is released.
//	The entry survives the buffer. Never-touched buffers give theirs
//	back.
//
// `setPurgeableState:Empty` returns the buffer's physical PAGES (that
// is what d5cc131 fixed, and the footprint measurements confirm it
// works) but it does not return the map ENTRY. So the purge fixed the
// visible half of this driver behaviour and left the invisible half —
// which is why toggling it changes the region slope not at all.
//
// The threshold is sharp and sits at 16 KB. At or above it the driver
// hands each buffer its own mapping and each leaks one entry, whatever
// the size. Below it buffers are suballocated out of driver arenas, and
// the leak becomes proportional to BYTES churned rather than to
// allocation count — roughly one stranded entry per ~115 KB. Both
// halves are unbounded; small buffers are cheaper, not free.
//
// WHY IT MATTERS. Nothing about it is free even though the footprint
// is flat. Measured on this machine, ramping one process to 1M regions:
//
//   - newBufferWithLength: p50 rises LINEARLY with map size, 4.0 us at
//     50k regions to 26.1 us at 1M (p99 27 us to 78 us). Release
//     latency stays flat, so it is the allocation path, not teardown.
//   - each leaked region costs ~2 kernel VM.map.entries at 64 bytes
//     each (zone element size read from zprint), so ~128 bytes of
//     WIRED kernel memory per leaked buffer. That memory is not charged
//     to the task's phys_footprint, which is exactly why the footprint
//     looked innocent.
//   - `vmmap --summary` on the process takes 28 s at 1M regions. Any
//     tool that walks the map — a footprint guard, a debugger, a crash
//     reporter — degrades with it.
//
// Process teardown, the one cost that would have been operationally
// nasty, is NOT a problem: 71 ms at 100k regions, 0.5 s at 1M, linear.
//
// THE FIX is to stop churning: a size-classed buffer reuse cache (R1b).
// TestVMRegionReuseCacheStopsTheGrowth pins both that it works and the
// condition on it — the size classes must be COARSE. Caching by exact
// size on the real mixed distribution produces thousands of one-shot
// classes that never hit, which does not slow the leak at all and
// balloons live bytes instead.

const (
	// vmLeakThreshold is the buffer size at or above which the driver
	// gives each allocation its own mapping — and so leaks one map
	// entry per released buffer. Measured, not documented by Apple.
	vmLeakThreshold = 16 << 10

	// vmChurnSize is the per-buffer size used by the churn tests: over
	// the threshold, so the leak is one entry per allocation and the
	// slope is readable in a couple of thousand allocations, but small
	// enough that a live ring of them is a rounding error.
	vmChurnSize = 64 << 10
)

// churnOpts configures one allocate/release churn run.
type churnOpts struct {
	rounds   int  // measurement rounds
	perRound int  // allocations per round
	liveSet  int  // buffers held live at once (keeps live bytes flat)
	size     int  // 0 => mixed log-uniform sizes
	touch    bool // write one word per 4 KB page through the CPU mapping
	purge    bool // SetPurgeOnRelease
	reuse    bool // recycle through a free list instead of releasing
	pow2     bool // round sizes up to a power of two (coarse classes)
}

// churn runs the workload and returns the IOAccelerator region count
// after each round. Live bytes are held flat by construction: a ring of
// liveSet buffers, one released for every one allocated.
func churn(t *testing.T, dev *Device, o churnOpts) []int {
	t.Helper()

	// The package exposes no getter for the purge flag, so restore the
	// default rather than the previous value — same convention as
	// reclaimFixture, and every test here sets it explicitly anyway.
	SetPurgeOnRelease(o.purge)
	t.Cleanup(func() { SetPurgeOnRelease(true) })

	// These runs measure the DRIVER's behaviour under churn, and the
	// `reuse`/`pow2` options model a cache in Go rather than using the
	// real one — that is the record of the experiment that established
	// ADR-015's design. The shipped cache would otherwise absorb the
	// churn and every arm would read zero, including the two that are
	// supposed to leak. TestBufferCacheStopsTheGrowth exercises the
	// real thing.
	wasLimit := BufferCacheLimit()
	SetBufferCacheLimit(0)
	t.Cleanup(func() { SetBufferCacheLimit(wasLimit) })

	r := rand.New(rand.NewSource(1))
	ring := make([]*Buffer, o.liveSet)
	pool := map[int][]*Buffer{}

	size := func() int {
		n := o.size
		if n == 0 { // log-uniform over [256 B, 4 MB], as a micro-step is
			lo, hi := math.Log(256), math.Log(4<<20)
			n = int(math.Exp(lo+r.Float64()*(hi-lo))) &^ 3
		}
		if o.pow2 {
			c := 256
			for c < n {
				c <<= 1
			}
			n = c
		}
		return n
	}
	get := func(n int) *Buffer {
		if o.reuse {
			if fl := pool[n]; len(fl) > 0 {
				b := fl[len(fl)-1]
				pool[n] = fl[:len(fl)-1]
				return b
			}
		}
		return dev.NewBuffer(n)
	}
	put := func(b *Buffer) {
		if o.reuse {
			pool[b.Len()] = append(pool[b.Len()], b)
			return
		}
		b.Release()
	}

	counts := make([]int, 0, o.rounds)
	for round := 0; round < o.rounds; round++ {
		for i := 0; i < o.perRound; i++ {
			slot := i % o.liveSet
			if ring[slot] != nil {
				put(ring[slot])
			}
			b := get(size())
			if o.touch {
				fs := b.FloatSlice()
				for p := 0; p < len(fs); p += 1024 { // one word per page
					fs[p] = 1
				}
			}
			ring[slot] = b
		}
		counts = append(counts, VMRegionSnapshot().Tag(VMTagIOAccelerator))
	}

	for _, b := range ring {
		if b != nil {
			b.Release()
		}
	}
	for _, fl := range pool {
		for _, b := range fl {
			b.Release()
		}
	}
	return counts
}

// slope is regions gained per allocation over the measured rounds,
// skipping round 0 so one-time warmup does not count as growth.
func slope(counts []int, perRound int) float64 {
	if len(counts) < 2 {
		return 0
	}
	gained := counts[len(counts)-1] - counts[0]
	return float64(gained) / float64(perRound*(len(counts)-1))
}

func vmDevice(t *testing.T) *Device {
	t.Helper()
	dev, err := NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(dev.Release)
	return dev
}

// TestVMRegionGrowthIsCPUTouchNotPurge is the core attribution.
//
// Three runs of the same fixed-size churn, differing only in whether
// the CPU touches the mapping and whether the release purges:
//
//	touch, purge     -> ~1 leaked region per allocation
//	touch, no purge  -> the SAME slope; the purge is not the cause
//	no touch         -> no growth at all; the touch is the cause
func TestVMRegionGrowthIsCPUTouchNotPurge(t *testing.T) {
	dev := vmDevice(t)
	base := churnOpts{rounds: 3, perRound: 2000, liveSet: 64, size: vmChurnSize}

	o := base
	o.touch, o.purge = true, true
	touchedPurged := slope(churn(t, dev, o), o.perRound)

	o = base
	o.touch, o.purge = true, false
	touchedUnpurged := slope(churn(t, dev, o), o.perRound)

	o = base
	o.touch, o.purge = false, true
	untouched := slope(churn(t, dev, o), o.perRound)

	t.Logf("regions per allocation: touch+purge=%.3f touch+nopurge=%.3f untouched=%.3f",
		touchedPurged, touchedUnpurged, untouched)

	// A CPU-touched buffer over the threshold strands its entry.
	if touchedPurged < 0.5 {
		t.Errorf("touched churn leaked %.3f regions/alloc, want ~1 — "+
			"if this dropped to zero the driver behaviour changed and "+
			"the reuse cache may no longer be load-bearing", touchedPurged)
	}
	// An untouched buffer gives it back. This is the control that makes
	// the result an attribution rather than an observation.
	if untouched > 0.05 {
		t.Errorf("untouched churn leaked %.3f regions/alloc, want ~0", untouched)
	}
	// The purge is innocent. Guarding the RATIO rather than the
	// difference keeps this meaningful on a loaded machine.
	if ratio := touchedUnpurged / touchedPurged; ratio < 0.8 || ratio > 1.25 {
		t.Errorf("purge changed the slope by %.2fx (purged=%.3f unpurged=%.3f); "+
			"it did not when this was measured, and if it does now the "+
			"page discard has become a second, separate cause",
			ratio, touchedPurged, touchedUnpurged)
	}
}

// TestVMRegionLeakThresholdIs16K pins the sharp size threshold. Above
// it every allocation leaks an entry; below it the driver suballocates
// and the leak is far slower (proportional to bytes, not to count).
//
// The threshold is what makes a size-classed cache worth building: the
// classes that matter are the ones at or above 16 KB.
func TestVMRegionLeakThresholdIs16K(t *testing.T) {
	dev := vmDevice(t)
	o := churnOpts{rounds: 3, perRound: 2000, liveSet: 64, touch: true, purge: true}

	o.size = vmLeakThreshold / 2
	below := slope(churn(t, dev, o), o.perRound)
	o.size = vmLeakThreshold
	at := slope(churn(t, dev, o), o.perRound)

	t.Logf("regions per allocation: %d B=%.3f  %d B=%.3f",
		vmLeakThreshold/2, below, vmLeakThreshold, at)

	if at < 0.5 {
		t.Errorf("at %d B leaked %.3f regions/alloc, want ~1", vmLeakThreshold, at)
	}
	if below > 0.3 {
		t.Errorf("at %d B leaked %.3f regions/alloc, want well under the "+
			"one-per-allocation rate above the threshold", vmLeakThreshold/2, below)
	}
}

// TestVMRegionReuseCacheStopsTheGrowth is the fix, and its condition.
//
// Recycling buffers through a free list instead of releasing them
// eliminates the growth outright — no allocation, no leaked entry. But
// it only works if the size classes are COARSE enough to actually hit.
// On the mixed distribution a real micro-step produces, caching by
// exact size yields thousands of single-use classes: the cache never
// hits, the leak continues at full rate, and live bytes grow instead.
// That failure mode is the whole reason this test carries three arms.
func TestVMRegionReuseCacheStopsTheGrowth(t *testing.T) {
	dev := vmDevice(t)
	base := churnOpts{rounds: 3, perRound: 2000, liveSet: 64, touch: true, purge: true}

	// Arm 1: mixed sizes, no cache — the status quo.
	o := base
	noCache := slope(churn(t, dev, o), o.perRound)

	// Arm 2: mixed sizes, cache keyed by EXACT size — the trap.
	o = base
	o.reuse = true
	exactCache := slope(churn(t, dev, o), o.perRound)

	// Arm 3: power-of-two size classes plus reuse — the fix.
	o = base
	o.reuse, o.pow2 = true, true
	classedCache := slope(churn(t, dev, o), o.perRound)

	t.Logf("regions per allocation: no-cache=%.3f exact-size-cache=%.3f pow2-cache=%.3f",
		noCache, exactCache, classedCache)

	if noCache < 0.3 {
		t.Fatalf("baseline churn leaked %.3f regions/alloc; the workload no "+
			"longer reproduces the bug, so the other arms prove nothing", noCache)
	}
	if classedCache > 0.05 {
		t.Errorf("pow2-classed reuse still leaked %.3f regions/alloc, want ~0 — "+
			"the R1b cache is supposed to eliminate this at source", classedCache)
	}
	// Exact-size caching is not a fix. Pinned so nobody "simplifies" the
	// classing away and quietly reintroduces the leak.
	if exactCache < 0.3 {
		t.Errorf("exact-size cache leaked only %.3f regions/alloc; it did not "+
			"help when measured, and if it does now the size distribution "+
			"assumption in this test has drifted", exactCache)
	}
}

// TestVMRegionRampCost is the operational-ceiling measurement: how
// allocation latency, the map walk, and teardown behave as the region
// count climbs. Opt-in — it takes minutes and drives one process to a
// million map entries.
//
//	GORCH_VM_RAMP=1000000 go test ./metal/ -run RampCost -v -timeout 30m
func TestVMRegionRampCost(t *testing.T) {
	target := 0
	if v := os.Getenv("GORCH_VM_RAMP"); v != "" {
		fmt.Sscanf(v, "%d", &target)
	}
	if target <= 0 {
		t.Skip("set GORCH_VM_RAMP=<target region count> to run")
	}
	dev := vmDevice(t)
	SetPurgeOnRelease(true)

	const liveSet, window = 256, 50000
	ring := make([]*Buffer, liveSet)
	lat := make([]time.Duration, 0, window)

	t.Logf("%-10s %-12s %-12s %-10s %s", "regions", "allocP50", "allocP99", "walk", "elapsed")
	start := time.Now()
	for n := 0; ; {
		for i := 0; i < window; i++ {
			slot := n % liveSet
			if ring[slot] != nil {
				ring[slot].Release()
			}
			t0 := time.Now()
			b := dev.NewBuffer(vmChurnSize)
			lat = append(lat, time.Since(t0))
			fs := b.FloatSlice()
			for p := 0; p < len(fs); p += 1024 {
				fs[p] = 1
			}
			ring[slot] = b
			n++
		}
		tw := time.Now()
		total := VMRegionSnapshot().Total
		walk := time.Since(tw)

		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		p := func(q float64) time.Duration { return lat[int(float64(len(lat)-1)*q)] }
		t.Logf("%-10d %-12s %-12s %-10s %s", total, p(0.5), p(0.99),
			walk.Round(time.Millisecond), time.Since(start).Round(time.Second))
		lat = lat[:0]

		if total >= target {
			break
		}
	}
	for _, b := range ring {
		if b != nil {
			b.Release()
		}
	}
}
