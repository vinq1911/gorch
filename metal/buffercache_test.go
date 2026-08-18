//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// Tests for the R1b size-classed buffer reuse cache (ADR-015).
//
// The cache exists because a CPU-touched shared MTLBuffer strands its
// IOAccelerator VM map entry when released (ADR-014), and a buffer that
// is never released strands nothing. Recycling is therefore a fix at
// the source — but it moves the framework from "every allocation is
// fresh, zero-filled, and exclusively ours" to "an allocation may be
// somebody else's memory from a moment ago", and that trade has three
// ways to go silently wrong. Each has a test here:
//
//  1. STALE BYTES. A recycled buffer that keeps its previous owner's
//     contents hands wrong numbers to any caller that writes only part
//     of its allocation. Worse than the leak, because it is invisible.
//  2. WRONG LENGTH. A 64 KB buffer handed out for a 40 KB request must
//     present as 40 KB, or MPS descriptors and slice views silently
//     take the wrong shape.
//  3. REUSE IN FLIGHT. Releasing a buffer while the GPU still reads it
//     is safe (command buffers retain their resources); REUSING it is
//     not, because the new owner overwrites what that work is reading.
//
// The fourth test is the point of the exercise: the region slope on the
// real micro-step size distribution.

// cacheFixture returns a device with its own empty cache, restoring
// every global knob afterwards.
func cacheFixture(t *testing.T) *Device {
	t.Helper()
	wasLimit := BufferCacheLimit()
	wasPerClass := int(cachePerClass.Load())
	wasAsync := AsyncEnabled()
	wasZero := cacheZeroing.Load()
	wasPoison := cachePoison.Load()

	dev, err := NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		SyncQueue()
		SetAsync(wasAsync)
		dev.Release() // drains and unregisters this device's cache
		SetBufferCacheLimit(wasLimit)
		SetBufferCachePerClass(wasPerClass)
		setCacheZeroingForTest(wasZero)
		setCachePoisonForTest(wasPoison)
		SetPurgeOnRelease(true)
	})
	SetPurgeOnRelease(true)
	ResetBufferCacheStats()
	return dev
}

// TestSizeClassesAreCoarse pins the classing rule itself, because the
// classing IS the fix: exact-size classes measured 0.570 regions per
// allocation against 0.003 for coarse ones (ADR-014).
func TestSizeClassesAreCoarse(t *testing.T) {
	cases := []struct{ in, want int }{
		{0, 256}, {1, 256}, {256, 256}, {257, 512},
		{4000, 4096}, {4096, 4096}, {4097, 8192},
		{16 << 10, 16 << 10}, {(16 << 10) + 1, 32 << 10},
		{2 << 20, 2 << 20},
		// Above 2 MB the classes step rather than double: a doubling
		// would turn a 51 MB frozen weight into 64 MB, 28 layers over.
		{(2 << 20) + 1, 4 << 20},
		{51 << 20, 52 << 20},
		{100 << 20, 100 << 20},
	}
	for _, c := range cases {
		if got := SizeClassFor(c.in); got != c.want {
			t.Errorf("SizeClassFor(%d) = %d, want %d", c.in, got, c.want)
		}
	}

	// The whole micro-step distribution must land in a handful of
	// classes. If this count explodes, the cache has become the
	// exact-size trap wearing the coarse-class name.
	seen := map[int]bool{}
	r := rand.New(rand.NewSource(1))
	lo, hi := math.Log(256), math.Log(4<<20)
	for i := 0; i < 20000; i++ {
		n := int(math.Exp(lo+r.Float64()*(hi-lo))) &^ 3
		seen[SizeClassFor(n)] = true
	}
	if len(seen) > 20 {
		t.Errorf("the log-uniform [256 B, 4 MB] micro-step distribution produced "+
			"%d distinct size classes, want a handful — thousands of one-shot "+
			"classes is exactly the failure mode ADR-014 measured at 0.570 "+
			"regions/alloc", len(seen))
	}
	t.Logf("micro-step distribution spans %d size classes", len(seen))
}

// mixedChurn runs the real micro-step allocation distribution —
// log-uniform [256 B, 4 MB], one CPU word written per 4 KB page, a ring
// of liveSet buffers so live bytes stay flat — through whatever
// allocation path is currently configured, and reports the region
// slope with allocation latency.
type churnResult struct {
	slope      float64
	p50, p99   time.Duration
	peakCached int64
	hits       int64
	misses     int64
	evictions  int64
}

func mixedChurn(t *testing.T, dev *Device, rounds, perRound, liveSet int) churnResult {
	t.Helper()
	r := rand.New(rand.NewSource(1))
	lo, hi := math.Log(256), math.Log(4<<20)
	ring := make([]*Buffer, liveSet)
	counts := make([]int, 0, rounds)
	lat := make([]time.Duration, 0, rounds*perRound)

	base := BufferCacheStatsSnapshot()
	for round := 0; round < rounds; round++ {
		for i := 0; i < perRound; i++ {
			slot := i % liveSet
			if ring[slot] != nil {
				ring[slot].Release()
				ring[slot] = nil
			}
			n := int(math.Exp(lo+r.Float64()*(hi-lo))) &^ 3
			t0 := time.Now()
			b := dev.NewBuffer(n)
			lat = append(lat, time.Since(t0))
			fs := b.FloatSlice()
			for p := 0; p < len(fs); p += 1024 { // one word per 4 KB page
				fs[p] = 1
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

	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	q := func(p float64) time.Duration { return lat[int(float64(len(lat)-1)*p)] }
	end := BufferCacheStatsSnapshot()
	return churnResult{
		slope:      slope(counts, perRound),
		p50:        q(0.5),
		p99:        q(0.99),
		peakCached: end.PeakCachedBytes,
		hits:       end.Hits - base.Hits,
		misses:     end.Misses - base.Misses,
		evictions:  end.Evictions - base.Evictions,
	}
}

// TestBufferCacheStopsTheGrowth is the headline result: the shipped
// cache, on the size distribution a real micro-step produces, with the
// CPU touches that cause the leak.
//
// The uncached baseline for this exact workload is 0.586 regions per
// allocation (ADR-014, and TestVMRegionReuseCacheStopsTheGrowth's first
// arm re-measures it every run). Anything near that means the cache is
// not absorbing the churn.
func TestBufferCacheStopsTheGrowth(t *testing.T) {
	dev := cacheFixture(t)
	const rounds, perRound, liveSet = 3, 2000, 64

	// Baseline arm, cache off: this is the workload the fix has to beat,
	// measured on the same machine in the same run rather than quoted.
	SetBufferCacheLimit(0)
	off := mixedChurn(t, dev, rounds, perRound, liveSet)

	SetBufferCacheLimit(defaultCacheBytes)
	ResetBufferCacheStats()
	on := mixedChurn(t, dev, rounds, perRound, liveSet)

	t.Logf("cache OFF: %.3f regions/alloc  p50 %v  p99 %v", off.slope, off.p50, off.p99)
	t.Logf("cache ON : %.3f regions/alloc  p50 %v  p99 %v  hits %d misses %d "+
		"evictions %d hit-rate %.3f peak cached %.1f MB (cap %.0f MB)",
		on.slope, on.p50, on.p99, on.hits, on.misses, on.evictions,
		float64(on.hits)/float64(on.hits+on.misses),
		float64(on.peakCached)/(1<<20), float64(defaultCacheBytes)/(1<<20))

	if off.slope < 0.3 {
		t.Fatalf("the uncached baseline leaked only %.3f regions/alloc; the driver "+
			"behaviour this cache exists for no longer reproduces, so the ON arm "+
			"proves nothing", off.slope)
	}
	if on.slope > 0.05 {
		t.Errorf("with the reuse cache the churn still leaked %.3f regions/alloc, "+
			"want < 0.05 (uncached %.3f) — the cache is not absorbing the "+
			"allocations it is supposed to", on.slope, off.slope)
	}
	// Live bytes must stay bounded. The exact-size trap "worked" on the
	// slope for a while by simply never releasing anything, reaching
	// 5 GB live in 12 k allocations; a bounded pool is what makes this a
	// fix rather than a different leak.
	if got := BufferCacheStatsSnapshot().CachedBytes; got > defaultCacheBytes {
		t.Errorf("cache holds %.1f MB, over its own %.0f MB cap",
			float64(got)/(1<<20), float64(defaultCacheBytes)/(1<<20))
	}
	if on.peakCached > defaultCacheBytes {
		t.Errorf("cached bytes peaked at %.1f MB, over the %.0f MB cap — the "+
			"eviction path is not keeping up with admission",
			float64(on.peakCached)/(1<<20), float64(defaultCacheBytes)/(1<<20))
	}
	// A coarse-classed cache on this distribution hits nearly always
	// once warm. A collapsed hit rate is the signature of exact-size
	// classing creeping back in, and it would show up here long before
	// the slope did.
	if hr := float64(on.hits) / float64(on.hits+on.misses); hr < 0.9 {
		t.Errorf("hit rate %.3f, want > 0.9 — with coarse classes this workload "+
			"should miss only while the pool warms up", hr)
	}
}

// TestBufferCacheHitRateDependsOnDrainFrequency measures the one place
// this design degrades, and pins the regime the trainer runs in.
//
// A buffer released while the in-flight gate is up cannot be handed out
// again until the queue is provably drained. In SYNC mode that is every
// dispatch, so the quarantine is invisible. In ASYNC mode — which the
// trainer uses — it is every SyncQueue, i.e. every syncForCPU or
// Tensor.Data() that finds a Metal-resident tensor. If those became
// rare the cache would degrade into a holding area: buffers go in,
// nothing comes out, and the allocations go to the driver as before.
//
// Measured here, 6000 mixed allocations with a dispatch every 8:
//
//	drain every       hit rate
//	1                 0.972
//	10                0.971
//	100               0.954
//	1000              0.374
//	never             0.000
//
// The degradation is graceful — the pool fills to its cap and starts
// evicting, so the worst case is the pre-R1b leak plus a bounded pool,
// not a cliff — but it is real. ADR-013's measurement that -accel=sync
// takes unpurged releases from 6284 to 845 says ~37% of releases
// already land in a drained window, which puts the trainer at the top
// of this table; BufferCacheStats().Quarantined in the micro-step log
// is what confirms it on real data.
func TestBufferCacheHitRateDependsOnDrainFrequency(t *testing.T) {
	dev := cacheFixture(t)
	queue := dev.NewCommandQueue()
	defer queue.Release()
	pipe, err := dev.CompileKernel(reclaimKernel, "scale_two")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Release()
	SetAsync(true)

	in, out := dev.NewBuffer(4096*4), dev.NewBuffer(4096*4)
	defer in.Release()
	defer out.Release()

	run := func(syncEvery int) BufferCacheStats {
		ResetBufferCacheStats()
		DrainBufferCache()
		r := rand.New(rand.NewSource(1))
		lo, hi := math.Log(256), math.Log(4<<20)
		ring := make([]*Buffer, 64)
		for i := 0; i < 6000; i++ {
			slot := i % 64
			if ring[slot] != nil {
				ring[slot].Release()
				ring[slot] = nil
			}
			n := int(math.Exp(lo+r.Float64()*(hi-lo))) &^ 3
			b := dev.NewBuffer(n)
			fs := b.FloatSlice()
			for p := 0; p < len(fs); p += 1024 {
				fs[p] = 1
			}
			ring[slot] = b
			if i%8 == 0 {
				queue.Dispatch1D(pipe, []*Buffer{in, out}, 4096)
			}
			if i%syncEvery == syncEvery-1 {
				SyncQueue()
			}
		}
		for _, b := range ring {
			if b != nil {
				b.Release()
			}
		}
		SyncQueue()
		return BufferCacheStatsSnapshot()
	}

	for _, syncEvery := range []int{1, 10, 100, 1000, 1 << 20} {
		s := run(syncEvery)
		t.Logf("drain every %-8d hit-rate %.3f (hits %d misses %d) cached %.1f MB "+
			"evictions %d", syncEvery, s.HitRate(), s.Hits, s.Misses,
			float64(s.CachedBytes)/(1<<20), s.Evictions)
		if syncEvery <= 10 && s.HitRate() < 0.9 {
			t.Errorf("hit rate %.3f with a drain every %d allocations, want > 0.9 — "+
				"the quarantine is holding buffers past the drain that should "+
				"release them", s.HitRate(), syncEvery)
		}
		// However sparse the drains, the pool must stay bounded: that is
		// what keeps the degraded case a slowdown rather than a leak.
		if s.CachedBytes > BufferCacheLimit() {
			t.Errorf("cache holds %d bytes over its %d cap at drain-every-%d",
				s.CachedBytes, BufferCacheLimit(), syncEvery)
		}
	}
}

// TestBufferCacheRecycledBufferNeverYieldsStaleBytes is invariant 1.
//
// The second half deliberately turns the zeroing off and asserts that
// the stale bytes DO come back. Without it this test would pass just as
// happily against a cache that never recycled anything, and the thing
// it is supposed to protect would be untested.
func TestBufferCacheRecycledBufferNeverYieldsStaleBytes(t *testing.T) {
	dev := cacheFixture(t)
	SetAsync(false)
	SyncQueue()
	// This test is about the zero-fill, and the poison would do the
	// zero-fill's job for it (GORCH_METAL_CACHE_POISON=1 overwrites the
	// whole class, so the previous owner's bytes would be gone whether
	// or not the zeroing ran). The fixture restores the setting.
	setCachePoisonForTest(false)

	const n = 64 << 10 // 16384 floats, a clean power-of-two class

	fill := func(b *Buffer, v float32) {
		s := b.FloatSlice()
		for i := range s {
			s[i] = v
		}
	}
	// recycle writes v into a fresh buffer, releases it, and takes
	// another of the same class back out. Fails the test if the cache
	// did not in fact hand back the same storage — a vacuous pass here
	// would be worse than a failure.
	//
	// Comparing MTLBuffer pointers is sound in this direction only: the
	// cache holds the buffer's retain, so the object cannot be
	// destroyed and its address handed to a new allocation while it
	// sits there. With the cache OFF the same comparison would be
	// meaningless — ObjC recycles the object address immediately.
	recycle := func(v float32, want int) *Buffer {
		t.Helper()
		a := dev.NewBuffer(n)
		fill(a, v)
		addr := a.ptr
		a.Release()
		b := dev.NewBuffer(want)
		if b.ptr != addr {
			b.Release()
			t.Fatalf("the cache did not recycle the buffer (%p -> %p); this test "+
				"cannot observe stale bytes if nothing is being reused", addr, b.ptr)
		}
		return b
	}

	b := recycle(123.5, n)
	for i, x := range b.FloatSlice() {
		if x != 0 {
			t.Fatalf("recycled buffer[%d] = %v, want 0 — a caller that writes only "+
				"part of its allocation would silently compute on the previous "+
				"tensor's data", i, x)
		}
	}
	b.Release()

	// A SMALLER request out of the SAME class must also come back clean
	// across its whole visible length — including the tail the previous
	// owner wrote and this one has not reached yet, which is precisely
	// where a partial-writer would read someone else's tensor.
	const short = n - (16 << 10) // 48 KB, still the 64 KB class
	if SizeClassFor(short) != SizeClassFor(n) {
		t.Fatalf("test assumes %d and %d share a class", short, n)
	}
	part := recycle(-7, short)
	if part.Len() != short {
		t.Errorf("Len() = %d, want %d", part.Len(), short)
	}
	for i, x := range part.FloatSlice() {
		if x != 0 {
			t.Fatalf("recycled short buffer[%d] = %v, want 0", i, x)
		}
	}
	part.Release()

	// Non-vacuity: with the zeroing off the stale bytes must reappear.
	// If they do not, the assertions above were not testing the zeroing.
	setCacheZeroingForTest(false)
	dirty := recycle(99.25, n)
	stale := 0
	for _, x := range dirty.FloatSlice() {
		if x == 99.25 {
			stale++
		}
	}
	dirty.Release()
	setCacheZeroingForTest(true)
	if stale == 0 {
		t.Error("with zeroing disabled the recycled buffer came back clean anyway — " +
			"something else is clearing it, so the zero-fill assertions above are " +
			"not evidence that the zero-fill works")
	}
	t.Logf("zeroing off: %d/%d words came back stale, as expected", stale, n/4)
}

// TestBufferCacheLengthIsTheRequestedLength is invariant 2, in both
// halves that matter: what Go sees, and what MPS sees.
func TestBufferCacheLengthIsTheRequestedLength(t *testing.T) {
	dev := cacheFixture(t)
	SetAsync(false)
	SyncQueue()
	queue := dev.NewCommandQueue()
	defer queue.Release()

	// 40 KB out of a 64 KB class.
	const req = 40 << 10
	if SizeClassFor(req) != 64<<10 {
		t.Fatalf("test assumes a 64 KB class for %d bytes, got %d", req, SizeClassFor(req))
	}
	b := dev.NewBuffer(req)
	if b.Len() != req {
		t.Errorf("Len() = %d, want %d — a recycled buffer must not leak its class "+
			"size to the caller", b.Len(), req)
	}
	if got := len(b.FloatSlice()); got != req/4 {
		t.Errorf("len(FloatSlice()) = %d, want %d", got, req/4)
	}
	if got := len(b.Uint16Slice()); got != req/2 {
		t.Errorf("len(Uint16Slice()) = %d, want %d", got, req/2)
	}
	if got := len(b.Uint32Slice()); got != req/4 {
		t.Errorf("len(Uint32Slice()) = %d, want %d", got, req/4)
	}
	if b.cap != 64<<10 {
		t.Errorf("cap = %d, want the 64 KB class", b.cap)
	}
	b.Release()

	// MPS on over-allocated buffers. 17x17 matrices are 1156 bytes and
	// land in the 2048 B class, so every operand buffer here is ~1.8x
	// the size MPS is told about. MPSMatrix takes rows/columns/rowBytes
	// from the descriptor and MPSGraphTensorData takes a shape, and
	// both are built from the caller's dimensions — but "a bigger
	// buffer is fine" is an assumption about Apple's code, so it gets
	// measured rather than asserted.
	const d = 17
	mkMat := func(seed int64) (*Buffer, []float32) {
		buf := dev.NewBuffer(d * d * 4)
		s := buf.FloatSlice()
		r := rand.New(rand.NewSource(seed))
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return buf, s
	}
	a, av := mkMat(1)
	bb, bv := mkMat(2)
	c := dev.NewBuffer(d * d * 4)
	defer a.Release()
	defer bb.Release()
	defer c.Release()
	if a.cap != 2048 {
		t.Fatalf("expected the 2048 B class for a %d-byte request, got %d", d*d*4, a.cap)
	}

	want := make([]float32, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			var acc float32
			for k := 0; k < d; k++ {
				acc += av[i*d+k] * bv[k*d+j]
			}
			want[i*d+j] = acc
		}
	}

	check := func(name string) {
		t.Helper()
		SyncQueue()
		got := c.FloatSlice()
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4 {
				t.Fatalf("%s: C[%d] = %v, want %v — MPS produced wrong numbers on an "+
					"over-allocated buffer, so the size class cannot be hidden from it",
					name, i, got[i], want[i])
			}
		}
	}
	queue.MatMul(a, bb, c, d, d, d)
	check("MPSMatrix")

	clear(c.FloatSlice())
	if err := queue.MatMulDT(a, bb, c, d, d, d, false, false, false, false); err != nil {
		t.Fatalf("MatMulDT: %v", err)
	}
	check("MPSGraph")
}

// TestBufferCacheDoesNotRecycleWhileTheGPUMayBeReading is invariant 3,
// the one that would be a race rather than a bug.
//
// A buffer released while a command buffer that encoded it is still in
// flight goes into the cache — correct, and better than the old path,
// which purged nothing and deferred the release. But its BYTES must
// stay off limits until the queue drains, because the next owner writes
// through them. The shim's drain generation is what enforces that.
func TestBufferCacheDoesNotRecycleWhileTheGPUMayBeReading(t *testing.T) {
	dev := cacheFixture(t)
	queue := dev.NewCommandQueue()
	defer queue.Release()
	pipe, err := dev.CompileKernel(reclaimKernel, "scale_two")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Release()

	SetAsync(true)
	SyncQueue() // start from a provably drained queue

	const n = 16384 // 64 KB, a clean class
	in := dev.NewBuffer(n * 4)
	out := dev.NewBuffer(n * 4)
	defer out.Release()
	src := in.FloatSlice()
	for i := range src {
		src[i] = float32(i%97) + 1
	}

	queue.Dispatch1D(pipe, []*Buffer{in, out}, n) // committed, NOT waited
	victim := in.ptr
	in.Release() // in flight → cache, quarantined

	// Any same-class request now must be served by a NEW allocation.
	reuse := dev.NewBuffer(n * 4)
	if reuse.ptr == victim {
		t.Fatalf("the cache handed back a buffer the GPU may still be reading (%p) — "+
			"the next owner's writes race an in-flight kernel and the failure mode "+
			"is silent wrong numbers", victim)
	}
	// Scribble: were the gate broken, this is what would corrupt `out`.
	for i, s := 0, reuse.FloatSlice(); i < len(s); i++ {
		s[i] = -12345
	}

	SyncQueue()
	for i, got := range out.FloatSlice() {
		if want := (float32(i%97) + 1) * 2; got != want {
			t.Fatalf("out[%d] = %v, want %v — the in-flight kernel read recycled memory",
				i, got, want)
		}
	}

	// And the quarantine must be a delay, not a discard: with the queue
	// drained, the very same storage has to come back. If it does not,
	// the cache is silently dropping every buffer released in flight,
	// which in async mode is nearly all of them.
	reuse.Release()
	after := dev.NewBuffer(n * 4)
	defer after.Release()
	if after.ptr != victim {
		t.Errorf("after the drain the quarantined buffer (%p) was not reused (%p) — "+
			"buffers released in flight must rejoin the pool, not fall out of it",
			victim, after.ptr)
	}
}

// TestBufferCacheEvictionFollowsTheReleaseGate: admission to the cache
// skips the purge on purpose (purging a buffer we intend to reuse would
// discard the very pages we are keeping), but EVICTION is a real
// destruction and must follow the existing discipline — purge when
// drained, defer when work is in flight, never abandon.
func TestBufferCacheEvictionFollowsTheReleaseGate(t *testing.T) {
	dev := cacheFixture(t)
	queue := dev.NewCommandQueue()
	defer queue.Release()
	pipe, err := dev.CompileKernel(reclaimKernel, "scale_two")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Release()

	// 1 MB cap, 64 KB buffers: 16 fit, the 17th evicts.
	const cap1MB = 1 << 20
	const n = 16384 // 64 KB
	SetBufferCacheLimit(cap1MB)

	SetAsync(true)
	SyncQueue()

	keep := dev.NewBuffer(n * 4)
	out := dev.NewBuffer(n * 4)
	defer keep.Release()
	defer out.Release()
	queue.Dispatch1D(pipe, []*Buffer{keep, out}, n) // queue now in flight

	beforeP, beforeD, beforeU := PurgeStats()
	bufs := make([]*Buffer, 24)
	for i := range bufs {
		bufs[i] = dev.NewBuffer(n * 4)
	}
	for _, b := range bufs {
		b.Release()
	}
	afterP, afterD, afterU := PurgeStats()

	if s := BufferCacheStatsSnapshot(); s.CachedBytes > cap1MB {
		t.Errorf("cache holds %d bytes over its %d cap", s.CachedBytes, cap1MB)
	}
	if afterD <= beforeD {
		t.Errorf("no eviction took the deferred branch (%d -> %d) — either nothing "+
			"was evicted, or evictions are bypassing the in-flight gate",
			beforeD, afterD)
	}
	if afterP != beforeP {
		t.Errorf("%d eviction(s) purged their pages while GPU work was in flight — "+
			"that is exactly the discard the gate exists to prevent", afterP-beforeP)
	}
	if afterU != beforeU {
		t.Errorf("%d eviction(s) abandoned their pages instead of deferring them",
			afterU-beforeU)
	}

	SyncQueue()
	if got := PendingReleaseBytes(); got != 0 {
		t.Errorf("%d bytes still pending after the drain — evicted buffers must not "+
			"be able to hide on the deferred list", got)
	}
}

// TestBufferCacheDrainReleasesEverything covers the allocation-failure
// path's new first move (drain the cache, then the deferred list, then
// retry) and the ceiling's escape hatch, both of which depend on the
// drain actually giving the memory back.
func TestBufferCacheDrainReleasesEverything(t *testing.T) {
	dev := cacheFixture(t)
	SetAsync(false)
	SyncQueue()

	const n = 64 << 10
	for i := 0; i < 8; i++ {
		dev.NewBuffer(n).Release()
	}
	s := BufferCacheStatsSnapshot()
	if s.CachedBytes == 0 {
		t.Fatal("nothing was cached; the drain has nothing to prove")
	}
	beforeP, _, _ := PurgeStats()
	freed := DrainBufferCache()
	if freed != s.CachedBytes {
		t.Errorf("drain freed %d bytes, cache held %d", freed, s.CachedBytes)
	}
	if got := BufferCacheStatsSnapshot(); got.CachedBytes != 0 || got.CachedBuffers != 0 {
		t.Errorf("after the drain the cache still holds %d bytes in %d buffers",
			got.CachedBytes, got.CachedBuffers)
	}
	// Drained buffers are destroyed, not merely forgotten: with the
	// queue drained they must have purged their pages on the way out.
	if afterP, _, _ := PurgeStats(); afterP <= beforeP {
		t.Errorf("purged-release count did not advance across the drain (%d -> %d) — "+
			"the cache dropped its buffers without releasing them", beforeP, afterP)
	}
	if got := BufferCacheLimit(); got == 0 {
		t.Error("draining must not disable the cache")
	}
	// And the cache must still work afterwards.
	b := dev.NewBuffer(n)
	b.Release()
	if BufferCacheStatsSnapshot().CachedBytes == 0 {
		t.Error("the cache stopped accepting buffers after a drain")
	}
}

// TestBufferCacheDisabledIsThePreR1bBehaviour keeps the off switch
// honest: it has to actually stop recycling, or the tests that measure
// the uncached driver behaviour (every arm of the ADR-014 suite) are
// silently measuring the cache instead.
func TestBufferCacheDisabledIsThePreR1bBehaviour(t *testing.T) {
	dev := cacheFixture(t)
	SetAsync(false)
	SyncQueue()
	SetBufferCacheLimit(0)

	const n = 64 << 10
	before := BufferCacheStatsSnapshot()
	dev.NewBuffer(n).Release()
	if got := BufferCacheStatsSnapshot().CachedBytes; got != 0 {
		t.Fatalf("cache holds %d bytes with the limit at 0", got)
	}
	b := dev.NewBuffer(n)
	defer b.Release()
	// Pointer identity cannot answer this: with the cache off the buffer
	// is genuinely destroyed, and ObjC hands the same object address
	// straight back to the next allocation. The hit counter can.
	if got := BufferCacheStatsSnapshot(); got.Hits != before.Hits {
		t.Errorf("%d cache hit(s) with the cache disabled", got.Hits-before.Hits)
	}
	// The size class must not be applied either: a disabled cache must
	// not leave the framework quietly over-allocating.
	c := dev.NewBuffer(3000)
	defer c.Release()
	if c.cap != 3000 {
		t.Errorf("allocated %d bytes for a 3000-byte request with the cache off, "+
			"want exact", c.cap)
	}
}

// BenchmarkBufferAlloc measures the allocation path with and without
// the cache at steady state. Reported rather than asserted — the
// interesting comparison is against the ADR-014 ramp table, where
// newBufferWithLength: p50 climbs from 4.0 us at 50 k regions to
// 26.1 us at 1 M because of the very leak this cache removes.
func BenchmarkBufferAlloc(b *testing.B) {
	dev, err := NewDevice()
	if err != nil {
		b.Fatal(err)
	}
	defer dev.Release()
	SetPurgeOnRelease(true)

	for _, tc := range []struct {
		name  string
		limit int64
	}{{"cached", defaultCacheBytes}, {"uncached", 0}} {
		b.Run(tc.name, func(b *testing.B) {
			was := BufferCacheLimit()
			SetBufferCacheLimit(tc.limit)
			defer SetBufferCacheLimit(was)
			// Warm the pool so the measurement is steady state.
			for i := 0; i < 64; i++ {
				dev.NewBuffer(64 << 10).Release()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				buf := dev.NewBuffer(64 << 10)
				buf.FloatSlice()[0] = 1 // the CPU touch that strands the entry
				buf.Release()
			}
		})
	}
}
