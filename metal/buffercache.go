//go:build darwin

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#include "shim.h"
*/
import "C"

import (
	"os"
	"sync"
	"sync/atomic"
)

// ---------------------------------------------------------------------------
// Size-classed buffer reuse cache (R1b, ADR-015)
// ---------------------------------------------------------------------------
//
// WHY. A shared MTLBuffer whose `contents` mapping the CPU has touched
// — a read is as bad as a write — strands its IOAccelerator VM map
// entry when it is released, for the life of the process (ADR-014).
// At or above 16 KB that is 1.004 stranded entries per allocation
// whatever the size; below it the driver suballocates and the leak
// tracks bytes churned instead, ~1 entry per 115 KB. On the log-uniform
// [256 B, 4 MB] distribution a micro-step actually produces the
// composite is 0.586 entries per allocation. Each entry costs ~128
// bytes of WIRED kernel memory that is charged to no task's
// phys_footprint, and newBufferWithLength: degrades linearly with map
// size — 6.5x from 50 k to 1 M regions.
//
// A buffer that is never released strands nothing. So: recycle.
//
// WHY THE CLASSES MUST BE COARSE. Caching by exact size looks like the
// same fix and is not. On the real mixed distribution it mints
// thousands of single-use classes that never hit: the slope stays at
// 0.570/alloc and live bytes balloon to 5 GB in 12 k allocations.
// TestVMRegionReuseCacheStopsTheGrowth pins all three arms.
//
// WHY THE TAIL IS NOT POWER-OF-TWO. Powers of two are right where the
// churn is, and wrong where the weights are: a frozen 51 MB weight
// tensor would round to 64 MB, and there are 28 layers of those. So the
// classes are powers of two up to pow2ClassLimit and multiples of
// pow2ClassLimit above it — bounded waste (< 2 MB) exactly where a
// doubling would be expensive, and still coarse enough to hit, since a
// 2 MB-granular bucket is a very coarse bucket for a 50 MB tensor.

const (
	// minClassBytes is the smallest class. Below the 16 KB threshold
	// the leak is proportional to bytes rather than to count, so tiny
	// buffers are cheap to churn — but they are also cheap to cache,
	// and 256 B is where the measured distribution starts.
	minClassBytes = 256

	// pow2ClassLimit is the size up to which classes double. Above it
	// they step by pow2ClassLimit. See the note above.
	pow2ClassLimit = 2 << 20

	// defaultCacheBytes is the default ceiling on total cached bytes.
	// The validated experiment settled at 15 classes / 176 buffers /
	// 257 MB on the micro-step distribution; this leaves headroom for
	// a wider live set without being a meaningful bite out of a 24 GB
	// machine's budget.
	defaultCacheBytes = 384 << 20

	// defaultCachePerClass caps how many buffers one class may hold, so
	// a burst in one size cannot evict every other class.
	defaultCachePerClass = 32

	// cacheMaxBufferDivisor bounds the largest cacheable buffer at
	// limit/divisor. Without it a single huge allocation would be
	// admitted and then immediately evict most of the pool — churn
	// with extra steps. Buffers past it are allocated at their exact
	// requested size and released normally: they are the model's
	// frozen weights, which are allocated once and never freed, so
	// they contribute nothing to the leak this cache exists to stop.
	cacheMaxBufferDivisor = 8
)

// SizeClassFor returns the class a request of n bytes is rounded up to.
// Exported for tests and for callers sizing a pool.
func SizeClassFor(n int) int {
	if n <= minClassBytes {
		return minClassBytes
	}
	if n <= pow2ClassLimit {
		c := minClassBytes
		for c < n {
			c <<= 1
		}
		return c
	}
	return (n + pow2ClassLimit - 1) &^ (pow2ClassLimit - 1)
}

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------

var (
	cacheLimitBytes atomic.Int64 // 0 disables the cache entirely
	cachePerClass   atomic.Int64
	cacheZeroing    atomic.Bool // hand out zeroed bytes; see zeroing note
	cachePoison     atomic.Bool // test aid; see setCachePoisonForTest
)

func init() {
	cacheLimitBytes.Store(defaultCacheBytes)
	cachePerClass.Store(defaultCachePerClass)
	cacheZeroing.Store(true)
	// GORCH_METAL_CACHE_POISON=1 makes every recycled buffer's bytes
	// PAST the requested length a loud NaN pattern instead of whatever
	// its previous owner left there. It exists so the whole test suite
	// can be run as one audit of the claim that the requested length is
	// the full range any consumer touches:
	//
	//	GORCH_METAL_CACHE_POISON=1 go test ./...
	//
	// Anything that reads past its own request turns numeric-garbage
	// instead of accidentally-plausible.
	if os.Getenv("GORCH_METAL_CACHE_POISON") == "1" {
		cachePoison.Store(true)
	}
}

// SetBufferCacheLimit caps the total bytes held by the buffer reuse
// cache. 0 disables the cache and drains what it holds — the pre-R1b
// behaviour, useful only for reproducing the VM region leak in tests.
// Default 384 MB.
//
// Cached bytes are real resident memory: they count against the
// SetLiveBufferLimit ceiling, which drains the cache before it panics.
func SetBufferCacheLimit(bytes int64) {
	if bytes < 0 {
		bytes = 0
	}
	cacheLimitBytes.Store(bytes)
	if bytes == 0 {
		DrainBufferCache()
		return
	}
	trimBufferCaches()
}

// BufferCacheLimit returns the current cap on total cached bytes.
func BufferCacheLimit() int64 { return cacheLimitBytes.Load() }

// SetBufferCachePerClass caps how many buffers a single size class may
// hold. Default 32; values below 1 disable caching just as a 0 byte
// limit does.
func SetBufferCachePerClass(n int) {
	if n < 0 {
		n = 0
	}
	cachePerClass.Store(int64(n))
	trimBufferCaches()
}

// setCacheZeroingForTest turns off the zero-fill applied to recycled
// buffers. Test-only: with it off, a recycled buffer hands its previous
// owner's bytes to its next one, which is a silent-wrong-answer bug.
// It exists so TestBufferCacheRecycledBufferIsZeroed can demonstrate
// that the zeroing is what prevents that, rather than asserting against
// a buffer that happened to be clean.
func setCacheZeroingForTest(on bool) { cacheZeroing.Store(on) }

// setCachePoisonForTest fills the bytes of a recycled buffer past the
// requested length with a NaN pattern. See the init note.
func setCachePoisonForTest(on bool) { cachePoison.Store(on) }

// poisonWord is a quiet-NaN bit pattern with a recognisable payload:
// anything that reads it as float32 propagates NaN, and anything that
// dumps it as hex is obvious in a log.
const poisonWord = uint32(0x7fc0dead)

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

var (
	cacheHits       atomic.Int64
	cacheMisses     atomic.Int64
	cacheEvictions  atomic.Int64
	cachedBytes     atomic.Int64
	peakCachedBytes atomic.Int64
)

// BufferCacheStats reports the reuse cache's behaviour. Hits and misses
// are cumulative since process start (ResetBufferCacheStats restarts
// them); the rest are instantaneous.
type BufferCacheStats struct {
	// Hits is allocations served from the cache — each one is a VM map
	// entry not stranded and a newBufferWithLength: not made.
	Hits int64
	// Misses is allocations that had to go to the driver.
	Misses int64
	// Evictions is cached buffers genuinely released to stay under the
	// cap. Each one strands an entry, so this is the residual leak.
	Evictions int64
	// CachedBytes/CachedBuffers/Classes are what the cache holds now.
	CachedBytes   int64
	CachedBuffers int
	Classes       int
	// Quarantined counts cached buffers that are NOT yet handable-out
	// because GPU work may still be reading them (see the epoch note on
	// bufferCache.get). A high steady-state figure means the drain
	// points are too sparse and the cache is acting as a holding area.
	Quarantined int
	// PeakCachedBytes is the high-water mark of CachedBytes.
	PeakCachedBytes int64
	// LimitBytes is the current cap.
	LimitBytes int64
}

// HitRate returns hits / (hits + misses); 0 when nothing was allocated.
func (s BufferCacheStats) HitRate() float64 {
	n := s.Hits + s.Misses
	if n == 0 {
		return 0
	}
	return float64(s.Hits) / float64(n)
}

// ResetBufferCacheStats zeroes the cumulative hit/miss/eviction counters
// and reseeds the cached-bytes peak with the current value.
func ResetBufferCacheStats() {
	cacheHits.Store(0)
	cacheMisses.Store(0)
	cacheEvictions.Store(0)
	peakCachedBytes.Store(cachedBytes.Load())
}

// ---------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------

// cachedBuf is a recycled MTLBuffer waiting for its next owner. It is
// deliberately NOT a *Buffer: the *Buffer handed back to the old owner
// is dead (its ptr nilled, its finalizer cleared) and a cache hit mints
// a fresh one, so a stale handle can never name a recycled allocation.
//
// It deliberately does not carry the `contents` mapping pointer either.
// Asking for that pointer is itself what strands the VM map entry, so a
// buffer that has never been mapped must stay unmapped — see
// adoptBuffer.
type cachedBuf struct {
	ptr C.MTLBufferRef
	// epoch is the drain generation this buffer's bytes become safe to
	// overwrite at. See bufferCache.put.
	epoch uint64
}

// bufferCache is one device's free list, keyed by size class.
//
// LOCK ORDER. c.mu is taken before the shim's pending lock (both
// metal_reuse_epoch and metal_drain_epoch take it). The shim never
// calls back into Go, so the reverse order does not exist and this
// cannot deadlock. Driver calls — the purge inside metal_release_buffer
// — are made OUTSIDE c.mu: eviction collects victims under the lock and
// releases them after dropping it.
type bufferCache struct {
	mu      sync.Mutex
	classes map[int][]cachedBuf
	bytes   int64
	// gen caches the last observed drain generation so the common hit
	// path does not need a cgo call. Only ever refreshed upward, and a
	// stale (lower) value only ever refuses a reuse that would have
	// been safe, so it cannot make an unsafe one look safe.
	gen uint64
}

var (
	cacheRegMu sync.Mutex
	cacheReg   []*bufferCache
)

func newBufferCache() *bufferCache {
	c := &bufferCache{classes: map[int][]cachedBuf{}}
	cacheRegMu.Lock()
	cacheReg = append(cacheReg, c)
	cacheRegMu.Unlock()
	return c
}

func (c *bufferCache) unregister() {
	cacheRegMu.Lock()
	for i, x := range cacheReg {
		if x == c {
			cacheReg = append(cacheReg[:i], cacheReg[i+1:]...)
			break
		}
	}
	cacheRegMu.Unlock()
}

func allCaches() []*bufferCache {
	cacheRegMu.Lock()
	out := append([]*bufferCache(nil), cacheReg...)
	cacheRegMu.Unlock()
	return out
}

// maxCacheable is the largest buffer the cache will admit.
func maxCacheable() int64 { return cacheLimitBytes.Load() / cacheMaxBufferDivisor }

// get pops a reusable buffer of exactly class bytes, or reports false.
//
// THE EPOCH GATE — the one place a subtle bug would hide. Releasing a
// buffer while GPU work is in flight is safe because command buffers
// retain their encoded resources until completion; REUSING its bytes is
// not, because the new owner overwrites what that work may still be
// reading, and the failure mode is silent wrong numbers. So a buffer
// that entered the cache while the in-flight gate was raised records
// the NEXT drain generation and is refused until the shim has actually
// reached it — i.e. until a waitUntilCompleted has returned, which on a
// single in-order queue covers every command buffer committed before
// the buffer was cached, which is all of the work that could name it.
//
// Popping only the FRONT is sound because epochs are assigned under
// c.mu, in the same critical section as the append (see put): the list
// is therefore sorted by epoch, and if the oldest entry is still
// quarantined none of the younger ones can be free.
func (c *bufferCache) get(class int) (C.MTLBufferRef, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	fl := c.classes[class]
	if len(fl) == 0 {
		return nil, false
	}
	if fl[0].epoch > c.gen {
		c.gen = uint64(C.metal_drain_epoch())
		if fl[0].epoch > c.gen {
			return nil, false // still in quarantine
		}
	}
	e := fl[0]
	fl[0] = cachedBuf{}
	c.classes[class] = fl[1:]
	c.bytes -= int64(class)
	cachedBytes.Add(-int64(class))
	return e.ptr, true
}

// put takes ownership of a released buffer's MTLBuffer, or refuses it.
// Refusing means the caller must release it the ordinary way.
//
// The reuse epoch is sampled INSIDE c.mu, immediately before the
// append. Sampling it outside would let two threads interleave —
// sample 100, drain, sample 101, append 101, append 100 — leaving the
// list unsorted and breaking get's front-only scan.
func (c *bufferCache) put(ptr C.MTLBufferRef, class int64) bool {
	limit := cacheLimitBytes.Load()
	perClass := int(cachePerClass.Load())
	if limit <= 0 || perClass <= 0 || class > maxCacheable() {
		return false
	}

	var victims []cachedBuf
	var victimClass []int64
	c.mu.Lock()
	k := int(class)
	fl := c.classes[k]
	if len(fl) >= perClass {
		c.mu.Unlock()
		return false
	}
	c.classes[k] = append(fl, cachedBuf{
		ptr:   ptr,
		epoch: uint64(C.metal_reuse_epoch()),
	})
	c.bytes += class
	tot := cachedBytes.Add(class)
	notePeakCached(tot)
	victims, victimClass = c.evictLocked(limit)
	c.mu.Unlock()

	for i, v := range victims {
		cacheEvictions.Add(1)
		releaseRawBuffer(v.ptr, victimClass[i])
	}
	return true
}

// evictLocked drops buffers until the cache is under limit, largest
// class first and oldest first within a class.
//
// Largest first because every eviction is a real release and therefore
// one more stranded VM map entry: freeing the bytes with the fewest
// releases is the whole point. Oldest first within a class because the
// front of the list is also the entry most likely to be out of
// quarantine, and evicting a quarantined entry is fine but pointless.
func (c *bufferCache) evictLocked(limit int64) ([]cachedBuf, []int64) {
	var victims []cachedBuf
	var sizes []int64
	for c.bytes > limit {
		biggest := 0
		for k, fl := range c.classes {
			if len(fl) > 0 && k > biggest {
				biggest = k
			}
		}
		if biggest == 0 {
			break
		}
		fl := c.classes[biggest]
		victims = append(victims, fl[0])
		sizes = append(sizes, int64(biggest))
		fl[0] = cachedBuf{}
		c.classes[biggest] = fl[1:]
		c.bytes -= int64(biggest)
		cachedBytes.Add(-int64(biggest))
	}
	return victims, sizes
}

// drain empties the cache, genuinely releasing everything it holds.
// Returns the bytes released.
func (c *bufferCache) drain() int64 {
	c.mu.Lock()
	taken := c.classes
	c.classes = map[int][]cachedBuf{}
	freed := c.bytes
	c.bytes = 0
	c.mu.Unlock()

	for k, fl := range taken {
		for _, e := range fl {
			cachedBytes.Add(-int64(k))
			cacheEvictions.Add(1)
			releaseRawBuffer(e.ptr, int64(k))
		}
	}
	return freed
}

func (c *bufferCache) stats(s *BufferCacheStats) {
	gen := uint64(C.metal_drain_epoch())
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, fl := range c.classes {
		if len(fl) == 0 {
			continue
		}
		s.Classes++
		s.CachedBuffers += len(fl)
		for _, e := range fl {
			if e.epoch > gen {
				s.Quarantined++
			}
		}
	}
}

func notePeakCached(cur int64) {
	for {
		old := peakCachedBytes.Load()
		if cur <= old || peakCachedBytes.CompareAndSwap(old, cur) {
			return
		}
	}
}

// releaseRawBuffer destroys an MTLBuffer through the ordinary release
// path — purge, in-flight gate, deferred list — and accounts the
// disposition. Every exit from the cache goes through here: a buffer
// leaving the cache is being genuinely freed, and must not skip any of
// the discipline a normal release follows.
func releaseRawBuffer(ptr C.MTLBufferRef, n int64) {
	switch C.metal_release_buffer(ptr) {
	case 1:
		purgedBytes.Add(n)
	case 2:
		deferredBytes.Add(n)
	default:
		unpurgedBytes.Add(n)
	}
}

// DrainBufferCache releases every buffer the reuse cache holds and
// returns the bytes freed. The trainer does not need this; it exists
// for the allocation-failure path, for the live-bytes ceiling, and for
// tests that need a known-empty starting point.
func DrainBufferCache() int64 {
	var freed int64
	for _, c := range allCaches() {
		freed += c.drain()
	}
	return freed
}

// trimBufferCaches re-applies the current caps after a knob change.
func trimBufferCaches() {
	limit := cacheLimitBytes.Load()
	perClass := int(cachePerClass.Load())
	for _, c := range allCaches() {
		c.mu.Lock()
		var victims []cachedBuf
		var sizes []int64
		for k, fl := range c.classes {
			for len(fl) > perClass {
				victims = append(victims, fl[0])
				sizes = append(sizes, int64(k))
				fl[0] = cachedBuf{}
				fl = fl[1:]
				c.bytes -= int64(k)
				cachedBytes.Add(-int64(k))
			}
			c.classes[k] = fl
		}
		v2, s2 := c.evictLocked(limit)
		victims = append(victims, v2...)
		sizes = append(sizes, s2...)
		c.mu.Unlock()
		for i, v := range victims {
			cacheEvictions.Add(1)
			releaseRawBuffer(v.ptr, sizes[i])
		}
	}
}

// BufferCacheStatsSnapshot returns the current state of the reuse
// cache. Cheap except for Quarantined/Classes, which walk the free
// lists — fine per optimizer step, not per allocation.
func BufferCacheStatsSnapshot() BufferCacheStats {
	s := BufferCacheStats{
		Hits:            cacheHits.Load(),
		Misses:          cacheMisses.Load(),
		Evictions:       cacheEvictions.Load(),
		CachedBytes:     cachedBytes.Load(),
		PeakCachedBytes: peakCachedBytes.Load(),
		LimitBytes:      cacheLimitBytes.Load(),
	}
	for _, c := range allCaches() {
		c.stats(&s)
	}
	return s
}
