//go:build darwin

package gorch

import "sync"

// AcquireFloat32 / ReleaseFloat32 are a tiny sync.Pool for transient
// float32 scratch buffers — the kind that ops allocate, write into,
// and discard within one call (GELU's `inner`, LayerNorm's `xNorm`,
// reshape scratch). They are safe to use across goroutines because
// sync.Pool handles the concurrency.
//
// Lifetime contract: caller must Release exactly once. Buffers
// returned via Acquire are NOT zeroed — the caller writes them
// fully before reading. Buffers returned via Release SHOULD have
// no further references; the next Acquire might hand them back.
//
// These are deliberately scoped to within-op scratch. Pooling
// activation tensors that escape an op needs explicit lifecycle
// (Release semantics on Tensor itself) which lives in a separate
// future change — see ADR-004.

var float32Pool sync.Pool

// AcquireFloat32 returns a float32 slice with len(buf) == n. Capacity
// may be larger if a bigger buffer was returned to the pool. The
// returned slice's contents are unspecified — do not rely on zero
// initialization.
func AcquireFloat32(n int) []float32 {
	if v := float32Pool.Get(); v != nil {
		buf := v.([]float32)
		if cap(buf) >= n {
			return buf[:n]
		}
		// Cached buffer too small — drop it back; caller gets a
		// fresh one. The discarded buf returns to GC.
	}
	return make([]float32, n)
}

// ReleaseFloat32 returns a buffer to the pool. The caller MUST not
// reference buf after this call.
func ReleaseFloat32(buf []float32) {
	if cap(buf) == 0 {
		return
	}
	float32Pool.Put(buf[:cap(buf)])
}
