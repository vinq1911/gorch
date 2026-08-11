//go:build darwin

package gorch

import "github.com/vinq1911/gorch/metal"

// Metal async dispatch mode — plan 0009 X2, risk R6.
//
// The X1K1 profile measured ~46% of the Metal block-step wall clock
// blocked in per-op waitUntilCompleted (each GPU op was a full command-
// buffer round trip). SetMetalAsync(true) switches the metal shim to
// commit-without-wait: GPU→GPU chains (matmul → bias → rmsnorm →
// softmax …) queue back-to-back with no host stall, and the host only
// blocks when it actually needs to READ GPU-written memory — enforced
// by syncForCPU calls at every CPU-compute branch in the op layer and
// by Tensor.Data()/At/Set for package-external readers (nn, optim,
// user code all read tensor contents through Data()).
//
// Default OFF: with async off every dispatch still waits, and
// syncForCPU is a no-op (one atomic load), so no existing behavior
// changes. The mode is opt-in for training loops; the e2e bench
// measures both modes and the parity test asserts identical results.
//
// Correctness contract (single global queue, single-threaded):
//   - every CPU read of possibly-Metal-resident tensor data inside
//     package gorch must be preceded by syncForCPU(inputs...)
//   - CPU writes into FRESH Metal buffers (ZerosLike outputs, fullLike)
//     need no sync — no in-flight GPU command references them
//   - buffer Release while GPU work is in flight is safe: command
//     buffers retain encoded resources until completion

// SetMetalAsync enables or disables commit-without-wait GPU dispatch.
// Turning it off synchronizes pending work first.
func SetMetalAsync(on bool) { metal.SetAsync(on) }

// MetalAsyncEnabled reports whether async dispatch mode is on.
func MetalAsyncEnabled() bool { return metal.AsyncEnabled() }

// SyncMetal blocks until all committed GPU work has completed. No-op
// when nothing is pending. Public escape hatch for callers that read
// unified memory through raw buffer slices instead of Tensor.Data().
func SyncMetal() { metal.SyncQueue() }

// syncForCPU waits for pending GPU work iff any of the given tensors
// is Metal-resident. Called at the top of every CPU-compute branch
// that reads tensor data. With async mode off (the default) this is a
// single atomic load per resident tensor.
func syncForCPU(ts ...*Tensor) {
	for _, t := range ts {
		if t != nil && t.buf != nil {
			metal.SyncQueue()
			return
		}
	}
}
