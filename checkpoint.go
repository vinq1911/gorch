//go:build darwin

package gorch

// Activation (gradient) checkpointing — trade compute for activation
// memory (Chen et al. 2016, "Training Deep Nets with Sublinear Memory
// Cost"; PyTorch's torch.utils.checkpoint).
//
// THE PROBLEM. Reverse-mode autodiff needs every intermediate a
// backward closure reads. A 28-layer Qwen3-0.6B micro-step at seq 1024
// retains ~11 GB of them — the attention score tensors alone are
// 16 heads · 1024 · 1024 · 4 B = 67 MB per layer, twice (pre- and
// post-softmax). That is legitimate, live memory: no leak to fix, and
// it is what caps the trainable (accum, seq) box on a 24 GB machine.
//
// THE TRADE. Run a segment's forward WITHOUT building a graph, keep
// only its INPUT, and install one GradFn whose backward re-runs the
// same forward WITH tracking, backprops through that one segment, and
// throws the local graph away. Peak activation memory becomes
// (one segment) + (one boundary tensor per segment) instead of
// (every layer). The cost is one extra forward per segment: ~+33% of a
// step's FLOPs in theory.
//
// WHY THIS IS UNUSUALLY SAFE HERE. torch.utils.checkpoint forks and
// restores RNG state because dropout must draw the same mask twice.
// gorch's Qwen block path has no dropout and no RNG at all (RMSNorm,
// GQA projections, RoPE, fused causal softmax, SwiGLU, LoRA matmuls —
// all deterministic), and the base model is frozen, so no optimizer
// state can shift between the forward and the recompute. The recompute
// is the forward.
//
// THE CONTRACT (violating any of these silently produces WRONG
// gradients, not a crash — see checkpoint_test.go, which tests each):
//
//  1. fn must be PURE: same input and same captured weights ⇒ same
//     output, no side effects, no RNG, no cache mutation.
//  2. Every differentiable tensor fn captures other than its argument
//     must be a LEAF (gradFn == nil) — a trainable parameter. Capturing
//     a non-leaf outer tensor would let the local backward walk into
//     the outer graph, whose contributions have not all been collected
//     yet: double accumulation.
//  3. Nothing may mutate x's storage, or any captured weight, between
//     the Checkpoint call and Backward. gorch has no version counter,
//     so this is unchecked. In the trainer this means: no optimizer
//     step inside the forward/backward window (Step runs after the
//     accumulation loop, which satisfies it).
//  4. Captured parameters must NOT also be listed as inputs of the
//     enclosing GradFn — the local backward already accumulates into
//     them exactly once.
//
// (1)-(4) were the review findings from the pre-implementation design
// review; each has a named regression test.

// CheckpointSegmentDone, when non-nil, is called at each point where a
// segment's intermediates have just become garbage: once after the
// no-grad forward of every segment, and once after every segment's
// recompute-and-backprop.
//
// WITHOUT IT, CHECKPOINTING SAVES ALMOST NOTHING ON THE METAL PATH —
// this is not a tuning knob (measured, 28 layers / seq 312: peak live
// Metal 7687 MB uncheckpointed vs 5391 MB checkpointed-without-hook, a
// 30% dent instead of the expected order-of-magnitude, while the
// physical footprint went UP because the recompute doubles allocation
// volume).
//
// The reason is that "not retained" and "freed" are different things
// here. Metal buffers are unified-memory allocations outside the Go
// heap: dropping the last reference to one exerts zero Go-heap
// pressure, so nothing triggers a GC, so the finalizer that releases it
// never runs. A whole 28-layer forward can go by without one
// collection, and live bytes track the CUMULATIVE allocation volume of
// the pass — which is the same whether or not a graph was retained.
// Checkpointing only converts into memory once each dead segment is
// actually collected before the next one allocates.
//
// The trainer installs a SyncMetal + GC + settle + GC flush here.
//
// Process-global, like the rest of the grad-mode state. Set it once at
// startup.
var CheckpointSegmentDone func()

// segmentDone fires the flush hook if one is installed.
func segmentDone() {
	if done := CheckpointSegmentDone; done != nil {
		done()
	}
}

// GraphSize returns the number of distinct tensors reachable from t
// through GradFn edges — the size of the autograd graph t is keeping
// alive, and therefore a direct proxy for the activation memory a
// backward pass from t would need.
//
// Diagnostic. Its reason for existing is that "checkpointing produces
// correct gradients but saves nothing" is a silent failure: the numbers
// stay right and only the footprint regresses, which no gradient test
// can see. Comparing GraphSize with checkpointing on and off makes that
// regression a unit test instead of a benchmark.
func GraphSize(t *Tensor) int {
	seen := make(map[*Tensor]bool)
	var walk func(*Tensor)
	walk = func(n *Tensor) {
		if n == nil || seen[n] {
			return
		}
		seen[n] = true
		if n.gradFn != nil {
			for _, in := range n.gradFn.inputs {
				walk(in)
			}
		}
	}
	walk(t)
	return len(seen)
}

// Checkpoint runs fn(x) with activation checkpointing: the forward
// builds no graph and retains no intermediates, and the returned
// tensor's backward re-runs fn with tracking to rebuild the local
// graph, backprops through it, and returns dL/dx.
//
// Gradients for parameters captured by fn are accumulated into their
// .grad fields by the local backward, exactly as an uncheckpointed
// pass would; x is the only input the enclosing graph sees. Outside a
// gradient-tracking scope Checkpoint is a plain fn(x) — there is no
// graph to save.
//
// name is diagnostic only (it prefixes the GradFn's name).
//
// See the package comment above for the four-point purity contract.
func Checkpoint(name string, x *Tensor, fn func(*Tensor) *Tensor) *Tensor {
	if !GradEnabled() {
		return fn(x)
	}

	// Forward with no graph. Run it on a DETACHED handle: ops branch on
	// x.RequiresGrad() to decide whether to build their backward
	// closure, and although SetGradFn would drop it, the closure still
	// gets allocated and still captures its forward tensors. Detaching
	// takes the branch away entirely, so the segment's intermediates are
	// unreferenced the moment the next op consumes them.
	var out *Tensor
	NoGrad(func() { out = fn(x.Detach()) })
	if out == nil {
		panic("gorch: Checkpoint fn returned nil")
	}
	// The segment's intermediates are unreachable as of this line — this
	// is the forward's peak, and on the Metal path the only place it can
	// be brought back down. See CheckpointSegmentDone.
	segmentDone()

	// Detach the output too: fn may legitimately return a tensor that
	// already carries a gradFn from an enclosing scope (or, degenerately,
	// x itself), and this node owns the output's autograd identity.
	out = out.Detach()
	// Direct field write rather than SetRequiresGrad. Equivalent HERE
	// (the early return above guarantees tracking is on at this point),
	// but SetRequiresGrad(true) is silently ignored inside a NoGrad
	// scope, so the moment anyone moves this line into the NoGrad
	// closure above — the natural-looking refactor — the flag stops
	// being set, the node stops being reachable as a grad-requiring
	// tensor, and every gradient upstream of this segment silently
	// becomes nil. The field write cannot fail that way.
	out.requiresGrad = true

	out.gradFn = &GradFn{
		name:   "Checkpoint/" + name,
		inputs: []*Tensor{x},
		backward: func(grad *Tensor) []*Tensor {
			// Recompute WITH tracking. We are called from backpropFrom,
			// which runs inside Backward's NoGrad scope, so tracking has
			// to be forced back on for the duration of the recompute.
			//
			// xl is a fresh detached handle on x's storage, not x itself.
			// That is what keeps the accounting straight: the local
			// backward writes xl.grad, and the enclosing backward remains
			// the only writer of x.grad (from the dx we return). Using x
			// directly would count dx twice.
			var xl, y *Tensor
			enableGrad(func() {
				xl = x.Detach()
				xl.requiresGrad = true
				y = fn(xl)
			})
			if y == nil {
				panic("gorch: Checkpoint fn returned nil on recompute")
			}
			if y == xl {
				// Identity segment: no graph, dL/dx is the incoming grad.
				return []*Tensor{grad}
			}

			// Local backward, untracked — same rule as the outer pass.
			// Save and restore y.grad rather than clobbering it: fn is
			// contractually pure, but if it ever returns a pre-existing
			// tensor, silently destroying that tensor's accumulated
			// gradient is the worst possible failure mode.
			savedGrad := y.grad
			NoGrad(func() {
				y.grad = grad
				backpropFrom(y)
				if y.gradFn == nil && y.requiresGrad {
					// fn returned a LEAF — a captured parameter passed
					// straight through. backpropFrom had no graph to walk,
					// so restoring savedGrad here would drop this segment's
					// entire contribution to that parameter, which an
					// uncheckpointed pass would have accumulated. Add it
					// instead. (Found by design review; the Qwen block
					// closures never do this, but Checkpoint is a general
					// primitive and the failure would be silent.)
					if savedGrad != nil {
						y.grad = savedGrad
						accumulateGrad(y.grad, grad)
					}
					return
				}
				y.grad = savedGrad
			})

			dx := xl.grad
			xl.grad = nil

			// Drop the local graph before the next segment recomputes.
			y, xl = nil, nil
			segmentDone()
			return []*Tensor{dx}
		},
	}
	return out
}
