//go:build darwin

package optim

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// ClipGradNorm rescales gradients in-place so that the global L2 norm
// across all params is at most maxNorm. Returns the original total norm
// (before clipping) so callers can log gradient-explosion events.
//
// Equivalent to PyTorch's torch.nn.utils.clip_grad_norm_:
//
//	totalNorm = sqrt( sum_p ||grad_p||² )
//	if totalNorm > maxNorm:
//	    clipCoef = maxNorm / (totalNorm + eps)
//	    for p: grad_p *= clipCoef
//
// Without this, transformer training routinely explodes — one bad batch
// produces a few extreme gradients which drive AdamW into a region the
// loss can't recover from. clip_grad_norm at maxNorm=1.0 is the default
// in nanoGPT, llama-recipes, and most public training scripts.
//
// Call this between loss.Backward() and optimizer.Step():
//
//	loss.Backward()
//	optim.ClipGradNorm(model.Parameters(), 1.0)
//	opt.Step()
//
// Plan 0001 Phase 1 item 13; called out in `0003-gemini-review.md` as a
// trivial-but-blocking missing piece for any serious training run.
func ClipGradNorm(params []*g.Tensor, maxNorm float32) (totalNorm float32) {
	const eps = 1e-6

	var sumSq float64
	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		for _, v := range grad.Data() {
			sumSq += float64(v) * float64(v)
		}
	}
	totalNorm = float32(math.Sqrt(sumSq))

	if totalNorm <= maxNorm {
		return totalNorm
	}

	clipCoef := maxNorm / (totalNorm + eps)
	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		gData := grad.Data()
		for j := range gData {
			gData[j] *= clipCoef
		}
	}
	return totalNorm
}
