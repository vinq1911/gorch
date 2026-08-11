//go:build darwin

package gorch

import "github.com/vinq1911/gorch/metal"

// Column-reduction + bias-add Metal dispatch — plan 0009 X2b (the
// Linear bias/db item plus K5's per-column template shared with
// rmsnorm_dgamma). These are the last per-Linear host loops that
// forced a GPU sync in every backward (db) and forward (bias add).

const colReduceThreadgroupSize = 256

// colSumPipelineReady reports whether the col_sum kernel was compiled.
func colSumPipelineReady() bool {
	if gpu == nil {
		return false
	}
	_, ok := gpu.pipelines["col_sum"]
	return ok
}

// ColSumMetal computes out[j] = Σ_i a[i,j] over an (rows, cols) Metal-
// resident tensor, returning a Metal-backed (1, cols) tensor — the
// Linear db reduction. Returns nil when a is not Metal-resident or the
// pipeline is unavailable (callers fall back to the CPU path).
func ColSumMetal(a *Tensor, rows, cols int) *Tensor {
	if a == nil || a.buf == nil || !colSumPipelineReady() {
		return nil
	}
	dev := gpu.Dev
	out := ZerosOnMetal(dev, 1, cols)

	dimsBuf := dev.NewBuffer(2 * 4)
	dims := dimsBuf.Uint32Slice()
	dims[0] = uint32(rows)
	dims[1] = uint32(cols)

	gpu.Queue.Dispatch1DThreadgroups(
		gpu.Pipe("col_sum"),
		[]*metal.Buffer{a.buf, dimsBuf, out.buf},
		cols,
		colReduceThreadgroupSize,
	)
	metalColReduceDispatches.Add(1)
	dimsBuf.Release()
	return out
}

// BiasAddInPlaceMetal adds a broadcast bias row (cols elements) to
// every row of the Metal-resident tensor out, in place, via the
// vec_bias_add kernel. Returns false when either tensor is not Metal-
// resident or the pipeline is unavailable — callers then run the CPU
// fallback. In-place is safe: each lane reads and writes only its own
// element.
func BiasAddInPlaceMetal(out, bias *Tensor) bool {
	if out == nil || bias == nil || out.buf == nil || bias.buf == nil || gpu == nil {
		return false
	}
	if _, ok := gpu.pipelines["vec_bias_add"]; !ok {
		return false
	}
	cols := bias.Size()
	dev := gpu.Dev
	colsBuf := dev.NewBuffer(4)
	colsBuf.Uint32Slice()[0] = uint32(cols)

	gpu.Queue.Dispatch1D(gpu.pipe("vec_bias_add"),
		[]*metal.Buffer{out.buf, bias.buf, out.buf, colsBuf}, out.Size())
	metalBiasAddDispatches.Add(1)
	colsBuf.Release()
	return true
}
