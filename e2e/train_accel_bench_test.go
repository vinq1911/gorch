//go:build darwin && e2e

package e2e

// Training-acceleration benchmark harness — plan 0009 phase X0.
//
// Measures the CPU-f32 baseline training step at the Qwen3-0.6B
// geometry and writes doc/training_accel_results.json (conventions
// mirror doc/metal_crossover_results.json). Later phases (X1–X4)
// append comparable rows under new phase labels.
//
// Geometry source: Qwen/Qwen3-0.6B config.json (HuggingFace snapshot):
// hidden_size 1024, num_attention_heads 16, num_key_value_heads 8,
// head_dim 128, intermediate_size 3072, num_hidden_layers 28,
// vocab_size 151936, rope_theta 1e6, rms_norm_eps 1e-6, SiLU (SwiGLU
// FFN), tied embeddings, no attention bias.
//
// NOTE on head_dim: 16 heads × 128 head_dim = 2048 ≠ hidden 1024. This
// is real Qwen3: q/k/v projection output dims follow the head config,
// not the hidden size — q_proj 1024→2048, k/v_proj 1024→1024 (8×128),
// o_proj 2048→1024. The nn.GQA module hardcodes headDim =
// dim/numQueryHeads (= 64 here), so it cannot express this geometry;
// the block below assembles attention from Linear + RMSNorm + RoPE +
// Permute/RepeatInterleave + batched matmuls + Softmax, mirroring
// nn/gqa.go's op-for-op structure (Full+Mul scaling, tiled bool
// causal mask + MaskFill, Softmax over the (heads*seq, seq) view).
//
// Divergences from the real Qwen3 block, documented per plan §2.1a:
//   - nn.Linear always has a bias; Qwen3 has no biases anywhere. The
//     bias adds a vector add per projection and a db reduction per
//     backward — small, but included in the measured numbers.
//   - Q/K per-head RMSNorm (Qwen3's q_norm/k_norm over head_dim) IS
//     included, applied on the (heads*seq, head_dim) reshaped view —
//     mathematically identical to the per-head norm.
//   - Weights are f32 (the checkpoint ships bf16); X0 measures today's
//     CPU f32 path per plan §2.2.
//   - The vocab used is 151936 + 16384 appended Mimi tokens = 168320,
//     per plan §0.
//
// pprof (deliverable 2): one CPU profile of 3 block steps at seq 1024
// is captured to $TRAIN_ACCEL_PPROF_OUT (default: os.TempDir()) and
// its top-10 flat% is parsed into the JSON via `go tool pprof`.
//
// Measured pprof top-10 flat% (X0 baseline run, 2026-08-11, 3 block
// steps at seq 1024, Apple M4; total samples 1440ms — exact rows also
// in the JSON's pprof_top10 field):
//
//	25.00%  runtime.cgocall           (Accelerate BLAS sgemm — all matmul fwd+bwd)
//	14.58%  <unknown>                 (unsymbolized Accelerate/vecLib frames)
//	 9.72%  gorch.Softmax             (fwd row loop ex-exp; 18.75% cum with exp)
//	 9.72%  math.archExp              (Softmax exp + SwiGLU sigmoid)
//	 9.03%  optim.(*AdamW).Step       (scalar moment loop, block params)
//	 6.94%  gorch.Permute             (head-split/concat copies)
//	 5.56%  gorch.Softmax.func1       (softmax backward dot/scale loop)
//	 5.56%  runtime.madvise           (allocator page churn from graph allocs)
//	 2.08%  gorch.MaskFill            (causal mask fill)
//	 2.08%  math.Exp inline shim      (11.81% cum — feeds archExp above)

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/metal"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// ---------- geometry (Qwen3-0.6B, see file comment) ----------

const (
	taHidden    = 1024
	taQHeads    = 16
	taKVHeads   = 8
	taHeadDim   = 128
	taQDim      = taQHeads * taHeadDim  // 2048
	taKVDim     = taKVHeads * taHeadDim // 1024
	taInter     = 3072
	taLayers    = 28
	taBaseVocab = 151936
	taMimiVocab = 16384
	taVocab     = taBaseVocab + taMimiVocab // 168320
	taRopeTheta = 1e6
	taSeed      = 42
	taWarmups   = 2
	taRuns      = 5
)

// ---------- helpers ----------

func taMedian(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func taMin(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func taMax(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func taMs(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

// taSeedTensor fills a fresh tensor with deterministic N(0, scale²)
// values from rng.
func taSeedTensor(rng *rand.Rand, scale float32, shape ...int) *g.Tensor {
	t := g.Zeros(shape...)
	d := t.Data()
	for i := range d {
		d[i] = float32(rng.NormFloat64()) * scale
	}
	return t
}

// taReseed overwrites a param's data deterministically in place.
func taReseed(rng *rand.Rand, t *g.Tensor, scale float32) {
	d := t.Data()
	for i := range d {
		d[i] = float32(rng.NormFloat64()) * scale
	}
}

// taSumScalars adds 1-element tensors into one scalar (for Backward
// over multi-output micro-benches).
func taSumScalars(ts ...*g.Tensor) *g.Tensor {
	out := ts[0]
	for _, t := range ts[1:] {
		out = g.Add(out, t)
	}
	return out
}

// taBenchOp times fwd and (Sum-loss +) backward of fn over
// taWarmups discarded warmups + taRuns measured runs, zeroing leaf
// grads between iterations. If fn's output is not scalar, a Sum loss
// is appended — its forward (Accelerate) and its backward (one Full
// alloc of the output shape) are counted in the bwd segment; the real
// training chain has an equivalent upstream-grad tensor, so this is
// representative, not free.
func taBenchOp(warm, runs int, leaves []*g.Tensor, fn func() *g.Tensor) (fwdMs, bwdMs float64) {
	var fwds, bwds []float64
	for i := 0; i < warm+runs; i++ {
		for _, l := range leaves {
			l.ZeroGrad()
		}
		t0 := time.Now()
		y := fn()
		t1 := time.Now()
		if y.Size() != 1 {
			y = g.Sum(y)
		}
		y.Backward()
		t2 := time.Now()
		if i >= warm {
			fwds = append(fwds, taMs(t1.Sub(t0)))
			bwds = append(bwds, taMs(t2.Sub(t1)))
		}
		// Untimed: collect the dead graph so Metal-buffer finalizers can
		// release unified memory between iterations (Go GC feels no
		// pressure from MTLBuffer bytes). No-op cost for CPU-only runs.
		runtime.GC()
	}
	return taMedian(fwds), taMedian(bwds)
}

// ---------- the transformer block at Qwen3-0.6B geometry ----------

type taBlock struct {
	attnNorm *nn.RMSNorm
	qNorm    *nn.RMSNorm // per-head RMSNorm over head_dim (Qwen3 q_norm)
	kNorm    *nn.RMSNorm // per-head RMSNorm over head_dim (Qwen3 k_norm)
	ffnNorm  *nn.RMSNorm
	wq       *nn.Linear // 1024 → 2048
	wk       *nn.Linear // 1024 → 1024
	wv       *nn.Linear // 1024 → 1024
	wo       *nn.Linear // 2048 → 1024
	gate     *nn.Linear // 1024 → 3072
	up       *nn.Linear // 1024 → 3072
	down     *nn.Linear // 3072 → 1024
	rope     *nn.RoPE
}

func newTABlock(rng *rand.Rand, maxSeq int) *taBlock {
	b := &taBlock{
		attnNorm: nn.NewRMSNorm(taHidden),
		qNorm:    nn.NewRMSNorm(taHeadDim),
		kNorm:    nn.NewRMSNorm(taHeadDim),
		ffnNorm:  nn.NewRMSNorm(taHidden),
		wq:       nn.NewLinear(taHidden, taQDim),
		wk:       nn.NewLinear(taHidden, taKVDim),
		wv:       nn.NewLinear(taHidden, taKVDim),
		wo:       nn.NewLinear(taQDim, taHidden),
		gate:     nn.NewLinear(taHidden, taInter),
		up:       nn.NewLinear(taHidden, taInter),
		down:     nn.NewLinear(taInter, taHidden),
		rope:     nn.NewRoPE(taHeadDim, maxSeq, taRopeTheta, nn.RopeLlama),
	}
	// Deterministic seeded weights (plan §2.1a). Norm gammas stay at
	// their ones init (deterministic already); Linear biases stay zero
	// (Qwen3 has no biases — see divergence note).
	for _, l := range []*nn.Linear{b.wq, b.wk, b.wv, b.wo, b.gate, b.up, b.down} {
		taReseed(rng, l.Weight, 0.02)
	}
	return b
}

func (b *taBlock) parameters() []*g.Tensor {
	var ps []*g.Tensor
	ps = append(ps, b.attnNorm.Parameters()...)
	ps = append(ps, b.qNorm.Parameters()...)
	ps = append(ps, b.kNorm.Parameters()...)
	ps = append(ps, b.ffnNorm.Parameters()...)
	for _, l := range []*nn.Linear{b.wq, b.wk, b.wv, b.wo, b.gate, b.up, b.down} {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}

// toMetal moves every block weight into Metal unified memory (plan
// 0009 X1: the X1K1 bench runs with weights and activations resident).
func (b *taBlock) toMetal(dev *metal.Device) {
	for _, l := range []*nn.Linear{b.wq, b.wk, b.wv, b.wo, b.gate, b.up, b.down} {
		l.ToMetal(dev)
	}
	for _, rn := range []*nn.RMSNorm{b.attnNorm, b.qNorm, b.kNorm, b.ffnNorm} {
		rn.Weight.ToMetal(dev)
	}
}

// forward runs one pre-norm transformer block: x + Attn(RMSNorm(x)),
// then h + SwiGLU-FFN(RMSNorm(h)). Attention mirrors nn/gqa.go op for
// op at the real Qwen3 projection dims and with Qwen3's q/k per-head
// RMSNorm. fused=false reproduces the exact X0 op chain (Full+Mul
// scale, tiled bool mask + MaskFill, Softmax on the (heads*seq, seq)
// view) for baseline canary re-runs; fused=true uses the plan-0009-K1
// g.CausalSoftmax fusion, matching post-X1K1 nn/gqa.go.
func (b *taBlock) forward(x *g.Tensor, fused bool) *g.Tensor {
	seq := x.Shape()[0]

	// --- attention ---
	xn := b.attnNorm.Forward(x)
	q := b.wq.Forward(xn) // (seq, 2048)
	k := b.wk.Forward(xn) // (seq, 1024)
	v := b.wv.Forward(xn) // (seq, 1024)

	qH := g.Permute(q.Reshape(seq, taQHeads, taHeadDim), []int{1, 0, 2})  // (16, seq, 128)
	kH := g.Permute(k.Reshape(seq, taKVHeads, taHeadDim), []int{1, 0, 2}) // (8, seq, 128)
	vH := g.Permute(v.Reshape(seq, taKVHeads, taHeadDim), []int{1, 0, 2}) // (8, seq, 128)

	// Qwen3 q_norm/k_norm: RMSNorm over head_dim, per (head, position)
	// row — identical math on the flattened 2-D view.
	qH = b.qNorm.Forward(qH.Reshape(taQHeads*seq, taHeadDim)).Reshape(taQHeads, seq, taHeadDim)
	kH = b.kNorm.Forward(kH.Reshape(taKVHeads*seq, taHeadDim)).Reshape(taKVHeads, seq, taHeadDim)

	qH = b.rope.Apply(qH, 0)
	kH = b.rope.Apply(kH, 0)

	// GQA expansion 8 KV heads → 16 Q heads (groupSize 2), the
	// nn/gqa.go RepeatInterleave pattern.
	group := taQHeads / taKVHeads
	kRep := g.RepeatInterleave(kH.Reshape(taKVHeads, 1, seq*taHeadDim), group).Reshape(taQHeads, seq, taHeadDim)
	vRep := g.RepeatInterleave(vH.Reshape(taKVHeads, 1, seq*taHeadDim), group).Reshape(taQHeads, seq, taHeadDim)

	scores := g.BatchedMatMulTransB(qH, kRep, taQHeads, seq, seq, taHeadDim) // (16, seq, seq)
	invScale := float32(1.0 / 11.313708498984761)                            // 1/sqrt(128)

	var soft *g.Tensor
	if fused {
		// K1 fused scale+mask+softmax (one op, one output tensor).
		soft = g.CausalSoftmax(scores, taQHeads, seq, invScale)
	} else {
		// X0 chain, op for op (Full+Mul scale, tiled bool mask +
		// MaskFill, Softmax on the flat view) — the baseline canary.
		scaleVec := g.Full(invScale, scores.Shape()...)
		scaled := g.Mul(scores, scaleVec)
		flat := scaled.Reshape(taQHeads*seq, seq)
		baseMask := g.CausalMask(seq)
		fullMask := make([]bool, taQHeads*seq*seq)
		for h := 0; h < taQHeads; h++ {
			copy(fullMask[h*seq*seq:(h+1)*seq*seq], baseMask)
		}
		masked := g.MaskFill(flat, fullMask, -1e9)
		soft = g.Softmax(masked).Reshape(taQHeads, seq, seq)
	}
	attn := g.BatchedMatMul(soft, vRep, taQHeads, seq, taHeadDim, seq) // (16, seq, 128)
	concat := g.Permute(attn, []int{1, 0, 2}).Reshape(seq, taQDim)     // (seq, 2048)
	attnOut := b.wo.Forward(concat)                                    // (seq, 1024)

	h := g.Add(x, attnOut)

	// --- SwiGLU FFN ---
	hn := b.ffnNorm.Forward(h)
	gateOut := b.gate.Forward(hn) // (seq, 3072)
	upOut := b.up.Forward(hn)     // (seq, 3072)
	act := g.SwiGLU(gateOut, upOut)
	ffnOut := b.down.Forward(act) // (seq, 1024)

	return g.Add(h, ffnOut)
}

// ---------- result schema (mirrors metal_crossover_results.json) ----------

type taBlockRow struct {
	Seq                 int     `json:"seq"`
	Warmups             int     `json:"warmups"`
	Runs                int     `json:"runs"`
	FwdMs               float64 `json:"fwd_ms"`
	BwdMs               float64 `json:"bwd_ms"`
	OptMs               float64 `json:"opt_ms"`
	TotalMs             float64 `json:"total_ms"`
	TotalMinMs          float64 `json:"total_min_ms"`
	TotalMaxMs          float64 `json:"total_max_ms"`
	AllocPerStepMB      float64 `json:"alloc_per_step_mb"`
	LiveGraphAfterFwdMB float64 `json:"live_graph_after_fwd_mb"`
	Note                string  `json:"note,omitempty"`
}

type taTailRow struct {
	Name    string  `json:"name"`
	Seq     int     `json:"seq"`
	Shape   string  `json:"shape"`
	FwdMs   float64 `json:"fwd_ms"`
	BwdMs   float64 `json:"bwd_ms"`
	TotalMs float64 `json:"total_ms"`
	Note    string  `json:"note,omitempty"`
}

type taFullStepRow struct {
	Seq              int     `json:"seq"`
	BlockMs          float64 `json:"block_ms"`
	Blocks28Ms       float64 `json:"blocks_28_ms"`
	EmbeddingMs      float64 `json:"embedding_ms"`
	LmHeadMs         float64 `json:"lm_head_ms"`
	LossMs           float64 `json:"loss_ms"`
	OptimizerTableMs float64 `json:"optimizer_table_ms"`
	TotalMs          float64 `json:"total_ms"`
	TotalS           float64 `json:"total_s"`
}

type taOpRow struct {
	Op           string  `json:"op"`
	Seq          int     `json:"seq"`
	Shape        string  `json:"shape"`
	FwdMs        float64 `json:"fwd_ms"`
	BwdMs        float64 `json:"bwd_ms"`
	CountPerStep int     `json:"count_per_step"`
	PerStepMs    float64 `json:"per_step_ms"`
	SharePct     float64 `json:"share_pct"`
}

type taMemoryRow struct {
	Seq                  int     `json:"seq"`
	LiveGraphBlockGB     float64 `json:"live_graph_after_fwd_gb_one_block"`
	Extrapolated28GB     float64 `json:"extrapolated_28_layer_graph_gb"`
	WeightsF32GB         float64 `json:"weights_f32_gb"`
	FitsIn24GBUnifiedMem bool    `json:"fits_24_gb"`
	Note                 string  `json:"note"`
}

type taPhase struct {
	Phase            string           `json:"phase"`
	Date             string           `json:"date"`
	Machine          string           `json:"machine"`
	MemoryGB         int              `json:"memory_gb"`
	GoVersion        string           `json:"go_version"`
	LoadAvgAtStart   string           `json:"load_avg_at_start"`
	MetalInitialized bool             `json:"metal_initialized"`
	Geometry         map[string]any   `json:"geometry"`
	BlockStepResults []taBlockRow     `json:"block_step_results"`
	CanaryResults    []taBlockRow     `json:"baseline_canary_results,omitempty"`
	TailResults      []taTailRow      `json:"tail_results"`
	FullStepEstimate []taFullStepRow  `json:"full_step_estimate"`
	PerOpResults     []taOpRow        `json:"per_op_results"`
	MemoryResults    []taMemoryRow    `json:"memory_results"`
	DispatchCounts   map[string]int64 `json:"dispatch_counts,omitempty"`
	PprofTop10       []string         `json:"pprof_top10"`
	MeasuredRanking  []string         `json:"measured_ranking_seq1500"`
	Notes            []string         `json:"notes"`
}

type taResultsFile struct {
	Hardware string    `json:"hardware"`
	Phases   []taPhase `json:"phases"`
}

// ---------- the harness ----------

func TestTrainAccelBench(t *testing.T) {
	smoke := os.Getenv("TA_SMOKE") != ""
	seqs := []int{512, 1024, 1500}
	perOpSeqs := []int{1024, 1500}
	warm, runs := taWarmups, taRuns
	if smoke {
		seqs = []int{64}
		perOpSeqs = []int{64}
		warm, runs = 1, 2
	}
	maxSeq := seqs[len(seqs)-1] + 1

	machine := taSysctl("machdep.cpu.brand_string")
	loadAvg := taSysctl("vm.loadavg")
	t.Logf("machine=%s load=%s", machine, loadAvg)

	phase := taPhase{
		Phase:            "X0-baseline",
		Date:             time.Now().Format("2006-01-02"),
		Machine:          machine,
		MemoryGB:         24,
		GoVersion:        runtime.Version(),
		LoadAvgAtStart:   loadAvg,
		MetalInitialized: false, // baseline = today's CPU f32 training path; no tensor is Metal-resident
		Geometry: map[string]any{
			"hidden": taHidden, "q_heads": taQHeads, "kv_heads": taKVHeads,
			"head_dim": taHeadDim, "q_dim": taQDim, "kv_dim": taKVDim,
			"ffn_inter": taInter, "layers": taLayers,
			"vocab": taVocab, "base_vocab": taBaseVocab, "mimi_vocab": taMimiVocab,
			"rope_theta": taRopeTheta, "rms_norm_eps": 1e-6,
			"note": "head_dim 128 != hidden/q_heads (64); q/k/v/o projection dims follow the head config per Qwen3-0.6B config.json: q 1024->2048, k/v 1024->1024, o 2048->1024",
		},
		Notes: []string{
			"single-sequence step (no batching), matching model/finetune.go's one-sequence-at-a-time loop",
			"block includes Qwen3 q_norm/k_norm; nn.Linear biases included (Qwen3 has none) — see harness file comment for divergences",
			"per-op rows are isolated fwd+bwd micro-benches at exact workload shapes; bwd segment includes the Sum-loss seed (representative of the upstream grad tensor in the real chain)",
			"full_step_estimate = 28*block_total + embedding + lm_head + cross_entropy + adamw_172m_table (per plan §2.1)",
			"share_pct is relative to the sum of per_op per_step_ms at that seq",
		},
	}

	rng := rand.New(rand.NewSource(taSeed))
	block := newTABlock(rng, maxSeq)
	params := block.parameters()
	blockParams := 0
	for _, p := range params {
		blockParams += p.Size()
	}
	phase.Geometry["block_params"] = blockParams
	phase.Geometry["embedding_params"] = taVocab * taHidden

	// ---- (a) single-block training step per seq ----
	for _, seq := range seqs {
		row := taBlockStepBench(t, block, rng, seq, warm, runs)
		phase.BlockStepResults = append(phase.BlockStepResults, row)
		t.Logf("block step seq=%d: fwd=%.1fms bwd=%.1fms opt=%.1fms total=%.1fms (alloc/step %.0f MB, live graph after fwd %.0f MB)",
			seq, row.FwdMs, row.BwdMs, row.OptMs, row.TotalMs, row.AllocPerStepMB, row.LiveGraphAfterFwdMB)

		// Memory extrapolation to the 28-layer graph (plan §2.2: the
		// full graph at seq 1500 is predicted not to fit 24 GB; the
		// harness measures one block and extrapolates rather than
		// swapping the machine to death).
		liveGB := row.LiveGraphAfterFwdMB / 1024
		weightsGB := float64(taBaseVocab+taMimiVocab)*taHidden*4/1e9 + float64(taLayers*blockParams)*4/1e9
		extrap := liveGB*taLayers + weightsGB
		phase.MemoryResults = append(phase.MemoryResults, taMemoryRow{
			Seq:                  seq,
			LiveGraphBlockGB:     liveGB,
			Extrapolated28GB:     extrap,
			WeightsF32GB:         weightsGB,
			FitsIn24GBUnifiedMem: extrap < 22, // leave ~2 GB headroom for OS
			Note:                 "extrapolated_28_layer_graph_gb = 28 x measured live autograd graph of one block (held after forward, post-GC) + f32 weights; excludes optimizer moments and backward transients",
		})
	}

	// ---- (b) tails per seq + the 172M-param AdamW table ----
	for _, seq := range seqs {
		tails := taTailBench(t, rng, seq, warm, runs)
		phase.TailResults = append(phase.TailResults, tails...)
	}
	adamwTail := taAdamWTableBench(t, rng, warm, runs)
	phase.TailResults = append(phase.TailResults, adamwTail)

	// ---- (c) full-step estimate = 28×block + tails ----
	for _, seq := range seqs {
		var blockMs float64
		for _, r := range phase.BlockStepResults {
			if r.Seq == seq {
				blockMs = r.TotalMs
			}
		}
		est := taFullStepRow{Seq: seq, BlockMs: blockMs, Blocks28Ms: blockMs * taLayers, OptimizerTableMs: adamwTail.TotalMs}
		for _, tr := range phase.TailResults {
			if tr.Seq != seq {
				continue
			}
			switch tr.Name {
			case "embedding_fwd_bwd":
				est.EmbeddingMs = tr.TotalMs
			case "lm_head_matmul_fwd_bwd":
				est.LmHeadMs = tr.TotalMs
			case "cross_entropy_fwd_bwd":
				est.LossMs = tr.TotalMs
			}
		}
		est.TotalMs = est.Blocks28Ms + est.EmbeddingMs + est.LmHeadMs + est.LossMs + est.OptimizerTableMs
		est.TotalS = est.TotalMs / 1000
		phase.FullStepEstimate = append(phase.FullStepEstimate, est)
		t.Logf("full-step estimate seq=%d: 28×block=%.0fms embed=%.0fms lm_head=%.0fms ce=%.0fms adamw=%.0fms → %.2fs",
			seq, est.Blocks28Ms, est.EmbeddingMs, est.LmHeadMs, est.LossMs, est.OptimizerTableMs, est.TotalS)
	}

	// ---- (d) per-op-class breakdown ----
	for _, seq := range perOpSeqs {
		ops := taPerOpBench(t, block, rng, seq, warm, runs)
		// Fold in the tails + optimizer as per-step rows so the table
		// covers the whole step.
		for _, tr := range phase.TailResults {
			if tr.Seq != seq && tr.Name != "adamw_172m_table" {
				continue
			}
			name := map[string]string{
				"embedding_fwd_bwd":      "embedding",
				"lm_head_matmul_fwd_bwd": "matmul_lm_head",
				"cross_entropy_fwd_bwd":  "loss_cross_entropy",
				"adamw_172m_table":       "optimizer_adamw_table",
			}[tr.Name]
			if name == "" {
				continue
			}
			ops = append(ops, taOpRow{Op: name, Seq: seq, Shape: tr.Shape,
				FwdMs: tr.FwdMs, BwdMs: tr.BwdMs, CountPerStep: 1, PerStepMs: tr.TotalMs})
		}
		for _, r := range phase.BlockStepResults {
			if r.Seq == seq {
				ops = append(ops, taOpRow{Op: "optimizer_adamw_block", Seq: seq,
					Shape:        fmt.Sprintf("%d params x 28", blockParams),
					CountPerStep: taLayers, PerStepMs: r.OptMs * taLayers})
			}
		}
		var total float64
		for _, o := range ops {
			total += o.PerStepMs
		}
		for i := range ops {
			ops[i].SharePct = ops[i].PerStepMs / total * 100
		}
		sort.Slice(ops, func(i, j int) bool { return ops[i].PerStepMs > ops[j].PerStepMs })
		phase.PerOpResults = append(phase.PerOpResults, ops...)
		for _, o := range ops {
			t.Logf("per-op seq=%d %-24s %8.1f ms/step (%5.1f%%)  [fwd %.1f + bwd %.1f ms ×%d]",
				seq, o.Op, o.PerStepMs, o.SharePct, o.FwdMs, o.BwdMs, o.CountPerStep)
		}
		if seq == 1500 || (smoke && seq == perOpSeqs[len(perOpSeqs)-1]) {
			for i, o := range ops {
				if i >= 5 {
					break
				}
				phase.MeasuredRanking = append(phase.MeasuredRanking,
					fmt.Sprintf("%d. %s %.0f ms/step (%.1f%%)", i+1, o.Op, o.PerStepMs, o.SharePct))
			}
		}
	}

	// ---- (2) pprof CPU profile of block steps at seq 1024 ----
	pprofSeq := 1024
	if smoke {
		pprofSeq = seqs[0]
	}
	phase.PprofTop10 = taCaptureProfile(t, block, rng, pprofSeq)

	if smoke {
		t.Log("TA_SMOKE set — skipping doc/training_accel_results.json write")
		return
	}

	// ---- (e) write/append the results JSON ----
	outPath := "../doc/training_accel_results.json"
	var file taResultsFile
	if raw, err := os.ReadFile(outPath); err == nil {
		if err := json.Unmarshal(raw, &file); err != nil {
			t.Fatalf("existing %s is not valid JSON: %v", outPath, err)
		}
	}
	file.Hardware = machine
	file.Phases = append(file.Phases, phase)
	raw, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		t.Fatalf("marshal results: %v", err)
	}
	if err := os.WriteFile(outPath, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", outPath, err)
	}
	t.Logf("results appended to %s (phase %s)", outPath, phase.Phase)
}

// taBlockStepBench measures one full block training step (forward +
// Sum loss + Backward + AdamW step) at the given seq.
func taBlockStepBench(t *testing.T, block *taBlock, rng *rand.Rand, seq, warm, runs int) taBlockRow {
	t.Helper()
	x := taSeedTensor(rng, 1.0, seq, taHidden)
	x.SetRequiresGrad(true) // embeddings are trainable → grad flows into the block input
	opt := optim.NewAdamW(block.parameters(), 1e-4, 0.01)

	var fwds, bwds, opts, totals, allocs []float64
	for i := 0; i < warm+runs; i++ {
		opt.ZeroGrad()
		x.ZeroGrad()
		var m0 runtime.MemStats
		runtime.ReadMemStats(&m0)
		t0 := time.Now()
		out := block.forward(x, false)
		t1 := time.Now()
		loss := g.Sum(out)
		loss.Backward()
		t2 := time.Now()
		opt.Step()
		t3 := time.Now()
		var m1 runtime.MemStats
		runtime.ReadMemStats(&m1)
		if i >= warm {
			fwds = append(fwds, taMs(t1.Sub(t0)))
			bwds = append(bwds, taMs(t2.Sub(t1)))
			opts = append(opts, taMs(t3.Sub(t2)))
			totals = append(totals, taMs(t3.Sub(t0)))
			allocs = append(allocs, float64(m1.TotalAlloc-m0.TotalAlloc)/1e6)
		}
	}

	// Live-graph footprint: hold the forward graph, GC, measure heap.
	runtime.GC()
	var h0 runtime.MemStats
	runtime.ReadMemStats(&h0)
	out := block.forward(x, false)
	runtime.GC()
	var h1 runtime.MemStats
	runtime.ReadMemStats(&h1)
	live := (float64(h1.HeapAlloc) - float64(h0.HeapAlloc)) / 1e6
	if live < 0 {
		live = 0
	}
	runtime.KeepAlive(out)
	runtime.GC()

	return taBlockRow{
		Seq: seq, Warmups: warm, Runs: runs,
		FwdMs: taMedian(fwds), BwdMs: taMedian(bwds), OptMs: taMedian(opts),
		TotalMs: taMedian(totals), TotalMinMs: taMin(totals), TotalMaxMs: taMax(totals),
		AllocPerStepMB: taMedian(allocs), LiveGraphAfterFwdMB: live,
	}
}

// taTailBench measures the non-per-layer tails at one seq: embedding
// fwd+bwd, lm_head matmul fwd+bwd, CrossEntropyLoss fwd+bwd.
func taTailBench(t *testing.T, rng *rand.Rand, seq, warm, runs int) []taTailRow {
	t.Helper()
	var rows []taTailRow

	// Embedding lookup over the 168320×1024 table. Backward allocates
	// the dense (vocab, hidden) grad and scatters — the plan §2.2
	// suspected hotspot.
	table := taSeedTensor(rng, 0.02, taVocab, taHidden)
	table.SetRequiresGrad(true)
	ids := make([]int, seq)
	for i := range ids {
		ids[i] = rng.Intn(taVocab)
	}
	fwd, bwd := taBenchOp(warm, runs, []*g.Tensor{table}, func() *g.Tensor {
		return g.EmbeddingLookup(table, ids)
	})
	rows = append(rows, taTailRow{Name: "embedding_fwd_bwd", Seq: seq,
		Shape: fmt.Sprintf("(%d ids) into (%d,%d)", seq, taVocab, taHidden),
		FwdMs: fwd, BwdMs: bwd, TotalMs: fwd + bwd,
		Note: "backward allocates + scatters a dense (vocab, hidden) f32 grad (~690 MB)"})
	table = nil
	runtime.GC()

	// lm_head: (seq,1024) @ (1024,168320) fwd+bwd via MatMul autograd.
	xl := taSeedTensor(rng, 1.0, seq, taHidden)
	xl.SetRequiresGrad(true)
	wHead := taSeedTensor(rng, 0.02, taHidden, taVocab)
	wHead.SetRequiresGrad(true)
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{xl, wHead}, func() *g.Tensor {
		return g.MatMul(xl, wHead)
	})
	rows = append(rows, taTailRow{Name: "lm_head_matmul_fwd_bwd", Seq: seq,
		Shape: fmt.Sprintf("(%d,%d)@(%d,%d)", seq, taHidden, taHidden, taVocab),
		FwdMs: fwd, BwdMs: bwd, TotalMs: fwd + bwd,
		Note: "bwd includes the dense (seq, vocab) upstream-grad tensor (Sum seed), as CE backward would produce"})
	xl, wHead = nil, nil
	runtime.GC()

	// CrossEntropyLoss over (seq, 168320) — fwd LogSoftmax, bwd
	// recomputes Softmax (the loss.go double-softmax the plan calls out).
	logits := taSeedTensor(rng, 1.0, seq, taVocab)
	logits.SetRequiresGrad(true)
	tgt := g.Zeros(seq, 1)
	for i := 0; i < seq; i++ {
		tgt.Data()[i] = float32(rng.Intn(taVocab))
	}
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{logits}, func() *g.Tensor {
		return g.CrossEntropyLoss(logits, tgt)
	})
	rows = append(rows, taTailRow{Name: "cross_entropy_fwd_bwd", Seq: seq,
		Shape: fmt.Sprintf("(%d,%d)", seq, taVocab),
		FwdMs: fwd, BwdMs: bwd, TotalMs: fwd + bwd,
		Note: "backward recomputes Softmax over the full logits (loss.go)"})
	logits = nil
	runtime.GC()

	return rows
}

// taAdamWTableBench measures one AdamW step over a 172M-param
// (vocab, hidden) table with a dense grad — the trainable-embedding
// worst case from plan §2.2.
func taAdamWTableBench(t *testing.T, rng *rand.Rand, warm, runs int) taTailRow {
	t.Helper()
	p := taSeedTensor(rng, 0.02, taVocab, taHidden)
	p.SetRequiresGrad(true)
	g.Sum(p).Backward() // dense all-ones grad; only the step is timed
	opt := optim.NewAdamW([]*g.Tensor{p}, 1e-4, 0.01)

	var steps []float64
	for i := 0; i < warm+runs; i++ {
		t0 := time.Now()
		opt.Step()
		t1 := time.Now()
		if i >= warm {
			steps = append(steps, taMs(t1.Sub(t0)))
		}
	}
	med := taMedian(steps)
	p = nil
	runtime.GC()
	return taTailRow{Name: "adamw_172m_table", Seq: 0,
		Shape: fmt.Sprintf("(%d,%d) = %dM params", taVocab, taHidden, taVocab*taHidden/1_000_000),
		FwdMs: 0, BwdMs: 0, TotalMs: med,
		Note: "scalar Go loop with math.Sqrt per element (optim/adamw.go)"}
}

// taPerOpBench measures isolated fwd+bwd per op class at exact
// per-block workload shapes; count_per_step = 28 (one instance per
// layer, with each class fn covering ALL instances of that class in
// one block).
func taPerOpBench(t *testing.T, block *taBlock, rng *rand.Rand, seq, warm, runs int) []taOpRow {
	t.Helper()
	var rows []taOpRow
	add := func(op, shape string, fwd, bwd float64) {
		rows = append(rows, taOpRow{Op: op, Seq: seq, Shape: shape,
			FwdMs: fwd, BwdMs: bwd, CountPerStep: taLayers, PerStepMs: (fwd + bwd) * taLayers})
	}

	// matmul_attn_proj: Wq, Wk, Wv on (seq,1024); Wo on (seq,2048).
	xn := taSeedTensor(rng, 1.0, seq, taHidden)
	xn.SetRequiresGrad(true)
	xo := taSeedTensor(rng, 1.0, seq, taQDim)
	xo.SetRequiresGrad(true)
	leaves := append([]*g.Tensor{xn, xo}, block.parameters()...)
	fwd, bwd := taBenchOp(warm, runs, leaves, func() *g.Tensor {
		return taSumScalars(
			g.Sum(block.wq.Forward(xn)), g.Sum(block.wk.Forward(xn)),
			g.Sum(block.wv.Forward(xn)), g.Sum(block.wo.Forward(xo)))
	})
	add("matmul_attn_proj", fmt.Sprintf("q/k/v (%d,1024)->2048/1024/1024, o (%d,2048)->1024", seq, seq), fwd, bwd)

	// matmul_ffn_proj: gate+up on (seq,1024), down on (seq,3072).
	xa := taSeedTensor(rng, 1.0, seq, taInter)
	xa.SetRequiresGrad(true)
	leaves = append([]*g.Tensor{xn, xa}, block.parameters()...)
	fwd, bwd = taBenchOp(warm, runs, leaves, func() *g.Tensor {
		return taSumScalars(
			g.Sum(block.gate.Forward(xn)), g.Sum(block.up.Forward(xn)),
			g.Sum(block.down.Forward(xa)))
	})
	add("matmul_ffn_proj", fmt.Sprintf("gate/up (%d,1024)->3072, down (%d,3072)->1024", seq, seq), fwd, bwd)

	// matmul_attn_batched: scores QK^T + attn@V.
	qH := taSeedTensor(rng, 1.0, taQHeads, seq, taHeadDim)
	qH.SetRequiresGrad(true)
	kH := taSeedTensor(rng, 1.0, taQHeads, seq, taHeadDim)
	kH.SetRequiresGrad(true)
	vH := taSeedTensor(rng, 1.0, taQHeads, seq, taHeadDim)
	vH.SetRequiresGrad(true)
	pa := taSeedTensor(rng, 0.01, taQHeads, seq, seq)
	pa.SetRequiresGrad(true)
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{qH, kH, vH, pa}, func() *g.Tensor {
		return taSumScalars(
			g.Sum(g.BatchedMatMulTransB(qH, kH, taQHeads, seq, seq, taHeadDim)),
			g.Sum(g.BatchedMatMul(pa, vH, taQHeads, seq, taHeadDim, seq)))
	})
	add("matmul_attn_batched", fmt.Sprintf("(16,%d,128)x(16,%d,128)^T + (16,%d,%d)x(16,%d,128)", seq, seq, seq, seq, seq), fwd, bwd)

	// softmax over the (16*seq, seq) attention view.
	sm := taSeedTensor(rng, 1.0, taQHeads*seq, seq)
	sm.SetRequiresGrad(true)
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{sm}, func() *g.Tensor {
		return g.Softmax(sm)
	})
	add("softmax", fmt.Sprintf("(%d,%d)", taQHeads*seq, seq), fwd, bwd)

	// mask_scale: Full+Mul scale, mask build+tile, MaskFill (gqa.go pattern).
	sc := taSeedTensor(rng, 1.0, taQHeads, seq, seq)
	sc.SetRequiresGrad(true)
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{sc}, func() *g.Tensor {
		scaleVec := g.Full(0.0883883476, sc.Shape()...)
		scaled := g.Mul(sc, scaleVec)
		baseMask := g.CausalMask(seq)
		fullMask := make([]bool, taQHeads*seq*seq)
		for h := 0; h < taQHeads; h++ {
			copy(fullMask[h*seq*seq:(h+1)*seq*seq], baseMask)
		}
		return g.MaskFill(scaled.Reshape(taQHeads*seq, seq), fullMask, -1e9)
	})
	add("mask_scale", fmt.Sprintf("Full+Mul+MaskFill on (16,%d,%d)", seq, seq), fwd, bwd)

	// permute_reshape: the block's full permute/repeat traffic —
	// 3 head-split permutes, 2 KV RepeatInterleaves, 1 output permute.
	q2 := taSeedTensor(rng, 1.0, seq, taQDim)
	q2.SetRequiresGrad(true)
	k2 := taSeedTensor(rng, 1.0, seq, taKVDim)
	k2.SetRequiresGrad(true)
	v2 := taSeedTensor(rng, 1.0, seq, taKVDim)
	v2.SetRequiresGrad(true)
	ao := taSeedTensor(rng, 1.0, taQHeads, seq, taHeadDim)
	ao.SetRequiresGrad(true)
	group := taQHeads / taKVHeads
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{q2, k2, v2, ao}, func() *g.Tensor {
		qh := g.Permute(q2.Reshape(seq, taQHeads, taHeadDim), []int{1, 0, 2})
		kh := g.Permute(k2.Reshape(seq, taKVHeads, taHeadDim), []int{1, 0, 2})
		vh := g.Permute(v2.Reshape(seq, taKVHeads, taHeadDim), []int{1, 0, 2})
		kr := g.RepeatInterleave(kh.Reshape(taKVHeads, 1, seq*taHeadDim), group)
		vr := g.RepeatInterleave(vh.Reshape(taKVHeads, 1, seq*taHeadDim), group)
		cc := g.Permute(ao, []int{1, 0, 2}).Reshape(seq, taQDim)
		return taSumScalars(g.Sum(qh), g.Sum(kr), g.Sum(vr), g.Sum(cc))
	})
	add("permute_reshape", fmt.Sprintf("3 head-splits + 2 kv-repeats + 1 concat at seq %d", seq), fwd, bwd)

	// rmsnorm: 2×(seq,1024) block norms + q_norm (16*seq,128) + k_norm (8*seq,128).
	nx := taSeedTensor(rng, 1.0, seq, taHidden)
	nx.SetRequiresGrad(true)
	nq := taSeedTensor(rng, 1.0, taQHeads*seq, taHeadDim)
	nq.SetRequiresGrad(true)
	nk := taSeedTensor(rng, 1.0, taKVHeads*seq, taHeadDim)
	nk.SetRequiresGrad(true)
	leaves = append([]*g.Tensor{nx, nq, nk}, block.parameters()...)
	fwd, bwd = taBenchOp(warm, runs, leaves, func() *g.Tensor {
		return taSumScalars(
			g.Sum(block.attnNorm.Forward(nx)), g.Sum(block.ffnNorm.Forward(nx)),
			g.Sum(block.qNorm.Forward(nq)), g.Sum(block.kNorm.Forward(nk)))
	})
	add("rmsnorm", fmt.Sprintf("2x(%d,1024) + (%d,128) + (%d,128)", seq, taQHeads*seq, taKVHeads*seq), fwd, bwd)

	// activation: SwiGLU on (seq, 3072).
	ga := taSeedTensor(rng, 1.0, seq, taInter)
	ga.SetRequiresGrad(true)
	ua := taSeedTensor(rng, 1.0, seq, taInter)
	ua.SetRequiresGrad(true)
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{ga, ua}, func() *g.Tensor {
		return g.SwiGLU(ga, ua)
	})
	add("activation_swiglu", fmt.Sprintf("(%d,%d)", seq, taInter), fwd, bwd)

	// rope: Q (16,seq,128) + K (8,seq,128).
	rq := taSeedTensor(rng, 1.0, taQHeads, seq, taHeadDim)
	rq.SetRequiresGrad(true)
	rk := taSeedTensor(rng, 1.0, taKVHeads, seq, taHeadDim)
	rk.SetRequiresGrad(true)
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{rq, rk}, func() *g.Tensor {
		return taSumScalars(g.Sum(block.rope.Apply(rq, 0)), g.Sum(block.rope.Apply(rk, 0)))
	})
	add("rope", fmt.Sprintf("(16,%d,128) + (8,%d,128)", seq, seq), fwd, bwd)

	runtime.GC()
	return rows
}

// taCaptureProfile records a runtime/pprof CPU profile of 3 block
// steps at the given seq and returns `go tool pprof -top` lines
// (deliverable 2 of plan X0).
func taCaptureProfile(t *testing.T, block *taBlock, rng *rand.Rand, seq int) []string {
	t.Helper()
	dir := os.Getenv("TRAIN_ACCEL_PPROF_OUT")
	if dir == "" {
		dir = os.TempDir()
	}
	path := filepath.Join(dir, fmt.Sprintf("train_accel_x0_block%d.pprof", seq))
	f, err := os.Create(path)
	if err != nil {
		t.Logf("pprof: cannot create %s: %v", path, err)
		return nil
	}
	defer f.Close()

	x := taSeedTensor(rng, 1.0, seq, taHidden)
	x.SetRequiresGrad(true)
	opt := optim.NewAdamW(block.parameters(), 1e-4, 0.01)
	// one unprofiled warmup
	opt.ZeroGrad()
	x.ZeroGrad()
	g.Sum(block.forward(x, false)).Backward()
	opt.Step()

	if err := pprof.StartCPUProfile(f); err != nil {
		t.Logf("pprof: start: %v", err)
		return nil
	}
	for i := 0; i < 3; i++ {
		opt.ZeroGrad()
		x.ZeroGrad()
		g.Sum(block.forward(x, false)).Backward()
		opt.Step()
	}
	pprof.StopCPUProfile()
	t.Logf("pprof: CPU profile of 3 block steps (seq %d) written to %s", seq, path)

	out, err := exec.Command("go", "tool", "pprof", "-top", "-nodecount=10", path).Output()
	if err != nil {
		t.Logf("pprof: go tool pprof failed (profile still on disk): %v", err)
		return []string{"profile at " + path + " (go tool pprof unavailable in test env)"}
	}
	var lines []string
	for _, ln := range strings.Split(string(out), "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		lines = append(lines, ln)
	}
	for _, ln := range lines {
		t.Logf("pprof: %s", ln)
	}
	return lines
}

// taSysctl returns `sysctl -n key` output, or "unknown".
func taSysctl(key string) string {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// ==================== Phase X1K1 (plan 0009 §3.2 + §3.3-K1) ====================
//
// GPU-resident autograd wiring + fused causal softmax. The bench runs
// the same block step with every weight and the input Metal-resident
// (fused=true → g.CausalSoftmax), plus a CPU baseline canary re-run
// per §2.1 (fused=false, weights on CPU — the exact X0 op chain).
//
// Revised gate (evidence-based, replaces the plan's X1-only ≥1.8×):
// block step at seq 1024 ≥2.0× vs the X0 baseline row; every fwd+bwd
// matmul above threshold dispatches MPS (asserted via
// g.ReadMetalDispatchCounts); seq-1500 memory extrapolation reported
// with Metal buffer bytes included (metal.LiveBufferBytes — Go's
// HeapAlloc cannot see MTLBuffer memory).

// taFlushGC runs GC cycles with short pauses so Metal buffer
// finalizers (async) release their buffers before a memory reading.
func taFlushGC() {
	for i := 0; i < 3; i++ {
		runtime.GC()
		time.Sleep(20 * time.Millisecond)
	}
	runtime.GC()
}

// taBlockStepBenchMetal is taBlockStepBench with weights+activations
// Metal-resident and the K1 fused softmax. The live-graph figure
// includes live Metal buffer bytes.
func taBlockStepBenchMetal(t *testing.T, block *taBlock, rng *rand.Rand, seq, warm, runs int) taBlockRow {
	t.Helper()
	dev := g.MetalDev()
	x := taSeedTensor(rng, 1.0, seq, taHidden).ToMetal(dev)
	x.SetRequiresGrad(true)
	opt := optim.NewAdamW(block.parameters(), 1e-4, 0.01)

	var fwds, bwds, opts, totals, allocs []float64
	for i := 0; i < warm+runs; i++ {
		opt.ZeroGrad()
		x.ZeroGrad()
		var m0 runtime.MemStats
		runtime.ReadMemStats(&m0)
		t0 := time.Now()
		out := block.forward(x, true)
		t1 := time.Now()
		loss := g.Sum(out)
		loss.Backward()
		t2 := time.Now()
		opt.Step()
		t3 := time.Now()
		var m1 runtime.MemStats
		runtime.ReadMemStats(&m1)
		if i >= warm {
			fwds = append(fwds, taMs(t1.Sub(t0)))
			bwds = append(bwds, taMs(t2.Sub(t1)))
			opts = append(opts, taMs(t3.Sub(t2)))
			totals = append(totals, taMs(t3.Sub(t0)))
			allocs = append(allocs, float64(m1.TotalAlloc-m0.TotalAlloc)/1e6)
		}
		// Untimed: release the dead step graph's Metal buffers. Go's GC
		// feels no pressure from MTLBuffer memory (it lives outside the
		// Go heap), so without this the loop accumulates multi-GB of
		// dead unified-memory buffers between collections.
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}

	// Live-graph footprint: hold the forward graph, flush finalizers,
	// measure Go heap + live Metal buffer bytes.
	taFlushGC()
	var h0 runtime.MemStats
	runtime.ReadMemStats(&h0)
	mb0 := metal.LiveBufferBytes()
	out := block.forward(x, true)
	taFlushGC()
	var h1 runtime.MemStats
	runtime.ReadMemStats(&h1)
	mb1 := metal.LiveBufferBytes()
	live := (float64(h1.HeapAlloc) - float64(h0.HeapAlloc) + float64(mb1-mb0)) / 1e6
	if live < 0 {
		live = 0
	}
	runtime.KeepAlive(out)
	taFlushGC()

	return taBlockRow{
		Seq: seq, Warmups: warm, Runs: runs,
		FwdMs: taMedian(fwds), BwdMs: taMedian(bwds), OptMs: taMedian(opts),
		TotalMs: taMedian(totals), TotalMinMs: taMin(totals), TotalMaxMs: taMax(totals),
		AllocPerStepMB: taMedian(allocs), LiveGraphAfterFwdMB: live,
		Note: "weights+activations Metal-resident, K1 fused causal softmax; live graph includes Metal buffer bytes; alloc/step is Go heap only",
	}
}

// taLmHeadBenchMetal measures the lm_head matmul fwd+bwd with x and W
// Metal-resident (258G-FMA shape at seq 1500 → MPS fwd+bwd).
func taLmHeadBenchMetal(t *testing.T, rng *rand.Rand, seq, warm, runs int) taTailRow {
	t.Helper()
	dev := g.MetalDev()
	xl := taSeedTensor(rng, 1.0, seq, taHidden).ToMetal(dev)
	xl.SetRequiresGrad(true)
	wHead := taSeedTensor(rng, 0.02, taHidden, taVocab).ToMetal(dev)
	wHead.SetRequiresGrad(true)
	fwd, bwd := taBenchOp(warm, runs, []*g.Tensor{xl, wHead}, func() *g.Tensor {
		return g.MatMul(xl, wHead)
	})
	taFlushGC()
	return taTailRow{Name: "lm_head_matmul_fwd_bwd", Seq: seq,
		Shape: fmt.Sprintf("(%d,%d)@(%d,%d)", seq, taHidden, taHidden, taVocab),
		FwdMs: fwd, BwdMs: bwd, TotalMs: fwd + bwd,
		Note: "x and W Metal-resident: fwd + both grads on MPS (Sum seed grad Metal-resident via fullLike)"}
}

// taX0Phase loads the X0-baseline phase from the results JSON.
func taX0Phase(t *testing.T, path string) *taPhase {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var file taResultsFile
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatalf("existing %s is not valid JSON: %v", path, err)
	}
	for i := range file.Phases {
		if file.Phases[i].Phase == "X0-baseline" {
			return &file.Phases[i]
		}
	}
	return nil
}

func TestTrainAccelBenchX1K1(t *testing.T) {
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	smoke := os.Getenv("TA_SMOKE") != ""
	seqs := []int{512, 1024, 1500}
	warm, runs := taWarmups, taRuns
	if smoke {
		seqs = []int{64}
		warm, runs = 1, 2
	}
	maxSeq := seqs[len(seqs)-1] + 1

	machine := taSysctl("machdep.cpu.brand_string")
	loadAvg := taSysctl("vm.loadavg")
	t.Logf("machine=%s load=%s", machine, loadAvg)

	outPath := "../doc/training_accel_results.json"
	x0 := taX0Phase(t, outPath)
	if x0 == nil && !smoke {
		t.Fatal("X0-baseline phase not found in results JSON — X1K1 speedups need the baseline")
	}
	x0Block := func(seq int) *taBlockRow {
		if x0 == nil {
			return nil
		}
		for i := range x0.BlockStepResults {
			if x0.BlockStepResults[i].Seq == seq {
				return &x0.BlockStepResults[i]
			}
		}
		return nil
	}

	phase := taPhase{
		Phase:            "X1K1",
		Date:             time.Now().Format("2006-01-02"),
		Machine:          machine,
		MemoryGB:         24,
		GoVersion:        runtime.Version(),
		LoadAvgAtStart:   loadAvg,
		MetalInitialized: true,
		Geometry: map[string]any{
			"hidden": taHidden, "q_heads": taQHeads, "kv_heads": taKVHeads,
			"head_dim": taHeadDim, "q_dim": taQDim, "kv_dim": taKVDim,
			"ffn_inter": taInter, "layers": taLayers,
			"vocab": taVocab, "base_vocab": taBaseVocab, "mimi_vocab": taMimiVocab,
			"rope_theta": taRopeTheta, "rms_norm_eps": 1e-6,
		},
		Notes: []string{
			"X1K1: weights+activations Metal-resident (residency propagation), batched matmul backward on MPS, Sum-loss grad seeded Metal-resident, K1 fused causal softmax kernels (softmax_causal_forward/softmax_backward)",
			"baseline_canary_results = same-session CPU re-run of the exact X0 op chain per plan §2.1 (drift >10% would invalidate cross-session comparison)",
			"live_graph_after_fwd includes live Metal buffer bytes (metal.LiveBufferBytes); Go HeapAlloc alone cannot see MTLBuffer memory",
			"tails: lm_head measured Metal-resident; cross-entropy and embedding remain CPU (K2/K3 out of scope for this phase)",
		},
	}

	// ---- (a) baseline canary: CPU, exact X0 op chain ----
	rngC := rand.New(rand.NewSource(taSeed))
	blockC := newTABlock(rngC, maxSeq)
	for _, seq := range seqs {
		row := taBlockStepBench(t, blockC, rngC, seq, warm, runs)
		row.Note = "CPU f32 canary (X0 op chain re-run under this session's load)"
		phase.CanaryResults = append(phase.CanaryResults, row)
		if base := x0Block(seq); base != nil {
			drift := (row.TotalMs - base.TotalMs) / base.TotalMs * 100
			note := fmt.Sprintf("canary seq=%d: %.0fms vs X0 %.0fms (drift %+.1f%%)", seq, row.TotalMs, base.TotalMs, drift)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
			if drift > 10 || drift < -10 {
				phase.Notes = append(phase.Notes, fmt.Sprintf("WARNING: canary drift at seq %d exceeds 10%% — machine load differs from X0 session; treat cross-session speedups with care", seq))
			}
		}
	}

	// ---- (b) X1K1 Metal-resident block step ----
	rng := rand.New(rand.NewSource(taSeed))
	block := newTABlock(rng, maxSeq)
	block.toMetal(g.MetalDev())
	for _, seq := range seqs {
		row := taBlockStepBenchMetal(t, block, rng, seq, warm, runs)
		phase.BlockStepResults = append(phase.BlockStepResults, row)
		msg := fmt.Sprintf("X1K1 block step seq=%d: fwd=%.1fms bwd=%.1fms opt=%.1fms total=%.1fms (live graph %.0f MB)",
			seq, row.FwdMs, row.BwdMs, row.OptMs, row.TotalMs, row.LiveGraphAfterFwdMB)
		t.Log(msg)
		if base := x0Block(seq); base != nil {
			speedup := base.TotalMs / row.TotalMs
			note := fmt.Sprintf("block-step speedup vs X0 at seq %d: %.2fx (%.0fms -> %.0fms)", seq, speedup, base.TotalMs, row.TotalMs)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
			// The ≥2.0× seq-1024 speed gate is a phase-acceptance
			// criterion recorded as a verdict, not a hard test failure —
			// the 2026-08-11 X1K1 session measured 1.06× vs the X0 row
			// under heavy external load (canary drift −37% at seq 1500,
			// formally invalidating cross-session comparison per §2.1);
			// the verdict lives in the JSON notes and plan §2.3. Hard
			// assertions here are the dispatch counters below and the
			// parity suites.
			if seq == 1024 && speedup < 2.0 {
				verdict := fmt.Sprintf("GATE VERDICT: block-step >=2.0x at seq 1024 NOT MET this session (%.2fx)", speedup)
				t.Log(verdict)
				phase.Notes = append(phase.Notes, verdict)
			}
		}
		// Same-session comparison (load-matched): X1K1 vs this session's
		// CPU canary of the exact X0 op chain.
		for _, cn := range phase.CanaryResults {
			if cn.Seq == seq {
				note := fmt.Sprintf("same-session speedup vs CPU canary at seq %d: %.2fx (%.0fms -> %.0fms)",
					seq, cn.TotalMs/row.TotalMs, cn.TotalMs, row.TotalMs)
				t.Log(note)
				phase.Notes = append(phase.Notes, note)
			}
		}

		// Memory extrapolation (28 layers + f32 weights), Metal-aware.
		liveGB := row.LiveGraphAfterFwdMB / 1024
		blockParams := 0
		for _, p := range block.parameters() {
			blockParams += p.Size()
		}
		weightsGB := float64(taVocab)*taHidden*4/1e9 + float64(taLayers*blockParams)*4/1e9
		extrap := liveGB*taLayers + weightsGB
		phase.MemoryResults = append(phase.MemoryResults, taMemoryRow{
			Seq:                  seq,
			LiveGraphBlockGB:     liveGB,
			Extrapolated28GB:     extrap,
			WeightsF32GB:         weightsGB,
			FitsIn24GBUnifiedMem: extrap < 22,
			Note:                 "as X0 but live graph includes Metal buffer bytes; K1 fusion eliminates the scale/mask/masked seq^2 intermediates (5 -> 2 live seq^2 tensors per layer)",
		})
	}

	// ---- (c) dispatch-counter gate at the largest non-smoke seq ----
	if !smoke {
		seq := 1024
		x := taSeedTensor(rng, 1.0, seq, taHidden).ToMetal(g.MetalDev())
		x.SetRequiresGrad(true)
		opt := optim.NewAdamW(block.parameters(), 1e-4, 0.01)
		opt.ZeroGrad()
		x.ZeroGrad()
		g.ResetMetalDispatchCounts()
		g.Sum(block.forward(x, true)).Backward()
		c := g.ReadMetalDispatchCounts()
		t.Logf("dispatch counts (1 block fwd+bwd, seq %d): matmul=%d batched=%d softmax=%d", seq, c.MatMul, c.BatchedMatMul, c.SoftmaxKernel)
		// 7 projection matmuls fwd (Wq,Wk,Wv,Wo,gate,up,down) + 14 bwd
		// (dx+dW each); 2 batched fwd (QK^T, attn@V) + 4 batched bwd;
		// 1 fused softmax fwd + 1 softmax bwd kernel.
		if c.MatMul < 21 {
			t.Errorf("X1K1 gate: expected >=21 MPS matmul dispatches in fwd+bwd, got %d", c.MatMul)
		}
		if c.BatchedMatMul < 6 {
			t.Errorf("X1K1 gate: expected >=6 MPS batched-matmul dispatches in fwd+bwd, got %d", c.BatchedMatMul)
		}
		if c.SoftmaxKernel < 2 {
			t.Errorf("X1K1 gate: expected >=2 fused-softmax kernel dispatches, got %d", c.SoftmaxKernel)
		}
		phase.DispatchCounts = map[string]int64{
			"block_fwd_bwd_seq1024_mps_matmul":         c.MatMul,
			"block_fwd_bwd_seq1024_mps_batched_matmul": c.BatchedMatMul,
			"block_fwd_bwd_seq1024_softmax_kernel":     c.SoftmaxKernel,
		}
		g.ResetMetalDispatchCounts()
		taFlushGC()
	}

	// ---- (d) tails: lm_head Metal-resident; embedding/CE/AdamW as X0 ----
	for _, seq := range seqs {
		tails := taTailBench(t, rng, seq, warm, runs)
		for i := range tails {
			if tails[i].Name == "lm_head_matmul_fwd_bwd" {
				tails[i] = taLmHeadBenchMetal(t, rng, seq, warm, runs)
			}
		}
		phase.TailResults = append(phase.TailResults, tails...)
		taFlushGC()
	}
	adamwTail := taAdamWTableBench(t, rng, warm, runs)
	phase.TailResults = append(phase.TailResults, adamwTail)

	// ---- (e) full-step estimate ----
	for _, seq := range seqs {
		var blockMs float64
		for _, r := range phase.BlockStepResults {
			if r.Seq == seq {
				blockMs = r.TotalMs
			}
		}
		est := taFullStepRow{Seq: seq, BlockMs: blockMs, Blocks28Ms: blockMs * taLayers, OptimizerTableMs: adamwTail.TotalMs}
		for _, tr := range phase.TailResults {
			if tr.Seq != seq {
				continue
			}
			switch tr.Name {
			case "embedding_fwd_bwd":
				est.EmbeddingMs = tr.TotalMs
			case "lm_head_matmul_fwd_bwd":
				est.LmHeadMs = tr.TotalMs
			case "cross_entropy_fwd_bwd":
				est.LossMs = tr.TotalMs
			}
		}
		est.TotalMs = est.Blocks28Ms + est.EmbeddingMs + est.LmHeadMs + est.LossMs + est.OptimizerTableMs
		est.TotalS = est.TotalMs / 1000
		phase.FullStepEstimate = append(phase.FullStepEstimate, est)
		t.Logf("X1K1 full-step estimate seq=%d: 28xblock=%.0fms embed=%.0fms lm_head=%.0fms ce=%.0fms adamw=%.0fms -> %.2fs",
			seq, est.Blocks28Ms, est.EmbeddingMs, est.LmHeadMs, est.LossMs, est.OptimizerTableMs, est.TotalS)
	}

	if smoke {
		t.Log("TA_SMOKE set — skipping doc/training_accel_results.json write")
		return
	}

	// ---- (f) append the phase row ----
	var file taResultsFile
	if raw, err := os.ReadFile(outPath); err == nil {
		if err := json.Unmarshal(raw, &file); err != nil {
			t.Fatalf("existing %s is not valid JSON: %v", outPath, err)
		}
	}
	file.Hardware = machine
	file.Phases = append(file.Phases, phase)
	raw, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		t.Fatalf("marshal results: %v", err)
	}
	if err := os.WriteFile(outPath, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", outPath, err)
	}
	t.Logf("results appended to %s (phase %s)", outPath, phase.Phase)
}
