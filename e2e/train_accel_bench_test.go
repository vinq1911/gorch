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

// freezeBase marks every Linear weight and bias non-trainable — the
// plan 0009 X3 frozen-path configuration (LoRA base frozen; the
// stand-in trainable set is the norm gammas plus the block input).
func (b *taBlock) freezeBase() {
	for _, l := range []*nn.Linear{b.wq, b.wk, b.wv, b.wo, b.gate, b.up, b.down} {
		l.Weight.SetRequiresGrad(false)
		l.Bias.SetRequiresGrad(false)
	}
}

// toMetalBF16 converts every Linear weight to bf16 storage and moves
// the block to Metal (plan 0009 X3-B2/B4): frozen weights bf16,
// biases and norm gammas f32. Linear.ToMetal is dtype-aware.
func (b *taBlock) toMetalBF16(dev *metal.Device) {
	for _, l := range []*nn.Linear{b.wq, b.wk, b.wv, b.wo, b.gate, b.up, b.down} {
		l.Weight = l.Weight.ToBF16() // fresh bf16 copy; requiresGrad=false
		l.ToMetal(dev)
	}
	for _, rn := range []*nn.RMSNorm{b.attnNorm, b.qNorm, b.kNorm, b.ffnNorm} {
		rn.Weight.ToMetal(dev)
	}
}

// trainableParameters filters parameters() by RequiresGrad — the
// optimizer set for the frozen-path modes.
func (b *taBlock) trainableParameters() []*g.Tensor {
	var ps []*g.Tensor
	for _, p := range b.parameters() {
		if p.RequiresGrad() {
			ps = append(ps, p)
		}
	}
	return ps
}

// linearWeightParams returns the frozen-set param count (the 7 Linear
// weights) — the bytes that move to bf16 in X3 memory accounting.
func (b *taBlock) linearWeightParams() int {
	n := 0
	for _, l := range []*nn.Linear{b.wq, b.wk, b.wv, b.wo, b.gate, b.up, b.down} {
		n += l.Weight.Size()
	}
	return n
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

// ==================== Phase X2 (plan 0009 §3.3 K2/K4/K7 + R6) ====================
//
// Second X2 wave: K7 vectorized AdamW (accelerate.AdamWStep), K4
// SiLU/SwiGLU Metal kernels + Accelerate fallback, K2 fused
// cross-entropy Metal kernels + vectorized CPU fallback, and the R6
// commit-without-wait async dispatch mode (g.SetMetalAsync). The bench
// measures the Metal block step in both sync and async dispatch modes
// (their delta IS the recovered per-op waitUntilCompleted cost), the
// Metal-resident CE tail, and the AdamW 172M table in scalar vs
// vectorized form (the K7 gate delta). Canary discipline per §2.1 —
// note that the canary CPU chain is no longer byte-identical to X0's:
// K4/K7 also sped up the *CPU* SwiGLU path (vectorized) while the
// canary forces the scalar AdamW; the residual difference is small
// (SwiGLU ≈5% of the CPU block step) and is flagged in the notes.

// taAdamWTableBenchScalar is taAdamWTableBench with the pre-K7 scalar
// Go loop forced — the K7 baseline side of the gate.
func taAdamWTableBenchScalar(t *testing.T, rng *rand.Rand, warm, runs int) taTailRow {
	t.Helper()
	prev := optim.UseScalarAdamW
	optim.UseScalarAdamW = true
	defer func() { optim.UseScalarAdamW = prev }()
	row := taAdamWTableBench(t, rng, warm, runs)
	row.Name = "adamw_172m_table_scalar"
	row.Note = "pre-K7 scalar Go loop forced via optim.UseScalarAdamW (K7 gate baseline)"
	return row
}

// taCEBenchMetal measures CrossEntropyLoss fwd+bwd with Metal-resident
// logits — the K2 fused-kernel path at the exact workload shape.
func taCEBenchMetal(t *testing.T, rng *rand.Rand, seq, warm, runs int) taTailRow {
	t.Helper()
	dev := g.MetalDev()
	logits := taSeedTensor(rng, 1.0, seq, taVocab).ToMetal(dev)
	logits.SetRequiresGrad(true)
	tgt := g.Zeros(seq, 1)
	for i := 0; i < seq; i++ {
		tgt.Data()[i] = float32(rng.Intn(taVocab))
	}
	fwd, bwd := taBenchOp(warm, runs, []*g.Tensor{logits}, func() *g.Tensor {
		return g.CrossEntropyLoss(logits, tgt)
	})
	taFlushGC()
	return taTailRow{Name: "cross_entropy_fwd_bwd", Seq: seq,
		Shape: fmt.Sprintf("(%d,%d)", seq, taVocab),
		FwdMs: fwd, BwdMs: bwd, TotalMs: fwd + bwd,
		Note: "K2 fused Metal kernels (logits Metal-resident): fwd logsumexp+target pick, bwd softmax-onehot from saved lse"}
}

// taDispatchOverheadBench measures the per-dispatch round-trip cost
// precisely (R6 deliverable): 200 tiny vec_mul kernel dispatches on
// 1k-element Metal buffers, sync mode (commit+wait each) vs async mode
// (commit all + one final wait). The per-dispatch difference is the
// pure waitUntilCompleted round trip that async mode recovers.
func taDispatchOverheadBench(t *testing.T) (syncPerDispatchMs, asyncPerDispatchMs float64) {
	t.Helper()
	dev := g.MetalDev()
	const n = 1024
	const iters = 200
	rng := rand.New(rand.NewSource(1))
	x := taSeedTensor(rng, 1.0, n).ToMetal(dev)
	y := taSeedTensor(rng, 1.0, n).ToMetal(dev)

	run := func() float64 {
		// warmup
		for i := 0; i < 10; i++ {
			_ = g.Mul(x, y) // both Metal-resident → vec_mul GPU dispatch
		}
		g.SyncMetal()
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			_ = g.Mul(x, y)
		}
		g.SyncMetal()
		return taMs(time.Since(t0)) / iters
	}

	syncPerDispatchMs = run()
	g.SetMetalAsync(true)
	asyncPerDispatchMs = run()
	g.SetMetalAsync(false)
	taFlushGC()
	return syncPerDispatchMs, asyncPerDispatchMs
}

// taX2PartialPath is the handoff file for the two-part X2 run (see
// TA_X2_PART below).
func taX2PartialPath() string {
	return filepath.Join(os.TempDir(), "ta_x2_partial_phase.json")
}

// TestTrainAccelBenchX2 supports TA_X2_PART=blocks|tails: the 2026-08-11
// bench session found the single full-length process reproducibly
// SIGKILLed at ~110s wall (isolated tails and a 150s CPU-burner both
// survive — cause undetermined, no jetsam/log trace; memory was 74%
// free). Splitting into two shorter processes (blocks ≈60s, tails ≈60s)
// with a temp-file phase handoff stays under the kill horizon. Unset
// runs everything in one process as originally written.
func TestTrainAccelBenchX2(t *testing.T) {
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	smoke := os.Getenv("TA_SMOKE") != ""
	part := os.Getenv("TA_X2_PART") // "", "blocks", "tails"
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
		t.Fatal("X0-baseline phase not found in results JSON — X2 speedups need the baseline")
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
	x0Full := func(seq int) *taFullStepRow {
		if x0 == nil {
			return nil
		}
		for i := range x0.FullStepEstimate {
			if x0.FullStepEstimate[i].Seq == seq {
				return &x0.FullStepEstimate[i]
			}
		}
		return nil
	}

	phase := taPhase{
		Phase:            "X2",
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
			"X2 second wave: K7 vectorized AdamW (accelerate.AdamWStep), K4 SiLU/SwiGLU Metal kernels + Accelerate CPU fallback, K2 fused cross-entropy Metal kernels + vectorized CPU fallback, R6 commit-without-wait async dispatch (g.SetMetalAsync) with wait-on-CPU-read fencing",
			"block_step_results carry TWO rows per seq: per-op sync dispatch and async dispatch (R6); the async row is the X2 configuration used in full_step_estimate, and the sync-vs-async delta is the recovered per-op waitUntilCompleted cost",
			"canary caveat: the CPU canary chain is NOT byte-identical to X0's code — K4 also vectorized the CPU SwiGLU (~5% of the CPU block step) and K2 vectorized the CPU CE fallback; the canary forces the pre-K7 scalar AdamW (optim.UseScalarAdamW) but cannot un-vectorize SwiGLU. Treat the canary as approximate load calibration",
			"tails: lm_head + cross-entropy measured Metal-resident (K2); a cross_entropy_fwd_bwd_cpu_vectorized row records the K2 CPU-fallback (acc_vexp) path; adamw_172m_table is the K7 vectorized step with adamw_172m_table_scalar as the pre-K7 baseline",
		},
	}

	rng := rand.New(rand.NewSource(taSeed))
	asyncRows := map[int]taBlockRow{}

	if part != "tails" {
		taX2BlocksPart(t, &phase, seqs, warm, runs, maxSeq, smoke, x0Block, asyncRows)
	}

	if part == "blocks" && !smoke {
		raw, err := json.MarshalIndent(phase, "", "  ")
		if err != nil {
			t.Fatalf("marshal partial phase: %v", err)
		}
		if err := os.WriteFile(taX2PartialPath(), append(raw, '\n'), 0o644); err != nil {
			t.Fatalf("write partial phase: %v", err)
		}
		t.Logf("TA_X2_PART=blocks: partial phase written to %s — run again with TA_X2_PART=tails", taX2PartialPath())
		return
	}
	if part == "tails" && !smoke {
		raw, err := os.ReadFile(taX2PartialPath())
		if err != nil {
			t.Fatalf("TA_X2_PART=tails needs the blocks part first: %v", err)
		}
		if err := json.Unmarshal(raw, &phase); err != nil {
			t.Fatalf("partial phase unmarshal: %v", err)
		}
		for _, r := range phase.BlockStepResults {
			if strings.Contains(r.Note, "async dispatch") {
				asyncRows[r.Seq] = r
			}
		}
		if len(asyncRows) == 0 {
			t.Fatal("partial phase has no async block rows")
		}
	}

	taX2TailsPart(t, &phase, rng, seqs, warm, runs, smoke, x0Block, x0Full, asyncRows, outPath, machine)
}

// taX2BlocksPart runs the canary, the sync/async Metal block steps,
// the dispatch-counter gates, and the R6 microbench (sections a–c of
// the X2 bench).
func taX2BlocksPart(t *testing.T, phase *taPhase, seqs []int, warm, runs, maxSeq int, smoke bool, x0Block func(int) *taBlockRow, asyncRows map[int]taBlockRow) {
	t.Helper()
	// ---- (a) baseline canary: CPU, X0 op chain (scalar AdamW forced) ----
	rngC := rand.New(rand.NewSource(taSeed))
	blockC := newTABlock(rngC, maxSeq)
	optim.UseScalarAdamW = true
	for _, seq := range seqs {
		row := taBlockStepBench(t, blockC, rngC, seq, warm, runs)
		row.Note = "CPU f32 canary (X0 op chain re-run under this session's load; scalar AdamW forced, CPU SwiGLU is K4-vectorized — see notes)"
		phase.CanaryResults = append(phase.CanaryResults, row)
		if base := x0Block(seq); base != nil {
			drift := (row.TotalMs - base.TotalMs) / base.TotalMs * 100
			note := fmt.Sprintf("canary seq=%d: %.0fms vs X0 %.0fms (drift %+.1f%%)", seq, row.TotalMs, base.TotalMs, drift)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
			if drift > 10 || drift < -10 {
				phase.Notes = append(phase.Notes, fmt.Sprintf("WARNING: canary drift at seq %d exceeds 10%% — cross-session comparison degraded per plan §2.1", seq))
			}
		}
	}
	optim.UseScalarAdamW = false

	// ---- (b) X2 Metal block step: sync dispatch, then async (R6) ----
	rng := rand.New(rand.NewSource(taSeed))
	block := newTABlock(rng, maxSeq)
	block.toMetal(g.MetalDev())
	for _, seq := range seqs {
		rowSync := taBlockStepBenchMetal(t, block, rng, seq, warm, runs)
		rowSync.Note = "X2 kernels (K1+K4 GPU, K7 vectorized AdamW), per-op sync dispatch"
		phase.BlockStepResults = append(phase.BlockStepResults, rowSync)

		g.SetMetalAsync(true)
		w0 := metal.SyncWaits.Load()
		rowAsync := taBlockStepBenchMetal(t, block, rng, seq, warm, runs)
		w1 := metal.SyncWaits.Load()
		g.SetMetalAsync(false)
		stepsRun := int64(warm + runs + 1) // +1: the live-graph forward
		rowAsync.Note = fmt.Sprintf("X2 kernels, R6 async dispatch (commit-without-wait); ~%d host sync waits per step", (w1-w0)/stepsRun)
		phase.BlockStepResults = append(phase.BlockStepResults, rowAsync)
		asyncRows[seq] = rowAsync

		t.Logf("X2 block step seq=%d: sync=%.1fms async=%.1fms (sync-async delta %.1fms; ~%d waits/step in async)",
			seq, rowSync.TotalMs, rowAsync.TotalMs, rowSync.TotalMs-rowAsync.TotalMs, (w1-w0)/stepsRun)
		if base := x0Block(seq); base != nil {
			note := fmt.Sprintf("block-step speedup vs X0 at seq %d: sync %.2fx, async %.2fx (%.0fms -> %.0fms/%.0fms)",
				seq, base.TotalMs/rowSync.TotalMs, base.TotalMs/rowAsync.TotalMs, base.TotalMs, rowSync.TotalMs, rowAsync.TotalMs)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
		}
		for _, cn := range phase.CanaryResults {
			if cn.Seq == seq {
				note := fmt.Sprintf("same-session speedup vs CPU canary at seq %d: sync %.2fx, async %.2fx",
					seq, cn.TotalMs/rowSync.TotalMs, cn.TotalMs/rowAsync.TotalMs)
				t.Log(note)
				phase.Notes = append(phase.Notes, note)
			}
		}

		liveGB := rowAsync.LiveGraphAfterFwdMB / 1024
		blockParams := 0
		for _, p := range block.parameters() {
			blockParams += p.Size()
		}
		weightsGB := float64(taVocab)*taHidden*4/1e9 + float64(taLayers*blockParams)*4/1e9
		extrap := liveGB*taLayers + weightsGB
		phase.MemoryResults = append(phase.MemoryResults, taMemoryRow{
			Seq: seq, LiveGraphBlockGB: liveGB, Extrapolated28GB: extrap,
			WeightsF32GB: weightsGB, FitsIn24GBUnifiedMem: extrap < 22,
			Note: "as X1K1 (live graph includes Metal buffer bytes); K4 backward recomputes sigma(x) instead of caching a (seq,3072) sigmoid slice per layer",
		})
	}

	// ---- (c) dispatch-counter gate ----
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
		t.Logf("dispatch counts (1 block fwd+bwd, seq %d): matmul=%d batched=%d softmax=%d silu=%d", seq, c.MatMul, c.BatchedMatMul, c.SoftmaxKernel, c.SiluKernel)
		if c.MatMul < 21 {
			t.Errorf("X2 gate: expected >=21 MPS matmul dispatches, got %d", c.MatMul)
		}
		if c.BatchedMatMul < 6 {
			t.Errorf("X2 gate: expected >=6 MPS batched-matmul dispatches, got %d", c.BatchedMatMul)
		}
		if c.SoftmaxKernel < 2 {
			t.Errorf("X2 gate: expected >=2 fused-softmax dispatches, got %d", c.SoftmaxKernel)
		}
		if c.SiluKernel < 2 {
			t.Errorf("X2 gate (K4): expected >=2 SwiGLU kernel dispatches (fwd+bwd), got %d", c.SiluKernel)
		}
		phase.DispatchCounts = map[string]int64{
			"block_fwd_bwd_seq1024_mps_matmul":         c.MatMul,
			"block_fwd_bwd_seq1024_mps_batched_matmul": c.BatchedMatMul,
			"block_fwd_bwd_seq1024_softmax_kernel":     c.SoftmaxKernel,
			"block_fwd_bwd_seq1024_silu_kernel":        c.SiluKernel,
		}
		g.ResetMetalDispatchCounts()

		// K2 CE dispatch check at a real shape.
		logits := taSeedTensor(rng, 1.0, 256, taVocab).ToMetal(g.MetalDev())
		logits.SetRequiresGrad(true)
		tgt := g.Zeros(256, 1)
		g.CrossEntropyLoss(logits, tgt).Backward()
		c = g.ReadMetalDispatchCounts()
		if c.CEKernel < 2 {
			t.Errorf("X2 gate (K2): expected 2 CE kernel dispatches (fwd+bwd), got %d", c.CEKernel)
		}
		phase.DispatchCounts["ce_fwd_bwd_seq256_ce_kernel"] = c.CEKernel
		g.ResetMetalDispatchCounts()
		taFlushGC()

		// R6 per-dispatch round-trip microbench.
		syncMs, asyncMs := taDispatchOverheadBench(t)
		note := fmt.Sprintf("R6 dispatch microbench (200x vec_mul on 1k elements): sync %.3f ms/dispatch vs async %.3f ms/dispatch -> waitUntilCompleted round trip ~%.3f ms recovered per GPU-GPU dispatch", syncMs, asyncMs, syncMs-asyncMs)
		t.Log(note)
		phase.Notes = append(phase.Notes, note)
	}
}

// ==================== Phase X2b (plan 0009 §3.3 K5/K6 + permute/repeat/bias residue) ====================
//
// Third X2 wave: the block-step CPU residue identified by the X2
// profile — permute_copy Metal kernel (Permute was 25% of residual CPU
// samples), K6 rope_apply fwd+bwd kernels with once-uploaded cos/sin
// tables, repeat_interleave fwd/bwd kernels (GQA KV expansion), K5
// rmsnorm_dgamma (removes the RMSNorm backward host loop), col_sum for
// Linear db + vec_bias_add for the Linear forward bias, and GPU-
// resident grad accumulation (vec_add in place). Together these retire
// nearly all of the ~21 residual host sync points per async block
// step. Bench structure mirrors X2 (canary, sync/async block rows,
// dispatch gates, tails, full-step estimates, X2 gate verdicts) plus
// Metal-resident per-op rows for the three newly-kerneled classes so
// per-item deltas vs the X0 per-op table are direct.

func TestTrainAccelBenchX2b(t *testing.T) {
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	smoke := os.Getenv("TA_SMOKE") != ""
	part := os.Getenv("TA_X2_PART") // "", "blocks", "tails"
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
		t.Fatal("X0-baseline phase not found in results JSON — X2b speedups need the baseline")
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
	x0Full := func(seq int) *taFullStepRow {
		if x0 == nil {
			return nil
		}
		for i := range x0.FullStepEstimate {
			if x0.FullStepEstimate[i].Seq == seq {
				return &x0.FullStepEstimate[i]
			}
		}
		return nil
	}

	phase := taPhase{
		Phase:            "X2b",
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
			"X2b third wave: permute_copy kernel (generic N-D gather), K6 rope_apply fwd+bwd (cos/sin uploaded once per module), repeat_interleave_fwd/bwd (GQA KV expansion), K5 rmsnorm_dgamma per-column reduction, col_sum Linear db + vec_bias_add Linear forward bias, and GPU-resident grad accumulation (in-place vec_add)",
			"block_step_results carry TWO rows per seq: per-op sync dispatch and async dispatch; the X2b configuration (and full_step_estimate) uses the better of the two per seq — async wins at seq <=1024 (~1 residual host wait/step), sync wins at seq 1500 where async reproducibly regresses (fresh-buffer zero-fill traffic vs saturated GPU memory bandwidth)",
			"canary caveat: the CPU canary chain is NOT byte-identical to X0's code — X2b also sped up the CPU Permute (contiguous run copies when the innermost dim is unpermuted) and the CPU linearDb (vDSP row accumulation), on top of X2's vectorized SwiGLU/CE; the canary forces scalar AdamW but cannot un-vectorize those. Treat the canary as approximate load calibration",
			"per_op_results here are Metal-resident micro-benches of the three newly-kerneled classes (permute_reshape, rope, rmsnorm) at exact workload shapes — compare directly against the X0 per-op rows of the same names",
		},
	}

	rng := rand.New(rand.NewSource(taSeed))
	asyncRows := map[int]taBlockRow{}

	if part != "tails" {
		taX2bBlocksPart(t, &phase, seqs, warm, runs, maxSeq, smoke, x0Block, asyncRows)
	}

	if part == "blocks" && !smoke {
		raw, err := json.MarshalIndent(phase, "", "  ")
		if err != nil {
			t.Fatalf("marshal partial phase: %v", err)
		}
		if err := os.WriteFile(taX2PartialPath(), append(raw, '\n'), 0o644); err != nil {
			t.Fatalf("write partial phase: %v", err)
		}
		t.Logf("TA_X2_PART=blocks: partial phase written to %s — run again with TA_X2_PART=tails", taX2PartialPath())
		return
	}
	if part == "tails" && !smoke {
		raw, err := os.ReadFile(taX2PartialPath())
		if err != nil {
			t.Fatalf("TA_X2_PART=tails needs the blocks part first: %v", err)
		}
		if err := json.Unmarshal(raw, &phase); err != nil {
			t.Fatalf("partial phase unmarshal: %v", err)
		}
		// Reconstruct the per-seq winner (better of sync/async — the
		// X2b configuration, see taX2bBlocksPart).
		for _, r := range phase.BlockStepResults {
			if best, ok := asyncRows[r.Seq]; !ok || r.TotalMs < best.TotalMs {
				asyncRows[r.Seq] = r
			}
		}
		if len(asyncRows) == 0 {
			t.Fatal("partial phase has no block rows")
		}
	}

	taX2TailsPart(t, &phase, rng, seqs, warm, runs, smoke, x0Block, x0Full, asyncRows, outPath, machine)
}

// taX2bBlocksPart: canary, sync/async Metal block steps, the extended
// dispatch-counter gate (incl. the X2b kernels), and Metal-resident
// per-op rows for the newly-kerneled classes.
func taX2bBlocksPart(t *testing.T, phase *taPhase, seqs []int, warm, runs, maxSeq int, smoke bool, x0Block func(int) *taBlockRow, asyncRows map[int]taBlockRow) {
	t.Helper()
	// ---- (a) baseline canary: CPU, X0 op chain (scalar AdamW forced) ----
	rngC := rand.New(rand.NewSource(taSeed))
	blockC := newTABlock(rngC, maxSeq)
	optim.UseScalarAdamW = true
	for _, seq := range seqs {
		row := taBlockStepBench(t, blockC, rngC, seq, warm, runs)
		row.Note = "CPU f32 canary (X0 op chain re-run under this session's load; scalar AdamW forced; CPU Permute/SwiGLU/CE are wave-vectorized — see notes)"
		phase.CanaryResults = append(phase.CanaryResults, row)
		if base := x0Block(seq); base != nil {
			drift := (row.TotalMs - base.TotalMs) / base.TotalMs * 100
			note := fmt.Sprintf("canary seq=%d: %.0fms vs X0 %.0fms (drift %+.1f%%)", seq, row.TotalMs, base.TotalMs, drift)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
			if drift > 10 || drift < -10 {
				phase.Notes = append(phase.Notes, fmt.Sprintf("WARNING: canary drift at seq %d exceeds 10%% — cross-session comparison degraded per plan §2.1", seq))
			}
		}
	}
	optim.UseScalarAdamW = false

	// ---- (b) X2b Metal block step: sync dispatch, then async ----
	rng := rand.New(rand.NewSource(taSeed))
	block := newTABlock(rng, maxSeq)
	block.toMetal(g.MetalDev())
	for _, seq := range seqs {
		rowSync := taBlockStepBenchMetal(t, block, rng, seq, warm, runs)
		rowSync.Note = "X2b kernels (K1+K4+K5+K6+permute/repeat/bias GPU, K7 vectorized AdamW), per-op sync dispatch"
		phase.BlockStepResults = append(phase.BlockStepResults, rowSync)

		g.SetMetalAsync(true)
		w0 := metal.SyncWaits.Load()
		rowAsync := taBlockStepBenchMetal(t, block, rng, seq, warm, runs)
		w1 := metal.SyncWaits.Load()
		g.SetMetalAsync(false)
		stepsRun := int64(warm + runs + 1) // +1: the live-graph forward
		rowAsync.Note = fmt.Sprintf("X2b kernels, R6 async dispatch (commit-without-wait); ~%d host sync waits per step", (w1-w0)/stepsRun)
		phase.BlockStepResults = append(phase.BlockStepResults, rowAsync)
		// The X2b configuration picks the better dispatch mode per seq
		// (async wins at seq <=1024; at seq 1500 async reproducibly
		// regresses — large fresh-buffer zero-fill traffic competes with
		// saturated GPU memory bandwidth — so sync wins there). Both
		// rows are recorded; the full-step estimate uses the winner.
		if rowSync.TotalMs < rowAsync.TotalMs {
			asyncRows[seq] = rowSync
		} else {
			asyncRows[seq] = rowAsync
		}

		t.Logf("X2b block step seq=%d: sync=%.1fms async=%.1fms (delta %.1fms; ~%d waits/step in async)",
			seq, rowSync.TotalMs, rowAsync.TotalMs, rowSync.TotalMs-rowAsync.TotalMs, (w1-w0)/stepsRun)
		if base := x0Block(seq); base != nil {
			note := fmt.Sprintf("block-step speedup vs X0 at seq %d: sync %.2fx, async %.2fx (%.0fms -> %.0fms/%.0fms)",
				seq, base.TotalMs/rowSync.TotalMs, base.TotalMs/rowAsync.TotalMs, base.TotalMs, rowSync.TotalMs, rowAsync.TotalMs)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
		}
		for _, cn := range phase.CanaryResults {
			if cn.Seq == seq {
				note := fmt.Sprintf("same-session speedup vs CPU canary at seq %d: sync %.2fx, async %.2fx",
					seq, cn.TotalMs/rowSync.TotalMs, cn.TotalMs/rowAsync.TotalMs)
				t.Log(note)
				phase.Notes = append(phase.Notes, note)
			}
		}

		liveGB := rowAsync.LiveGraphAfterFwdMB / 1024
		blockParams := 0
		for _, p := range block.parameters() {
			blockParams += p.Size()
		}
		weightsGB := float64(taVocab)*taHidden*4/1e9 + float64(taLayers*blockParams)*4/1e9
		extrap := liveGB*taLayers + weightsGB
		phase.MemoryResults = append(phase.MemoryResults, taMemoryRow{
			Seq: seq, LiveGraphBlockGB: liveGB, Extrapolated28GB: extrap,
			WeightsF32GB: weightsGB, FitsIn24GBUnifiedMem: extrap < 22,
			Note: "as X2 (live graph includes Metal buffer bytes)",
		})
	}

	// ---- (b2) Metal-resident per-op rows for the newly-kerneled classes ----
	perOpSeqs := []int{1024, 1500}
	if smoke {
		perOpSeqs = seqs
	}
	for _, seq := range perOpSeqs {
		rows := taPerOpBenchMetalX2b(t, block, rng, seq, warm, runs)
		phase.PerOpResults = append(phase.PerOpResults, rows...)
		for _, o := range rows {
			t.Logf("per-op(metal) seq=%d %-18s %8.1f ms/step  [fwd %.1f + bwd %.1f ms x%d]",
				seq, o.Op, o.PerStepMs, o.FwdMs, o.BwdMs, o.CountPerStep)
		}
	}

	// ---- (c) dispatch-counter gate ----
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
		t.Logf("dispatch counts (1 block fwd+bwd, seq %d): matmul=%d batched=%d softmax=%d silu=%d permute=%d rope=%d repeat=%d colreduce=%d biasadd=%d",
			seq, c.MatMul, c.BatchedMatMul, c.SoftmaxKernel, c.SiluKernel, c.PermuteKernel, c.RopeKernel, c.RepeatKernel, c.ColReduce, c.BiasAdd)
		if c.MatMul < 21 {
			t.Errorf("X2b gate: expected >=21 MPS matmul dispatches, got %d", c.MatMul)
		}
		if c.BatchedMatMul < 6 {
			t.Errorf("X2b gate: expected >=6 MPS batched-matmul dispatches, got %d", c.BatchedMatMul)
		}
		if c.SoftmaxKernel < 2 {
			t.Errorf("X2b gate: expected >=2 fused-softmax dispatches, got %d", c.SoftmaxKernel)
		}
		if c.SiluKernel < 2 {
			t.Errorf("X2b gate (K4): expected >=2 SwiGLU kernel dispatches, got %d", c.SiluKernel)
		}
		// X2b kernels: 4 fwd + 4 bwd permutes, 2+2 RoPE, 2+2 repeats,
		// 4 dgamma + 7 db col_sums, 7 forward bias adds.
		if c.PermuteKernel < 8 {
			t.Errorf("X2b gate: expected >=8 permute_copy dispatches, got %d", c.PermuteKernel)
		}
		if c.RopeKernel < 4 {
			t.Errorf("X2b gate (K6): expected >=4 rope_apply dispatches, got %d", c.RopeKernel)
		}
		if c.RepeatKernel < 4 {
			t.Errorf("X2b gate: expected >=4 repeat_interleave dispatches, got %d", c.RepeatKernel)
		}
		if c.ColReduce < 11 {
			t.Errorf("X2b gate (K5+db): expected >=11 col-reduction dispatches (4 dgamma + 7 db), got %d", c.ColReduce)
		}
		if c.BiasAdd < 7 {
			t.Errorf("X2b gate: expected >=7 vec_bias_add dispatches, got %d", c.BiasAdd)
		}
		phase.DispatchCounts = map[string]int64{
			"block_fwd_bwd_seq1024_mps_matmul":         c.MatMul,
			"block_fwd_bwd_seq1024_mps_batched_matmul": c.BatchedMatMul,
			"block_fwd_bwd_seq1024_softmax_kernel":     c.SoftmaxKernel,
			"block_fwd_bwd_seq1024_silu_kernel":        c.SiluKernel,
			"block_fwd_bwd_seq1024_permute_kernel":     c.PermuteKernel,
			"block_fwd_bwd_seq1024_rope_kernel":        c.RopeKernel,
			"block_fwd_bwd_seq1024_repeat_kernel":      c.RepeatKernel,
			"block_fwd_bwd_seq1024_colreduce_kernel":   c.ColReduce,
			"block_fwd_bwd_seq1024_biasadd_kernel":     c.BiasAdd,
		}
		g.ResetMetalDispatchCounts()
		taFlushGC()
	}
}

// taPerOpBenchMetalX2b measures the three newly-kerneled op classes
// with Metal-resident inputs at exact workload shapes — rows named to
// match the X0 per-op table for direct per-item deltas.
func taPerOpBenchMetalX2b(t *testing.T, block *taBlock, rng *rand.Rand, seq, warm, runs int) []taOpRow {
	t.Helper()
	dev := g.MetalDev()
	var rows []taOpRow
	add := func(op, shape string, fwd, bwd float64) {
		rows = append(rows, taOpRow{Op: op, Seq: seq, Shape: shape,
			FwdMs: fwd, BwdMs: bwd, CountPerStep: taLayers, PerStepMs: (fwd + bwd) * taLayers})
	}

	// permute_reshape (Metal): the block's full permute/repeat traffic.
	q2 := taSeedTensor(rng, 1.0, seq, taQDim).ToMetal(dev)
	q2.SetRequiresGrad(true)
	k2 := taSeedTensor(rng, 1.0, seq, taKVDim).ToMetal(dev)
	k2.SetRequiresGrad(true)
	v2 := taSeedTensor(rng, 1.0, seq, taKVDim).ToMetal(dev)
	v2.SetRequiresGrad(true)
	ao := taSeedTensor(rng, 1.0, taQHeads, seq, taHeadDim).ToMetal(dev)
	ao.SetRequiresGrad(true)
	group := taQHeads / taKVHeads
	fwd, bwd := taBenchOp(warm, runs, []*g.Tensor{q2, k2, v2, ao}, func() *g.Tensor {
		qh := g.Permute(q2.Reshape(seq, taQHeads, taHeadDim), []int{1, 0, 2})
		kh := g.Permute(k2.Reshape(seq, taKVHeads, taHeadDim), []int{1, 0, 2})
		vh := g.Permute(v2.Reshape(seq, taKVHeads, taHeadDim), []int{1, 0, 2})
		kr := g.RepeatInterleave(kh.Reshape(taKVHeads, 1, seq*taHeadDim), group)
		vr := g.RepeatInterleave(vh.Reshape(taKVHeads, 1, seq*taHeadDim), group)
		cc := g.Permute(ao, []int{1, 0, 2}).Reshape(seq, taQDim)
		return taSumScalars(g.Sum(qh), g.Sum(kr), g.Sum(vr), g.Sum(cc))
	})
	add("permute_reshape", fmt.Sprintf("3 head-splits + 2 kv-repeats + 1 concat at seq %d (Metal)", seq), fwd, bwd)

	// rope (Metal): Q (16,seq,128) + K (8,seq,128).
	rq := taSeedTensor(rng, 1.0, taQHeads, seq, taHeadDim).ToMetal(dev)
	rq.SetRequiresGrad(true)
	rk := taSeedTensor(rng, 1.0, taKVHeads, seq, taHeadDim).ToMetal(dev)
	rk.SetRequiresGrad(true)
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{rq, rk}, func() *g.Tensor {
		return taSumScalars(g.Sum(block.rope.Apply(rq, 0)), g.Sum(block.rope.Apply(rk, 0)))
	})
	add("rope", fmt.Sprintf("(16,%d,128) + (8,%d,128) (Metal)", seq, seq), fwd, bwd)

	// rmsnorm (Metal): 2 block norms + q_norm + k_norm — dgamma now on GPU.
	nx := taSeedTensor(rng, 1.0, seq, taHidden).ToMetal(dev)
	nx.SetRequiresGrad(true)
	nq := taSeedTensor(rng, 1.0, taQHeads*seq, taHeadDim).ToMetal(dev)
	nq.SetRequiresGrad(true)
	nk := taSeedTensor(rng, 1.0, taKVHeads*seq, taHeadDim).ToMetal(dev)
	nk.SetRequiresGrad(true)
	leaves := append([]*g.Tensor{nx, nq, nk}, block.parameters()...)
	fwd, bwd = taBenchOp(warm, runs, leaves, func() *g.Tensor {
		return taSumScalars(
			g.Sum(block.attnNorm.Forward(nx)), g.Sum(block.ffnNorm.Forward(nx)),
			g.Sum(block.qNorm.Forward(nq)), g.Sum(block.kNorm.Forward(nk)))
	})
	add("rmsnorm", fmt.Sprintf("2x(%d,1024) + (%d,128) + (%d,128) (Metal, K5 dgamma)", seq, taQHeads*seq, taKVHeads*seq), fwd, bwd)

	taFlushGC()
	return rows
}

// taX2TailsPart runs the tail benches, full-step estimates, gate
// verdicts, and the results-JSON append (sections d–g of the X2
// bench).
func taX2TailsPart(t *testing.T, phase *taPhase, rng *rand.Rand, seqs []int, warm, runs int, smoke bool, x0Block func(int) *taBlockRow, x0Full func(int) *taFullStepRow, asyncRows map[int]taBlockRow, outPath, machine string) {
	t.Helper()

	// ---- (d) tails ----
	for _, seq := range seqs {
		tails := taTailBench(t, rng, seq, warm, runs)
		for i := range tails {
			switch tails[i].Name {
			case "lm_head_matmul_fwd_bwd":
				tails[i] = taLmHeadBenchMetal(t, rng, seq, warm, runs)
			case "cross_entropy_fwd_bwd":
				// Keep the (now K2-vectorized) CPU row under a new name…
				tails[i].Name = "cross_entropy_fwd_bwd_cpu_vectorized"
				tails[i].Note = "K2 CPU fallback: Accelerate acc_vexp path with saved logsumexp (no double softmax)"
			}
		}
		phase.TailResults = append(phase.TailResults, tails...)
		// …and measure the Metal CE as the primary row.
		phase.TailResults = append(phase.TailResults, taCEBenchMetal(t, rng, seq, warm, runs))
		taFlushGC()
	}
	adamwScalar := taAdamWTableBenchScalar(t, rng, warm, runs)
	phase.TailResults = append(phase.TailResults, adamwScalar)
	adamwTail := taAdamWTableBench(t, rng, warm, runs)
	adamwTail.Note = "K7 vectorized Accelerate step (acc_adamw_step)"
	phase.TailResults = append(phase.TailResults, adamwTail)
	k7note := fmt.Sprintf("K7 gate: 172M-param AdamW step %.0fms (scalar) -> %.0fms (vectorized), %.2fx", adamwScalar.TotalMs, adamwTail.TotalMs, adamwScalar.TotalMs/adamwTail.TotalMs)
	t.Log(k7note)
	phase.Notes = append(phase.Notes, k7note)

	// ---- (e) full-step estimate (async block rows + Metal CE + K7 AdamW) ----
	for _, seq := range seqs {
		blockMs := asyncRows[seq].TotalMs
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
		t.Logf("X2 full-step estimate seq=%d: 28xblock=%.0fms embed=%.0fms lm_head=%.0fms ce=%.0fms adamw=%.0fms -> %.2fs",
			seq, est.Blocks28Ms, est.EmbeddingMs, est.LmHeadMs, est.LossMs, est.OptimizerTableMs, est.TotalS)
		if base := x0Full(seq); base != nil {
			note := fmt.Sprintf("full-step speedup vs X0 at seq %d: %.2fx (%.1fs -> %.1fs)", seq, base.TotalMs/est.TotalMs, base.TotalS, est.TotalS)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
		}
	}

	// ---- (f) gate verdicts ----
	if !smoke {
		if base := x0Block(1024); base != nil {
			sp := base.TotalMs / asyncRows[1024].TotalMs
			verdict := "NOT MET"
			if sp >= 3.5 {
				verdict = "MET"
			}
			phase.Notes = append(phase.Notes, fmt.Sprintf("GATE (plan X2, block >=3.5x at seq 1024 vs X0): %s (%.2fx)", verdict, sp))
		}
		if base := x0Full(1024); base != nil {
			for _, est := range phase.FullStepEstimate {
				if est.Seq == 1024 {
					sp := base.TotalMs / est.TotalMs
					verdict := "NOT MET"
					if sp >= 4.0 {
						verdict = "MET"
					}
					phase.Notes = append(phase.Notes, fmt.Sprintf("GATE (plan X2, full-step >=4x at seq 1024 vs X0): %s (%.2fx)", verdict, sp))
				}
			}
		}
	}

	if smoke {
		t.Log("TA_SMOKE set — skipping doc/training_accel_results.json write")
		return
	}

	// ---- (g) append the phase row ----
	var file taResultsFile
	if raw, err := os.ReadFile(outPath); err == nil {
		if err := json.Unmarshal(raw, &file); err != nil {
			t.Fatalf("existing %s is not valid JSON: %v", outPath, err)
		}
	}
	file.Hardware = machine
	file.Phases = append(file.Phases, *phase)
	raw, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		t.Fatalf("marshal results: %v", err)
	}
	if err := os.WriteFile(outPath, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", outPath, err)
	}
	t.Logf("results appended to %s (phase %s)", outPath, phase.Phase)
}

// ==================== Phase X3 (plan 0009 §3.4: bf16 frozen path) ====================
//
// bf16 storage + MPS-dtyped (MPSGraph, ADR-012) matmul for the frozen
// path: the 7 Linear weights per block are bf16 Metal-resident and
// FROZEN (RequiresGrad false — dW GEMMs skipped entirely), biases and
// norm gammas stay f32 (the stand-in trainable set; the real workload
// trains LoRA A/B + embedding rows). Activations remain f32 — bf16
// activations are deferred until A-parity per §3.4.
//
// The bench records per seq: the CPU canary (X0 chain), a frozen-f32
// CONTROL block (same frozen config, f32 weights — isolates the
// bf16-matmul effect from the frozen-dW effect), and the bf16-frozen
// block — each in sync AND async dispatch (the X2b finding: async
// reproducibly regresses at seq 1500; both modes recorded, the winner
// is the X3 configuration). Tails: lm_head with a bf16 FROZEN head
// (fwd dtyped 258G-FMA matmul, bwd dx only), Metal CE, K7 AdamW table.
// Memory rows account block weights at 2 B/param (B5).
//
// TA_X3_PART=canary|control|bf16|tails splits the run into FOUR
// processes: this session's SIGKILL horizon was measured at ~60 s wall
// (tighter than the ~110 s documented on TestTrainAccelBenchX2), and
// the two-way blocks/tails split no longer fits under it. Each part
// accumulates the phase in the same temp handoff file; "tails"
// finalizes and appends to the results JSON. Unset runs everything in
// one process (TA_SMOKE-sized runs only).

// taBlockStepBenchFrozen is taBlockStepBenchMetal with the optimizer
// over the trainable subset only (frozen-path modes) and a caller-
// supplied row note.
func taBlockStepBenchFrozen(t *testing.T, block *taBlock, rng *rand.Rand, seq, warm, runs int, note string) taBlockRow {
	t.Helper()
	dev := g.MetalDev()
	x := taSeedTensor(rng, 1.0, seq, taHidden).ToMetal(dev)
	x.SetRequiresGrad(true)
	opt := optim.NewAdamW(block.trainableParameters(), 1e-4, 0.01)

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
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}

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
		Note: note,
	}
}

// taLmHeadBenchMetalBF16 measures the lm_head matmul with the head
// weight bf16 Metal-resident and FROZEN: forward is the 258G-FMA
// dtyped MPS matmul, backward computes dx only (dW skipped — the
// frozen-operand fast path in the bf16 matmul autograd).
func taLmHeadBenchMetalBF16(t *testing.T, rng *rand.Rand, seq, warm, runs int) taTailRow {
	t.Helper()
	dev := g.MetalDev()
	xl := taSeedTensor(rng, 1.0, seq, taHidden).ToMetal(dev)
	xl.SetRequiresGrad(true)
	wHead := taSeedTensor(rng, 0.02, taHidden, taVocab).ToBF16().ToMetal(dev)
	fwd, bwd := taBenchOp(warm, runs, []*g.Tensor{xl}, func() *g.Tensor {
		return g.MatMul(xl, wHead)
	})
	taFlushGC()
	return taTailRow{Name: "lm_head_matmul_fwd_bwd", Seq: seq,
		Shape: fmt.Sprintf("(%d,%d)@(%d,%d) bf16 frozen W", seq, taHidden, taHidden, taVocab),
		FwdMs: fwd, BwdMs: bwd, TotalMs: fwd + bwd,
		Note: "W bf16 Metal-resident frozen (X3-B4 dtyped MPS path): fwd + dx dtyped, dW GEMM skipped entirely"}
}

func TestTrainAccelBenchX3(t *testing.T) {
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	if !g.MetalBF16MatMulSupported() {
		t.Skip("MPS bf16 matmul unsupported (ADR-012 runtime probe) — X3 bench needs the dtyped path")
	}
	smoke := os.Getenv("TA_SMOKE") != ""
	part := os.Getenv("TA_X3_PART") // "", "canary", "control", "bf16", "tails"
	seqs := []int{512, 1024, 1500}
	warm, runs := taWarmups, taRuns
	if smoke {
		seqs = []int{64}
		warm, runs = 1, 2
	}
	maxSeq := seqs[len(seqs)-1] + 1

	machine := taSysctl("machdep.cpu.brand_string")
	loadAvg := taSysctl("vm.loadavg")
	t.Logf("machine=%s load=%s part=%q", machine, loadAvg, part)

	outPath := "../doc/training_accel_results.json"
	x0 := taX0Phase(t, outPath)
	if x0 == nil && !smoke {
		t.Fatal("X0-baseline phase not found in results JSON — X3 speedups need the baseline")
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
	x0Full := func(seq int) *taFullStepRow {
		if x0 == nil {
			return nil
		}
		for i := range x0.FullStepEstimate {
			if x0.FullStepEstimate[i].Seq == seq {
				return &x0.FullStepEstimate[i]
			}
		}
		return nil
	}

	phase := taPhase{
		Phase:            "X3",
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
			"X3 bf16 frozen path (plan 0009 §3.4): 7 Linear weights per block bf16 Metal-resident + FROZEN (dW skipped), biases/norm gammas f32 trainable, activations f32; bf16 matmuls via the MPSGraph dtyped path (ADR-012: MPSMatrix hard-asserts on MPSDataTypeBFloat16, tier-b chosen; bf16 cast to f32 inside the graph = f32 accumulation by construction)",
			"block_step_results carry THREE configs per seq, sync+async each: CPU canary, frozen-f32 CONTROL (isolates the frozen-dW effect), and bf16-frozen (adds the bf16 matmul + half weight bytes); the X3 configuration takes the better dispatch mode per seq (X2b async-at-1500 regression re-checked here)",
			"NOTE the frozen configs run FEWER FLOPs than X0/X2b all-trainable blocks (7 dW GEMMs skipped) — that is the workload's real shape (LoRA base frozen), but vs-X0 gate ratios below compare frozen-path steps against the all-trainable X0 baseline; the frozen-f32 control row is the honest attribution basis for the bf16-specific gain",
			"tails: lm_head with bf16 FROZEN head weight (fwd dtyped 258G FMA, bwd dx only, dW skipped); cross-entropy Metal-resident (K2); adamw_172m_table is the K7 vectorized step (embedding table stays f32 trainable in the workload)",
			"memory: weights_f32_gb column reports the X3 weight bytes (block Linear weights at 2 B/param + f32 embedding table/norms/biases); B5 companion test e2e/bf16_memory_test.go pins the full 0.6B set at 1.23 GB bf16 vs 2.45 GB f32",
		},
	}

	rng := rand.New(rand.NewSource(taSeed))
	bestRows := map[int]taBlockRow{}

	// Resume the accumulated phase for every part after the first.
	if !smoke && part != "" && part != "canary" {
		raw, err := os.ReadFile(taX2PartialPath())
		if err != nil {
			t.Fatalf("TA_X3_PART=%s needs the earlier parts first: %v", part, err)
		}
		if err := json.Unmarshal(raw, &phase); err != nil {
			t.Fatalf("partial phase unmarshal: %v", err)
		}
	}

	if part == "" || part == "canary" {
		taX3CanaryPart(t, &phase, seqs, warm, runs, maxSeq, x0Block)
	}
	if part == "" || part == "control" {
		rngF := rand.New(rand.NewSource(taSeed))
		blockF := newTABlock(rngF, maxSeq)
		blockF.freezeBase()
		blockF.toMetal(g.MetalDev())
		taX3BlockModePart(t, &phase, blockF, rngF, seqs, warm, runs, "frozen-f32 control", false, x0Block, bestRows)
	}
	if part == "" || part == "bf16" {
		rngB := rand.New(rand.NewSource(taSeed))
		blockB := newTABlock(rngB, maxSeq)
		blockB.freezeBase()
		blockB.toMetalBF16(g.MetalDev())
		taX3BlockModePart(t, &phase, blockB, rngB, seqs, warm, runs, "bf16-frozen", true, x0Block, bestRows)
		if !smoke {
			taX3DispatchGate(t, &phase, blockB, rngB)
		}
	}

	if !smoke && part != "" && part != "tails" {
		raw, err := json.MarshalIndent(phase, "", "  ")
		if err != nil {
			t.Fatalf("marshal partial phase: %v", err)
		}
		if err := os.WriteFile(taX2PartialPath(), append(raw, '\n'), 0o644); err != nil {
			t.Fatalf("write partial phase: %v", err)
		}
		t.Logf("TA_X3_PART=%s: partial phase written to %s", part, taX2PartialPath())
		return
	}
	// Reconstruct the per-seq winner among the bf16-frozen rows when
	// they came from an earlier process.
	if len(bestRows) == 0 {
		for _, r := range phase.BlockStepResults {
			if !strings.Contains(r.Note, "bf16-frozen") {
				continue
			}
			if best, ok := bestRows[r.Seq]; !ok || r.TotalMs < best.TotalMs {
				bestRows[r.Seq] = r
			}
		}
	}
	if len(bestRows) == 0 {
		t.Fatal("no bf16-frozen block rows — run the earlier TA_X3_PART parts first")
	}

	// ---- tails ----
	// Unlike the X2/X2b tails, the CPU lm_head micro-bench is skipped
	// (it alone costs ~30 s of wall across the three seqs and the X0
	// row already documents it) — X3 measures the bf16-frozen lm_head,
	// the CPU-vectorized + Metal CE, the embedding, and the K7 table.
	for _, seq := range seqs {
		tails := taX3TailBench(t, rng, seq, warm, runs)
		phase.TailResults = append(phase.TailResults, tails...)
		phase.TailResults = append(phase.TailResults, taLmHeadBenchMetalBF16(t, rng, seq, warm, runs))
		phase.TailResults = append(phase.TailResults, taCEBenchMetal(t, rng, seq, warm, runs))
		taFlushGC()
	}
	adamwTail := taAdamWTableBench(t, rng, warm, runs)
	adamwTail.Note = "K7 vectorized Accelerate step (embedding table stays f32 trainable in the workload)"
	phase.TailResults = append(phase.TailResults, adamwTail)

	// ---- full-step estimate (bf16-frozen winner rows) + gate verdicts ----
	for _, seq := range seqs {
		blockMs := bestRows[seq].TotalMs
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
		t.Logf("X3 full-step estimate seq=%d: 28xblock=%.0fms embed=%.0fms lm_head=%.0fms ce=%.0fms adamw=%.0fms -> %.2fs",
			seq, est.Blocks28Ms, est.EmbeddingMs, est.LmHeadMs, est.LossMs, est.OptimizerTableMs, est.TotalS)
		if base := x0Full(seq); base != nil {
			note := fmt.Sprintf("full-step speedup vs X0 at seq %d: %.2fx (%.1fs -> %.1fs)", seq, base.TotalMs/est.TotalMs, base.TotalS, est.TotalS)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
		}
	}
	if !smoke {
		if base := x0Block(1024); base != nil {
			sp := base.TotalMs / bestRows[1024].TotalMs
			verdict := "NOT MET"
			if sp >= 3.5 {
				verdict = "MET"
			}
			phase.Notes = append(phase.Notes, fmt.Sprintf("GATE (plan X2 carry-over, block >=3.5x at seq 1024 vs X0): %s (%.2fx, bf16-frozen config — see the FLOP-count caveat note)", verdict, sp))
		}
		if base := x0Full(1024); base != nil {
			for _, est := range phase.FullStepEstimate {
				if est.Seq == 1024 {
					sp := base.TotalMs / est.TotalMs
					verdict := "NOT MET"
					if sp >= 4.0 {
						verdict = "MET"
					}
					phase.Notes = append(phase.Notes, fmt.Sprintf("GATE (plan X2 carry-over, full-step >=4x at seq 1024 vs X0): %s (%.2fx)", verdict, sp))
				}
			}
		}
	}

	if smoke {
		t.Log("TA_SMOKE set — skipping doc/training_accel_results.json write")
		return
	}

	// ---- append the phase row ----
	var file taResultsFile
	if raw, err := os.ReadFile(outPath); err == nil {
		if err := json.Unmarshal(raw, &file); err != nil {
			t.Fatalf("existing %s is not valid JSON: %v", outPath, err)
		}
	}
	file.Hardware = machine
	file.Phases = append(file.Phases, phase)
	rawOut, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		t.Fatalf("marshal results: %v", err)
	}
	if err := os.WriteFile(outPath, append(rawOut, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", outPath, err)
	}
	t.Logf("results appended to %s (phase %s)", outPath, phase.Phase)
}

// taX3CanaryPart: CPU canary, X0 op chain (scalar AdamW forced).
func taX3CanaryPart(t *testing.T, phase *taPhase, seqs []int, warm, runs, maxSeq int, x0Block func(int) *taBlockRow) {
	t.Helper()
	rngC := rand.New(rand.NewSource(taSeed))
	blockC := newTABlock(rngC, maxSeq)
	optim.UseScalarAdamW = true
	for _, seq := range seqs {
		row := taBlockStepBench(t, blockC, rngC, seq, warm, runs)
		row.Note = "CPU f32 canary (X0 op chain re-run under this session's load; scalar AdamW forced; CPU Permute/SwiGLU/CE are wave-vectorized — see X2b notes)"
		phase.CanaryResults = append(phase.CanaryResults, row)
		if base := x0Block(seq); base != nil {
			drift := (row.TotalMs - base.TotalMs) / base.TotalMs * 100
			note := fmt.Sprintf("canary seq=%d: %.0fms vs X0 %.0fms (drift %+.1f%%)", seq, row.TotalMs, base.TotalMs, drift)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
			if drift > 10 || drift < -10 {
				phase.Notes = append(phase.Notes, fmt.Sprintf("WARNING: canary drift at seq %d exceeds 10%% — cross-session comparison degraded per plan §2.1", seq))
			}
		}
	}
	optim.UseScalarAdamW = false
}

// taX3BlockModePart benches one frozen block config (sync + async per
// seq), appends its rows/notes, and — when record is set — the per-seq
// winner and the bf16-aware memory rows.
func taX3BlockModePart(t *testing.T, phase *taPhase, block *taBlock, rng *rand.Rand, seqs []int, warm, runs int, label string, record bool, x0Block func(int) *taBlockRow, bestRows map[int]taBlockRow) {
	t.Helper()
	for _, seq := range seqs {
		rowSync := taBlockStepBenchFrozen(t, block, rng, seq, warm, runs,
			label+" (frozen base, K1..K7 kernels), per-op sync dispatch")
		phase.BlockStepResults = append(phase.BlockStepResults, rowSync)

		g.SetMetalAsync(true)
		w0 := metal.SyncWaits.Load()
		rowAsync := taBlockStepBenchFrozen(t, block, rng, seq, warm, runs, "")
		w1 := metal.SyncWaits.Load()
		g.SetMetalAsync(false)
		stepsRun := int64(warm + runs + 1) // +1: the live-graph forward
		rowAsync.Note = fmt.Sprintf("%s (frozen base), R6 async dispatch; ~%d host sync waits per step", label, (w1-w0)/stepsRun)
		phase.BlockStepResults = append(phase.BlockStepResults, rowAsync)

		winner := rowAsync
		if rowSync.TotalMs < rowAsync.TotalMs {
			winner = rowSync
		}
		if record {
			bestRows[seq] = winner
		}
		t.Logf("X3 %s block step seq=%d: sync=%.1fms async=%.1fms", label, seq, rowSync.TotalMs, rowAsync.TotalMs)
		if base := x0Block(seq); base != nil {
			note := fmt.Sprintf("%s block-step vs X0 at seq %d: sync %.2fx, async %.2fx (%.0fms -> %.0fms/%.0fms)",
				label, seq, base.TotalMs/rowSync.TotalMs, base.TotalMs/rowAsync.TotalMs, base.TotalMs, rowSync.TotalMs, rowAsync.TotalMs)
			t.Log(note)
			phase.Notes = append(phase.Notes, note)
		}
		for _, cn := range phase.CanaryResults {
			if cn.Seq == seq {
				note := fmt.Sprintf("%s same-session vs CPU canary at seq %d: sync %.2fx, async %.2fx",
					label, seq, cn.TotalMs/rowSync.TotalMs, cn.TotalMs/rowAsync.TotalMs)
				t.Log(note)
				phase.Notes = append(phase.Notes, note)
			}
		}

		if record {
			// Memory row with bf16-aware weight accounting: block
			// Linear weights at 2 B/param, everything else f32.
			liveGB := winner.LiveGraphAfterFwdMB / 1024
			linW := block.linearWeightParams()
			blockParams := 0
			for _, p := range block.parameters() {
				blockParams += p.Size()
			}
			weightsGB := float64(taVocab)*taHidden*4/1e9 +
				float64(taLayers)*(float64(linW)*2+float64(blockParams-linW)*4)/1e9
			extrap := liveGB*taLayers + weightsGB
			phase.MemoryResults = append(phase.MemoryResults, taMemoryRow{
				Seq: seq, LiveGraphBlockGB: liveGB, Extrapolated28GB: extrap,
				WeightsF32GB: weightsGB, FitsIn24GBUnifiedMem: extrap < 22,
				Note: "X3: weights_f32_gb column = X3 weight bytes (28x block Linear weights bf16 at 2 B/param + f32 embedding table/norms/biases); activations stay f32 in this phase, so the live-graph savings vs X2b come from skipped dW allocations, not activation dtype",
			})
		}
	}
}

// taX3DispatchGate asserts the bf16-frozen dispatch shape at seq 1024.
func taX3DispatchGate(t *testing.T, phase *taPhase, blockB *taBlock, rngB *rand.Rand) {
	t.Helper()
	seq := 1024
	x := taSeedTensor(rngB, 1.0, seq, taHidden).ToMetal(g.MetalDev())
	x.SetRequiresGrad(true)
	x.ZeroGrad()
	g.ResetMetalDispatchCounts()
	g.Sum(blockB.forward(x, true)).Backward()
	c := g.ReadMetalDispatchCounts()
	t.Logf("X3 dispatch counts (1 bf16-frozen block fwd+bwd, seq %d): bf16_matmul=%d f32_matmul=%d batched=%d softmax=%d silu=%d permute=%d rope=%d repeat=%d colreduce=%d biasadd=%d",
		seq, c.BF16MatMul, c.MatMul, c.BatchedMatMul, c.SoftmaxKernel, c.SiluKernel, c.PermuteKernel, c.RopeKernel, c.RepeatKernel, c.ColReduce, c.BiasAdd)
	// 7 forward projections (dtyped TransB) + 7 backward dx (dtyped
	// plain; dW skipped, weights frozen) = 14 bf16 matmuls; the f32
	// MPS matmul count drops to 0 (every 2-D matmul in the block has
	// a bf16 weight operand).
	if c.BF16MatMul < 14 {
		t.Errorf("X3 gate: expected >=14 dtyped bf16 MPS matmuls (7 fwd + 7 dx), got %d", c.BF16MatMul)
	}
	if c.BatchedMatMul < 6 {
		t.Errorf("X3 gate: expected >=6 MPS batched matmuls (f32 attention), got %d", c.BatchedMatMul)
	}
	if c.SoftmaxKernel < 2 || c.SiluKernel < 2 {
		t.Errorf("X3 gate: fused softmax/silu kernels missing: softmax=%d silu=%d", c.SoftmaxKernel, c.SiluKernel)
	}
	phase.DispatchCounts = map[string]int64{
		"block_fwd_bwd_seq1024_bf16_matmul":        c.BF16MatMul,
		"block_fwd_bwd_seq1024_mps_matmul":         c.MatMul,
		"block_fwd_bwd_seq1024_mps_batched_matmul": c.BatchedMatMul,
		"block_fwd_bwd_seq1024_softmax_kernel":     c.SoftmaxKernel,
		"block_fwd_bwd_seq1024_silu_kernel":        c.SiluKernel,
		"block_fwd_bwd_seq1024_permute_kernel":     c.PermuteKernel,
		"block_fwd_bwd_seq1024_rope_kernel":        c.RopeKernel,
		"block_fwd_bwd_seq1024_repeat_kernel":      c.RepeatKernel,
		"block_fwd_bwd_seq1024_colreduce_kernel":   c.ColReduce,
		"block_fwd_bwd_seq1024_biasadd_kernel":     c.BiasAdd,
	}
	g.ResetMetalDispatchCounts()
	taFlushGC()
}

// taX3TailBench: the embedding and CPU-vectorized-CE tails only (the
// CPU lm_head micro-bench is deliberately skipped in X3 — see the
// caller comment).
func taX3TailBench(t *testing.T, rng *rand.Rand, seq, warm, runs int) []taTailRow {
	t.Helper()
	var rows []taTailRow

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
		Note: "as X0 (table stays f32 trainable in the workload)"})
	table = nil
	runtime.GC()

	logits := taSeedTensor(rng, 1.0, seq, taVocab)
	logits.SetRequiresGrad(true)
	tgt := g.Zeros(seq, 1)
	for i := 0; i < seq; i++ {
		tgt.Data()[i] = float32(rng.Intn(taVocab))
	}
	fwd, bwd = taBenchOp(warm, runs, []*g.Tensor{logits}, func() *g.Tensor {
		return g.CrossEntropyLoss(logits, tgt)
	})
	rows = append(rows, taTailRow{Name: "cross_entropy_fwd_bwd_cpu_vectorized", Seq: seq,
		Shape: fmt.Sprintf("(%d,%d)", seq, taVocab),
		FwdMs: fwd, BwdMs: bwd, TotalMs: fwd + bwd,
		Note: "K2 CPU fallback (acc_vexp path)"})
	logits = nil
	runtime.GC()

	return rows
}
