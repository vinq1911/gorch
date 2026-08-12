//go:build darwin

package qwen

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
)

// Prompt token ids — MUST match model/export_qwen_fixtures.py (the
// fixture file also carries {tag}_ids; loadPromptIDs cross-checks).
var promptShort = []int{
	785, 6722, 315, 9625, 374, 264, 6662, 3283, 3881, 369, 1947, 11,
	3607, 323, 13078, 13,
}

var promptLong = []int{
	34253, 4128, 4119, 1882, 1467, 438, 23700, 315, 43179, 11211, 13,
	8886, 3950, 374, 23844, 311, 264, 27950, 4621, 11, 23507, 1526,
	1657, 13617, 315, 6529, 323, 5395, 44804, 34447, 11, 323, 5499,
	27348, 1182, 8630, 279, 34918, 311, 7023, 1128, 4041, 1790, 13,
	5737, 291, 65489, 6529, 13248, 1376, 323, 897, 14629, 3941, 5203,
	315, 3239, 14629, 11, 892, 14035, 15504, 279, 6500, 429, 3078,
	460, 46719, 47116, 1969, 2506, 304, 4938, 13, 74856, 2309, 70547,
	16164, 1973, 553, 41396, 13530, 315, 11744, 11, 323, 77178,
	48723, 13598, 92495, 15175, 3941, 220, 17, 23, 41315, 13617, 13,
	576, 3974, 13876, 38835, 34208, 916, 220, 16, 18, 1602, 15678,
	12590, 11, 220, 19, 20, 21, 3039, 11, 304, 279, 9255, 12406, 315,
	220, 17, 15, 17, 21, 13,
}

var (
	qwenOnce   sync.Once
	qwenCached *Model
	qwenErr    error
)

// loadModelCached loads Qwen3-0.6B from the real checkpoint once per
// test binary (the 1.2 GB bf16 parse is the slow part).
func loadModelCached(t testing.TB) *Model {
	if _, err := FindCheckpoint(); err != nil {
		t.Skipf("qwen checkpoint not available (set QWEN3_MODEL): %v", err)
	}
	qwenOnce.Do(func() { qwenCached, qwenErr = LoadPretrained() })
	if qwenErr != nil {
		t.Fatalf("LoadPretrained: %v", qwenErr)
	}
	return qwenCached
}

var (
	fixOnce   sync.Once
	fixCached *model.SafetensorsFile
	fixErr    error
)

func loadFixtures(t testing.TB) *model.SafetensorsFile {
	fixOnce.Do(func() {
		fixCached, fixErr = model.LoadSafetensors("../testdata/qwen_fixtures.safetensors")
	})
	if fixErr != nil {
		t.Fatalf("load fixtures (regenerate with model/export_qwen_fixtures.py): %v", fixErr)
	}
	return fixCached
}

func fixture(t testing.TB, sf *model.SafetensorsFile, name string) *g.Tensor {
	tt, ok := sf.Tensors[name]
	if !ok {
		t.Fatalf("fixture %q missing", name)
	}
	return tt
}

// promptIDs returns the hardcoded prompt after cross-checking it
// against the ids recorded in the fixture file — catches silent drift
// between the export script and this test.
func promptIDs(t testing.TB, sf *model.SafetensorsFile, tag string) []int {
	var ids []int
	switch tag {
	case "short":
		ids = promptShort
	case "long":
		ids = promptLong
	default:
		t.Fatalf("unknown prompt tag %q", tag)
	}
	ref := fixture(t, sf, tag+"_ids").Data()
	if len(ref) != len(ids) {
		t.Fatalf("%s: fixture has %d ids, test hardcodes %d", tag, len(ref), len(ids))
	}
	for i := range ids {
		if int(ref[i]) != ids[i] {
			t.Fatalf("%s: id %d is %d in fixture, %d in test", tag, i, int(ref[i]), ids[i])
		}
	}
	return ids
}

// stageCheck is one golden comparison — copied verbatim from
// audio/mimi/encoder_test.go (the P2 tolerance-precedent helpers):
//  1. plan metric |a−b|/(|b|+1e-5) ≤ relBigGate restricted to |b| ≥ 1e-2,
//  2. mixed tolerance |a−b| ≤ absTol + relTol·|b| over all elements.
type stageCheck struct {
	stage          string
	got, ref       []float32
	relBigGate     float64
	absTol, relTol float64
	maxAbs, relAll float64
	relBig, mixed  float64
	failure        string
}

func (c *stageCheck) evaluate() bool {
	c.maxAbs, c.relAll, c.relBig, c.mixed = 0, 0, 0, 0
	for i := range c.got {
		d := math.Abs(float64(c.got[i]) - float64(c.ref[i]))
		ab := math.Abs(float64(c.ref[i]))
		if d > c.maxAbs {
			c.maxAbs = d
		}
		if rel := d / (ab + 1e-5); rel > c.relAll {
			c.relAll = rel
		}
		if ab >= 1e-2 {
			if rel := d / (ab + 1e-5); rel > c.relBig {
				c.relBig = rel
			}
		}
		if m := d / (c.absTol + c.relTol*ab); m > c.mixed {
			c.mixed = m
		}
	}
	c.failure = ""
	if c.relBig > c.relBigGate {
		c.failure = fmt.Sprintf("%s: max relative error %.3g > %.0e budget on |ref| >= 1e-2", c.stage, c.relBig, c.relBigGate)
	}
	if c.mixed > 1 {
		c.failure += fmt.Sprintf("; %s: mixed tolerance violated: max |a-b|/(%.0e + %.0e*|b|) = %.3g > 1",
			c.stage, c.absTol, c.relTol, c.mixed)
	}
	return c.failure == ""
}

func (c *stageCheck) log(t *testing.T) {
	t.Helper()
	t.Logf("%s: max abs err %.3g; plan metric max rel err %.3g (all), %.3g (|ref|>=1e-2); worst ratio %.3g vs |a-b| <= %.0e + %.0e*|b|",
		c.stage, c.maxAbs, c.relAll, c.relBig, c.mixed, c.absTol, c.relTol)
}

// runGolden evaluates the checks produced by compute; on a gate
// failure it recomputes ONCE and requires the failure to reproduce
// (Accelerate's threaded GEMM occasionally splits work differently
// under load; a transient f32 summation-order blip does not repeat —
// see audio/mimi/encoder_test.go for the full rationale).
func runGolden(t *testing.T, compute func() []*stageCheck) {
	t.Helper()
	checks := compute()
	ok := true
	for _, c := range checks {
		if !c.evaluate() {
			ok = false
		}
		c.log(t)
	}
	if ok {
		return
	}
	t.Logf("gate failure — recomputing once to rule out load-induced BLAS nondeterminism")
	checks = compute()
	for _, c := range checks {
		pass := c.evaluate()
		c.log(t)
		if !pass {
			t.Error(c.failure)
		}
	}
}

// Tolerance policy. Plan 0008 §2.7 says "start at relBig 5e-4 / absTol
// 1e-5 / relTol 5e-4 per stage, logits at 2e-3" — those gates hold for
// the layer-0 norm/attn/block stages but are tighter than f32 noise
// where two effects dominate:
//
//  1. RoPE table ulp noise (l0_q / l0_k): HF materialises inv_freq via
//     float32 pow (torch `base ** t`), which is 1-ulp irreproducible
//     without bit-exact Sleef; the resulting θ differences are bounded
//     by ulp(θ) ≈ |θ|·6e-8 rad and read as ≤1.3e-4 absolute on rotated
//     Q/K at position ≤128. Downstream stages (l0_attn on) wash the
//     phase noise out — measured 8e-5 relBig at l0_attn on the long
//     prompt — so the q/k gates get an absTol of 2e-4 and everything
//     after stays on the plan budget.
//  2. Depth accumulation (l13/l27/final_norm/logits). Measured
//     justification (2026-08-11, transformers 4.57.1): running the HF
//     reference against ITSELF with eager vs sdpa attention (same
//     weights, same framework) gives relBig(|ref|≥1e-2) noise of
//     8.3e-4/1.4e-3 at layer 13 (short/long prompt), 7.3e-3/1.49e-2 at
//     layer 27, and 3.0e-3/1.1e-3 on last-position logits. The Go
//     port's deviation from the eager reference is inside that
//     envelope at every depth (l13 8.2e-4/1.3e-3, l27 5.3e-3/1.46e-2,
//     logits 2.0e-3/1.4e-3), so the deep gates below sit one notch
//     above the reference's own backend noise. The greedy contract is
//     pinned separately by the exact top-5 id match.
const (
	relBigGate = 5e-4
	absTol     = 1e-5
	relTol     = 5e-4
)

// stageGates maps stages with a measured looser budget to
// {relBigGate, absTol, relTol}; all other stages use the plan default.
var stageGates = map[string][3]float64{
	"l0_q":       {5e-3, 2e-4, 5e-4},
	"l0_k":       {5e-3, 2e-4, 5e-4},
	"l13_out":    {2e-3, 1e-4, 1e-3},
	"l27_out":    {2e-2, 1e-3, 1e-2},
	"final_norm": {5e-3, 2e-4, 2e-3},
	"logits":     {5e-3, 1e-4, 5e-3},
}

// goldenStages runs the per-stage parity pyramid for one prompt. The
// stage computations are the exact op sequence Model.Forward executes
// (Embed → blocks → norm → tied head), unrolled so intermediate
// tensors can be compared.
func goldenStages(t *testing.T, tag string) {
	m := loadModelCached(t)
	sf := loadFixtures(t)
	ids := promptIDs(t, sf, tag)

	runGolden(t, func() []*stageCheck {
		var checks []*stageCheck
		add := func(stage string, got []float32) {
			rg, at, rt := relBigGate, absTol, relTol
			if g, ok := stageGates[stage]; ok {
				rg, at, rt = g[0], g[1], g[2]
			}
			ref := fixture(t, sf, tag+"_"+stage).Data()
			if len(got) != len(ref) {
				t.Fatalf("%s_%s: got %d elements, fixture has %d", tag, stage, len(got), len(ref))
			}
			checks = append(checks, &stageCheck{stage: tag + "_" + stage, got: got, ref: ref,
				relBigGate: rg, absTol: at, relTol: rt})
		}

		emb := m.Embed.Forward(ids)
		add("embeddings", emb.Data())

		h0n := m.Blocks[0].NormAttn.Forward(emb)
		add("l0_norm", h0n.Data())

		qH, kH, _ := m.Blocks[0].Attn.ProjectQKV(h0n, 0)
		add("l0_q", qH.Data())
		add("l0_k", kH.Data())

		attn := m.Blocks[0].Attn.Forward(h0n, 0)
		add("l0_attn", attn.Data())

		h := emb
		for i, blk := range m.Blocks {
			h = blk.Forward(h, 0)
			switch i {
			case 0:
				add("l0_out", h.Data())
			case 13:
				add("l13_out", h.Data())
			case 27:
				add("l27_out", h.Data())
			}
		}

		hn := m.Norm.Forward(h)
		add("final_norm", hn.Data())

		seq, dim := hn.Shape()[0], hn.Shape()[1]
		last := g.NewTensor(hn.Data()[(seq-1)*dim:seq*dim], 1, dim)
		logits := g.MatMulTransB(last, m.Embed.Weight)
		add("logits", logits.Data())
		return checks
	})
}

func TestQwenGoldenShort(t *testing.T) { goldenStages(t, "short") }
func TestQwenGoldenLong(t *testing.T)  { goldenStages(t, "long") }

// topK returns the indices of the k largest values.
func topK(logits []float32, k int) []int {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	return idx[:k]
}

// TestQwenGoldenTop5 — the §2.7 logits argmax gate: the Go top-5 token
// ids must match the HF reference top-5 for both prompts (as ordered
// lists — greedy argmax is index 0).
func TestQwenGoldenTop5(t *testing.T) {
	m := loadModelCached(t)
	sf := loadFixtures(t)
	for _, tag := range []string{"short", "long"} {
		ids := promptIDs(t, sf, tag)
		logits := m.Forward(ids, 0)
		seq, vocab := logits.Shape()[0], logits.Shape()[1]
		lastRow := logits.Data()[(seq-1)*vocab : seq*vocab]
		got := topK(lastRow, 5)
		ref := topK(fixture(t, sf, tag+"_logits").Data(), 5)
		for i := range ref {
			if got[i] != ref[i] {
				t.Errorf("%s: top-5 mismatch at rank %d: got id %d, reference %d (got %v, want %v)",
					tag, i, got[i], ref[i], got, ref)
				break
			}
		}
		t.Logf("%s: top-5 ids %v (match)", tag, got)
	}
}

// TestQwenForwardCachedMatchesFull — ForwardCached-vs-full-forward
// equivalence (plan §2.7): the cached path (staircase prefill chunks +
// single-token steps) must reproduce the full-forward logits at every
// checked position. Prefill split 8+5, then 3 single-token steps, so
// the multi-token staircase mask, chunk append, and incremental step
// are all exercised.
//
// Gate note: the plan's "tolerance 1e-5" is not attainable between two
// f32 paths whose GEMMs sum in different orders (batched full-seq
// attention vs per-KV-head cached GEMMs) across 28 layers — measured
// divergence on the logits is ≤5e-5 absolute / ≤3e-5 relative (vs
// 2e-3 relative for Go-vs-HF parity). Gate: |a−b| ≤ 5e-5 + 1e-4·|b|.
//
// Flake discipline (2026-08-12 investigation). The earlier
// "load-induced BLAS reduction reordering, min-over-attempts clears
// it" comment was measured and found wrong in two ways:
//
//  1. Accelerate's sgemm on Apple Silicon (measured on M4, macOS 26.x)
//     occasionally computes a tile-shaped patch (~192 elements, or one
//     output row) of ONE GEMM call in an alternate bit-state for
//     bit-identical inputs — flip-flopping between exactly two stable
//     states, more often under host load, and the alternate state can
//     be STICKY for a process's lifetime (observed: one process
//     failing 3/3 attempts at ratio 6-64x while sibling processes
//     under the same load pass 3/3 bit-identically; go test -race
//     clean; inputs at the diverging op verified bit-identical, so
//     this is not a gorch memory bug). f64-reference conditioning
//     analysis showed the normal state matches the f64 dot product
//     while the alternate state lands 12-22x OUTSIDE the maximal f32
//     summation-order error bound u·K·Σ|a_i·b_i| — i.e. a
//     reduced-precision (bf16-like) tile, a platform defect rather
//     than benign reduction reordering. One such ~1% hidden-state
//     perturbation amplifies through the remaining layers into the
//     observed 60x logit ratios (which is also why widening the
//     tolerance is not a viable fix).
//  2. The old min-over-attempts retry recomputed only the CACHED path
//     against a full-forward reference computed once — a noisy draw on
//     the full side could never clear, turning transient platform
//     noise into a stable-looking failure.
//
// Discipline now: each attempt is self-contained (recomputes BOTH the
// full forward and the cached staircase); after 3 failed in-process
// attempts the test re-execs itself once in a fresh subprocess
// (process restart empirically leaves the sticky BLAS mode behind) and
// accepts that verdict, logging the recovery loudly. Set
// QWEN_CACHED_DIAG=1 for layer-divergence forensics on failure.
const qwenCachedSubprocEnv = "QWEN_CACHED_EQUIV_SUBPROC"

func TestQwenForwardCachedMatchesFull(t *testing.T) {
	m := loadModelCached(t)
	ids := promptShort // 16 tokens

	type step struct {
		toks []int
		pos  int // absolute position of the produced logits row
	}
	steps := []step{
		{ids[:8], 7},
		{ids[8:13], 12},
		{ids[13:14], 13},
		{ids[14:15], 14},
		{ids[15:16], 15},
	}
	worstOf := func(got, ref []float32) float64 {
		var worst float64
		for i := range got {
			d := math.Abs(float64(got[i]) - float64(ref[i]))
			if m := d / (5e-5 + 1e-4*math.Abs(float64(ref[i]))); m > worst {
				worst = m
			}
		}
		return worst
	}
	// attempt recomputes the full-forward reference AND the cached
	// staircase — both sides of the comparison are fresh draws, so a
	// transient BLAS event on either side clears on retry.
	attempt := func() (ratios []float64, full *g.Tensor) {
		full = m.Forward(ids, 0)
		vocab := full.Shape()[1]
		cache := m.NewCache()
		for _, s := range steps {
			logits := m.ForwardCached(s.toks, cache)
			ref := full.Data()[s.pos*vocab : (s.pos+1)*vocab]
			ratios = append(ratios, worstOf(logits.Data(), ref))
		}
		if cache.Len() != len(ids) {
			t.Errorf("cache.Len() = %d, want %d", cache.Len(), len(ids))
		}
		return ratios, full
	}
	pass := func(ratios []float64) bool {
		for _, r := range ratios {
			if r > 1 {
				return false
			}
		}
		return true
	}

	var last []float64
	var lastFull *g.Tensor
	for a := 1; a <= 3; a++ {
		last, lastFull = attempt()
		for i, s := range steps {
			t.Logf("attempt %d pos %d: worst |a-b|/(5e-5 + 1e-4*|b|) = %.3g", a, s.pos, last[i])
		}
		if pass(last) {
			if a > 1 {
				t.Logf("recovered on attempt %d — transient platform BLAS nondeterminism, not a cache defect", a)
			}
			return
		}
		t.Logf("attempt %d failed the gate", a)
	}

	if os.Getenv("QWEN_CACHED_DIAG") != "" {
		chunks := make([][2]int, len(steps))
		off := 0
		for i, st := range steps {
			chunks[i] = [2]int{off, off + len(st.toks)}
			off += len(st.toks)
		}
		diagnoseCachedDivergence(t, m, ids, lastFull, chunks)
	}

	if os.Getenv(qwenCachedSubprocEnv) != "" {
		// We ARE the quarantine subprocess: report the failure for real.
		for i, s := range steps {
			if last[i] > 1 {
				t.Errorf("pos %d: cached logits diverge from full forward beyond 5e-5 + 1e-4·|b|: ratio %.3g (persists in a fresh process)", s.pos, last[i])
			}
		}
		return
	}

	// Quarantine re-exec: the sticky alternate BLAS mode is per-process
	// state; a fresh process is a fresh draw. Exactly one child, scoped
	// to this test only, recursion blocked by the env sentinel.
	t.Logf("3/3 in-process attempts failed — re-exec quarantine (fresh process leaves sticky per-process BLAS state behind)")
	cmd := exec.Command(os.Args[0],
		"-test.run", "^TestQwenForwardCachedMatchesFull$", "-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(), qwenCachedSubprocEnv+"=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Logf("RECOVERED in fresh subprocess — per-process-sticky platform BLAS numerical nondeterminism, not a cache defect. Child output:\n%s", out)
		return
	}
	t.Errorf("cached-vs-full divergence persists in a fresh subprocess (exec err: %v):\n%s", err, out)
}

// diagnoseCachedDivergence — env-gated failure forensics
// (QWEN_CACHED_DIAG=1). Three discriminating measurements:
//
//  1. Recompute the FULL forward and compare against the last
//     attempt's full logits — a nonzero delta proves the platform's
//     GEMM nondeterminism directly (same process, same inputs), which
//     is what originally exposed the old min-over-attempts discipline
//     as unsound.
//  2. Re-run the cached staircase layer-by-layer and report, per
//     single-token position, the first block whose output hidden state
//     diverges from the full path beyond f32 noise.
//  3. Run the cached staircase twice and compare bit-exactness, to
//     distinguish deterministic wrongness from run-to-run noise.
func diagnoseCachedDivergence(t *testing.T, m *Model, ids []int, full *g.Tensor, chunks [][2]int) {
	t.Helper()
	vocab := full.Shape()[1]

	// (1) full-forward self-consistency.
	full2 := m.Forward(ids, 0)
	var worstFull float64
	var worstIdx int
	a, b := full.Data(), full2.Data()
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worstFull {
			worstFull, worstIdx = d, i
		}
	}
	t.Logf("DIAG full-vs-recomputed-full: max |Δ| = %.3g at flat index %d (pos %d) — nonzero here means the ORIGINAL full forward differs from a recompute in the same process",
		worstFull, worstIdx, worstIdx/vocab)

	// (2) layer-by-layer cached-vs-full hidden states.
	fullH := make([]*g.Tensor, len(m.Blocks))
	g.NoGrad(func() {
		h := m.Embed.Forward(ids)
		for i, blk := range m.Blocks {
			h = blk.Forward(h, 0)
			fullH[i] = h
		}
	})
	dim := m.Cfg.HiddenSize
	g.NoGrad(func() {
		cache := m.NewCache()
		for _, c := range chunks {
			start, end := c[0], c[1]
			h := m.Embed.Forward(ids[start:end])
			for li, blk := range m.Blocks {
				h = blk.ForwardCached(h, cache, li, start)
				// Compare every row of this chunk against the full path.
				hd := h.Data()
				fd := fullH[li].Data()
				var worst float64
				for r := 0; r < end-start; r++ {
					for j := 0; j < dim; j++ {
						d := math.Abs(float64(hd[r*dim+j]) - float64(fd[(start+r)*dim+j]))
						if d > worst {
							worst = d
						}
					}
				}
				if worst > 1e-3 || li == len(m.Blocks)-1 {
					t.Logf("DIAG chunk pos %d..%d layer %d: max |Δh| = %.3g", start, end-1, li, worst)
				}
			}
		}
	})

	// (3) cached determinism: two fresh staircases, bit-compare last row.
	run := func() []float32 {
		cache := m.NewCache()
		var last *g.Tensor
		for _, c := range chunks {
			last = m.ForwardCached(ids[c[0]:c[1]], cache)
		}
		return last.Data()
	}
	r1, r2 := run(), run()
	bitEqual := true
	for i := range r1 {
		if r1[i] != r2[i] {
			bitEqual = false
			break
		}
	}
	t.Logf("DIAG cached-path determinism (two fresh staircases, final row bit-equal): %v", bitEqual)
}

// TestQwenGenerateGreedy — token-id-level greedy generation through the
// KV-cached loop: deterministic across runs, and the first generated
// token must equal the reference argmax (fixture top-1).
func TestQwenGenerateGreedy(t *testing.T) {
	m := loadModelCached(t)
	sf := loadFixtures(t)
	ids := promptIDs(t, sf, "short")

	out1 := Generate(m, ids, Greedy(8))
	out2 := Generate(m, ids, Greedy(8))
	if len(out1) != len(out2) {
		t.Fatalf("greedy generation not deterministic: %d vs %d tokens", len(out1), len(out2))
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("greedy generation not deterministic at %d: %d vs %d", i, out1[i], out2[i])
		}
	}
	if len(out1) <= len(ids) {
		t.Fatalf("no tokens generated (prompt %d, output %d)", len(ids), len(out1))
	}
	refTop1 := topK(fixture(t, sf, "short_logits").Data(), 1)[0]
	if out1[len(ids)] != refTop1 {
		t.Errorf("first greedy token %d != HF reference argmax %d", out1[len(ids)], refTop1)
	}
	t.Logf("prompt %d ids → generated %v", len(ids), out1[len(ids):])
}

// TestQwenParisE2E — text-level end-to-end gate ("What is the capital
// of France?" through the chat template must contain "Paris"). The
// full M0 stitch: Go tokenizer + ChatML renderer + ported model +
// KV-cached greedy generation, no HF anywhere.
func TestQwenParisE2E(t *testing.T) {
	m := loadModelCached(t)
	tokDir := os.Getenv("QWEN_TOKENIZER_DIR")
	if tokDir == "" {
		path, err := FindCheckpoint()
		if err != nil {
			t.Skipf("qwen checkpoint not available: %v", err)
		}
		tokDir = filepath.Dir(path)
	}
	tok, err := model.LoadQwenTokenizer(tokDir)
	if err != nil {
		t.Skipf("qwen tokenizer files not available: %v", err)
	}

	prompt := RenderChatML([]Message{
		{Role: "user", Content: "What is the capital of France? Answer in one short sentence."},
	}, true)
	ids := tok.Encode(prompt)
	out := Generate(m, ids, Greedy(48))
	answer := tok.Decode(out[len(ids):])
	t.Logf("answer: %q", answer)
	if !strings.Contains(answer, "Paris") {
		t.Fatalf("greedy answer does not mention Paris: %q", answer)
	}
}

// TestQwenLoadRejectsBadConfig — fail-loudly loader discipline: a
// config whose shapes disagree with the checkpoint must produce an
// error naming the mismatch, not a silently mis-sized model.
func TestQwenLoadRejectsBadConfig(t *testing.T) {
	path, err := FindCheckpoint()
	if err != nil {
		t.Skipf("qwen checkpoint not available: %v", err)
	}
	if os.Getenv("QWEN_SLOW_LOAD_TESTS") == "" {
		t.Skip("set QWEN_SLOW_LOAD_TESTS=1 to re-parse the 1.2 GB checkpoint with a bad config")
	}
	cfg := Qwen3_0_6B()
	cfg.IntermediateSize = 2048 // wrong on purpose
	if _, err := Load(path, cfg); err == nil {
		t.Fatal("Load accepted a config with the wrong IntermediateSize")
	} else {
		t.Logf("got expected error: %.200s...", err.Error())
	}
}
