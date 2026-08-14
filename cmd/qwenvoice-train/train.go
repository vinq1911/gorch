//go:build darwin

package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"syscall"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/metal"
	"github.com/vinq1911/gorch/model/qwen"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// loadVoiceModel loads the real Qwen3-0.6B checkpoint (optionally
// depth-truncated) and applies the M1 trainable surgery. With -accel
// != off it configures the plan 0009 X4 GPU+bf16 path: Metal init +
// ADR-012 probe, native-bf16 frozen weights, everything Metal-resident,
// a lowered matmul threshold (short sequences and LoRA-rank matmuls
// must stay on the resident GPU path instead of the widen-per-call
// fallback), and async command-buffer dispatch unless -accel=sync
// (the X2b/X3 finding: async wins at seq ≤1024 — the trainer's cap —
// and regresses at 1500).
func loadVoiceModel(c cliConfig) (*qwen.VoiceModel, error) {
	accel := c.accel != "off"
	if c.accel != "off" && c.accel != "sync" && c.accel != "async" {
		return nil, fmt.Errorf("unknown -accel %q (want async | sync | off)", c.accel)
	}
	if accel {
		if _, err := g.InitMetal(); err != nil {
			return nil, fmt.Errorf("-accel=%s needs Metal (use -accel=off for the CPU f32 path): %w", c.accel, err)
		}
		if !qwen.AccelSupported() {
			return nil, fmt.Errorf("MPS bf16 matmul unsupported on this machine (ADR-012 probe failed) — use -accel=off")
		}
		g.MatMulMetalThreshold = c.metalMinMM
		g.SetMetalAsync(c.accel == "async")
		if unsafeNoPurge {
			metal.SetPurgeOnRelease(false)
		}
		// Enforce the footprint ceiling CONTINUOUSLY, not just at
		// micro-step boundaries. The boundary check below runs vmmap after
		// the flush, by which point the dangerous peak is already gone:
		// measured at 28 layers / seq 512 / accum 1, live Metal buffers
		// peaked at 12.3 GB mid-micro-step while the boundary sample read
		// 1.9 GB. Metal buffers are the bulk of the footprint, so a
		// per-allocation ceiling on them is what actually stands between a
		// too-large (accum, seq) and a jetsam event that takes the desktop
		// down with it.
		//
		// Headroom: the ceiling covers Metal only, so leave room for the
		// Go heap and the loaded weights that are already counted in it.
		if c.rssLimitMB > 0 {
			metal.SetLiveBufferLimit(int64(c.rssLimitMB) << 20)
		}
	}

	path, err := qwen.FindCheckpoint()
	if err != nil {
		return nil, err
	}
	cfg := qwen.Qwen3_0_6B()
	loadFull, loadTrunc := qwen.Load, qwen.LoadTruncated
	if accel {
		loadFull, loadTrunc = qwen.LoadNative, qwen.LoadTruncatedNative
	}
	var m *qwen.Model
	if c.layers > 0 && c.layers < cfg.NumLayers {
		cfg.NumLayers = c.layers
		m, err = loadTrunc(path, cfg)
	} else {
		m, err = loadFull(path, cfg)
	}
	if err != nil {
		return nil, err
	}
	vm := qwen.NewVoiceModel(m, qwen.VoiceConfig{
		LoRARank:        c.loraR,
		LoRAAlpha:       float32(c.loraAlpha),
		LoRALayers:      c.loraLayers,
		CheckpointEvery: c.ckptEvery,
	})
	if c.ckptEvery > 0 && c.ckptFlush {
		// Bound peak footprint DURING the backward pass, not just across
		// micro-steps. A recomputed segment's Metal buffers are freed by
		// finalizers, which need a GC; those buffers exert no Go-heap
		// pressure, so nothing triggers one. Without this hook the 28
		// segments' recompute graphs pile up inside a single backward and
		// checkpointing saves nothing.
		g.CheckpointSegmentDone = segmentFlush
	}
	if accel {
		vm.ToMetal(g.MetalDev())
		// Drop the load-time transients (the pre-load random f32 init,
		// the safetensors decode buffers, and the pre-ToMetal CPU
		// slices) before the first step: the training process must
		// enter the loop at its steady-state footprint, not its load
		// peak — under external memory pressure the difference is what
		// jetsam kills (plan 0009 §2.5 SIGKILL class).
		//
		// runtime.GC() alone is NOT enough here: it collects the
		// garbage but leaves the freed spans in the Go heap, so RSS
		// stays at the load peak (the scavenger returns pages to the
		// OS only lazily). debug.FreeOSMemory() forces that return.
		// Load transients are large — the safetensors decode buffers
		// plus the pre-ToMetal CPU copy of every tensor — so this is
		// worth its one-off STW cost exactly once, before the loop.
		flushMetalGraph()
		debug.FreeOSMemory()
	}
	return vm, nil
}

func loadDataset(c cliConfig) (*data.TokenDataset, map[string]float64, error) {
	if c.dataDir == "" {
		return nil, nil, fmt.Errorf("missing -data")
	}
	shards, err := filepath.Glob(filepath.Join(c.dataDir, "*.bin"))
	if err != nil || len(shards) == 0 {
		return nil, nil, fmt.Errorf("no shards in %s", c.dataDir)
	}
	ratios, err := parseTaskRatios(c.taskRatios)
	if err != nil {
		return nil, nil, err
	}
	ds, err := data.LoadTokenDataset(shards, data.TokenDatasetConfig{
		MaxTrainSeq: c.maxSeq,
		TaskRatios:  ratios,
		Seed:        c.seed,
	})
	return ds, ratios, err
}

// lrAt is the cosine-with-warmup schedule as a pure function of the
// optimizer step (1-based) — resumable from the step counter alone.
func lrAt(step, warmup, total int, minFrac float64) float64 {
	if warmup > 0 && step <= warmup {
		return float64(step) / float64(warmup)
	}
	if total <= warmup {
		return 1
	}
	prog := float64(step-warmup) / float64(total-warmup)
	if prog > 1 {
		prog = 1
	}
	return minFrac + (1-minFrac)*0.5*(1+math.Cos(math.Pi*prog))
}

// flushMetalGraph frees the dead autograd graph's Metal buffers
// between accum micro-steps. Metal-backed activations exert no
// Go-heap pressure, and MTLBuffer release runs via finalizers — which
// need a GC to be queued and a beat to actually run (the e2e bench's
// taFlushGC lesson; a single GC leaves the release lagging the
// allocation rate). Cost ~25 ms vs a ~1.5 s micro-step.
//
// THE FENCE IS NOT OPTIONAL IN ASYNC MODE (2026-08-12 post-mortem).
// A command buffer retains every MTLBuffer it encoded until it
// completes. Async dispatch (plan 0009 X2) never blocks the host, so
// the CPU queues an entire micro-step of backward work — 28 layers ×
// ~15 ops — and moves straight on to the next micro-step, allocating
// again while nothing has retired. GC cannot release those buffers:
// they are owned by the Metal command queue, not by Go references.
// Unfenced accum-8 @ seq-1024 reached 57-61 GB RSS on a 24 GB machine
// and was jetsam-killed three times, taking the desktop down with it.
//
// SyncMetal blocks on the last committed command buffer; a single
// queue completes in commit order, so that one wait drains every
// earlier buffer and bounds live GPU memory to ONE micro-step. The
// async win is preserved where it matters (the ~420 chained GPU ops
// inside a micro-step still queue without host stalls); only the
// micro-step boundary is serialized.

// unsafeNoFence disables the micro-step GPU fence. DIAGNOSTIC ONLY —
// it reproduces the unbounded-retention failure described above.
// Never set it for a real run.
var unsafeNoFence = false

// memTrace prints one memory line per accumulation micro-step.
var memTrace = false

// unsafeNoPurge disables the shared-buffer page discard performed when
// a Metal buffer is released. DIAGNOSTIC ONLY — it reproduces the
// footprint-grows-by-cumulative-allocation-volume behaviour that made
// gradient accumulation unbounded. Never set it for a real run.
var unsafeNoPurge = false

// memTraceRegions additionally dumps vmmap's per-region-type breakdown
// each micro-step — the only way to attribute footprint that neither
// the Go heap nor gorch's MTLBuffer accounting can see.
var memTraceRegions = false

// vmmapRegions returns vmmap --summary's region-type table, which
// attributes the physical footprint to categories (IOAccelerator,
// MALLOC, VM_ALLOCATE, ...). Slow; diagnostic only.
func vmmapRegions() string {
	out, err := exec.Command("vmmap", "--summary", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return "vmmap failed: " + err.Error()
	}
	var b strings.Builder
	inTable := false
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "REGION TYPE") {
			inTable = true
		}
		if inTable {
			if strings.TrimSpace(line) == "" {
				break
			}
			b.WriteString("      " + line + "\n")
		}
	}
	return b.String()
}

func flushMetalGraph() {
	if !unsafeNoFence {
		g.SyncMetal() // retire in-flight command buffers → release their MTLBuffers
	}
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

// segmentFlushSleep is the settle beat between the two GCs of
// segmentFlush. Shorter than flushMetalGraph's 10 ms because this runs
// once per checkpoint SEGMENT (28 times per micro-step at
// -checkpoint-every 1, where 10 ms would be 0.3 s of pure sleeping).
// segmentFlushMS exposes that beat as a flag (-checkpoint-flush-ms):
// the right value is a property of how fast the runtime's finalizer
// goroutine drains a segment's buffers, which is worth being able to
// measure rather than guess. Measured at 28 layers, seq 1024, accum 1,
// checkpoint-every 1 — peak physical footprint over the micro-step:
//
//	beat   footprint   step
//	 2 ms    12.9 GB    ceiling abort
//	10 ms     7.4 GB    10.1 s
//
// 10 ms it is. The step-time cost of the extra sleeping is ~6%.
var segmentFlushMS = 10

// segmentFlush is flushMetalGraph's per-segment sibling, installed as
// g.CheckpointSegmentDone. It runs after each recomputed segment's
// backward, and it is what makes checkpointing actually save anything:
// the recompute allocates a full block's activations 28 times per
// micro-step, and unless each one is genuinely freed before the next
// begins, live Metal bytes climb exactly as they did without
// checkpointing.
//
// THE SECOND GC IS NOT OPTIONAL (measured). MTLBuffer release runs from
// a GC FINALIZER, and finalizers are queued by a GC but executed
// asynchronously on the runtime's finalizer goroutine. One bare
// runtime.GC() therefore returns before a single buffer has actually
// been released: at 28 layers / seq 312 that left the peak at 5391 MB
// (vs 7687 MB uncheckpointed) — a third of the expected saving —
// because release lagged allocation all the way down the stack. GC,
// yield long enough for the finalizer goroutine to drain, GC again.
// Same lesson as flushMetalGraph's 10 ms beat, one scope smaller.
func segmentFlush() {
	if !unsafeNoFence {
		g.SyncMetal() // retire command buffers → drop their MTLBuffer refs
	}
	beat := time.Duration(segmentFlushMS) * time.Millisecond
	runtime.GC()
	time.Sleep(beat)
	runtime.GC()
	time.Sleep(beat)
}

// rssMB is the process PEAK resident size. Useful as a high-water
// report, but NOT as a guard input: it can only rise, so a large load
// transient would trip a steady-state ceiling forever after (and it
// cannot show a reclaim working).
func rssMB() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return float64(ru.Maxrss) / (1 << 20) // darwin reports bytes
}

// currentRSSMB reports the process's PHYSICAL FOOTPRINT in MB — the
// metric macOS jetsam actually acts on, and the only one that sees
// GPU/IOAccelerator memory.
//
// Do NOT use ps RSS here (2026-08-12 post-mortem, second finding).
// Measured simultaneously on this trainer at accum 2 / seq 512:
//
//	ps rss             781 MB
//	physical footprint 12.7 GB
//
// A 16x blind spot: an RSS-based ceiling sat at 6.8 GB while the
// process was being SIGKILLed. Metal buffers are unified-memory
// allocations that never appear in RSS. vmmap is slow-ish (it walks
// the region map) but this runs once per optimizer step, and a step
// costs seconds.
func currentRSSMB() float64 {
	out, err := exec.Command("vmmap", "--summary", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return rssMB()
	}
	for _, line := range strings.Split(string(out), "\n") {
		if !strings.HasPrefix(strings.TrimSpace(line), "Physical footprint:") {
			continue
		}
		return parseFootprintMB(strings.TrimSpace(line[strings.Index(line, ":")+1:]))
	}
	return rssMB()
}

// parseFootprintMB converts vmmap's "12.7G" / "781.2M" / "512K" to MB.
func parseFootprintMB(v string) float64 {
	if v == "" {
		return 0
	}
	mult := 1.0
	switch v[len(v)-1] {
	case 'G':
		mult, v = 1024, v[:len(v)-1]
	case 'M':
		mult, v = 1, v[:len(v)-1]
	case 'K':
		mult, v = 1.0/1024, v[:len(v)-1]
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0
	}
	return f * mult
}

func train(c cliConfig) error {
	if c.outDir == "" {
		return fmt.Errorf("missing -out")
	}
	if err := os.MkdirAll(c.outDir, 0755); err != nil {
		return err
	}
	nn.AlwaysComputeLinearDW = !c.dwSkip

	if c.accel != "off" && c.genEvery > 0 {
		// The KV-cached greedy decode is a CPU path; per-token single-row
		// matmuls against bf16 Metal weights would take the widen-per-call
		// fallback. Regeneration telemetry belongs to eval (-accel=off).
		fmt.Println("warning: -gen-every disabled under -accel (decode path is CPU; run -mode eval separately)")
		c.genEvery = 0
	}

	t0 := time.Now()
	vm, err := loadVoiceModel(c)
	if err != nil {
		return err
	}
	ckpt := "off"
	if c.ckptEvery > 0 {
		ckpt = fmt.Sprintf("every %d (flush=%v)", c.ckptEvery, c.ckptFlush)
	}
	fmt.Printf("model loaded in %.1fs (layers=%d lora-r=%d lora-layers=%d dw-skip=%v accel=%s checkpoint=%s)\n",
		time.Since(t0).Seconds(), vm.Base.Cfg.NumLayers, c.loraR, c.loraLayers, c.dwSkip, c.accel, ckpt)

	ds, ratios, err := loadDataset(c)
	if err != nil {
		return err
	}
	fmt.Printf("dataset: %d samples, tasks %v\n", ds.Len(), ds.Tasks())

	_, params := vm.TrainableParams()
	lora := params[:len(params)-1]
	ext := params[len(params)-1:]
	var trainable int
	for _, p := range params {
		trainable += p.Size()
	}
	fmt.Printf("trainable params: %d (%d LoRA tensors + ext rows)\n", trainable, len(lora))

	opt := optim.NewAdamWGroups([]optim.ParamGroup{
		{Params: lora, LR: float32(c.lrLoRA)},
		{Params: ext, LR: float32(c.lrExt)},
	}, float32(c.wd))

	startStep := 0
	var microStep int64
	if c.resume != "none" {
		ckptPath := c.resume
		if c.resume == "auto" {
			if p, ok := qwen.LatestCheckpoint(c.outDir); ok {
				ckptPath = p
			} else {
				ckptPath = ""
			}
		}
		if ckptPath != "" {
			meta, err := qwen.LoadCheckpoint(ckptPath, vm, opt)
			if err != nil {
				return fmt.Errorf("resume from %s: %w", ckptPath, err)
			}
			if meta.DatasetSeed != c.seed {
				return fmt.Errorf("resume: checkpoint dataset seed %d != -seed %d", meta.DatasetSeed, c.seed)
			}
			if err := ds.Restore(data.DatasetState{Seed: meta.DatasetSeed, Draws: meta.DatasetDraws}); err != nil {
				return err
			}
			startStep = meta.Step
			microStep = meta.MicroStep
			fmt.Printf("resumed from %s at step %d (micro %d)\n", ckptPath, startStep, microStep)
		}
	}

	lossLog, err := os.OpenFile(filepath.Join(c.outDir, "losses.tsv"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer lossLog.Close()
	// Header only when the log is genuinely empty. Keying on
	// startStep==0 appended a fresh header on every relaunch of a run
	// that died before its first checkpoint, which then broke naive
	// "tail -1 | cut -f1" progress parsing (2026-08-12 post-mortem).
	if fi, err := lossLog.Stat(); err == nil && fi.Size() == 0 {
		fmt.Fprintln(lossLog, "step\tloss\tlr\ttok_per_s\trss_mb\tper_task")
	}

	one := func(scale float32, s *g.Tensor) *g.Tensor {
		return g.Mul(s, g.NewTensor([]float32{scale}, 1))
	}

	trainStart := time.Now()
	var tokensSeen int64
	for step := startStep + 1; step <= c.steps; step++ {
		lr := lrAt(step, c.warmup, c.steps, c.minLRFrac)
		opt.SetLR(float32(c.lrLoRA * lr))

		taskLoss := map[string]float64{}
		taskCount := map[string]int{}
		var stepLoss float64
		stepStart := time.Now()
		var stepTokens int64
		for a := 0; a < c.accum; a++ {
			tokens, sup, task := ds.Sample()
			microStep++
			if len(sup) == 0 {
				continue
			}
			loss := vm.SupervisedLoss(tokens, sup)
			raw := float64(loss.Data()[0])
			one(1/float32(c.accum), loss).Backward()
			stepLoss += raw
			taskLoss[task] += raw
			taskCount[task]++
			stepTokens += int64(len(tokens))
			if c.accel != "off" {
				flushMetalGraph()
			}
			// Per-micro-step memory trace. The three numbers together
			// localize any retention: heapAlloc is the Go heap,
			// metalLive is gorch's own MTLBuffer accounting, and
			// dtGraphs counts compiled MPSGraph executables — the one
			// consumer of footprint that neither of the first two can
			// see (2026-08-13: footprint compounded per micro-step while
			// both heapAlloc and metalLive sat flat).
			if memTrace {
				var ms runtime.MemStats
				runtime.ReadMemStats(&ms)
				peak, alloc, n := metal.BufferStats()
				fmt.Printf("    micro %d/%d seq %d phys %.0f MB heap %.0f MB metalLive %.0f MB peakLive %.0f MB alloc %.0f MB (%d bufs) dtGraphs %d\n",
					a+1, c.accum, len(tokens), currentRSSMB(),
					float64(ms.HeapAlloc)/(1<<20),
					float64(metal.LiveBufferBytes())/(1<<20),
					float64(peak)/(1<<20), float64(alloc)/(1<<20), n,
					metal.DTGraphCacheLen())
				pg, un := metal.PurgeStats()
				fmt.Printf("      purgedReleases %d unpurged %d\n", pg, un)
				metal.ResetBufferStats()
				if memTraceRegions {
					fmt.Print(vmmapRegions())
				}
			}
			// Footprint must be checked per MICRO-step, not per
			// optimizer step: the growth happens across the accumulation
			// micro-steps, so a once-per-step check never gets to run —
			// the OS SIGKILLs us first (measured: exit 137 before the
			// first step completed at accum 8 / seq 1024).
			if c.rssLimitMB > 0 {
				if cur := currentRSSMB(); cur > float64(c.rssLimitMB) {
					return fmt.Errorf("footprint ceiling exceeded mid-step: %.0f MB > limit %d MB "+
						"(step %d, micro-step %d/%d, seq %d) — lower -accum / -max-seq",
						cur, c.rssLimitMB, step, a+1, c.accum, c.maxSeq)
				}
			}
		}
		if c.clip > 0 {
			optim.ClipGradNorm(params, float32(c.clip))
		}
		opt.Step()
		opt.ZeroGrad()
		tokensSeen += stepTokens

		stepLoss /= float64(c.accum)
		perTask := ""
		for _, task := range ds.Tasks() {
			if n := taskCount[task]; n > 0 {
				perTask += fmt.Sprintf("%s=%.4f(%d) ", task, taskLoss[task]/float64(n), n)
			}
		}
		tokPerS := float64(stepTokens) / time.Since(stepStart).Seconds()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		mb := func(v uint64) float64 { return float64(v) / (1 << 20) }
		fmt.Printf("step %4d/%d loss %.4f lr %.2e %s %.0f tok/s phys %.0f MB %.1fs\n",
			step, c.steps, stepLoss, c.lrLoRA*lr, perTask, tokPerS,
			currentRSSMB(), time.Since(stepStart).Seconds())
		// Where does the footprint live? Go heap vs everything else
		// (Metal buffers are unified-memory allocations outside the Go
		// heap, so a growing gap means GPU-side retention).
		fmt.Printf("    mem: heapAlloc %.0f heapSys %.0f heapIdle %.0f heapReleased %.0f metalLive %.0f MB\n",
			mb(ms.HeapAlloc), mb(ms.HeapSys), mb(ms.HeapIdle), mb(ms.HeapReleased),
			float64(metal.LiveBufferBytes())/(1<<20))
		fmt.Fprintf(lossLog, "%d\t%.6f\t%.4e\t%.1f\t%.0f\t%s\n",
			step, stepLoss, c.lrLoRA*lr, tokPerS, rssMB(), perTask)

		// Self-guard: fail loudly rather than letting the OS jetsam us
		// (which on a saturated 24 GB machine takes the desktop with
		// it — 2026-08-12 post-mortem). Checkpoints already on disk
		// stay valid; the operator lowers accum/max-seq and resumes.
		if cur := currentRSSMB(); c.rssLimitMB > 0 && cur > float64(c.rssLimitMB) {
			lossLog.Sync()
			return fmt.Errorf("RSS ceiling exceeded: current %.0f MB > limit %d MB at step %d (peak %.0f MB) "+
				"(lower -accum / -max-seq, or raise -rss-limit-mb if the machine really has the headroom)",
				cur, c.rssLimitMB, step, rssMB())
		}

		if c.genEvery > 0 && step%c.genEvery == 0 {
			logRegen(vm, ds, 0, c.evalMaxNew)
		}
		if c.saveEvery > 0 && (step%c.saveEvery == 0 || step == c.steps) {
			st := ds.State()
			meta := qwen.CheckpointMeta{
				Step: step, MicroStep: microStep,
				BaseLR:      float32(c.lrLoRA * lr),
				DatasetSeed: st.Seed, DatasetDraws: st.Draws,
				TaskRatios: ratios,
			}
			if p, err := qwen.SaveCheckpoint(c.outDir, vm, opt, meta, 3); err != nil {
				return err
			} else {
				fmt.Printf("saved %s\n", p)
			}
		}
	}
	elapsed := time.Since(trainStart)
	fmt.Printf("done: %d steps in %s (%.1f tok/s avg, peak rss %.0f MB)\n",
		c.steps-startStep, elapsed.Round(time.Second),
		float64(tokensSeen)/elapsed.Seconds(), rssMB())
	return nil
}

// logRegen prints one sample's greedy span regeneration (telemetry).
func logRegen(vm *qwen.VoiceModel, ds *data.TokenDataset, idx, maxNew int) {
	tokens, sup, task := ds.Get(idx)
	if len(sup) == 0 {
		return
	}
	prompt, target := regenSplit(tokens, sup)
	if len(target) > maxNew {
		target = target[:maxNew]
	}
	gen := vm.GenerateGreedy(prompt, len(target), nil)
	match := 0
	for i := range gen {
		if i < len(target) && gen[i] == target[i] {
			match++
		}
	}
	fmt.Printf("  regen[%d,%s]: %d/%d target tokens matched\n", idx, task, match, len(target))
}

// regenSplit derives the regeneration prompt and target span from a
// sample: prompt = everything up to the first supervised prediction,
// target = the supervised targets (positions s0+1 .. lastSup+1).
func regenSplit(tokens, sup []int) (prompt, target []int) {
	s0, sN := sup[0], sup[len(sup)-1]
	return tokens[:s0+1], tokens[s0+1 : sN+2]
}
