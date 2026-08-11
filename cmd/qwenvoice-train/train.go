//go:build darwin

package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"syscall"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model/qwen"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
)

// loadVoiceModel loads the real Qwen3-0.6B checkpoint (optionally
// depth-truncated) and applies the M1 trainable surgery.
func loadVoiceModel(c cliConfig) (*qwen.VoiceModel, error) {
	path, err := qwen.FindCheckpoint()
	if err != nil {
		return nil, err
	}
	cfg := qwen.Qwen3_0_6B()
	var m *qwen.Model
	if c.layers > 0 && c.layers < cfg.NumLayers {
		cfg.NumLayers = c.layers
		m, err = qwen.LoadTruncated(path, cfg)
	} else {
		m, err = qwen.Load(path, cfg)
	}
	if err != nil {
		return nil, err
	}
	return qwen.NewVoiceModel(m, qwen.VoiceConfig{
		LoRARank:   c.loraR,
		LoRAAlpha:  float32(c.loraAlpha),
		LoRALayers: c.loraLayers,
	}), nil
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

func rssMB() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return float64(ru.Maxrss) / (1 << 20) // darwin reports bytes
}

func train(c cliConfig) error {
	if c.outDir == "" {
		return fmt.Errorf("missing -out")
	}
	if err := os.MkdirAll(c.outDir, 0755); err != nil {
		return err
	}
	nn.AlwaysComputeLinearDW = !c.dwSkip

	t0 := time.Now()
	vm, err := loadVoiceModel(c)
	if err != nil {
		return err
	}
	fmt.Printf("model loaded in %.1fs (layers=%d lora-r=%d lora-layers=%d dw-skip=%v)\n",
		time.Since(t0).Seconds(), vm.Base.Cfg.NumLayers, c.loraR, c.loraLayers, c.dwSkip)

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
	if startStep == 0 {
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
		fmt.Printf("step %4d/%d loss %.4f lr %.2e %s %.0f tok/s rss %.0f MB %.1fs\n",
			step, c.steps, stepLoss, c.lrLoRA*lr, perTask, tokPerS,
			rssMB(), time.Since(stepStart).Seconds())
		fmt.Fprintf(lossLog, "%d\t%.6f\t%.4e\t%.1f\t%.0f\t%s\n",
			step, stepLoss, c.lrLoRA*lr, tokPerS, rssMB(), perTask)

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
