//go:build darwin

// Command qwenvoice-train is the plan 0008 §3.6 M1 trainer: LoRA +
// vocab-extension finetuning of a frozen Qwen3-0.6B over packed
// multi-task token shards, single-sequence forward/backward with
// gradient accumulation, cosine+warmup schedule, grad clipping,
// checkpoint/resume, and per-task telemetry.
//
// Modes:
//
//	-mode make-overfit  build the overfit-100 shard set from the 30
//	                    committed real-world clips (§3.6 exit gate)
//	-mode train         train (resumes from -out's latest checkpoint)
//	-mode eval          greedy span-regeneration + supervised-loss eval
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type cliConfig struct {
	mode       string
	dataDir    string
	outDir     string
	realworld  string
	tokDir     string
	steps      int
	accum      int
	lrLoRA     float64
	lrExt      float64
	warmup     int
	minLRFrac  float64
	clip       float64
	wd         float64
	seed       int64
	maxSeq     int
	loraR      int
	loraAlpha  float64
	loraLayers int
	layers     int
	resume     string
	saveEvery  int
	genEvery   int
	evalMaxNew int
	taskRatios string
	dwSkip     bool
	counts     string
	accel      string
	metalMinMM int
}

func main() {
	var c cliConfig
	flag.StringVar(&c.mode, "mode", "train", "train | make-overfit | eval")
	flag.StringVar(&c.dataDir, "data", "", "directory containing token shards (*.bin + *.idx)")
	flag.StringVar(&c.outDir, "out", "", "checkpoint/log directory")
	flag.StringVar(&c.realworld, "realworld", "audio/testdata/realworld", "committed real-world clip dir (make-overfit)")
	flag.StringVar(&c.tokDir, "tokenizer-dir", "", "Qwen tokenizer dir (default: QWEN_TOKENIZER_DIR or checkpoint dir)")
	flag.IntVar(&c.steps, "steps", 300, "total optimizer steps")
	flag.IntVar(&c.accum, "accum", 8, "gradient-accumulation sequences per optimizer step")
	flag.Float64Var(&c.lrLoRA, "lr", 1e-4, "base LR (LoRA A/B group)")
	flag.Float64Var(&c.lrExt, "lr-ext", 5e-4, "LR for the extended-embedding rows group")
	flag.IntVar(&c.warmup, "warmup", 20, "linear warmup steps")
	flag.Float64Var(&c.minLRFrac, "min-lr-frac", 0.1, "cosine floor as a fraction of base LR")
	flag.Float64Var(&c.clip, "clip", 1.0, "grad-norm clip (0 disables)")
	flag.Float64Var(&c.wd, "wd", 0.0, "AdamW weight decay")
	flag.Int64Var(&c.seed, "seed", 42, "dataset sampling seed")
	flag.IntVar(&c.maxSeq, "max-seq", 256, "MaxTrainSeq truncation cap")
	flag.IntVar(&c.loraR, "lora-r", 8, "LoRA rank")
	flag.Float64Var(&c.loraAlpha, "lora-alpha", 16, "LoRA alpha")
	flag.IntVar(&c.loraLayers, "lora-layers", 0, "adapt only the top N layers (0 = all)")
	flag.IntVar(&c.layers, "layers", 0, "truncate the base model to N layers (0 = full depth)")
	flag.StringVar(&c.resume, "resume", "auto", "auto | none | /path/to/ckpt.safetensors")
	flag.IntVar(&c.saveEvery, "save-every", 50, "checkpoint every N optimizer steps")
	flag.IntVar(&c.genEvery, "gen-every", 0, "log a fixed-prompt greedy regeneration every N steps (0 = off)")
	flag.IntVar(&c.evalMaxNew, "eval-max-new", 200, "max regenerated tokens per sample in eval")
	flag.StringVar(&c.taskRatios, "task-ratios", "", "task sampling weights, e.g. listen=0.4,speak=0.4,chain=0.2")
	flag.BoolVar(&c.dwSkip, "dw-skip", true, "skip dW GEMMs for frozen Linear weights (measurement toggle)")
	flag.StringVar(&c.counts, "counts", "40,40,20", "make-overfit sample counts: listen,speak,chain")
	flag.StringVar(&c.accel, "accel", "async", "GPU+bf16 path (plan 0009 X4): async | sync | off (off = the plain CPU f32 path)")
	flag.IntVar(&c.metalMinMM, "metal-min-matmul", 8_000_000, "MatMulMetalThreshold (FMA count) used when -accel is on; keeps short-sequence and LoRA matmuls on the resident GPU path")
	flag.Parse()

	var err error
	switch c.mode {
	case "make-overfit":
		err = makeOverfit(c)
	case "train":
		err = train(c)
	case "eval":
		err = evalRegen(c)
	default:
		err = fmt.Errorf("unknown -mode %q", c.mode)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "qwenvoice-train: %v\n", err)
		os.Exit(1)
	}
}

// parseTaskRatios parses "a=0.4,b=0.6" (empty → nil = uniform).
func parseTaskRatios(s string) (map[string]float64, error) {
	if s == "" {
		return nil, nil
	}
	out := map[string]float64{}
	for _, part := range strings.Split(s, ",") {
		kv := strings.SplitN(strings.TrimSpace(part), "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("bad task ratio %q", part)
		}
		w, err := strconv.ParseFloat(kv[1], 64)
		if err != nil {
			return nil, fmt.Errorf("bad task ratio %q: %w", part, err)
		}
		out[kv[0]] = w
	}
	return out, nil
}
