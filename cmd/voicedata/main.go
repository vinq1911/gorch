//go:build darwin

// Command voicedata builds the plan 0008 §4.2 corpus→shard pipeline:
// per clip, audio → 24 kHz mono (audio.Resample) → mimi.EncodeWindowed
// → Quantizer.Encode(·, 8) → qwen.AudioFrameIDs → ChatML sample with
// §3.3 supervised spans → packed token shards (data.WriteTokenShard).
//
// Three corpus modes build the Stage-A subset (plan §4.3):
//
//	-corpus librispeech  LISTEN: a deterministic ~10 h subset of
//	                     LibriSpeech train-clean-100. Speaker ids are
//	                     sorted numerically ascending; chapters and
//	                     utterances sorted within; clips longer than
//	                     -max-clip-sec (9.5 s, §4.1) or assembling past
//	                     -max-seq are skipped; selection stops when the
//	                     kept audio reaches -target-hours. Input WAVs
//	                     come from audio/voicedata/convert_librispeech.sh
//	                     (ffmpeg FLAC→WAV at native 16 kHz; this tool
//	                     resamples 16k→24k). The -eval-speakers highest-
//	                     numbered speakers are reserved: excluded from
//	                     training selection, and the held-out LISTEN
//	                     eval manifest draws 50 ≤8 s utterances from
//	                     them — dev-clean is not on disk, so held-out
//	                     train-clean-100 speakers substitute for the
//	                     plan's dev-clean set (documented deviation).
//	                     Measured note: the ≤9.5 s cap keeps only ~9%
//	                     of corpus hours (train-clean-100 clips are
//	                     long), so the "10 h" subset is bounded by the
//	                     whole corpus at ~8-9 h kept.
//	-corpus ljspeech     SPEAK: the deterministic even-index half
//	                     (~12 h) of LJSpeech metadata.csv order, voice
//	                     tag <|voice:lj|>; 50 eval sentences are drawn
//	                     evenly-strided from the odd-index (held-out)
//	                     half.
//	-corpus textreplay   TEXT: ~500 chat samples answered by the frozen
//	                     base Qwen3-0.6B itself (greedy, ChatML with the
//	                     enable_thinking=False prologue) over a
//	                     deterministic seed prompt list (prompts.go) —
//	                     the §4.3 frozen-brain replay guard.
//
// Shards (large) go to -out; manifests + stats (small, committed) go
// to -manifests. Training never tokenizes: shards are self-contained.
package main

import (
	"flag"
	"fmt"
	"os"
)

type cliConfig struct {
	corpus       string
	src          string
	wavDir       string
	out          string
	manifestDir  string
	tokDir       string
	mimiPath     string
	workers      int
	targetHours  float64
	maxClipSec   float64
	maxSeq       int
	evalCount    int
	evalSpeakers int
	evalMaxSec   float64
	numSamples   int
	maxNew       int
	limit        int
}

func main() {
	var c cliConfig
	flag.StringVar(&c.corpus, "corpus", "", "librispeech | ljspeech | textreplay")
	flag.StringVar(&c.src, "src", "", "corpus root (librispeech: .../LibriSpeech/train-clean-100; ljspeech: .../LJSpeech-1.1)")
	flag.StringVar(&c.wavDir, "wav-dir", "", "converted WAV root for librispeech (default ~/speech-corpora/LibriSpeech-wav16k)")
	flag.StringVar(&c.out, "out", "", "shard output directory")
	flag.StringVar(&c.manifestDir, "manifests", "audio/voicedata/manifests", "manifest/stats output directory")
	flag.StringVar(&c.tokDir, "tokenizer-dir", "", "Qwen tokenizer dir (default: QWEN_TOKENIZER_DIR or checkpoint dir)")
	flag.StringVar(&c.mimiPath, "mimi", "", "mimi model.safetensors (default: MIMI_MODEL or HF cache)")
	flag.IntVar(&c.workers, "workers", 5, "parallel Mimi encode workers")
	flag.Float64Var(&c.targetHours, "target-hours", 10, "librispeech: stop after this many kept audio hours")
	flag.Float64Var(&c.maxClipSec, "max-clip-sec", 9.5, "librispeech LISTEN clip cap in seconds (plan §4.1)")
	flag.IntVar(&c.maxSeq, "max-seq", 1024, "drop samples assembling to more tokens than this (MaxTrainSeq)")
	flag.IntVar(&c.evalCount, "eval-count", 50, "held-out eval manifest size")
	flag.IntVar(&c.evalSpeakers, "eval-speakers", 12, "librispeech: reserve the N highest-numbered speakers for eval")
	flag.Float64Var(&c.evalMaxSec, "eval-max-sec", 8.0, "librispeech eval utterance duration cap (plan §4.4)")
	flag.IntVar(&c.numSamples, "n", 500, "textreplay: number of chat samples")
	flag.IntVar(&c.maxNew, "max-new", 60, "textreplay: max generated tokens per answer")
	flag.IntVar(&c.limit, "limit", 0, "debug: process at most N clips/prompts (0 = no limit)")
	flag.Parse()

	if c.out == "" {
		fatal(fmt.Errorf("missing -out"))
	}
	if err := os.MkdirAll(c.out, 0755); err != nil {
		fatal(err)
	}
	if err := os.MkdirAll(c.manifestDir, 0755); err != nil {
		fatal(err)
	}

	var err error
	switch c.corpus {
	case "librispeech":
		err = buildLibriSpeech(c)
	case "ljspeech":
		err = buildLJSpeech(c)
	case "textreplay":
		err = buildTextReplay(c)
	default:
		err = fmt.Errorf("unknown -corpus %q (want librispeech | ljspeech | textreplay)", c.corpus)
	}
	if err != nil {
		fatal(err)
	}
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "voicedata: %v\n", err)
	os.Exit(1)
}
