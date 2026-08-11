//go:build darwin

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/model/qwen"
)

// Overfit-100 set construction (plan 0008 §3.6 M1 exit gate): 40
// LISTEN + 40 SPEAK pairs built from the 30 committed real-world
// digit clips (their Mimi tokens are committed in tokens.safetensors,
// 8 codebooks × T frames per clip) plus 20 short synthetic text CHAIN
// samples that reuse a clip's audio as the spoken span. Content
// quality is irrelevant — the gate tests training mechanics.

var overfitDigits = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
var overfitVoices = []string{"alloy", "echo", "shimmer"}

// clipIDs decodes one committed clip's (8, T) code tensor into
// frame-major interleaved extended-vocab ids.
func clipIDs(sf *model.SafetensorsFile, name string) ([]int, error) {
	t, ok := sf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("clip %q missing from tokens.safetensors", name)
	}
	shape := t.Shape()
	if len(shape) != 2 || shape[0] != qwen.AudioNumCodebooks {
		return nil, fmt.Errorf("clip %q has shape %v, want (8, T)", name, shape)
	}
	cb, T := shape[0], shape[1]
	codes := make([][]int, cb)
	for i := 0; i < cb; i++ {
		codes[i] = make([]int, T)
		for j := 0; j < T; j++ {
			codes[i][j] = int(t.Data()[i*T+j])
		}
	}
	return qwen.AudioFrameIDs(codes), nil
}

// tokenizerDir resolves the Qwen tokenizer file directory.
func tokenizerDir(c cliConfig) (string, error) {
	if c.tokDir != "" {
		return c.tokDir, nil
	}
	if d := os.Getenv("QWEN_TOKENIZER_DIR"); d != "" {
		return d, nil
	}
	ckpt, err := qwen.FindCheckpoint()
	if err != nil {
		return "", err
	}
	return filepath.Dir(ckpt), nil
}

// overfitBuilder assembles ChatML-framed samples with spliced audio
// ids and supervised spans.
type overfitBuilder struct {
	tok     *model.QwenTokenizer
	userPre []int // "<|im_start|>user\n"
	mid     []int // "<|im_end|>\n<|im_start|>assistant\n"
	imEnd   int
	samples []data.ShardSample
}

func newOverfitBuilder(tok *model.QwenTokenizer) *overfitBuilder {
	return &overfitBuilder{
		tok:     tok,
		userPre: tok.Encode("<|im_start|>user\n"),
		mid:     tok.Encode("<|im_end|>\n<|im_start|>assistant\n"),
		imEnd:   qwen.StopTokenImEnd,
	}
}

// add appends one sample: user span (text and/or audio ids), then an
// assistant span; supervised positions grade every prediction from
// the first assistant token through the final <|im_end|>, except that
// supStartTok positions the span start (index INTO the assistant
// slice) — used by SPEAK to skip grading the <|speak|><|voice|>
// prefix per §3.3.
func (b *overfitBuilder) add(task string, user, assistant []int, supStartTok int) {
	toks := make([]int, 0, len(b.userPre)+len(user)+len(b.mid)+len(assistant)+1)
	toks = append(toks, b.userPre...)
	toks = append(toks, user...)
	toks = append(toks, b.mid...)
	aStart := len(toks)
	toks = append(toks, assistant...)
	toks = append(toks, b.imEnd)
	// Position i is graded on predicting token i+1: grading targets
	// assistant[supStartTok..] plus the final <|im_end|> means
	// positions [aStart+supStartTok-1, len(toks)-1).
	b.samples = append(b.samples, data.ShardSample{
		Tokens: toks,
		Task:   task,
		Supervised: []data.Span{{
			Start: aStart + supStartTok - 1,
			End:   len(toks) - 1,
		}},
	})
}

// makeOverfit builds the 100-sample shard set into -data.
func makeOverfit(c cliConfig) error {
	if c.dataDir == "" {
		return fmt.Errorf("make-overfit requires -data")
	}
	tokDir, err := tokenizerDir(c)
	if err != nil {
		return err
	}
	tok, err := model.LoadQwenTokenizer(tokDir)
	if err != nil {
		return err
	}
	sf, err := model.LoadSafetensors(filepath.Join(c.realworld, "tokens.safetensors"))
	if err != nil {
		return err
	}

	type clip struct {
		word, voice string
		audio       []int
	}
	var clips []clip
	for _, w := range overfitDigits {
		for _, v := range overfitVoices {
			ids, err := clipIDs(sf, w+"_"+v)
			if err != nil {
				return err
			}
			clips = append(clips, clip{word: w, voice: v, audio: ids})
		}
	}
	sort.Slice(clips, func(i, j int) bool {
		if clips[i].word != clips[j].word {
			return clips[i].word < clips[j].word
		}
		return clips[i].voice < clips[j].voice
	})

	var nListen, nSpeak, nChain int
	if _, err := fmt.Sscanf(c.counts, "%d,%d,%d", &nListen, &nSpeak, &nChain); err != nil {
		return fmt.Errorf("bad -counts %q (want listen,speak,chain): %w", c.counts, err)
	}

	b := newOverfitBuilder(tok)

	// LISTEN: user = <|listen|> + audio + <|audio_end|>, assistant =
	// transcript. Clips cycle; repeats are identical pairs (consistent
	// targets).
	for i := 0; i < nListen; i++ {
		cl := clips[i%len(clips)]
		user := append([]int{qwen.TokListen}, cl.audio...)
		user = append(user, qwen.TokAudioEnd)
		b.add("listen", user, tok.Encode(cl.word), 0)
	}

	// SPEAK: user = text naming word+voice (unique prompt per
	// clip), assistant = <|speak|><|voice:az|> + audio + <|audio_end|>.
	// Supervised span starts at the audio (skips the speak/voice
	// prefix, §3.3).
	for i := 0; i < nSpeak; i++ {
		cl := clips[i%len(clips)]
		user := tok.Encode(fmt.Sprintf("Say the word %s in the %s voice.", cl.word, cl.voice))
		assistant := []int{qwen.TokSpeak, qwen.TokVoiceAz}
		assistant = append(assistant, cl.audio...)
		assistant = append(assistant, qwen.TokAudioEnd)
		b.add("speak", user, assistant, 2)
	}

	// CHAIN: text question → text answer → spoken answer reusing a
	// clip's audio. Question text is unique per sample.
	for i := 0; i < nChain; i++ {
		cl := clips[(i*3+1)%len(clips)]
		user := tok.Encode(fmt.Sprintf("Task %d: which digit is written %q? Answer, then say it in the %s voice.",
			i, cl.word, cl.voice))
		assistant := tok.Encode(fmt.Sprintf("The digit is %s.", cl.word))
		assistant = append(assistant, qwen.TokSpeak, qwen.TokVoiceAz)
		assistant = append(assistant, cl.audio...)
		assistant = append(assistant, qwen.TokAudioEnd)
		b.add("chain", user, assistant, 0)
	}

	if err := os.MkdirAll(c.dataDir, 0755); err != nil {
		return err
	}
	out := filepath.Join(c.dataDir, "overfit100.bin")
	if err := data.WriteTokenShard(out, b.samples); err != nil {
		return err
	}
	var maxLen int
	for _, s := range b.samples {
		if len(s.Tokens) > maxLen {
			maxLen = len(s.Tokens)
		}
	}
	fmt.Printf("wrote %s: %d samples (%d listen / %d speak / %d chain), longest %d tokens\n",
		out, len(b.samples), nListen, nSpeak, nChain, maxLen)
	return nil
}
