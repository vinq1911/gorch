//go:build darwin

package qwen

import "fmt"

// Vocab-extension token-id map (plan 0008 §3.2 — normative).
//
// The base Qwen2.5/3 vocabulary occupies ids 0..151,935. Appended
// audio tokens map Mimi RVQ codes with 8 codebooks × 2048 codes, in
// Quantizer.Encode level order (codebook 0 = semantic, 1..7 acoustic
// residuals). Frame layout is 8 tokens per 12.5 Hz frame, codebook
// order 0..7, flat interleave, NO acoustic delay pattern (single-
// stream AR model).
const (
	// BaseVocabSize is the Qwen2.5/3 family vocabulary (ids 0..151,935).
	BaseVocabSize = 151936

	// AudioNumCodebooks and AudioCodebookSize fix the audio-token grid:
	// Mimi codebooks 0..7 (level 0 semantic), 2048 codes each.
	AudioNumCodebooks = 8
	AudioCodebookSize = 2048

	// AudioTokenBase is the id of codebook 0, code 0:
	// id = AudioTokenBase + codebook*2048 + code.
	AudioTokenBase = BaseVocabSize // 151,936

	// NumAudioTokens spans ids 151,936..168,319.
	NumAudioTokens = AudioNumCodebooks * AudioCodebookSize // 16,384

	// Control/special ids appended after the audio block (from 168,320).
	TokListen   = 168320 // <|listen|>
	TokSpeak    = 168321 // <|speak|>
	TokAudioEnd = 168322 // <|audio_end|>
	TokVoiceAz  = 168323 // <|voice:az|>
	TokVoiceLj  = 168324 // <|voice:lj|>

	// NumReservedTokens ids (168,325..168,335) are reserved for future
	// control tokens.
	NumReservedTokens = 11

	// ExtVocabSize is the extended vocabulary: 151,936 base + 16,384
	// audio + 5 specials + 11 reserved = 168,336.
	ExtVocabSize = 168336

	// NumExtTokens is the number of appended (trainable) rows.
	NumExtTokens = ExtVocabSize - BaseVocabSize // 16,400
)

// AudioTokenID maps a (codebook, code) pair to its extended-vocab id.
func AudioTokenID(codebook, code int) int {
	if codebook < 0 || codebook >= AudioNumCodebooks {
		panic(fmt.Sprintf("qwen: audio codebook %d out of range [0, %d)", codebook, AudioNumCodebooks))
	}
	if code < 0 || code >= AudioCodebookSize {
		panic(fmt.Sprintf("qwen: audio code %d out of range [0, %d)", code, AudioCodebookSize))
	}
	return AudioTokenBase + codebook*AudioCodebookSize + code
}

// AudioTokenOf inverts AudioTokenID: returns (codebook, code, true)
// when id is an audio token, (0, 0, false) otherwise.
func AudioTokenOf(id int) (codebook, code int, ok bool) {
	if id < AudioTokenBase || id >= AudioTokenBase+NumAudioTokens {
		return 0, 0, false
	}
	off := id - AudioTokenBase
	return off / AudioCodebookSize, off % AudioCodebookSize, true
}

// AudioFrameIDs flattens Mimi codes (numCodebooks, T) — the exact
// shape Quantizer.Encode returns — into frame-major interleaved
// extended-vocab ids: frame t emits codebooks 0..n-1 consecutively.
func AudioFrameIDs(codes [][]int) []int {
	if len(codes) == 0 || len(codes) > AudioNumCodebooks {
		panic(fmt.Sprintf("qwen: AudioFrameIDs requires 1..%d codebook rows, got %d", AudioNumCodebooks, len(codes)))
	}
	T := len(codes[0])
	out := make([]int, 0, len(codes)*T)
	for t := 0; t < T; t++ {
		for cb := 0; cb < len(codes); cb++ {
			out = append(out, AudioTokenID(cb, codes[cb][t]))
		}
	}
	return out
}
