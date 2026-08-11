//go:build darwin

package qwen

import "testing"

// TestVocabExtIDMap pins the plan §3.2 normative token-id map.
func TestVocabExtIDMap(t *testing.T) {
	if BaseVocabSize != 151936 || AudioTokenBase != 151936 {
		t.Fatal("base vocab / audio base mismatch")
	}
	if NumAudioTokens != 16384 || AudioTokenBase+NumAudioTokens != 168320 {
		t.Fatalf("audio block: %d tokens ending at %d", NumAudioTokens, AudioTokenBase+NumAudioTokens)
	}
	if TokListen != 168320 || TokSpeak != 168321 || TokAudioEnd != 168322 ||
		TokVoiceAz != 168323 || TokVoiceLj != 168324 {
		t.Fatal("special ids off the normative map")
	}
	if TokVoiceLj+1+NumReservedTokens != ExtVocabSize || ExtVocabSize != 168336 {
		t.Fatalf("vocab end: %d reserved after %d gives %d, want 168336",
			NumReservedTokens, TokVoiceLj, TokVoiceLj+1+NumReservedTokens)
	}
	if NumExtTokens != 16400 {
		t.Fatalf("NumExtTokens = %d", NumExtTokens)
	}

	if id := AudioTokenID(0, 0); id != 151936 {
		t.Fatalf("AudioTokenID(0,0) = %d", id)
	}
	if id := AudioTokenID(7, 2047); id != 168319 {
		t.Fatalf("AudioTokenID(7,2047) = %d", id)
	}
	if id := AudioTokenID(1, 3); id != 151936+2048+3 {
		t.Fatalf("AudioTokenID(1,3) = %d", id)
	}
	for _, id := range []int{151936, 168319, 151936 + 2048*3 + 77} {
		cb, code, ok := AudioTokenOf(id)
		if !ok || AudioTokenID(cb, code) != id {
			t.Fatalf("AudioTokenOf(%d) does not invert AudioTokenID: (%d,%d,%v)", id, cb, code, ok)
		}
	}
	if _, _, ok := AudioTokenOf(151935); ok {
		t.Fatal("base-vocab id classified as audio")
	}
	if _, _, ok := AudioTokenOf(168320); ok {
		t.Fatal("special id classified as audio")
	}
}

// TestAudioFrameIDs — flat frame-major interleave, codebook order
// 0..7, no delay pattern (plan §3.2).
func TestAudioFrameIDs(t *testing.T) {
	codes := [][]int{{1, 2}, {3, 4}, {5, 6}} // (3 codebooks, T=2)
	got := AudioFrameIDs(codes)
	want := []int{
		AudioTokenID(0, 1), AudioTokenID(1, 3), AudioTokenID(2, 5),
		AudioTokenID(0, 2), AudioTokenID(1, 4), AudioTokenID(2, 6),
	}
	if len(got) != len(want) {
		t.Fatalf("len %d", len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frame id %d: %d != %d", i, got[i], want[i])
		}
	}
}
