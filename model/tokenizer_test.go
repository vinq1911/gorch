//go:build darwin

package model

import (
	"testing"
)

func TestSimpleTokenizer(t *testing.T) {
	text := "hello world"
	tok := NewSimpleTokenizer(text)

	// Should have unique chars: ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
	if tok.VocabSize() != 8 {
		t.Fatalf("vocab size = %d, want 8", tok.VocabSize())
	}

	ids := tok.Encode("hello")
	if len(ids) != 5 {
		t.Fatalf("encode 'hello' = %d tokens, want 5", len(ids))
	}

	decoded := tok.Decode(ids)
	if decoded != "hello" {
		t.Fatalf("decode = %q, want 'hello'", decoded)
	}
}

func TestSimpleTokenizerRoundtrip(t *testing.T) {
	corpus := "the quick brown fox jumps over the lazy dog"
	tok := NewSimpleTokenizer(corpus)

	tests := []string{
		"the",
		"fox",
		"the quick brown fox",
		" ",
	}

	for _, text := range tests {
		ids := tok.Encode(text)
		decoded := tok.Decode(ids)
		if decoded != text {
			t.Fatalf("roundtrip %q: got %q", text, decoded)
		}
	}
}

func TestSplitIntoWords(t *testing.T) {
	words := splitIntoWords("hello world foo")
	// "hello", " world", " foo"
	if len(words) != 3 {
		t.Fatalf("got %d words, want 3: %v", len(words), words)
	}
	if words[0] != "hello" {
		t.Fatalf("word[0] = %q, want 'hello'", words[0])
	}
	if words[1] != " world" {
		t.Fatalf("word[1] = %q, want ' world'", words[1])
	}
}

func TestBPEMerge(t *testing.T) {
	// Minimal BPE test with hand-crafted merges
	tok := &BPETokenizer{
		Encoder: map[string]int{
			"h":  0,
			"e":  1,
			"l":  2,
			"o":  3,
			"he": 4,
			"ll": 5,
		},
		Decoder: map[int]string{
			0: "h", 1: "e", 2: "l", 3: "o", 4: "he", 5: "ll",
		},
		BPERanks: map[[2]string]int{
			{"h", "e"}: 0, // highest priority merge
			{"l", "l"}: 1,
		},
	}
	tok.initByteEncoding()

	// BPE on "hello" chars: h, e, l, l, o
	chars := []rune{'h', 'e', 'l', 'l', 'o'}
	result := tok.bpe(chars)

	// Should merge h+e → "he", l+l → "ll", leaving: ["he", "ll", "o"]
	if len(result) != 3 {
		t.Fatalf("bpe result = %v, want 3 tokens", result)
	}
	if result[0] != "he" || result[1] != "ll" || result[2] != "o" {
		t.Fatalf("bpe = %v, want [he, ll, o]", result)
	}
}

func TestBPETokenizer_EncodeBatch_MatchesEncode(t *testing.T) {
	tok := &BPETokenizer{
		Encoder:    map[string]int{"he": 1, "ll": 2, "o": 3, "Ġworld": 4, "Ġ": 5, "world": 6},
		Decoder:    map[int]string{},
		BPERanks:   map[[2]string]int{{"h", "e"}: 0, {"l", "l"}: 1},
		ByteEncode: make(map[byte]rune),
	}
	for i := 0; i < 256; i++ {
		tok.ByteEncode[byte(i)] = rune(i)
	}

	texts := []string{"hello", "world", "hello world", "", "hello"}
	got := tok.EncodeBatch(texts)
	if len(got) != len(texts) {
		t.Fatalf("len=%d; want %d", len(got), len(texts))
	}
	for i, text := range texts {
		want := tok.Encode(text)
		if !equalIntSlice(got[i], want) {
			t.Errorf("EncodeBatch[%d]=%v; Encode(%q)=%v", i, got[i], text, want)
		}
	}
}

func TestBPETokenizer_EncodeBatch_Empty(t *testing.T) {
	tok := &BPETokenizer{}
	if got := tok.EncodeBatch(nil); len(got) != 0 {
		t.Errorf("nil input: got len=%d", len(got))
	}
	if got := tok.EncodeBatch([]string{}); len(got) != 0 {
		t.Errorf("empty input: got len=%d", len(got))
	}
}

func equalIntSlice(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
