//go:build darwin

package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/text/unicode/norm"
)

// defaultQwenTokenizerDir is the local HF cache snapshot of Qwen/Qwen3-0.6B
// (vocab.json + merges.txt + tokenizer_config.json); override with the
// QWEN_TOKENIZER_DIR env var.
const defaultQwenTokenizerDir = "/Users/tuomas/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

// qwenTokenizerDir resolves the directory holding the Qwen tokenizer files,
// following the checkpoint-path convention used by the mimi tests.
func qwenTokenizerDir() string {
	if dir := os.Getenv("QWEN_TOKENIZER_DIR"); dir != "" {
		return dir
	}
	return defaultQwenTokenizerDir
}

func qwenTokenizerForTest(t testing.TB) *QwenTokenizer {
	t.Helper()
	dir := qwenTokenizerDir()
	if _, err := os.Stat(filepath.Join(dir, "vocab.json")); err != nil {
		t.Skipf("qwen tokenizer files not available at %s (set QWEN_TOKENIZER_DIR): %v", dir, err)
	}
	tok, err := LoadQwenTokenizer(dir)
	if err != nil {
		t.Fatalf("LoadQwenTokenizer(%s): %v", dir, err)
	}
	return tok
}

type qwenTokFixtureCase struct {
	Name string `json:"name"`
	Text string `json:"text"`
	IDs  []int  `json:"ids"`
}

type qwenTokFixtures struct {
	Model string               `json:"model"`
	Cases []qwenTokFixtureCase `json:"cases"`
}

func loadQwenTokFixtures(t testing.TB) *qwenTokFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/qwen_tokenizer_fixtures.json")
	if err != nil {
		t.Fatalf("read tokenizer fixtures: %v", err)
	}
	var fx qwenTokFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("parse tokenizer fixtures: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("tokenizer fixtures contain no cases")
	}
	return &fx
}

func idsEqual(a, b []int) bool {
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

// firstDiff returns the first index at which a and b differ (or the shorter
// length if one is a prefix of the other).
func firstDiff(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// TestQwenTokenizerGoldenEncode is the plan 0008 §2.5 acceptance gate:
// exact id-sequence equality against HF Qwen3-0.6B on every fixture case.
func TestQwenTokenizerGoldenEncode(t *testing.T) {
	tok := qwenTokenizerForTest(t)
	fx := loadQwenTokFixtures(t)

	failed := 0
	for _, c := range fx.Cases {
		got := tok.Encode(c.Text)
		if !idsEqual(got, c.IDs) {
			failed++
			if failed <= 10 {
				d := firstDiff(got, c.IDs)
				lo := d - 3
				if lo < 0 {
					lo = 0
				}
				hiG, hiW := d+4, d+4
				if hiG > len(got) {
					hiG = len(got)
				}
				if hiW > len(c.IDs) {
					hiW = len(c.IDs)
				}
				t.Errorf("case %q: id mismatch at %d (got len %d, want len %d)\n  text: %q\n  got[%d:%d]:  %v\n  want[%d:%d]: %v",
					c.Name, d, len(got), len(c.IDs), c.Text, lo, hiG, got[lo:hiG], lo, hiW, c.IDs[lo:hiW])
			}
		}
	}
	if failed > 0 {
		t.Fatalf("%d/%d fixture cases mismatched (showing first 10)", failed, len(fx.Cases))
	}
}

// TestQwenTokenizerRoundTrip checks the §2.5 round-trip property:
// Decode(Encode(x)) == NFC(x) for every fixture case.
func TestQwenTokenizerRoundTrip(t *testing.T) {
	tok := qwenTokenizerForTest(t)
	fx := loadQwenTokFixtures(t)

	failed := 0
	for _, c := range fx.Cases {
		got := tok.Decode(tok.Encode(c.Text))
		want := norm.NFC.String(c.Text)
		if got != want {
			failed++
			if failed <= 10 {
				t.Errorf("case %q: round-trip mismatch\n  in:   %q\n  want: %q\n  got:  %q", c.Name, c.Text, want, got)
			}
		}
	}
	if failed > 0 {
		t.Fatalf("%d/%d round-trip cases mismatched (showing first 10)", failed, len(fx.Cases))
	}
}

// TestQwenTokenizerGoldenDecode decodes each golden id sequence directly and
// expects the NFC-normalized source text back (byte-level BPE is lossless).
func TestQwenTokenizerGoldenDecode(t *testing.T) {
	tok := qwenTokenizerForTest(t)
	fx := loadQwenTokFixtures(t)

	failed := 0
	for _, c := range fx.Cases {
		got := tok.Decode(c.IDs)
		want := norm.NFC.String(c.Text)
		if got != want {
			failed++
			if failed <= 10 {
				t.Errorf("case %q: decode mismatch\n  want: %q\n  got:  %q", c.Name, want, got)
			}
		}
	}
	if failed > 0 {
		t.Fatalf("%d/%d decode cases mismatched (showing first 10)", failed, len(fx.Cases))
	}
}

// TestQwenTokenizerVocab pins the verified §0.2 facts: base vocab 151,643,
// 26 added specials, full vocab size 151,669 distinct ids ending at 151,668.
func TestQwenTokenizerVocab(t *testing.T) {
	tok := qwenTokenizerForTest(t)
	if got := tok.VocabSize(); got != 151669 {
		t.Errorf("VocabSize() = %d, want 151669 (151,643 base + 26 added)", got)
	}
	for text, want := range map[string]int{
		"<|endoftext|>": 151643,
		"<|im_start|>":  151644,
		"<|im_end|>":    151645,
		"<think>":       151667,
		"</think>":      151668,
	} {
		ids := tok.Encode(text)
		if len(ids) != 1 || ids[0] != want {
			t.Errorf("Encode(%q) = %v, want [%d]", text, ids, want)
		}
	}
}
