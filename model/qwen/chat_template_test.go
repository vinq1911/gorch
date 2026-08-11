//go:build darwin

package qwen

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vinq1911/gorch/model"
)

// defaultQwenTokenizerDir mirrors the model package test convention: local HF
// cache snapshot of Qwen/Qwen3-0.6B, overridable with QWEN_TOKENIZER_DIR.
const defaultQwenTokenizerDir = "/Users/tuomas/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

type chatFixtureCase struct {
	Name                string    `json:"name"`
	Messages            []Message `json:"messages"`
	AddGenerationPrompt bool      `json:"add_generation_prompt"`
	Rendered            string    `json:"rendered"`
	IDs                 []int     `json:"ids"`
}

type chatFixtures struct {
	Model string            `json:"model"`
	Cases []chatFixtureCase `json:"cases"`
}

func loadChatFixtures(t testing.TB) *chatFixtures {
	t.Helper()
	data, err := os.ReadFile("../testdata/qwen_chat_template_fixtures.json")
	if err != nil {
		t.Fatalf("read chat template fixtures: %v", err)
	}
	var fx chatFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("parse chat template fixtures: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("chat template fixtures contain no cases")
	}
	return &fx
}

// TestChatTemplateGoldenRendered checks the fixed ChatML renderer against
// HF apply_chat_template(enable_thinking=False) string output (plan §2.6).
func TestChatTemplateGoldenRendered(t *testing.T) {
	fx := loadChatFixtures(t)
	for _, c := range fx.Cases {
		got := RenderChatML(c.Messages, c.AddGenerationPrompt)
		if got != c.Rendered {
			t.Errorf("case %q: rendered mismatch\n  want: %q\n  got:  %q", c.Name, c.Rendered, got)
		}
	}
}

// TestChatTemplateGoldenIDs is the §2.6 acceptance gate: exact token-id match
// of the rendered template against the HF apply_chat_template goldens.
func TestChatTemplateGoldenIDs(t *testing.T) {
	dir := os.Getenv("QWEN_TOKENIZER_DIR")
	if dir == "" {
		dir = defaultQwenTokenizerDir
	}
	if _, err := os.Stat(filepath.Join(dir, "vocab.json")); err != nil {
		t.Skipf("qwen tokenizer files not available at %s (set QWEN_TOKENIZER_DIR): %v", dir, err)
	}
	tok, err := model.LoadQwenTokenizer(dir)
	if err != nil {
		t.Fatalf("LoadQwenTokenizer(%s): %v", dir, err)
	}

	fx := loadChatFixtures(t)
	for _, c := range fx.Cases {
		got := tok.Encode(RenderChatML(c.Messages, c.AddGenerationPrompt))
		if len(got) != len(c.IDs) {
			t.Errorf("case %q: id length mismatch: got %d, want %d\n  got:  %v\n  want: %v",
				c.Name, len(got), len(c.IDs), got, c.IDs)
			continue
		}
		for i := range got {
			if got[i] != c.IDs[i] {
				t.Errorf("case %q: id mismatch at %d: got %d, want %d", c.Name, i, got[i], c.IDs[i])
				break
			}
		}
	}
}

// TestChatTemplateStopTokens pins the §2.6 stop-token ids.
func TestChatTemplateStopTokens(t *testing.T) {
	if StopTokenImEnd != 151645 {
		t.Errorf("StopTokenImEnd = %d, want 151645", StopTokenImEnd)
	}
	if StopTokenEndOfText != 151643 {
		t.Errorf("StopTokenEndOfText = %d, want 151643", StopTokenEndOfText)
	}
}
