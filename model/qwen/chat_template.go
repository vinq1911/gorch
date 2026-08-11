//go:build darwin

// Package qwen hosts the Qwen3 model port (plan 0008). This file implements
// the fixed ChatML renderer for the no-tools subset of the Qwen3 chat
// template, always with enable_thinking=False semantics (plan §2.6).
package qwen

import "strings"

// Stop token ids for ChatML generation (plan §2.6): generation ends at
// <|im_end|> (primary EOS) or <|endoftext|>.
const (
	StopTokenImEnd     = 151645 // <|im_end|>
	StopTokenEndOfText = 151643 // <|endoftext|>
)

// Message is one chat turn: Role is "system", "user", or "assistant".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// RenderChatML renders messages exactly like HF apply_chat_template for
// Qwen3-0.6B with tokenize=False, enable_thinking=False, and no tools.
// When addGenerationPrompt is true, the assistant prologue including the
// fixed empty think block (enable_thinking=False) is appended.
//
// Semantics ported from the Qwen3 Jinja template, non-tool branches only:
//   - a leading system message renders as <|im_start|>system\n{c}<|im_end|>\n
//   - user (and non-first system) turns render verbatim
//   - assistant turns strip an embedded <think>...</think> block; assistant
//     turns after the last user message re-emit reasoning per the template's
//     last-query rules, matching enable_thinking=False output exactly
func RenderChatML(messages []Message, addGenerationPrompt bool) string {
	var b strings.Builder

	// last_query_index: index of the last user message (template's ns.last_query_index
	// with the tool-response clause dropped — no tools in this subset).
	lastQuery := len(messages) - 1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastQuery = i
			break
		}
	}

	for i, m := range messages {
		switch {
		case m.Role == "system" && i == 0:
			b.WriteString("<|im_start|>system\n")
			b.WriteString(m.Content)
			b.WriteString("<|im_end|>\n")
		case m.Role == "user" || m.Role == "system":
			b.WriteString("<|im_start|>")
			b.WriteString(m.Role)
			b.WriteString("\n")
			b.WriteString(m.Content)
			b.WriteString("<|im_end|>\n")
		case m.Role == "assistant":
			content := m.Content
			reasoning := ""
			if idx := strings.LastIndex(content, "</think>"); idx >= 0 {
				pre := strings.TrimRight(content[:idx], "\n")
				if j := strings.LastIndex(pre, "<think>"); j >= 0 {
					pre = pre[j+len("<think>"):]
				}
				reasoning = strings.TrimLeft(pre, "\n")
				content = strings.TrimLeft(content[idx+len("</think>"):], "\n")
			}
			b.WriteString("<|im_start|>assistant\n")
			if i > lastQuery && (i == len(messages)-1 || reasoning != "") {
				b.WriteString("<think>\n")
				b.WriteString(strings.Trim(reasoning, "\n"))
				b.WriteString("\n</think>\n\n")
				b.WriteString(strings.TrimLeft(content, "\n"))
			} else {
				b.WriteString(content)
			}
			b.WriteString("<|im_end|>\n")
		}
	}

	if addGenerationPrompt {
		// enable_thinking=False: fixed empty think prologue on generated turns.
		b.WriteString("<|im_start|>assistant\n<think>\n\n</think>\n\n")
	}
	return b.String()
}
