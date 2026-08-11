#!/usr/bin/env python
"""Export Qwen3 tokenizer + chat-template golden fixtures (plan 0008 §2.5/§2.6).

Runs the HF Qwen3-0.6B tokenizer (transformers, ace_step env) over an
adversarial text corpus and dumps (text, ids) pairs to JSON for the Go port's
golden tests, plus apply_chat_template goldens for the fixed ChatML renderer.

Usage:
    /opt/homebrew/Caskroom/miniconda/base/envs/ace_step/bin/python \
        model/export_qwen_tokenizer_fixtures.py

Outputs (committed):
    model/testdata/qwen_tokenizer_fixtures.json
    model/testdata/qwen_chat_template_fixtures.json

Note: plan §2.5 calls for 1k random LibriSpeech transcript lines; the corpus
was not present at ~/speech-corpora/LibriSpeech at export time, so the script
substitutes 1,000 deterministically generated varied English sentences
(seeded RNG over word/phrase banks with numbers, punctuation, contractions).
"""

import json
import os
import random
import sys
import unicodedata

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")


def adversarial_cases():
    cases = []

    def add(name, text):
        cases.append((name, text))

    # --- ASCII prose ---
    add("ascii_simple", "Hello, world!")
    add("ascii_prose", "The quick brown fox jumps over the lazy dog.")
    add(
        "ascii_paragraph",
        "Tokenizers are tricky. Byte-level BPE operates on UTF-8 bytes, "
        "mapped through a printable alphabet; merges are ranked and applied "
        "greedily until no adjacent pair remains mergeable.",
    )
    add("ascii_single_word", "unbelievable")
    add("ascii_capitalized", "THE QUICK Brown fOX")
    add("empty", "")
    add("single_space", " ")
    add("single_char", "a")

    # --- Contractions (the (?i:'s|'t|'re|'ve|'m|'ll|'d) branch) ---
    add("contraction_basic", "I'm sure it's John's dog, isn't it?")
    add("contraction_all", "he's don't we're I've I'm they'll she'd")
    add("contraction_upper", "HE'S DON'T WE'RE I'VE I'M THEY'LL SHE'D")
    add("contraction_mixed_case", "It'S wEiRd, IsN'T iT? You'Ll see. THEY'RE here. I'D go. WE'VE won.")
    add("contraction_curly_quote", "it’s not a contraction to the regex")
    add("contraction_leading", "'s 't 're alone at start")
    add("contraction_no_word", "the cat 'll not merge")

    # --- Digit runs (regex takes \p{N} one digit at a time) ---
    add("digits_run", "1234567890")
    add("digits_pi", "pi is 3.14159265358979")
    add("digits_grouped", "In 2026 we sold 42,000 units for $1,234.56 total.")
    add("digits_mixed_word", "abc123def456")
    add("digits_unicode", "١٢٣ arabic-indic and １２３ fullwidth")
    add("digits_superscript", "x² + y³ = z⁴")

    # --- CJK ---
    add("cjk_chinese", "今天天气真好，我们去公园散步吧。")
    add("cjk_japanese", "東京タワーはとても高いですね。")
    add("cjk_korean", "안녕하세요, 오늘 날씨가 좋네요.")
    add("cjk_mixed_ascii", "GPU加速で10x speedupを達成した。really 真的吗?")

    # --- Emoji / ZWJ ---
    add("emoji_basic", "thumbs up \U0001f44d and fire \U0001f525!")
    add("emoji_zwj_family", "family: \U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 end")
    add("emoji_skin_tone", "wave \U0001f44b\U0001f3fd bye")
    add("emoji_flag", "flags \U0001f1eb\U0001f1ee \U0001f1ef\U0001f1f5 done")
    add("emoji_vs16", "heart ❤️ plain ❤ done")
    add("emoji_keycap", "press 1️⃣ now")

    # --- NFC vs NFD pairs (normalizer must fold these to identical ids) ---
    nfc_cafe = "café costs 3 euros"
    add("nfc_cafe", nfc_cafe)
    add("nfd_cafe", unicodedata.normalize("NFD", nfc_cafe))
    nfc_angstrom = "Ångström units"
    add("nfc_angstrom", nfc_angstrom)
    add("nfd_angstrom", unicodedata.normalize("NFD", nfc_angstrom))
    nfc_viet = "tiếng Việt rất hay"
    add("nfc_vietnamese", nfc_viet)
    add("nfd_vietnamese", unicodedata.normalize("NFD", nfc_viet))
    add("nfkc_not_applied", "ﬁne ligature stays ① circled stays")

    # --- Space / newline edge cases (lookahead rewrite territory) ---
    add("space_leading", "  hello")
    add("space_trailing", "hello  ")
    add("space_multi_between", "a  b   c    d")
    add("space_only_runs", "     ")
    add("space_before_digit", "x 5 y 42")
    add("space_before_punct", "wait ... what ?!")
    add("newline_single", "line one\nline two")
    add("newline_runs", "para one\n\n\npara two")
    add("newline_crlf", "win\r\nline\r\n\r\nend")
    add("newline_cr_only", "old\rmac\r\rstyle")
    add("space_then_newline", "trailing spaces   \nnext")
    add("newline_then_space", "indent\n    code\n\t\ttabbed")
    add("tabs", "col1\tcol2\t\tcol3")
    add("mixed_ws_soup", " \t \n\t x \t\n  y  \r\n\t")
    add("ws_at_end_lookahead", "word   ")
    add("ws_mid_lookahead", "word   next")
    add("nbsp", "non breaking space")
    add("ideographic_space", "中　文 ideographic")
    add("zero_width_space", "zero​width")

    # --- Punctuation clusters (alt 4: ' ?[^\s\p{L}\p{N}]+[\r\n]*') ---
    add("punct_cluster", "really?!?!?!")
    add("punct_space_prefix", "end . start ,, next !!")
    add("punct_newline_suffix", "done!!!\nnew")
    add("punct_symbols", "a+b=c; d*e/f % g^h & i|j")
    add("punct_brackets", "[foo](bar){baz}<qux>")
    add("punct_quotes", '"quoted" and \'single\' and `backtick`')
    add("punct_unicode", "em—dash «guillemets» …ellipsis")
    add("leading_punct_word", "-hyphenated _underscored @mentioned #tagged")

    # --- ChatML strings with specials ---
    add(
        "chatml_simple",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    )
    add(
        "chatml_system",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHi there!<|im_end|>\n"
        "<|im_start|>assistant\nHello! How can I help?<|im_end|>\n",
    )
    add("special_endoftext", "before<|endoftext|>after")
    add("special_think", "<think>\nreasoning here\n</think>\n\nanswer")
    add("special_tool", "<tool_call>\n{\"name\": \"f\"}\n</tool_call> and <tool_response>ok</tool_response>")
    add("special_fim", "<|fim_prefix|>def f():<|fim_suffix|>    return x<|fim_middle|>")
    add("special_vision", "<|vision_start|><|image_pad|><|vision_end|> caption")
    add("special_adjacent", "<|im_start|><|im_end|><|endoftext|>")
    add("special_broken", "<|im_start|user\nnot a special<|im_end\n<| im_start |>")
    add("special_partial_overlap", "<|im_star<|im_start|>t|>")
    add("special_object_ref", "<|object_ref_start|>cat<|object_ref_end|> <|box_start|>1 2<|box_end|>")
    add("special_quad_repo", "<|quad_start|>q<|quad_end|> <|repo_name|>r<|file_sep|>s<|fim_pad|>")
    add("special_video_vision_pad", "<|video_pad|> and <|vision_pad|> pads")

    return cases


# Word banks for the deterministic sentence generator (LibriSpeech substitute).
_SUBJECTS = [
    "the captain", "an old fisherman", "my sister", "the committee", "a stray dog",
    "the young engineer", "our neighbor", "the orchestra", "a distant relative",
    "the night watchman", "her grandmother", "the two travelers", "a curious child",
    "the shopkeeper", "an ambitious student", "the mountain guide", "the pilot",
    "a weary soldier", "the lighthouse keeper", "the visiting professor",
]
_VERBS = [
    "carried", "discovered", "remembered", "painted", "repaired", "described",
    "abandoned", "collected", "measured", "questioned", "followed", "ignored",
    "celebrated", "examined", "borrowed", "promised", "delivered", "arranged",
    "predicted", "watched",
]
_OBJECTS = [
    "the ancient map", "a basket of apples", "the broken clock", "her favorite book",
    "the wooden bridge", "a letter from abroad", "the silver coin", "his fishing net",
    "the garden gate", "a bundle of firewood", "the church bell", "the empty harbor",
    "a forgotten melody", "the winding road", "the morning newspaper",
    "a jar of honey", "the old piano", "the stone wall", "a pair of boots",
    "the captain's log",
]
_TAILS = [
    "before sunrise", "without a word", "during the storm", "in the village square",
    "after the long winter", "beside the river", "with great care",
    "despite the rain", "on the seventh day", "near the railway station",
    "under the oak tree", "for twenty-seven years", "at half past nine",
    "while the others slept", "against all advice", "in early September",
    "with 3 companions", "for $14.99", "at 6:45 in the morning",
    "along the coastal path",
]
_OPENERS = [
    "", "Yesterday, ", "To everyone's surprise, ", "As the story goes, ",
    "In those days, ", "Later that evening, ", "According to the report, ",
    "Once again, ", "Strangely enough, ", "By all accounts, ",
]
_SECOND = [
    "", " Nobody believed it at first.", " It wasn't easy.", " They'll never forget it.",
    " That's how it began.", " The rest is history.", " No one else had noticed.",
    " It had taken 14 attempts.", " She'd warned them twice.",
    " The villagers didn't mind.",
]


def synthetic_sentences(n=1000, seed=8):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s = (
            rng.choice(_OPENERS)
            + rng.choice(_SUBJECTS)
            + " "
            + rng.choice(_VERBS)
            + " "
            + rng.choice(_OBJECTS)
            + " "
            + rng.choice(_TAILS)
        )
        s = s[0].upper() + s[1:] + rng.choice([".", ".", ".", "!", "?"])
        s += rng.choice(_SECOND)
        out.append(("sentence_%04d" % i, s))
    return out


def chat_conversations():
    return [
        (
            "conv2_system_user",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            True,
        ),
        (
            "conv3_user_asst_user",
            [
                {"role": "user", "content": "Hi! My name is Tuomas."},
                {"role": "assistant", "content": "Hello Tuomas! How can I help you today?"},
                {"role": "user", "content": "Tell me a short joke about tokenizers."},
            ],
            True,
        ),
        (
            "conv5_multi_turn",
            [
                {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
                {"role": "user", "content": "Name a prime number greater than 10."},
                {"role": "assistant", "content": "11 is a prime number greater than 10."},
                {"role": "user", "content": "And one greater than 100?"},
                {"role": "assistant", "content": "101 is prime."},
            ],
            False,
        ),
        (
            "conv3_think_history",
            [
                {"role": "user", "content": "What is 12*12?"},
                {"role": "assistant", "content": "<think>\n12*12 = 144\n</think>\n\nIt is 144."},
                {"role": "user", "content": "And 13*13?"},
            ],
            True,
        ),
        (
            "conv5_ends_user_gen",
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
                {"role": "user", "content": "e"},
            ],
            True,
        ),
    ]


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    cases = adversarial_cases() + synthetic_sentences()
    fixture_cases = []
    for name, text in cases:
        ids = tok.encode(text, add_special_tokens=False)
        fixture_cases.append({"name": name, "text": text, "ids": ids})

    tok_out = {
        "model": "Qwen/Qwen3-0.6B",
        "transformers_version": __import__("transformers").__version__,
        "note": "LibriSpeech absent at export time; sentence_* cases are seeded synthetic English sentences.",
        "cases": fixture_cases,
    }
    tok_path = os.path.join(OUT_DIR, "qwen_tokenizer_fixtures.json")
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok_out, f, ensure_ascii=False, indent=1)
    print("wrote %s (%d cases)" % (tok_path, len(fixture_cases)))

    chat_cases = []
    for name, messages, agp in chat_conversations():
        rendered = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=agp,
            enable_thinking=False,
        )
        ids = tok.encode(rendered, add_special_tokens=False)
        chat_cases.append(
            {
                "name": name,
                "messages": messages,
                "add_generation_prompt": agp,
                "rendered": rendered,
                "ids": ids,
            }
        )

    chat_out = {
        "model": "Qwen/Qwen3-0.6B",
        "transformers_version": __import__("transformers").__version__,
        "enable_thinking": False,
        "cases": chat_cases,
    }
    chat_path = os.path.join(OUT_DIR, "qwen_chat_template_fixtures.json")
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(chat_out, f, ensure_ascii=False, indent=1)
    print("wrote %s (%d cases)" % (chat_path, len(chat_cases)))


if __name__ == "__main__":
    sys.exit(main())
