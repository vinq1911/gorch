# Plan 0008: Qwen voice LoRA — port Qwen3-0.6B into gorch, chain-of-modality adaptation to Mimi speech tokens, conversational voice demo

Status: **proposed**
Depends on: plan 0006 (Mimi native encoder — **done**), plan 0007 (Mimi native decoder — **done**), plan 0009 (bf16 + Metal training acceleration — **in flight**; this plan's M2+ training throughput and M4 decode rate depend on it, and this plan does NOT design that work).

Goal: a turn-based voice-to-voice demo running entirely in Go on one Mac: mic → gorch Mimi encoder → a LoRA-adapted Qwen instruct model that listens (Mimi tokens → transcript), thinks (text answer), and speaks (Mimi tokens) in one autoregressive pass → gorch `DecodeStream` → speaker.

Non-goals: full-duplex/overlapping speech (Moshi-style dual-stream), multi-speaker voice cloning, realtime *generation* guarantees (turn-based with honest latency), training-throughput engineering (plan 0009), quantized serving (plan 0005, parked).

---

## 0. Ground truth established during exploration

### 0.1 What gorch has today (verified in source)

- **`nn/rmsnorm.go`** — RMSNorm with configurable `Eps` field, full autograd, and a Metal fast path (`RMSNormForwardMetal`). Reusable as-is for both the per-layer norms and Qwen3's per-head QK-norm (it requires 2-D input `(M, dim)`; QK-norm application reshapes `(heads·seq, headDim)` — see §2.3).
- **`nn/gqa.go`** — GQA with RoPE hook, `RepeatInterleave` KV expansion, autograd-aware causal mask/softmax. Two hard-coded assumptions that break for Qwen: `headDim = dim / numQueryHeads` (`NewGQA`) and **no cached decode path** (`Forward(x, startPos)` recomputes full attention; there is no `GQA.ForwardCached`). Also no QK-norm hook, and `nn.Linear` always has a bias tensor.
- **`nn/rope.go`** — precomputed cos/sin tables, `Base` parametrized (Llama-3 uses 500k; Qwen uses 1e6 — just a constructor arg), `RopeLlama` half-rotation style = HF's `rotate_half`, which is what `modeling_qwen2/3.py`'s `apply_rotary_pos_emb` uses. Autograd backward implemented. **Reusable as-is.**
- **`nn/kvcache.go`** — per-layer flat `[]float32` K/V append cache, `Dim` uniform across layers. Works for Qwen (kvDim = numKV·headDim is uniform), but nothing consumes it except the GPT-2 `MultiHeadAttention.ForwardCached` path.
- **`nn/moe.go`** — `Expert` is *exactly* a Qwen MLP: `Wgate/Wup/Wdown` + `g.SwiGLU(gate, up)` (root `silu.go` has autograd-aware `SiLU`/`SwiGLU`). The dense Qwen FFN is an `Expert` without the router.
- **`model/mythos/`** — the modern-block assembly precedent: pre-norm block `h = h + Attn(RMSNorm(h)); h = h + FFN(RMSNorm(h))` (`block.go`), shared RoPE threaded through blocks, tied LM head via `g.MatMulTransB(h, Embed.Weight)` with an autograd re-emission through `MatMul(h, Transpose2D(W))` when `GradEnabled()` (`mythos.go:117-129`). This exact pattern is what the Qwen top-level model copies.
- **`model/safetensors.go`** — streaming safetensors reader with **F32/F16/BF16 → f32 decode** (`decodeBF16` verified) and `SaveSafetensors` (f32 only). Loader precedent in `model/gpt2_loader.go` (name-mapped `copyTensor`, transposition handling, `TieLMHeadToEmbedding`).
- **`model/tokenizer.go`** — GPT-2-style BPE, but with a **simplified word splitter** (`splitIntoWords` splits on spaces only — explicitly not the GPT-2 regex) and silent unknown-token dropping. **Not reusable for Qwen** beyond the byte-encoder table and merge-loop shape; see §2.5.
- **`model/generate.go` + `model/finetune.go`** — sampling (greedy/temp/top-k/top-p), KV-cached generation loop (GPT-typed), `CausalLMLoss` (single-sequence, shift-by-one, `g.CrossEntropyLoss` on 2-D logits). All GPT-typed; Qwen needs its own thin equivalents.
- **`optim/adamw.go`** — AdamW with `m`/`v` state **in memory only; no serialization** → checkpoint/resume gap confirmed.
- **`data/dataloader.go`** — `Dataset` interface is fixed-shape float vectors (`Get(i) (input, target []float32)`). **No token-sequence loader exists** → gap confirmed.
- **`nn/attention_batch.go` / `gpt.go:EncodeBatch`** — `ForwardBatched` is MHA-only and inference-only → batched-autograd gap confirmed. This plan sidesteps it (gradient accumulation over single sequences, §3.6) rather than closing it.
- **LoRA: nothing exists** (`grep -ri lora nn/ model/` → only a mythos comment about deferred depth-wise adapters).
- **Matmuls route through Apple Accelerate** (`ops.go:603` → `accelerate.Sgemm`, AMX-backed) — this materially shapes the throughput estimates in §7.
- **`audio/mimi/`** — encoder (plan 0006) and decoder (plan 0007) are done and judged: `Quantizer.Encode(latent, numQuantizers)` returns `(numQuantizers, T)` codes, **level 0 = semantic codebook, levels 1..31 acoustic residual chain, prefix-consistent** (`Encode(x,8)` = first 8 rows of `Encode(x,32)`), `codebookSize = 2048`. `Stream.Push` = 80 ms/1920-sample chunks → one 12.5 Hz latent frame; `DecodeStream.Push([8]codes)` → 1920 samples; measured ~8.8 ms/chunk encode, ~9.5 ms/frame decode (0006/0007 acceptance). `Quantizer.Decode` accepts 1..32 levels, so **speaking with fewer than 8 codebooks is a legal quality/throughput knob**.
- **`audio/realworld/` + `e2e/mimi_realworld_test.go`** — the judged-evidence conventions this plan reuses: Azure `gpt-realtime` websocket factory (`generate_azure_speech.py`), faster-whisper transcription (`transcribe_check.py`), homophone-aware verdict TSVs (`verdict.py`, GPT-5.6-confirmed ruling), committed tokens + transcripts making Go tests self-contained offline.

### 0.2 Qwen architecture facts — verified

Sources: `transformers/models/qwen{2,3}/modeling_*.py` in the installed ace_step env, plus `config.json` fetched from `Qwen/Qwen3-0.6B` and `Qwen/Qwen2.5-0.5B-Instruct` on 2026-08-11.

| Fact | **Qwen3-0.6B** | **Qwen2.5-0.5B-Instruct** |
|---|---|---|
| hidden_size | 1024 | 896 |
| layers | 28 | 24 |
| Q / KV heads (GQA) | 16 / 8 | 14 / 2 |
| head_dim | **128, explicit config field** — `q_proj: 1024→16·128=2048` (attention inner dim ≠ hidden!) | 64 (= 896/14, derived) |
| attention bias | `attention_bias: false` — **no biases anywhere in attention** | **q/k/v_proj `bias=True`** (hard-coded in `modeling_qwen2.py:134-136`), `o_proj bias=False` |
| QK-norm | **yes** — `q_norm`/`k_norm` = RMSNorm(head_dim=128), applied per-head after projection, before RoPE (`modeling_qwen3.py:183-201`) | no |
| MLP | SwiGLU gate/up/down, no bias, `act=silu`, intermediate **3072** | same shape, intermediate **4864** |
| RoPE | theta **1,000,000**, `rope_scaling: null`, HF `rotate_half` (= gorch `RopeLlama`) | theta 1,000,000, same style |
| RMSNorm eps | 1e-6 | 1e-6 |
| tied embeddings | **true** | **true** |
| vocab_size | 151,936 (ids 151,643–151,935 are specials/padding; `<|endoftext|>`=151643, `<|im_start|>`=151644, `<|im_end|>`=151645; eos = [151645, 151643]) | same 151,936, same specials |
| max context | 40,960 | 32,768 |
| sliding window | null / unused (`use_sliding_window: false`) | 32,768 but `use_sliding_window: false` → **plain causal for both** |
| dtype on disk | bf16, single `model.safetensors` | bf16, single file |
| params | ≈596 M total (≈440 M non-embedding; embed 151,936×1024 ≈ 155.6 M, tied) | ≈494 M total (≈358 M non-embedding) |
| chat template | ChatML `<|im_start|>role\n…<|im_end|>\n`; 4.2 KB Jinja with tool-call branches **and thinking-mode branches** (`<think>…</think>`; `enable_thinking=False` emits an empty think block) | plain ChatML, no thinking |
| tokenizer | byte-level BPE, **NFC normalization first** (`tokenization_qwen2.py:338`), pre-tokenize regex `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, GPT-2 byte↔unicode table, ~151k merges, 26 added special tokens, `split_special_tokens=false` | identical tokenizer family (same files) |

Weights download (~1.5 GB bf16) is an M0 implementation step; only configs were fetched during planning.

---

## A / §1. Model choice and module mapping

### 1.1 Recommendation: **Qwen3-0.6B**, with the block implemented config-driven so Qwen2.5-0.5B is a fallback, not a rewrite

Reasons, in order:

1. **The chain task lives or dies on the text brain.** CHAIN quality = answer quality; Qwen3-0.6B is a full generation newer and markedly stronger as an instruct model at ~the same cost. The speech adapters are trained by us either way; the frozen brain is what we can't improve.
2. **QK-norm is a finetuning-stability gift.** We are appending ~16.5k fresh embedding rows and pushing OOD token statistics through frozen attention. Per-head RMSNorm on q/k bounds attention logits regardless of what the new embeddings do — exactly the failure mode vocab-extension finetunes hit.
3. **No attention biases** → cleaner LoRA math and loader (Qwen2.5's QKV biases are loadable into gorch's always-bias `Linear`, but Qwen3 just zero-freezes them).
4. **Compute cost is a wash.** Per-token MACs ≈ 440M + 156M (head) = **596M** (Qwen3) vs 358M + 136M = **494M** (Qwen2.5) — 17% apart; both are bandwidth-bound at decode (§7) where the gap is the same ratio. Choosing the smaller model does not change any go/no-go outcome; if M0's benchmark misses the budget it misses for both, and the real levers are in §5.4.
5. **Same tokenizer, same vocab, same ChatML** → the tokenizer port and vocab surgery are identical for both; zero waste if we swap.

Costs accepted: explicit `head_dim=128 ≠ hidden/heads` (breaks `NewGQA`'s assumption — needed extension anyway for a cached path), QK-norm (one extra module, golden-tested), thinking-mode chat template (we always render with `enable_thinking=False`, emitting the fixed empty `<think>\n\n</think>\n\n` prologue in assistant turns — pinned by a golden template test).

Fallback trigger: if M0's benchmark (§2.7) shows Qwen3-0.6B decode < 8 tok/s f32 where Qwen2.5-0.5B's geometry would clear a knob-assisted budget, swap = new config + new golden fixtures, ~1 day.

### 1.2 Module mapping

| Qwen3 component | gorch today | Status |
|---|---|---|
| token embedding (151,936×1024) | `nn.Embedding` + `g.EmbeddingLookup` (autograd) | **exists** |
| RMSNorm (eps 1e-6) ×2/layer + final | `nn/rmsnorm.go` | **exists** (set `Eps`) |
| GQA, 16Q/8KV, head_dim 128, no bias | `nn/gqa.go` | **needs extension**: explicit `HeadDim` (q/o at 2048 inner dim), optional bias, QK-norm hook, `ForwardCached` |
| per-head QK RMSNorm(128) | `nn/rmsnorm.go` on reshaped view | **needs thin wiring** (inside extended GQA) |
| RoPE theta 1e6, rotate_half | `nn/rope.go` (`RopeLlama`, `Base` arg) | **exists** |
| SwiGLU MLP 1024→3072→1024 | `nn.Expert` (moe.go) minus router; `g.SwiGLU` | **needs thin new type** (`nn.SwiGLUFFN`, ~40 LOC, or reuse `Expert` directly) |
| pre-norm residual block | `model/mythos/block.go` pattern | **new assembly**, precedent exists |
| tied LM head | `mythos.go` MatMulTransB + autograd re-emission | **exists as pattern** |
| KV cache | `nn/kvcache.go` (uniform kvDim ✓) | **exists**; consumer (`ForwardCached`) is new |
| bf16 safetensors load | `model/safetensors.go` `decodeBF16` | **exists** |
| tokenizer (NFC + Qwen regex + 151k BPE + specials) | `model/tokenizer.go` (wrong splitter, no specials, no NFC) | **new** (§2.5) |
| chat template renderer | — | **new** (small; fixed template, not a Jinja engine) |
| generation loop for Qwen type | `model/generate.go` (GPT-typed) | **new thin port** (sampling helpers reusable) |
| LoRA | — | **new** (§3.1) |
| extended-vocab embedding/head w/ row-masked training | — | **new** (§3.2) |
| packed multi-task token loader | `data/` (fixed-vector only) | **new** (§3.3) |
| checkpoint/resume incl. AdamW state | `SaveSafetensors` only | **new** (§3.4) |
| masked/gathered LM loss | `g.CrossEntropyLoss` + autograd `g.Gather` | **new thin composition** (§3.5) |

---

## B / §2. M0 — port Qwen3-0.6B, golden parity, benchmark

Package layout: `model/qwen/` (config.go, block.go, qwen.go, load.go, generate.go, bench), `model/qwen_tokenizer.go` (+`chat_template.go`) in `model/` beside the GPT-2 tokenizer.

### 2.1 Loader — `model/qwen/load.go` (~200 LOC)

- Download `model.safetensors`, `config.json`, `tokenizer.json`/`vocab.json`+`merges.txt`, `tokenizer_config.json`, `generation_config.json` via the `downloadIfMissing` precedent (`model/download.go`).
- HF names: `model.embed_tokens.weight`, `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`, `…self_attn.{q,k,v,o}_proj.weight`, `…self_attn.{q,k}_norm.weight`, `…mlp.{gate,up,down}_proj.weight`, `model.norm.weight`. **Tied-weight handling:** with `tie_word_embeddings: true` the file typically has no `lm_head.weight`; load must (a) alias the head to the embedding tensor (mythos pattern), (b) tolerate a present `lm_head.weight` by verifying it equals `embed_tokens` (some exports materialize it).
- HF `nn.Linear` stores `(out, in)` — same as gorch `Linear.Weight`, **no transpose** (unlike GPT-2 Conv1D). Zero + freeze all gorch bias tensors (Qwen3 has none); for a Qwen2.5 fallback, load q/k/v biases.
- Memory note: f32 in RAM = 2.38 GB; streaming reader keeps transients bounded (safetensors.go doc).

### 2.2 Block assembly — `model/qwen/block.go`, `qwen.go` (~250 LOC)

Mirror mythos: `h = h + Attn(RMSNorm(h), startPos)`, `h = h + FFN(RMSNorm(h))`; final RMSNorm; tied head with the `GradEnabled()` MatMul re-emission. Config struct carries every §0.2 field plus `AttnBias bool`, `UseQKNorm bool`, explicit `HeadDim` — this is what makes Qwen2.5 a config swap.

### 2.3 GQA extension — `nn/gqa.go` (~150 LOC delta)

- `NewGQAConfig{Dim, NumQ, NumKV, HeadDim, Bias bool}`: `Wq: dim→numQ·headDim`, `Wk/Wv: dim→numKV·headDim`, `Wo: numQ·headDim→dim`. Keep `NewGQA` delegating for back-compat (mythos tests pin it).
- QK-norm hook: optional `QNorm, KNorm *RMSNorm` (dim=headDim), applied to the `(heads·seq, headDim)` reshaped view after projection, **before** RoPE — matching `modeling_qwen3.py` order exactly (project → view heads → q_norm/k_norm → transpose → RoPE).
- **`ForwardCached(x, cache, layerIdx, startPos)`**: project new tokens' K/V post-RoPE (RoPE at absolute positions `cache.Len()..`), append flat `(newTokens, numKV·headDim)` rows to `nn.KVCache` (cache created with `dim = numKV·headDim`), attend queries against full cached K/V with GQA group expansion, no mask needed for single-token steps / staircase mask for multi-token prefill chunks. Inference-only (NoGrad), mirrors `MultiHeadAttention.ForwardCached` (attention.go:310) structurally.

### 2.4 Attention-scale/masking notes
Scaling `head_dim^-0.5` (=1/√128) — gorch GQA already scales by `1/sqrt(headDim)` ✓. Causal mask full (no sliding window) ✓.

### 2.5 Qwen tokenizer port — `model/qwen_tokenizer.go` (~450 LOC + tests). **Azure-codex delegation target** (§6)

Scope (each item is real work; the GPT-2 tokenizer shortcuts on all of them):
1. **NFC normalization** (`golang.org/x/text/unicode/norm` — new module dependency, or a vendored minimal NFC table; decide at implementation, prefer x/text).
2. **Pre-tokenizer regex** `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`. Go's `regexp` (RE2) supports `\p{L}/\p{N}` but **not** lookahead `(?!\S)`; port as a hand-written scanner (recommended — deterministic, fast, no dep) implementing the alternation with the lookahead rewritten as "whitespace run: if followed by non-space, yield all but last space". This subtlety is precisely why golden tests arbitrate.
3. Byte-level BPE over ~151k merges with rank-map merging (GPT-2 loop shape reusable; use a proper priority scan, and cache word→tokens like HF does for throughput).
4. **Special-token handling**: split protected specials out of the text before BPE (all 26 `added_tokens_decoder` entries; `split_special_tokens=false`), exact-id emission, plus our appended audio/control specials (§3.2).
5. Decode = inverse byte-map with UTF-8 fallback (GPT-2 path reusable).

**Acceptance gate: golden tests vs HF.** `model/export_qwen_tokenizer_fixtures.py` (ace_step env) dumps `(text, ids)` pairs to a TSV/safetensors fixture for: ASCII prose, contractions, digit runs (regex splits every digit — matters for numbers in answers), CJK, emoji/ZWJ, NFC-vs-NFD input pairs, leading/trailing/multiple spaces, newline runs, ChatML strings with specials, 1k random LibriSpeech transcript lines. Go test: exact id-sequence equality on every case + round-trip `Decode(Encode(x)) == NFC(x)`.

### 2.6 Chat template — `model/qwen/chat_template.go` (~80 LOC)

Not a Jinja engine. A fixed renderer for the no-tools, non-thinking subset: `<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n` (+ the Qwen3 empty-think prologue `<think>\n\n</think>\n\n` on *generated* turns per `enable_thinking=False` semantics). Golden fixture: `tokenizer.apply_chat_template(...)` dumps from HF for 2-, 3-, 5-turn conversations; exact token-id match. Stop tokens: 151645 (`<|im_end|>`), 151643.

### 2.7 Golden parity fixtures + generation benchmark

`model/export_qwen_fixtures.py` (mirrors `audio/export_mimi_fixtures.py` discipline): run HF Qwen3-0.6B **in float32** on 2 fixed prompts (16 and 128 tokens), dump per-stage activations to `model/testdata/qwen_fixtures.safetensors`: embeddings out; layer 0 — post-input-norm, q/k post-qknorm+rope, attn out, post-mlp; layers 13 and 27 block outputs; final norm out; logits rows (last position, full 151,936). Tolerance policy per `encoder_test.go` precedent (dual-metric: relative on `|ref|≥1e-2` + mixed `|a−b| ≤ absTol + relTol·|b|`), budgets loosened one notch vs Mimi (bf16-decoded weights, 28-layer depth): start at relBig 5e-4 / absTol 1e-5 / relTol 5e-4 per stage, logits at 2e-3 relative + **argmax-match on the top-5** for both prompts. Test pyramid: tokenizer goldens → per-stage parity → `ForwardCached`-vs-full-forward equivalence (exact same logits path, tolerance 1e-5) → e2e greedy generation: "What is the capital of France?" through the chat template must contain "Paris".

**Benchmark (`BenchmarkQwenDecode`, `BenchmarkQwenPrefill`)** — the demo's THE-unknown measurement, M0 exit artifact: (a) cached decode tok/s at contexts 128/512/1024/2048, (b) prefill tok/s at 512/1024, f32 CPU Accelerate. Recorded in the plan's M0 results table and fed into §5.3's budget.

#### M0 measured results (2026-08-11, model half)

Apple M4, f32 CPU Accelerate, `model/qwen` KV-cached path (`go test ./model/qwen -bench …`). Median of 3 runs; machine carried background load (load average ~5–6) throughout, run-to-run spread was ±2%.

| Benchmark | Context / tokens | tok/s (median of 3) | runs |
|---|---|---|---|
| `BenchmarkQwenDecode` | 128 | **17.1** | 17.10 / 17.47 / 17.13 |
| `BenchmarkQwenDecode` | 512 | **15.0** | 15.05 / 14.98 / 14.77 |
| `BenchmarkQwenDecode` | 1024 | **12.1** | 11.94 / 12.36 / 12.10 |
| `BenchmarkQwenDecode` | 2048 | **8.7** | 8.65 / 8.82 / 8.67 |
| `BenchmarkQwenPrefill` | 512 | **351** | 348 / 354 / 351 |
| `BenchmarkQwenPrefill` | 1024 | **280** | 285 / 277 / 280 |

Reading against §7's honesty estimates: decode lands exactly in the predicted 8–15 f32 band (upper edge at short context), prefill slightly under the 400–1,000 band at 280–350 tok/s. §5.2 budget with R≈15/P≈350 at demo-typical context: TTFA ≈ 0.6 + 500/350 + 63/15 + 0.01 ≈ **6.2 s**; a 5 s answer (500 audio tokens) speaks in ~33–58 s depending on context growth — "works but patient", as §5.2 predicted; the §5.4 knob ladder plus plan 0009 remain the path to real-time.

Parity measured (Go vs HF float32 eager reference, `model/testdata/qwen_fixtures.safetensors`, prompts 16/128 tokens): embeddings bit-exact; layer-0 norm ≤3e-7 rel; layer-0 q/k post-qknorm+rope ≤1.3e-4 abs (RoPE table f32-pow ulp noise, washed out by attention: layer-0 attn/block out ≤1.4e-4 rel on |ref|≥1e-2); layer-13 out ≤1.3e-3 rel, layer-27 out ≤1.5e-2 rel, final norm ≤4.2e-3 rel, last-position logits ≤3e-3 rel with exact top-5 id match on both prompts. For calibration, the HF reference against itself (eager vs sdpa backends, same weights) shows 1.4e-3 / 1.5e-2 / 3e-3 noise at layers 13 / 27 / logits — the Go port sits inside the reference's own backend-noise envelope, so the deep-stage gates in `model/qwen/qwen_test.go` were set one notch above that envelope (deviation from the §2.7 starting budgets, justification recorded in the test file). ForwardCached-vs-full-forward: ≤5e-5 abs / ≤3e-5 rel on logits (gate 5e-5 + 1e-4·|b|; the plan's 1e-5 is unattainable between two differently-ordered f32 GEMM paths).

M0 effort: loader+block+cached-gen ~4-6 agent-days; tokenizer (codex-assisted) ~2-4; fixtures+bench ~2.

---

## C / §3. M1 — training infrastructure

### 3.1 `nn/lora.go` — `LoRALinear` (~180 LOC + tests)

```go
type LoRALinear struct {
    Base  *Linear   // frozen: Weight/Bias RequiresGrad=false
    A     *g.Tensor // (r, in)  — N(0, 0.02) init
    B     *g.Tensor // (out, r) — zero init ⇒ adapter starts as identity
    Alpha float32; R int
}
// Forward: y = Base(x) + MatMul(MatMul(x, Aᵀ), Bᵀ) · (Alpha/R)
```

- Composition of existing autograd ops (`MatMul`, `Transpose2D`, `Mul`, `Add`) — no new autograd primitives. Frozen base still propagates dX (Linear backward already computes input grads; freezing just makes dW dead — see §3.6 for skipping it).
- `Freeze()` / `Merge()` (fold `B·A·α/r` into `Base.Weight` for zero-overhead inference; keep unmerged copies for resumed training) / `Parameters()` returns `{A, B}` only.
- Wrap targets: q/k/v/o + gate/up/down on all 28 layers. r=16, α=32 default (≈9.6 M trainable for Qwen3 geometry; recompute the precise table at implementation).
- Unit gates: zero-init ⇒ bit-identical to base forward; grad-check A/B against finite differences on a 8×16 toy; merged-vs-unmerged parity 1e-6.

### 3.2 Vocab-extension surgery — `nn/extembed.go` (~200 LOC)

**Design: split tensors, not row masking.** `ExtendedEmbedding{Base *g.Tensor /*frozen (151936,1024)*/, Ext *g.Tensor /*trainable (N_new,1024)*/}`.

- Lookup: partition ids by range, `EmbeddingLookup` per partition (autograd already flows only into tensors that require grad), reassemble rows in order (composition via `Gather`/`ScatterAdd`, both autograd-aware; or a dedicated fused lookup, implementer's choice with the grad-flow test as the gate).
- **Tied head with the same split**: `logits = concat(MatMulTransB(h, Base) /*no grad into Base*/, MatMul(h, Transpose2D(Ext)) /*grad into Ext*/)` along vocab dim. This gives the "only appended rows train" property **structurally** — no per-row gradient masking code to get wrong, and AdamW state is sized to `Ext` (67 MB f32) not the full 172M-row matrix. A `RowMaskedGrad` fallback is explicitly rejected for optimizer-state bloat and masking-bug risk.
- Init of `Ext`: mean of base embedding rows + N(0, 0.02) noise (standard vocab-extension init; keeps new-token logits in-distribution at step 0).

**Token id map** (constants in `model/qwen/vocabext.go`):
- Audio: id = `151936 + codebook*2048 + code`, codebooks 0..7 (0 = semantic — preserving `Quantizer.Encode` level order), ids 151,936–168,319 (16,384).
- Specials from 168,320: `<|listen|>`, `<|speak|>`, `<|audio_end|>`, `<|voice:az|>`, `<|voice:lj|>`, + 11 reserved → new vocab **168,336**.
- Frame layout: 8 tokens per 12.5 Hz frame, codebook order 0..7, **flat interleave, no acoustic delay pattern** (single-stream AR model; Moshi's delay exists for its parallel heads — flagged for the gpt-5.6 design review, §6).

### 3.3 Packed multi-task loader — `data/tokens.go` (~250 LOC)

- On-disk: pre-tokenized shards `{name}.bin` (uint32 LE token ids) + `{name}.idx` JSON: per-sample `{offset, len, task, lossMask spans}`. Built offline by `cmd/voicedata` (§4.2) — training never tokenizes.
- `TokenDataset.Sample(rng) (tokens []int, supervised []int, task string)`: task-ratio-weighted sampling, truncation to `MaxTrainSeq` (1024, §7), `supervised` = indices of positions whose *next-token* prediction is graded (LISTEN: transcript span; SPEAK: audio+`<|audio_end|>` span; CHAIN: transcript+answer+audio spans; never the prompt/user spans).
- Deterministic under seed; epoch bookkeeping for resume.

### 3.4 Checkpoint/resume — `model/qwen/checkpoint.go` (~150 LOC) + `optim/adamw.go` delta (~40 LOC)

- Save: LoRA A/B + `Ext` + AdamW `m`/`v` (trainable params only) + `{step, epoch offsets, rng seed, lr-schedule position, task ratios}` JSON sidecar. One safetensors per checkpoint (~trainable 27 M params → ~330 MB with optimizer state); keep last 3.
- `AdamW` needs `StateTensors()`/`LoadState()` accessors (m/v exposure) — verified absent today.
- Gate: kill-and-resume mid-run reproduces the uninterrupted loss curve to 1e-4 over 20 steps.

### 3.5 Loss with position gathering — `model/qwen/loss.go` (~60 LOC)

Instead of full-sequence logits: `hSup = g.Gather(h_finalnorm, supervisedIdx)` (autograd ✓) → head matmul on gathered rows only → `g.CrossEntropyLoss(logitsSup, targets)`. This is simultaneously the **loss mask** and the fix for the 168k-vocab logits memory blowup (§7: full-seq logits at 1024×168,336 = 690 MB f32 *per copy*; gathered at ~400 supervised positions = 270 MB — and CE-chunking over supervised positions ×4 chunks if needed).

### 3.6 Trainer — `cmd/qwenvoice-train/` (~300 LOC)

Single-sequence forward/backward + **gradient accumulation** (accum 8–16) instead of batched autograd (gap stays open; plan 0009 owns real batching if it wants it). Grad clip (`optim/clip.go` exists), cosine schedule (`optim/scheduler.go` exists), telemetry: per-step loss by task tag, tokens/s, RSS, periodic fixed-prompt generations dumped to a log dir. Frozen-weight dW skipping: `Linear` backward computes dW unconditionally today — add a `Weight.RequiresGrad()==false ⇒ skip dW GEMM` fast path in `nn/module.go` backward (~15 LOC, ~25% step-time saving, §7).

**M1 exit gate — overfit-100:** 100 held pairs (40 LISTEN / 40 SPEAK / 20 CHAIN mini-samples), train to supervised-token loss < 0.1; greedy regeneration reproduces ≥90/100 target spans exactly; checkpoint-resume gate (§3.4); LoRA-zero parity gate (§3.1).

M1 effort: ~5-7 agent-days.

---

## D / §4. M2/M3 — data and training

### 4.1 Token-count math (12.5 Hz × 8 codebooks = **100 audio tokens/s**)

| Corpus | Hours | Audio tokens | Text tokens | Role |
|---|---|---|---|---|
| LibriSpeech train-clean-100 (gold transcripts, free) | 100.6 h | 36.2 M | ≈1.3 M | LISTEN |
| — 10 h starter subset (throughput knob) | 10 h | 3.6 M | 130 k | LISTEN v0 |
| LJSpeech 1.1 (public domain, 13,100 clips, ≤10 s) | 23.9 h | 8.6 M | ≈330 k | SPEAK grounding, voice `<|voice:lj|>` |
| Azure canonical-voice TTS (gpt-realtime, one fixed voice) | 2–5 h | 0.7–1.8 M | — | SPEAK/CHAIN target voice `<|voice:az|>` |
| CHAIN dialogs (2,000 QA) | ~5 h audio | ≈1.8 M | ≈120 k | CHAIN |

Sequence shapes: LISTEN sample = user `<|listen|>` + T·8 audio + `<|audio_end|>` + assistant transcript (~1,300 tokens at the LibriSpeech ~12.7 s mean → **cap LISTEN utterances at ≤9.5 s (≤760 audio tokens)** to fit MaxTrainSeq 1024; drop or split longer ones — keeps ~70% of clips). SPEAK sample = user text + assistant `<|speak|><|voice:x|>` + T·8 audio + `<|audio_end|>` (LJSpeech mean 6.6 s → ~660 audio + ~25 text ≈ 700 ✓). CHAIN sample = user `<|listen|>` + Q audio (~3 s = 300) + assistant: transcript (~15) + answer text (~40) + `<|speak|><|voice:az|>` + answer audio (~5 s = 500) + `<|audio_end|>` ≈ **~900 tokens** ✓.

**Voice reconciliation:** the demo's canonical voice is one fixed Azure gpt-realtime voice (`<|voice:az|>`) because chain answer-speech targets can only be synthesized by an available TTS; LJSpeech (`<|voice:lj|>`) supplies the *bulk* text→speech statistics 24 h cheap. Voice special tokens disambiguate; the demo always prompts `<|voice:az|>`. Knob: if voice mixing degrades az-voice quality, generate +5 h of az-voice reading LJSpeech text (§9 R6).

### 4.2 Data pipeline — `cmd/voicedata/` + `audio/voicedata/*.py`

Go tool: WAV → `Resample`(→24 kHz) → `mimi.Encode` → `Quantizer.Encode(·, 8)` → audio ids → assemble ChatML sample with the ported tokenizer → append to shard. LibriSpeech ships FLAC → offline `ffmpeg` convert step (script, documented; Go stays WAV-only). LJSpeech is 22.05 kHz WAV → resample path must support 22.05→24 k (verify `audio/resample.go` ratio generality early; extend if fixed-ratio, ~1 day).
CHAIN factory (extends `audio/realworld/` scripts): (1) question texts: 2,000 short factual/conversational questions, generated by the base Qwen itself + a curated seed list; (2) question *speech*: gpt-realtime, rotating voices (listening must be speaker-robust), curated with `transcribe_check.py` exactly like the realworld loop; (3) answers: base Qwen3-0.6B (the frozen brain — guarantees the chain target is *reachable* by the model), ≤40 tokens, greedy; (4) answer speech: gpt-realtime canonical voice, ASR-verified. All cached/committed as shards + manifests so training is offline-reproducible.

### 4.3 Curriculum and mixing

- **Stage A (adapters first)** — LISTEN 45% / SPEAK 45% / TEXT-replay 10% (text replay = chat samples answered by the base model itself, guarding the frozen-brain behavior *through the active adapters*). LISTEN input is full 8-codebook from day 1 (matches demo input); a semantic-only-listen variant is trained as an auxiliary task at 10% weight so the latency knob (§5.4) exists without retraining.
- **Stage B (chain second)** — CHAIN 30% / LISTEN 25% / SPEAK 35% / TEXT 10%.
- LR 1e-4 (LoRA) / 5e-4 (Ext rows) — two param groups (AdamW delta: per-group lr, ~20 LOC), cosine to 10%, warmup 300 steps. Budget: Stage A ≈ 20 M supervised-weighted tokens, Stage B ≈ 10 M (see §7 for wall-clock and the 0009 dependency).

### 4.4 Eval gates (judged-evidence conventions from `audio/realworld/`)

- **LISTEN gate:** 50 held-out LibriSpeech dev-clean utterances ≤8 s: word accuracy ≥80% vs gold transcripts (scripted WER, verdict-style TSV committed).
- **SPEAK gate:** 50 held-out sentences → generated tokens → native `Decoder.Decode` → faster-whisper → word accuracy ≥80%; TSV + decoded WAVs committed (the audible artifact, 0007 discipline).
- **CHAIN gate (M3):** 30 held-out spoken questions → full chain → (a) transcript word-accuracy ≥80%, (b) decoded answer speech transcribed, then **answer correctness fuzzy-graded by gpt-5.6** (`verdict.py` extended: judge prompt = question + reference answer + transcribed spoken answer → OK/MISS + reason; committed TSV; judge rulings are evidence, thresholds are the gate) ≥70% OK.
- **Ablation vs no-pretrained baseline:** equal-trainable-parameter from-scratch model (~10-15 M params, same data/steps) on LISTEN+SPEAK; report the word-accuracy delta. Demonstrates (or falsifies) frozen-text-brain transfer.

M2 effort ~6-8 agent-days + training wall-clock; M3 ~4-6 + factory wall-clock.

---

## E / §5. M4 — turn-based demo

### 5.1 Loop — `cmd/qwenvoice/` (~350 LOC)

Mic capture (CoreAudio via `ffmpeg`/`sox` subprocess piping 24 kHz s16le — subprocess acceptable for a demo) → 80 ms chunks → `mimi.Stream.Push` live during speech (encode is realtime: 8.8 ms/80 ms). **End-of-utterance:** RMS-energy silence heuristic — endpoint after 600 ms below threshold following ≥300 ms above. Then: build ChatML prompt (running dialog, `<|listen|>` + audio ids), prefill, generate with per-span stop tokens (transcript → shown as caption; answer text → caption; `<|speak|><|voice:az|>` → audio ids streamed **8-at-a-time** into `DecodeStream.Push` → speaker as generated, stop at `<|audio_end|>`/`<|im_end|>`). Turn appended to history; context trimmed to 2,048 tokens (drop-oldest-turn; audio spans of old turns replaced by their transcripts — cheap context compression, big win at 100 tok/s audio).

### 5.2 Latency budget arithmetic

For a 3 s question and a 5 s answer, with R = decode tok/s and P = prefill tok/s (both measured in M0 §2.7):

- encode: overlapped with speech, tail ≤ 9 ms; silence detector: +600 ms (heuristic floor)
- prefill ≈ (system+history+300 audio tokens) / P ≈ 500/P — at P≈500 tok/s: ~1 s
- transcript+answer text ≈ 55 tokens / R
- **time-to-first-audio** ≈ 0.6 + 500/P + (55+8)/R + 9.5 ms
- answer speech: 500 tokens; real-time playback without gaps needs **R ≥ 100 tok/s**; below that, either generate-then-play (adds 500/R) or paced playback with buffering ratio R/100.

At the honest f32 estimate R ≈ 8–15 tok/s (§7): TTFA ≈ 6–10 s and the 5 s answer takes 33–63 s to speak — **demo works but is patient**; the gate is set accordingly and the knobs below plus plan 0009 close the gap.

### 5.3 M0 benchmark contract
`BenchmarkQwenDecode` defines R(context) and P; M4's budget table is filled from measurements, no hand-waving. Re-run after 0009 lands (bf16 weights + Metal GEMV directly multiply R — decode is bandwidth-bound, §7).

### 5.4 Knobs if R is short (in order)
1. **bf16 weights via plan 0009** (~2× R — bandwidth-bound).
2. **Speak with 4 codebooks** (legal: `Quantizer.Decode` accepts 1..32 prefix; train SPEAK with an auxiliary 4-cb task tag) → halves speak tokens.
3. **Semantic-only LISTEN** (1 codebook, 12.5 tok/s input) → prefill and context shrink 8× on the listen side; trained as auxiliary (§4.3).
4. **Qwen2.5-0.5B swap** (−17% FLOPs/bytes) or depth truncation + re-finetune.
5. **Greedy decoding only** (skip 168k softmax sampling machinery).

M4 gate: live 10-turn session, ≥7 turns judged-correct end-to-end, TTFA and speak-rate recorded against the budget.

---

## F / §6. Azure-assisted coding protocol

**Split rule:** gpt-5.3-codex gets work that is (a) self-contained in 1–3 files, (b) has a reference implementation to mimic, (c) is arbitrated by committed golden tests — codex output is an **untrusted draft**; only tests confer trust. Subagents keep everything touching gorch autograd/loader integration or cross-package invariants.

| → gpt-5.3-codex | → subagents |
|---|---|
| Qwen tokenizer port (§2.5; reference: tokenization_qwen2.py; gate: golden id-equality) | GQA extension + ForwardCached (autograd + cache invariants) |
| chat-template renderer (§2.6; gate: apply_chat_template goldens) | block assembly, loader, tied-head wiring |
| fixture-export scripts | LoRALinear + grad-check tests, ExtendedEmbedding |
| data-prep scripts (flac convert, manifests, CHAIN factory extensions) | packed loader, checkpoint/resume, trainer |
| WER/verdict scoring scripts (extend verdict.py) | demo loop, streaming integration |

**Loop:** (1) subagent writes the golden test FIRST from HF reference dumps and commits fixtures; (2) draft request: `POST ${AZURE_OPENAI_URI_OPENAI}openai/responses?api-version=2025-04-01-preview`, header `api-key: ${AZURE_OPENAI_API_KEY}`, body `{"model":"gpt-5.3-codex","input": <task brief + reference excerpts + test file>}`; (3) run tests, feed failures back verbatim, iterate; (4) **after 3 failed iterations the subagent takes over**. **Never paste secrets, tokens, or env values into prompts.**
**gpt-5.6-terra design-review checkpoints**: (a) end of M1: chain sample format + no-delay interleave decision (§3.2) + LoRA/vocab-surgery numerics; (b) pre-M3: CHAIN curriculum + judge-prompt wording. Review output is advisory; plan gates arbitrate.

---

## G / §7. Compute honesty (M4-class Mac, 24 GB)

**Decode:** bandwidth-bound GEMV. Bytes/token f32 ≈ (440 M non-emb + 172 M head incl. extension) × 4 B ≈ **2.45 GB**. At ~120 GB/s the ceiling is ~49 tok/s; gorch CPU realism ⇒ **expect 8–15 tok/s f32**; bf16 weights (0009) ⇒ ceiling ~98, expect 15–30; Metal GEMV pushes toward the ceiling. M0's benchmark replaces estimates with measurements.

**Prefill:** compute-bound sgemm, ≈1.2 GFLOP/token; Accelerate sustains ~0.5–1.4 TFLOPS f32 ⇒ **400–1,000 tok/s** — not the bottleneck.

**LoRA training step, f32 CPU:** seq 1024, frozen-dW skipping ⇒ ≈ **3.7 GFLOP/token** ⇒ 3.8 TFLOP/step; at 300–600 effective GFLOPS ⇒ **6–13 s per step**, ~80–170 k tokens/hour. Stage A's 20 M tokens ⇒ **5–10 days f32 CPU — not acceptable as the primary path.** Consequences: (a) M2 first-signal runs use the 10 h subset + LJSpeech half (≈6 M tokens ⇒ 1.5–3 days) and must show monotone eval improvement before any full run; (b) the full Stage A/B budget is **gated on plan 0009** (4–6× step speedup ⇒ ~1–2 days); (c) corpus knobs stay wired as config.

**Memory at 24 GB, seq 1024 (f32):** weights 2.45 GB + Ext/LoRA/optimizer ≈ 0.5 GB; attention scores ≈ 3.8 GB; MLP/hidden ≈ 1.7 GB; gathered-loss logits ≈ 0.3–0.7 GB ⇒ **≈9–10 GB peak** — fits with headroom. Seq 2048 does NOT fit (≈15 GB attention term) ⇒ MaxTrainSeq = 1024 hard cap until 0009's memory work.

---

## H / §8. Phases

| Phase | Deliverables | Gate | Effort (agent-days) | Depends on |
|---|---|---|---|---|
| **M0 port** | loader, block, GQA ext (+QK-norm, ForwardCached), tokenizer port, chat template, parity fixtures, decode/prefill benchmark | per-stage parity in tolerance; tokenizer/template golden equality; "Paris" e2e; **benchmark table recorded** | 8–12 | — |
| **M1 training infra** | LoRALinear, ExtendedEmbedding+id map, packed loader, checkpoint/resume, gathered loss, trainer, frozen-dW skip | overfit-100 (≥90/100 exact spans, loss<0.1); resume parity; LoRA-zero parity | 5–7 | M0 |
| **M2 adapters** | data pipeline (cmd/voicedata, resample check), LISTEN+SPEAK Stage A on subset→full | LISTEN ≥80% word acc; SPEAK ≥80% whisper word acc; ablation table | 6–8 + wall-clock | M1; full runs gated on **0009** |
| **M3 chain** | CHAIN factory (2k dialogs), Stage B, judge harness | chain ≥70% judged OK; transcript ≥80% | 4–6 + factory/wall-clock | M2, Azure quota |
| **M4 demo** | mic loop, silence endpoint, streaming speak, context trimming, latency table | 10-turn live session ≥7 correct; TTFA/speak-rate vs budget | 4–5 | M3; rate knobs may pull 0009 |
| **M5 stretch** | Moshi-MLX side-by-side eval, multi-turn memory polish, az-voice-only retrain | comparative writeup | 3–4 | M4 |

## §9. Risks, ranked

1. **TTS direction is hard at 0.6B+LoRA.** Mitigations: single canonical voice; LJSpeech bulk grounding + az-voice targets; raise LoRA r to 32/64 on upper layers before concluding failure; last-resort: unfreeze top-4 blocks (config flag). Early signal: SPEAK overfit-100 and the 10 h run — if whisper can't transcribe *overfit* samples, stop and redesign (semantic-first two-stage speak) before burning compute.
2. **Chain latency / decode rate** (§7): §5.4 knob ladder + 0009; the M0 benchmark makes this measured, not discovered.
3. **Tokenizer-port fidelity** (151k vocab, RE2-incompatible lookahead, NFC): golden-test-first, adversarial fixtures, codex iteration cap; residual risk contained because training and inference share the same Go tokenizer — only HF-parity of fixtures needs exactness.
4. **Exposure bias in the chain**: Stage B variant with 30% model-generated transcripts (regenerated each 2k steps — cheap DAgger flavor); demo displays the transcript so failures are diagnosable.
5. **Context length at 8 cb/frame**: utterance caps (§4.1), transcript-substitution history compression (§5.1), semantic-only-listen escape hatch.
6. **Azure factory throughput/content-filter churn** (4k+ clips): overnight batches with the existing retry loop, incremental ASR-verify; budget 2–4 wall-clock days; LibriSpeech/LJSpeech carry M2 regardless.
7. **Training memory regressions** at 28 layers: RSS telemetry, seq-512 canary config.
8. **f32 CPU too slow and 0009 slips**: M2 subset run sized to complete on f32 alone; full-corpus and M3 shift right rather than die.

## Critical files for implementation

- `nn/gqa.go` — the attention module to extend (explicit head_dim, QK-norm hook, ForwardCached)
- `model/mythos/block.go` (+`mythos.go`) — the modern pre-norm block + tied-head assembly precedent
- `model/safetensors.go` — bf16→f32 loader the Qwen weight loader builds on
- `model/tokenizer.go` — the BPE machinery the Qwen tokenizer port replaces/extends (golden-tested vs tokenization_qwen2.py)
- `audio/mimi/quantizer.go` — token id semantics (8×2048, semantic level 0, prefix-consistent) fixing the vocab-extension map and codebook-count knobs
