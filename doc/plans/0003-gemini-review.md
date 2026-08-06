# Plan 0003: Review of external advice on scaling gorch toward GPT-4-class LLMs

**Status:** review notes; conclusions folded into plans 0001 and 0002.
**Source:** A Gemini-written advisory on what gorch would need to "support
GPT-4." Captured here so the rationale for our diverging from some of its
recommendations is durable.
**Last updated:** 2026-04-29

## What the advisory got right (and where we already agree)

| Claim | Verdict | Action |
| --- | --- | --- |
| FlashAttention-style fused Q@K^T → mask → softmax → @V kernel keeping QK in threadgroup memory | **Right and valuable.** gorch has `ScaledMatMul` + a basic batched MHA but no single fused Metal kernel for the full attention block. | Land as a Phase 5 item alongside non-MatMul GPU autograd. |
| bf16 / fp16 weight storage doubles bandwidth and effective memory | **Right.** Already on gorch's public roadmap. | Tracked in plan `0002-bf16-support.md`. |
| Quantized weight serving (int4 / Q4_K) for fitting big models on consumer hardware | **Right** for serving / inference. | Out of v1 scope; separate "quantization for serving" track if/when we want llama.cpp interop. |
| RoPE and MoE belong in the roadmap | **Right.** | Both in plan 0001 Phase 1. |

## Where the advisory is behind reality

These items are presented as future work but already exist in gorch.

| Claim | Reality | Reference |
| --- | --- | --- |
| "Implement Zero-Copy unified memory using MTLStorageModeShared" | gorch already does this from day one. README is explicit. | ADR-001 |
| "Add a KVCache type to your nn package" | KV cache exists and is integrated into GPT forward; speedup measured. | `33d97c3`, ADR-011 |
| "Add a GGUF Loader" | gorch loads HuggingFace safetensors (incl. bf16 read), with a streaming loader that halves peak RSS. Safetensors covers Llama 3, Mistral, Qwen, DeepSeek. GGUF is mostly a llama.cpp format. | `c99ccce` |

## Where the advisory is technically wrong

These are recommendations we are explicitly **not** following. Documenting
why so we don't accidentally re-litigate them later.

### "Implement complex-number math for RoPE natively in your Metal shaders"

Wrong direction. Production RoPE implementations (nanoGPT, llama.cpp,
vLLM, HuggingFace transformers) precompute real-valued `(cos, sin)` tables
on host and apply RoPE as **two element-wise multiplies plus a half-vector
swap**. No `view_as_complex`, no polar arithmetic, no complex-typed Metal
buffers. The PyTorch `torch.polar` calls in OpenMythos are an
implementation choice, not a requirement of the algorithm.

Plan 0001 follows the standard real-valued approach.

### "MoE expert parallelism via multiple Metal command queues"

Conflates two unrelated optimizations.

- **Expert parallelism** in MoE means executing different experts on
  different devices, partitioning the experts across hardware. Not
  applicable to a single-GPU Mac.
- **What single-device MoE actually does** is _batched expert dispatch_:
  gather routed tokens by destination expert → grouped matmul (one large
  block-sparse op or a loop over experts each with small contiguous batches)
  → scatter back. This is what NVIDIA's TransformerEngine,
  Megatron-LM's MoE, and DeepSpeed-MoE all do.
- **Multiple Metal command queues** are useful for **I/O / compute overlap**
  (one queue copying weights while another runs matmul). They do not
  parallelize independent expert computations on the same physical GPU
  cores, because the cores are shared.

Plan 0001 specifies batched expert dispatch. Plan does **not** allocate
multiple command queues for MoE.

### "GPT-4 supporting engine" as a stated goal

GPT-4's architecture is not public. There's nothing concrete to support.

The honest re-framing is **"GPT-4-class open LLMs"**: Llama 3 8B / 70B,
Mistral / Mixtral, DeepSeek-V2 and V3, Qwen 2.5. All are decoder-only
transformers and share the same primitive set:

- Linear, Embedding (already in gorch)
- RMSNorm (planned)
- Multi-head attention with KV cache (gorch has standard MHA + cache)
- GQA or MLA (planned — both)
- RoPE (planned)
- SwiGLU FFN, optionally MoE (planned)

In other words the same Phase 1 list that unblocks `mythos_tiny` also
unblocks running Llama 3 8B inference on gorch with no further primitives
needed. That is the practical "GPT-4-class" target.

## What the advisory missed

Items absent from the Gemini advisory but central to actually shipping a
trainable / serveable LLM stack on gorch.

1. **Non-MatMul GPU autograd is still CPU-only.** ADR-009-fix is explicit:
   LayerNorm/Softmax/GELU backward run on CPU. For training transformer
   shapes >1G FMAs this is a bigger bottleneck than a missing FlashAttention
   kernel. Should be tackled before bf16, since it has no design risk and
   immediate measurable speedup.
2. **AdamW + gradient clipping.** Adam exists, weight decay does not. No
   `clip_grad_norm`. Both are blocking for any serious training run; both
   are in plan 0001 Phase 1.
3. **Distinction between training and serving precision.** The advisory
   blends "use bf16 to fit GPT-4 in VRAM" (serving) with "support training
   on Mac hardware" (gorch's stated value prop). bf16 helps both; int4
   quantization is serving-only. Plan 0002 calls this out explicitly.
4. **Streaming dataset loaders for training at scale.** TinyStories
   (~500 MB) fits in RAM; FineWeb-Edu (~1.3T tokens) does not. Plan 0001
   defers this to Phase 5 but flags it.

## Net effect on our existing plans

- One genuine addition: native Metal **FlashAttention-2** kernel as a
  Phase 5 item, alongside non-MatMul GPU autograd and bf16. Group as
  "Metal kernel performance pass," ~6–8 weeks of upstream gorch work in
  parallel with mythos v1.
- No changes to Phases 1–4 of plan 0001.
- bf16 plan 0002 unchanged.

## Lessons for future external advisories

When external sources recommend changes to gorch, check before acting:

1. **What does gorch already have?** README + `doc/decisions.md` + recent
   commit log is usually faster than re-deriving.
2. **Is the recommendation production-standard or bespoke?** "Complex-
   number RoPE in Metal" sounds plausible if you only read the OpenMythos
   PyTorch code; reading nanoGPT or llama.cpp shows the simpler real-valued
   path is what everyone ships.
3. **Does the recommendation conflate orthogonal concerns?** MoE expert
   parallelism vs Metal command queues is the textbook example.
4. **Is the goal even well-defined?** "Support GPT-4" is not a goal; "run
   Llama 3 8B inference" or "fine-tune DeepSeek-V2 on a domain corpus" are
   goals.
