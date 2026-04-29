# Plan 0002: bf16 / fp16 dtype support in gorch

**Status:** proposed
**Tracks alongside:** README roadmap item "fp16/bf16 dtype support
(~2× memory + ~2× compute available on Apple Silicon)" — already public.
**Last updated:** 2026-04-29

## Why

- **Throughput**: Apple Silicon GPU has roughly 2× fp16/bf16 throughput vs
  fp32 across MPS matmul, vDSP element-wise, and SIMD-group fast math.
- **Memory**: 2× reduction in tensor footprint. For LLMs this is the
  difference between fitting Llama 3 8B at fp32 (32 GB activations + weights)
  and fitting it on a 32 GB unified-memory Mac with headroom for KV cache.
- **Pretrained weights are already bf16/fp16 on disk.** Loading them into
  fp32 today is wasted memory; gorch already _reads_ the bf16 dtype but
  promotes everything to fp32 internally.

## Why bf16 over fp16

bf16 is preferable for training because it has the same exponent range as
fp32 (8 bits) and so doesn't need loss scaling. fp16's narrower exponent
range causes gradient underflow without `GradScaler`-style infrastructure.

For inference both are fine; for training, bf16 is the default in PyTorch
and JAX and should be in gorch too.

## What this is not

- Not "support all numerical dtypes." Just bf16 (and fp16 as a near-free
  add-on once bf16 storage exists).
- Not int8/int4 quantization — separate track. This is for native-precision
  training and inference; quantization is a serving optimization.
- Not "automatic mixed precision" with op allowlists. v1 = pick a dtype at
  tensor creation, all ops respect it. AMP can come later.

## Scope of work

This is invasive — touches every kernel in gorch. Estimate 2–3 weeks.

### 1. Tensor dtype field (`tensor.go`)

`Tensor` currently stores `[]float32`. Choices:

A. **Add `Dtype` field + a parallel `data16 []uint16`**, dispatch on dtype
   in every op. Simpler, lower risk.
B. **Generic over numeric type**. Cleaner, but Go generics across CGo
   boundaries is painful and gorch's autograd graph type would explode.

Choose A. The dispatch overhead is negligible compared to op cost, and the
diff stays a clean "every op grows a dtype switch."

### 2. Per-backend kernels

| Backend | Work |
| --- | --- |
| Metal | MPS already supports bf16 matmul. Custom `.metal` kernels need bf16 variants — `vec_gelu`, `vec_softmax`, `add`, `mul`, etc. ~1 wk of shader work. |
| Accelerate (CPU) | `cblas_sgemm` is fp32 only. bf16 matmul on CPU = upcast to fp32 + cblas + downcast. Acceptable; Mac GPU is the real path anyway. |
| Pure-Go fallback | Just bit-reinterpretation; trivial. |

### 3. Autograd

Backward passes must respect dtype. Most ops just propagate; the gotchas:

- Loss values typically computed in fp32 for stability even with bf16
  weights. Match PyTorch's default: cross-entropy in fp32 over bf16 logits.
- Reduction ops (sum, mean) accumulate in fp32 and cast to output dtype.
  Standard mixed-precision pattern.

### 4. Optimizer state

AdamW first/second moment tensors should be **fp32** even when weights are
bf16, otherwise momentum loses precision over many steps. PyTorch does this
by default (`optimizer.state['exp_avg']` is fp32).

This adds memory back: 1B params at bf16 + fp32 Adam state = 8 GB
(2 + 4 + 4 = 10 GB actually). bf16 saves the activation/forward memory,
not the optimizer memory. Worth being honest about.

### 5. Safetensors

gorch already _reads_ bf16 from disk and promotes to fp32. Stop promoting;
keep it native bf16. Saving bf16 needs a write path (probably already
written but un-tested for bf16; verify).

## Order of operations

1. **Decide dtype representation** (Plan A vs B above). Half a day.
2. **Tensor dtype field + creation API**. `g.NewTensorBF16(...)`,
   `g.NewTensorF32(...)`. 1 day.
3. **Bit reinterpretation helpers** (`bf16FromF32`, `bf16ToF32`). Half a
   day.
4. **Pure-Go backend dtype dispatch** for every op. 2–3 days. This is the
   fp32 baseline against which Metal and Accelerate will be checked.
5. **Accelerate backend**: upcast/downcast wrappers around fp32 BLAS for
   bf16 inputs. 2 days.
6. **Metal backend**: bf16 variants of all custom kernels. 1 wk.
7. **Autograd correctness pass**: every op's backward run in bf16, compared
   to fp32 reference on a representative input. Allow 5e-3 relative error
   for bf16 (per PyTorch's tolerances). 3 days.
8. **AdamW master-fp32 state**. 1 day.
9. **Safetensors round-trip test**: load bf16 model, save it, byte-compare.
   1 day.
10. **End-to-end test**: GPT-2 inference in bf16 vs fp32, generated text
    should be identical or nearly so for greedy decode. 1 day.

Total estimate: **2–3 weeks**.

## Interaction with the OpenMythos plan

bf16 is **not** on the v1 critical path. `mythos_tiny` (~5–10 M params)
trains fine in fp32 within 24 h on M-series. bf16 becomes important when
we go past `mythos_tiny`:

- `mythos_1b`: fp32 training is impractical. bf16 mandatory.
- Any Llama 3 8B / Mistral 7B inference at full speed: bf16 mandatory.

Planning order: do bf16 _after_ Phase 4 of the mythos port if the v1 result
justifies a Phase 5 attempt; or do it _before_ if the surrounding gorch
needs it for unrelated reasons (Llama 3 inference, fine-tuning bigger
GPT-2 variants).

## Open questions

1. Does gorch's MPS shim already expose bf16-typed `MPSGraphTensor`?
   Verify before committing to scope.
2. fp16 simultaneously, or bf16-only first? Probably both at once since the
   storage path is identical and only the kernel variants differ. Loss
   scaling for fp16 training is then a separate small PR.
3. What does the public API look like for picking dtype? Tensor-creation
   override, or a global "default dtype" setter? PyTorch has both (`torch.float32`
   default + `torch.set_default_dtype`). Suggest matching that.

## Risk

- **Cascading test breakage.** Every op test today assumes fp32. They'll
  need a `t.Run("fp32", ...)` / `t.Run("bf16", ...)` table-driven structure.
  Estimate +30% test churn during this work.
- **Numerical regressions in non-LM workloads.** MNIST and Fashion-MNIST
  CNNs may need fp32 to keep their accuracy numbers. Plan: dtype is opt-in,
  not a default.

## First PR

`gorch/feature/bf16-tensor`: just the dtype field, the bf16 storage type,
and `NewTensorBF16` constructor. No ops yet — pure data representation.
This shakes out the API decisions (Plan A vs B, default dtype, etc.) before
any kernel work begins.
