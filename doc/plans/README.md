# Plans

Forward-looking design notes that are too speculative to be ADRs (which
record decisions already made and accepted) but durable enough to outlive
a chat session.

Each plan starts with a status (`proposed` / `in progress` / `superseded`),
captures the goal, the trade-offs considered, and explicit non-goals. When
a plan becomes a decision, port the relevant pieces to `doc/decisions.md`
as an ADR and mark the plan superseded.

| File | Topic |
| --- | --- |
| `0001-openmythos-port.md` | Roadmap for porting OpenMythos to Go on gorch (mythos_tiny on TinyStories as v1) |
| `0002-bf16-support.md` | bf16/fp16 dtype support track in gorch (parallel to mythos work) |
| `0003-gemini-review.md` | Review of an external advisory on scaling gorch toward GPT-4-class LLMs |
| `0004-flashattention2.md` | Native Metal FlashAttention-2 + non-matmul GPU autograd (LayerNorm/Softmax/GELU/Embedding backward Metal kernels) — closes the GPU training regression at large shapes |
| `0005-quantization-serving.md` | int8/int4 quantization for serving large pretrained LLMs on memory-constrained Macs (parked unless serving is a stated goal) |
| `0006-mimi-native-encoder.md` | Native Mimi audio-codec encoder inference in Go (Conv1d/ELU/axis reductions, WAV+resampler, SEANet+transformer port, streaming; RVQ optional) — removes Python from the audio feature pipeline |
| `0007-mimi-native-decoder.md` | Native Mimi decoder in Go (ConvTranspose1d + streaming overlap-add; tokens → 24 kHz audio) — removes the reference Python decoder, the last non-Go step in the audio path |
| `0009-training-acceleration-execution.md` | Execution plan operationalizing 0002+0004 for the plan-0008 LoRA workload: GPU-resident autograd, ranked Metal kernel set, bf16 frozen path, Azure-codex kernel protocol; FA2 deferred with arithmetic |
