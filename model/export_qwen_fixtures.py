"""Export golden parity fixtures for the native Qwen3-0.6B port
(plan doc/plans/0008-qwen-voice-lora.md §2.7).

Runs HF Qwen3-0.6B in FLOAT32 (eager attention) on two fixed token-id
prompts and records per-stage activations so the Go implementation in
model/qwen can be validated stage by stage:

    {tag}_ids         the prompt token ids (as f32 ints)          (S,)
    {tag}_embeddings  model.model.embed_tokens output             (S, 1024)
    {tag}_l0_norm     layer 0 input_layernorm output              (S, 1024)
    {tag}_l0_q        layer 0 Q after q_norm + RoPE               (16, S, 128)
    {tag}_l0_k        layer 0 K after k_norm + RoPE               (8, S, 128)
    {tag}_l0_attn     layer 0 self_attn output (post o_proj,
                      pre-residual)                               (S, 1024)
    {tag}_l0_out      layer 0 block output (post-mlp residual)    (S, 1024)
    {tag}_l13_out     layer 13 block output                       (S, 1024)
    {tag}_l27_out     layer 27 block output                       (S, 1024)
    {tag}_final_norm  model.model.norm output                     (S, 1024)
    {tag}_logits      LAST-position logits row only (size budget:
                      full-sequence logits at a 151,936 vocab
                      would be ~78 MB for the long prompt)        (151936,)

Prompts (token ids hardcoded — the tokenizer half of M0 is a separate
deliverable; parity works at the token-id level):

    short  16 ids  "The capital of France is a grand city known for
                    art, food and wine."
    long  128 ids  a 128-token paragraph about transformer decoding

The q/k capture monkeypatches modeling_qwen3.apply_rotary_pos_emb
(module-level function, not hookable) and records the first call per
forward — layer 0, matching the "post-qknorm+rope" stage the Go GQA
exposes via ProjectQKV.

Run with the ace_step env (transformers 4.57.1 / torch 2.9.1 at
recording time); the checkpoint comes from the HF hub cache:

    /opt/homebrew/Caskroom/miniconda/base/envs/ace_step/bin/python \
        model/export_qwen_fixtures.py [out_dir]

Writes model/testdata/qwen_fixtures.safetensors (~7 MB, budget ≤8 MB).
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

MODEL_ID = "Qwen/Qwen3-0.6B"

PROMPTS = {
    "short": [
        785, 6722, 315, 9625, 374, 264, 6662, 3283, 3881, 369, 1947, 11,
        3607, 323, 13078, 13,
    ],
    "long": [
        34253, 4128, 4119, 1882, 1467, 438, 23700, 315, 43179, 11211, 13,
        8886, 3950, 374, 23844, 311, 264, 27950, 4621, 11, 23507, 1526,
        1657, 13617, 315, 6529, 323, 5395, 44804, 34447, 11, 323, 5499,
        27348, 1182, 8630, 279, 34918, 311, 7023, 1128, 4041, 1790, 13,
        5737, 291, 65489, 6529, 13248, 1376, 323, 897, 14629, 3941, 5203,
        315, 3239, 14629, 11, 892, 14035, 15504, 279, 6500, 429, 3078,
        460, 46719, 47116, 1969, 2506, 304, 4938, 13, 74856, 2309, 70547,
        16164, 1973, 553, 41396, 13530, 315, 11744, 11, 323, 77178,
        48723, 13598, 92495, 15175, 3941, 220, 17, 23, 41315, 13617, 13,
        576, 3974, 13876, 38835, 34208, 916, 220, 16, 18, 1602, 15678,
        12590, 11, 220, 19, 20, 21, 3039, 11, 304, 279, 9255, 12406, 315,
        220, 17, 15, 17, 21, 13,
    ],
}

assert len(PROMPTS["short"]) == 16 and len(PROMPTS["long"]) == 128


@torch.no_grad()
def stages(model, ids: list) -> dict:
    x = torch.tensor([ids], dtype=torch.long)
    captured = {}
    hooks = []

    def keep(name):
        def hook(_mod, _inp, out):
            # Attention modules return (attn_out, attn_weights); decoder
            # layers in transformers ≥4.54 return the bare hidden-states
            # tensor. Either way the payload is (1, S, hidden).
            t = out if isinstance(out, torch.Tensor) else out[0]
            assert t.dim() == 3 and t.shape[0] == 1, (name, t.shape)
            captured.setdefault(name, t.detach()[0])

        return hook

    core = model.model
    hooks.append(core.embed_tokens.register_forward_hook(keep("embeddings")))
    hooks.append(core.layers[0].input_layernorm.register_forward_hook(keep("l0_norm")))
    hooks.append(core.layers[0].self_attn.register_forward_hook(keep("l0_attn")))
    hooks.append(core.layers[0].register_forward_hook(keep("l0_out")))
    hooks.append(core.layers[13].register_forward_hook(keep("l13_out")))
    hooks.append(core.layers[27].register_forward_hook(keep("l27_out")))
    hooks.append(core.norm.register_forward_hook(keep("final_norm")))

    # Layer-0 Q/K post-qknorm+rope: apply_rotary_pos_emb is module-level;
    # capture its first invocation (layer 0) per forward.
    orig_rope = qwen3_mod.apply_rotary_pos_emb

    def patched(q, k, cos, sin, *a, **kw):
        qr, kr = orig_rope(q, k, cos, sin, *a, **kw)
        captured.setdefault("l0_q", qr.detach()[0])  # (16, S, 128)
        captured.setdefault("l0_k", kr.detach()[0])  # (8, S, 128)
        return qr, kr

    qwen3_mod.apply_rotary_pos_emb = patched
    try:
        out = model(x)
    finally:
        qwen3_mod.apply_rotary_pos_emb = orig_rope
        for h in hooks:
            h.remove()

    captured["logits"] = out.logits.detach()[0, -1]  # last position, full vocab
    captured["ids"] = torch.tensor(ids, dtype=torch.float32)
    return captured


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "testdata"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
    )
    assert model.config.tie_word_embeddings
    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)

    tensors = {}
    for tag, ids in PROMPTS.items():
        st = stages(model, ids)
        for name, val in st.items():
            tensors[f"{tag}_{name}"] = val.to(torch.float32).contiguous()
        top5 = torch.topk(st["logits"], 5)
        print(f"{tag}: S={len(ids)} top5 ids {top5.indices.tolist()} "
              f"logits {[round(v, 4) for v in top5.values.tolist()]}")

    path = out_dir / "qwen_fixtures.safetensors"
    save_file(tensors, str(path))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(tensors)} tensors)")
    for k in sorted(tensors):
        print(f"  {k} {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
