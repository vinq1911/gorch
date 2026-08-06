"""Export golden fixtures for the native Mimi encoder (plan 0006, Phase 0).

Runs the kyutai/mimi ENCODER (transformers reference; generated with
transformers 4.57.1, torch 2.9.1) on deterministic inputs and records
per-stage activations so the Go implementation in audio/mimi can be
validated stage by stage:

    {sig}_pcm          input PCM, 24 kHz mono                    (L,)
    {sig}_seanet       model.encoder(x)                          (512, T25)
    {sig}_layer0       after transformer layer 0 (forward hook)  (T25, 512)
    {sig}_transformer  model.encoder_transformer(...)            (T25, 512)
    {sig}_latent       after model.downsample                    (512, T12.5)
    {sig}_stream_latent  chunked encode, 1920-sample chunks via
                         MimiConv1dPaddingCache + sliding KV     (512, T12.5)
    {sig}_pooled       mean+std pool of latent.T, unbiased=False (1024,)

Signals: "chirp" (2 s seeded chirp+noise) and "long" (12 s, 300 frames
at 25 Hz > sliding window 250, exercising the window).

IMPORTANT sliding-window caveat (verified empirically against
transformers 4.57.1): the reference OFFLINE path (sdpa/eager) does NOT
apply the 250-frame sliding-window mask -- `create_causal_mask` builds a
plain causal mask and sdpa ignores the per-layer `sliding_window`
argument -- while the STREAMING path (DynamicCache with sliding layers)
evicts keys beyond 250 frames. For the 12 s signal the two therefore
diverge from frame 250 on (max abs diff ~8e-3 at the transformer
output). Since the Go implementation applies the window in both modes
(the intended Mimi semantics), two extra fixtures record the offline
reference run under an explicit sliding-window attention mask:

    long_transformer_win  offline transformer w/ sliding-window mask (T25, 512)
    long_latent_win       downsample of the above                    (512, T12.5)

Golden tests should compare windowed Go offline output against
`long_*_win` (and `long_stream_latent`), not `long_transformer` /
`long_latent`, for frames >= 250.

Also writes audio/testdata/mimi_encoder_keys.txt: sorted state-dict keys
(encoder.*, encoder_transformer.*, downsample.*) with shapes, the
loader-expectation manifest for Phase 2.

Usage:
    python audio/export_mimi_fixtures.py [out_dir]

Resample fixtures (scipy resample_poly pairs) are exported by a separate
script and intentionally NOT produced here.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import MimiModel
from transformers.cache_utils import DynamicCache
from transformers.models.mimi.modeling_mimi import MimiConv1dPaddingCache

SR = 24000
CHUNK = 1920  # 80 ms: 2 SEANet frames -> 1 latent frame at 12.5 Hz


def make_signal(seconds: float, seed: int) -> np.ndarray:
    """Seeded chirp + noise, 24 kHz mono float32 in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float64) / SR
    # Exponential chirp 100 Hz -> 8 kHz over the full duration.
    f0, f1 = 100.0, 8000.0
    phase = 2 * np.pi * f0 * (seconds / np.log(f1 / f0)) * (
        np.exp(t / seconds * np.log(f1 / f0)) - 1.0
    )
    sig = 0.5 * np.sin(phase)
    sig += 0.35 * np.sin(2 * np.pi * 440.0 * t)
    sig += 0.05 * rng.standard_normal(n)
    return sig.astype(np.float32)


@torch.no_grad()
def offline_stages(model: MimiModel, pcm: np.ndarray, mask_4d=None) -> dict:
    """Per-stage offline activations. mask_4d optionally overrides the
    transformer attention mask (used for the sliding-window variant)."""
    x = torch.from_numpy(pcm)[None, None, :]
    seanet = model.encoder(x)  # (1, 512, T25)

    layer0_out = {}

    def hook(_mod, _inp, out):
        layer0_out["h"] = out[0].detach()

    handle = model.encoder_transformer.layers[0].register_forward_hook(hook)
    kwargs = {"use_cache": False}
    if mask_4d is not None:
        kwargs["attention_mask"] = mask_4d
    trans = model.encoder_transformer(seanet.transpose(1, 2), **kwargs)[0]
    handle.remove()

    latent = model.downsample(trans.transpose(1, 2))  # (1, 512, T12.5)
    return {
        "seanet": seanet[0],  # (512, T25)
        "layer0": layer0_out["h"][0],  # (T25, 512)
        "transformer": trans[0],  # (T25, 512)
        "latent": latent[0],  # (512, T12.5)
    }


def sliding_window_mask(T: int, window: int, dtype) -> torch.Tensor:
    """Additive 4D mask: key j visible to query i iff i - window < j <= i."""
    q = torch.arange(T)[:, None]
    k = torch.arange(T)[None, :]
    allowed = (k <= q) & (k > q - window)
    mask = torch.zeros(T, T, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]  # (1, 1, T, T)


def new_padding_cache(model: MimiModel) -> MimiConv1dPaddingCache:
    """Mirror MimiModel.encode()'s streaming padding-cache construction."""
    pads, modes, chans = [], [], []
    for name in model.encoder._mimiconv1d_layer_names:
        conv = model.encoder.get_submodule(name)
        pads.append(conv.padding_total)
        modes.append(conv.pad_mode)
        chans.append(conv.in_channels)
    pads.append(model.downsample.padding_total)
    modes.append(model.downsample.pad_mode)
    chans.append(model.downsample.in_channels)
    return MimiConv1dPaddingCache(
        num_layers=len(pads),
        per_layer_padding=pads,
        per_layer_padding_mode=modes,
        per_layer_in_channels=chans,
    )


@torch.no_grad()
def stream_latent(model: MimiModel, pcm: np.ndarray) -> torch.Tensor:
    """Chunked pre-quantizer latent: transformers' streaming path
    (padding cache + sliding-window KV cache), 1920-sample chunks."""
    assert len(pcm) % CHUNK == 0, "streaming fixture requires whole 80 ms chunks"
    pad_cache = new_padding_cache(model)
    kv_cache = DynamicCache(config=model.config)
    outs = []
    for start in range(0, len(pcm), CHUNK):
        x = torch.from_numpy(pcm[start : start + CHUNK])[None, None, :]
        h = model.encoder(x, padding_cache=pad_cache)
        h = model.encoder_transformer(
            h.transpose(1, 2), past_key_values=kv_cache, use_cache=True
        )[0]
        outs.append(model.downsample(h.transpose(1, 2), padding_cache=pad_cache))
    return torch.cat(outs, dim=2)[0]  # (512, T12.5)


def pool(latent_tc: torch.Tensor) -> torch.Tensor:
    """Mean+std pooling over time, (T, 512) -> (1024,).
    Matches audio/export_fsdd_mimi.py::pool (unbiased=False)."""
    mean = latent_tc.mean(dim=0)
    std = (
        latent_tc.std(dim=0, unbiased=False)
        if latent_tc.shape[0] > 1
        else torch.zeros_like(mean)
    )
    return torch.cat([mean, std])


def dump_keys(model: MimiModel, path: Path) -> None:
    prefixes = ("encoder.", "encoder_transformer.", "downsample.")
    sd = model.state_dict()
    lines = [
        f"{k} {tuple(sd[k].shape)}"
        for k in sorted(sd)
        if k.startswith(prefixes)
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(lines)} keys)")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "testdata"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = MimiModel.from_pretrained("kyutai/mimi").eval()
    window = model.config.sliding_window  # 250

    signals = {
        "chirp": make_signal(2.0, seed=1234),  # 50 frames at 25 Hz
        "long": make_signal(12.0, seed=5678),  # 300 frames > window
    }

    tensors = {}
    for name, pcm in signals.items():
        stages = offline_stages(model, pcm)
        tensors[f"{name}_pcm"] = torch.from_numpy(pcm)
        for stage, val in stages.items():
            tensors[f"{name}_{stage}"] = val.contiguous()
        tensors[f"{name}_stream_latent"] = stream_latent(model, pcm).contiguous()
        tensors[f"{name}_pooled"] = pool(stages["latent"].T)

        off_vs_stream = (
            (tensors[f"{name}_latent"] - tensors[f"{name}_stream_latent"])
            .abs()
            .max()
            .item()
        )
        print(f"{name}: T25={stages['seanet'].shape[1]} "
              f"T12.5={stages['latent'].shape[1]} "
              f"offline-vs-stream max|d|={off_vs_stream:.3e}")

    # Sliding-window offline variant for the long signal (see module docstring).
    pcm = signals["long"]
    T25 = tensors["long_seanet"].shape[1]
    mask = sliding_window_mask(T25, window, torch.float32)
    win = offline_stages(model, pcm, mask_4d=mask)
    tensors["long_transformer_win"] = win["transformer"].contiguous()
    tensors["long_latent_win"] = win["latent"].contiguous()
    win_vs_stream = (
        (tensors["long_latent_win"] - tensors["long_stream_latent"]).abs().max().item()
    )
    print(f"long: windowed-offline vs stream max|d|={win_vs_stream:.3e} "
          f"(default offline ignores the window; see docstring)")

    tensors = {k: v.detach().to(torch.float32).contiguous() for k, v in tensors.items()}
    fix_path = out_dir / "mimi_fixtures.safetensors"
    save_file(tensors, str(fix_path))
    total = fix_path.stat().st_size
    print(f"wrote {fix_path} ({total / 1e6:.1f} MB, {len(tensors)} tensors)")
    for k in sorted(tensors):
        print(f"  {k} {tuple(tensors[k].shape)}")

    dump_keys(model, out_dir / "mimi_encoder_keys.txt")


if __name__ == "__main__":
    main()
