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

Phase 7 (RVQ quantizer) additions — int codes stored as float32
int-values (safetensors via this writer is f32-only):

    {sig}_codes        model.encode(x).audio_codes, all 32
                       quantizers (offline path)                 (32, T12.5)
    {sig}_codes8       same with num_quantizers=8 (Moshi's       (8, T12.5)
                       operating point)
    {sig}_quantized    model.quantizer.decode of the 8-codebook
                       codes: the decoder-side quantized latent  (512, T12.5)

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

Plan 0007 Phase D0 additions (separate output files; the encoder
fixture file above regenerates byte-identically): DECODER golden
stages into audio/testdata/mimi_decoder_fixtures.safetensors plus the
loader manifest audio/testdata/mimi_decoder_keys.txt -- see the
"Plan 0007 Phase D0" section below for the tensor list, size-budget
trims, and the measured Python decode baselines.

Usage:
    python audio/export_mimi_fixtures.py [out_dir]

Resample fixtures (scipy resample_poly pairs) are exported by a separate
script and intentionally NOT produced here.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
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


def dump_keys(model: MimiModel, path: Path, prefixes: tuple) -> None:
    sd = model.state_dict()
    lines = [
        f"{k} {tuple(sd[k].shape)}"
        for k in sorted(sd)
        if k.startswith(prefixes)
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(lines)} keys)")


# ---------------------------------------------------------------------------
# Plan 0007 Phase D0: DECODER golden fixtures
# (audio/testdata/mimi_decoder_fixtures.safetensors, separate file so the
# encoder fixtures above stay byte-identical).
#
# Decode path (MimiModel._decode_frame, verified against transformers
# 4.57.1): quantizer.decode -> upsample (depthwise ConvTranspose1d,
# 12.5 Hz -> 25 Hz) -> decoder_transformer -> decoder (SEANet).
#
# Tensor list (T = token count, so 2T transformer frames, 1920*T samples):
#
#     {sig}_dec_upsampled        model.upsample(quantized)         (512, 2T)
#     chirp_dec_layer0           after decoder_transformer layer 0 (2T, 512)
#     {sig}_dec_transformer      decoder_transformer output        (2T, 512)
#     {sig}_dec_seanet0          after decoder.layers.0            (1024, 2T)
#     chirp_dec_stage1           after decoder.layers.3            (512, 16T)
#     chirp_dec_stage2           after decoder.layers.6            (256, 96T)
#     {sig}_dec_wav              final waveform                    (1920*T,)
#     long_dec_transformer_win   offline w/ explicit sliding-window
#                                mask (same caveat as the encoder:
#                                the default offline path ignores
#                                the 250-frame window)              (300, 512)
#     long_dec_wav_win           full decode under the windowed
#                                transformer output                 (288000,)
#     rw_{clip}_dec_wav          reference model.decode of the
#                                committed real-world Go tokens
#                                (testdata/realworld/
#                                tokens.safetensors) for zero_alloy,
#                                five_echo, nine_shimmer            (1920*T,)
#     ct_{i}_{in,w,b,out}        ConvTranspose1d layout-pinning
#                                cases + ct_manifest (see
#                                convtranspose_cases below)
#
# Size-budget trims (plan 0007 SS5.1 budget <=10 MB): dec_layer0 and
# dec_stage1 are chirp-only (like dec_stage2) -- the long signal's
# unique value is the >250-frame sliding-window behaviour, fully
# covered by long_dec_transformer{,_win} / long_dec_wav{,_win}; the
# SEANet stages are time-invariant convs whose code path chirp covers.
# With long_dec_layer0 + long_dec_stage1 the file would be ~15.2 MB.
#
# Python decode baselines (Phase D0, recorded 2026-08-07; transformers
# 4.57.1, torch 2.9.1, CPU, Apple M4; best/mean of repeated runs on the
# long fixture signal, 150 tokens x 8 codebooks = 12 s of audio):
#
#   offline model.decode(150 tokens): best-of-3 466.9 ms
#       (runs: 535.2 / 466.9 / 490.2 ms)
#   per-token incremental decode, KV cache (decoder_past_key_values=
#       DynamicCache, one token per call; HF streams only the
#       transformer KV, not conv state): mean 33.95 ms, median
#       31.88 ms, p95 44.73 ms per token (= per 80 ms of audio)
#   per-token full-prefix re-decode (no-cache fallback): mean
#       260.96 ms, median 240.91 ms, 433.3 ms at T=150
#
# These numbers are duplicated in doc/plans/0007-mimi-native-decoder.md SS7.
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_stages(model: MimiModel, codes: torch.Tensor, mask_4d=None) -> dict:
    """Per-stage decoder activations for codes (1, nq, T) int64. mask_4d
    optionally overrides the decoder-transformer attention mask (used
    for the sliding-window variant), mirroring offline_stages."""
    quantized = model.quantizer.decode(codes)  # (1, 512, T)
    upsampled = model.upsample(quantized)  # (1, 512, 2T)

    layer0_out = {}

    def hook(_mod, _inp, out):
        layer0_out["h"] = out[0].detach()

    handle = model.decoder_transformer.layers[0].register_forward_hook(hook)
    kwargs = {"use_cache": False}
    if mask_4d is not None:
        kwargs["attention_mask"] = mask_4d
    trans = model.decoder_transformer(upsampled.transpose(1, 2), **kwargs)[0]
    handle.remove()

    seanet_out = {}

    def conv_hook(idx):
        def h(_mod, _inp, out):
            seanet_out[idx] = out.detach()

        return h

    handles = [
        model.decoder.layers[i].register_forward_hook(conv_hook(i))
        for i in (0, 3, 6)
    ]
    wav = model.decoder(trans.transpose(1, 2))  # (1, 1, 1920*T)
    for h in handles:
        h.remove()

    assert wav.shape[-1] == 1920 * codes.shape[-1], "decode length != 1920*T"
    return {
        "dec_upsampled": upsampled[0],  # (512, 2T)
        "dec_layer0": layer0_out["h"][0],  # (2T, 512)
        "dec_transformer": trans[0],  # (2T, 512)
        "dec_seanet0": seanet_out[0][0],  # (1024, 2T)
        "dec_stage1": seanet_out[3][0],  # (512, 16T)
        "dec_stage2": seanet_out[6][0],  # (256, 96T)
        "dec_wav": wav[0, 0],  # (1920*T,)
    }


def convtranspose_cases() -> dict:
    """Small random ConvTranspose1d cases pinning PyTorch's exact
    weight-layout convention -- weight (inC, outC/groups, k), NOT
    Conv1d's (outC, inC, k) -- and output semantics (padding=0, so
    L -> (L-1)*stride + k) for the D1 Go op. (kernel, stride, groups)
    cover all five Mimi decoder geometries plus a grouped case:
    (4,2,1), (4,2,C) depthwise bias-free like the upsample, (16,8,1),
    (12,6,1), (3,1,1), (8,4,2). ct_manifest rows are
    [kernel, stride, groups, has_bias] per case i."""
    gen = torch.Generator().manual_seed(20260807)
    #        k  s  g  inC outC  L  bias
    cases = [
        (4, 2, 1, 3, 5, 9, True),
        (4, 2, 6, 6, 6, 9, False),  # depthwise, bias-free (the upsample)
        (16, 8, 1, 4, 3, 5, True),
        (12, 6, 1, 3, 2, 7, True),
        (3, 1, 1, 2, 3, 8, True),
        (8, 4, 2, 4, 6, 6, True),
    ]
    out = {}
    manifest = torch.zeros(len(cases), 4)
    for i, (k, s, g, in_c, out_c, length, has_bias) in enumerate(cases):
        x = torch.randn(1, in_c, length, generator=gen)
        w = torch.randn(in_c, out_c // g, k, generator=gen)
        b = torch.randn(out_c, generator=gen) if has_bias else None
        y = torch.conv_transpose1d(x, w, b, stride=s, groups=g)
        assert y.shape == (1, out_c, (length - 1) * s + k)
        out[f"ct_{i}_in"] = x
        out[f"ct_{i}_w"] = w
        if has_bias:
            out[f"ct_{i}_b"] = b
        out[f"ct_{i}_out"] = y
        manifest[i] = torch.tensor([k, s, g, float(has_bias)])
    out["ct_manifest"] = manifest
    return out


@torch.no_grad()
def decoder_fixtures(model: MimiModel, tensors: dict, window: int) -> dict:
    """All plan-0007 D0 decoder fixtures, from the encoder run's
    {sig}_codes8 plus the committed real-world Go-produced tokens."""
    dec = {}
    keep = {
        "chirp": (
            "dec_upsampled",
            "dec_layer0",
            "dec_transformer",
            "dec_seanet0",
            "dec_stage1",
            "dec_stage2",
            "dec_wav",
        ),
        # Size-budget trim, see the section comment above.
        "long": ("dec_upsampled", "dec_transformer", "dec_seanet0", "dec_wav"),
    }
    for name, stages_kept in keep.items():
        codes8 = tensors[f"{name}_codes8"].to(torch.long)[None]  # (1, 8, T)
        stages = decode_stages(model, codes8)
        for st in stages_kept:
            dec[f"{name}_{st}"] = stages[st]
        # The hooked pipeline must reproduce the public API exactly.
        ref = model.decode(codes8).audio_values[0, 0]
        assert torch.equal(stages["dec_wav"], ref), f"{name}: hooked wav != decode()"
        print(f"{name}: dec_wav {tuple(stages['dec_wav'].shape)}")

    # Sliding-window offline variant for the long signal: same caveat as
    # the encoder (default offline path ignores the 250-frame window).
    codes8 = tensors["long_codes8"].to(torch.long)[None]
    t25 = 2 * codes8.shape[-1]
    mask = sliding_window_mask(t25, window, torch.float32)
    win = decode_stages(model, codes8, mask_4d=mask)
    dec["long_dec_transformer_win"] = win["dec_transformer"]
    dec["long_dec_wav_win"] = win["dec_wav"]
    win_vs_plain = (
        (dec["long_dec_transformer_win"] - dec["long_dec_transformer"])
        .abs()
        .max()
        .item()
    )
    print(f"long: windowed vs plain dec_transformer max|d|={win_vs_plain:.3e}")

    # Reference decodes of the committed real-world Go-produced tokens
    # (the exact tensors roundtrip_decode.py feeds to whisper).
    rw_path = Path(__file__).parent / "testdata" / "realworld" / "tokens.safetensors"
    rw = load_file(str(rw_path))
    for clip in ("zero_alloy", "five_echo", "nine_shimmer"):
        codes = rw[clip].to(torch.long)[None]  # (1, 8, T)
        wav = model.decode(codes).audio_values[0, 0]
        assert wav.shape[-1] == 1920 * codes.shape[-1]
        dec[f"rw_{clip}_dec_wav"] = wav
        print(f"rw_{clip}: {tuple(codes[0].shape)} codes -> {wav.shape[-1]} samples")

    dec.update(convtranspose_cases())
    return dec


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

        # Phase 7: discrete RVQ codes + quantized latent. model.encode's
        # default (non-streaming) path is exactly the offline stages
        # above followed by quantizer.encode.
        with torch.no_grad():
            x = torch.from_numpy(pcm)[None, None, :]
            codes32 = model.encode(x, return_dict=True).audio_codes  # (1, 32, T)
            codes8 = model.encode(x, num_quantizers=8, return_dict=True).audio_codes
            quantized = model.quantizer.decode(codes8)  # (1, 512, T)
        assert torch.equal(codes8, codes32[:, :8]), "RVQ prefix property violated"
        tensors[f"{name}_codes"] = codes32[0].to(torch.float32)
        tensors[f"{name}_codes8"] = codes8[0].to(torch.float32)
        tensors[f"{name}_quantized"] = quantized[0]
        print(f"{name}: codes {tuple(codes32[0].shape)} "
              f"codes8 {tuple(codes8[0].shape)} "
              f"quantized {tuple(quantized[0].shape)}")

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

    dump_keys(
        model,
        out_dir / "mimi_encoder_keys.txt",
        ("encoder.", "encoder_transformer.", "downsample."),
    )

    # --- Plan 0007 Phase D0: decoder fixtures (separate file) ---
    dec = decoder_fixtures(model, tensors, window)
    dec = {k: v.detach().to(torch.float32).contiguous() for k, v in dec.items()}
    dec_path = out_dir / "mimi_decoder_fixtures.safetensors"
    save_file(dec, str(dec_path))
    total = dec_path.stat().st_size
    print(f"wrote {dec_path} ({total / 1e6:.1f} MB, {len(dec)} tensors)")
    for k in sorted(dec):
        print(f"  {k} {tuple(dec[k].shape)}")

    dump_keys(
        model,
        out_dir / "mimi_decoder_keys.txt",
        ("decoder.", "decoder_transformer.", "upsample."),
    )


if __name__ == "__main__":
    main()
