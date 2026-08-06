#!/usr/bin/env python3
"""Generate golden fixtures for the audio package tests (plan 0006 P4).

Writes:
  testdata/resample_fixtures.safetensors
      - scipy.signal.resample_poly input/output pairs (float32) for
        8000/16000/48000/44100 -> 24000
      - ground-truth float32 decodes (as read by python-soundfile) of every
        generated test WAV, stored interleaved
  testdata/*.wav
      - tiny WAV files covering PCM16 mono/stereo, PCM24, PCM32, float32,
        float64, WAVE_FORMAT_EXTENSIBLE variants, and a PCM16 file with
        JUNK/LIST chunks spliced in before the data chunk

Run once with an env that has numpy, scipy, soundfile, safetensors:

  /opt/homebrew/Caskroom/miniconda/base/envs/ace_step/bin/python \
      audio/make_resample_fixtures.py

Library versions are recorded in the safetensors metadata.
Generated with scipy 1.15.3 / numpy 2.2.6 / soundfile 0.13.x.
"""

import os
import struct

import numpy as np
import scipy
import scipy.signal as sps
import soundfile as sf
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "testdata")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260806)
tensors = {}


def make_signal(sr, n):
    """Seeded multi-tone + noise test signal, peak-normalized to 0.9."""
    t = np.arange(n) / sr
    comps = [(0.5, 440.0, 0.1), (0.3, 1234.5, 1.7), (0.15, 2999.0, 2.9)]
    sig = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in comps)
    sig = sig + 0.05 * rng.standard_normal(n)
    sig = 0.9 * sig / np.max(np.abs(sig))
    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Resampler fixtures: scipy.signal.resample_poly with default window
# ('kaiser', 5.0). Lengths are deliberately "awkward" (not multiples of the
# down factor) to exercise the ceil() in the output-length formula.
#
# The input is float32 (stored as such, exactly representable), but the
# reference is computed through scipy's float64 path: for float32 input,
# resample_poly casts the filter to float32 and accumulates in float32 with
# a compiler-dependent summation order, which is not reproducible bit-for-bit.
# The Go resampler accumulates in float64, so the float64 reference is the
# correct ground truth (the two scipy paths differ by ~1e-7 absolute anyway).
# ---------------------------------------------------------------------------
for sr_in, n in [(8000, 4001), (16000, 8003), (48000, 24007), (44100, 22063)]:
    x = make_signal(sr_in, n)
    y = sps.resample_poly(x.astype(np.float64), 24000, sr_in)
    tensors[f"resample_{sr_in}_in"] = x
    tensors[f"resample_{sr_in}_out"] = y.astype(np.float32)
    print(f"resample {sr_in} -> 24000: in {len(x)}, out {len(y)}")

# ---------------------------------------------------------------------------
# WAV format fixtures. Ground truth is what python-soundfile reads back as
# float32 (libsndfile normalizes ints by 2^(bits-1)), which is exactly what
# the Go reader must reproduce for FSDD parity.
# ---------------------------------------------------------------------------
WAV_SR = 8000
N_WAV = 256
mono = make_signal(WAV_SR, N_WAV)
stereo = np.stack([mono, make_signal(WAV_SR, N_WAV)], axis=1)

wav_specs = [
    ("pcm16_mono.wav", mono, "PCM_16", "WAV"),
    ("pcm16_stereo.wav", stereo, "PCM_16", "WAV"),
    ("pcm24_mono.wav", mono, "PCM_24", "WAV"),
    ("pcm32_mono.wav", mono, "PCM_32", "WAV"),
    ("float32_mono.wav", mono, "FLOAT", "WAV"),
    ("float64_mono.wav", mono, "DOUBLE", "WAV"),
    ("wavex_pcm16_mono.wav", mono, "PCM_16", "WAVEX"),
    ("wavex_float32_mono.wav", mono, "FLOAT", "WAVEX"),
]
for name, data, subtype, fmt in wav_specs:
    path = os.path.join(OUT, name)
    sf.write(path, data, WAV_SR, subtype=subtype, format=fmt)
    back, got_sr = sf.read(path, dtype="float32", always_2d=False)
    assert got_sr == WAV_SR
    key = "wav_" + name[:-4]
    tensors[key] = np.ascontiguousarray(back, dtype=np.float32).reshape(-1)
    print(f"{name}: {os.path.getsize(path)} bytes, {tensors[key].size} samples")

# ---------------------------------------------------------------------------
# Chunk-walk robustness: splice a JUNK chunk (odd size -> exercises the pad
# byte) and a LIST/INFO chunk between 'fmt ' and 'data'.
# ---------------------------------------------------------------------------
with open(os.path.join(OUT, "pcm16_mono.wav"), "rb") as f:
    src = f.read()
i = src.find(b"data")
assert i > 12
junk = b"JUNK" + struct.pack("<I", 5) + b"junk!" + b"\x00"  # odd size + pad
list_body = b"INFO" + b"ISFT" + struct.pack("<I", 8) + b"gorch\x00\x00\x00"
list_chunk = b"LIST" + struct.pack("<I", len(list_body)) + list_body
out = src[:i] + junk + list_chunk + src[i:]
out = out[:4] + struct.pack("<I", len(out) - 8) + out[8:]
extra_path = os.path.join(OUT, "pcm16_mono_extra_chunks.wav")
with open(extra_path, "wb") as f:
    f.write(out)
back, got_sr = sf.read(extra_path, dtype="float32", always_2d=False)
assert got_sr == WAV_SR
np.testing.assert_array_equal(back, tensors["wav_pcm16_mono"])
tensors["wav_pcm16_mono_extra_chunks"] = tensors["wav_pcm16_mono"]
print(f"pcm16_mono_extra_chunks.wav: {os.path.getsize(extra_path)} bytes")

fx_path = os.path.join(OUT, "resample_fixtures.safetensors")
save_file(
    tensors,
    fx_path,
    metadata={
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "generator": "audio/make_resample_fixtures.py",
    },
)
print(f"{fx_path}: {os.path.getsize(fx_path)} bytes, {len(tensors)} tensors")
