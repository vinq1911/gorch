"""INDEPENDENT CROSS-CHECK decode of gorch-produced Mimi tokens.

The production tokens→audio path is native Go (audio/mimi Decoder,
plan 0007) — see TestMimiRealWorldNativeRoundtrip and
audio/testdata/realworld/native_roundtrip_transcripts.tsv for the
primary round-trip evidence. This script is retained only as the
independent cross-check: it decodes the same committed tokens with the
REFERENCE (transformers) Mimi decoder, an implementation gorch shares
no code with, producing the evidence behind roundtrip_transcripts.tsv.

Reads audio/testdata/realworld/tokens.safetensors — 8-codebook Mimi
codes produced NATIVELY BY GO (e2e/mimi_realworld_test.go with
GORCH_MIMI_WRITE_TOKENS=1) — decodes each clip's tokens back to a
waveform with the reference Mimi decoder, and writes the
reconstructions to a scratch dir for transcription.

Then (run separately, needs the faster-whisper env):
    python audio/realworld/transcribe_check.py <scratch>/roundtrip \
      | python audio/realworld/verdict.py > audio/testdata/realworld/roundtrip_transcripts.tsv

Usage: python roundtrip_decode.py <tokens.safetensors> <outdir>
"""
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import MimiModel

tokens_path, outdir = sys.argv[1], Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

model = MimiModel.from_pretrained("kyutai/mimi").eval()
tensors = load_file(tokens_path)

for name, codes in sorted(tensors.items()):
    c = codes.to(torch.long).unsqueeze(0)  # (1, 8, T)
    with torch.no_grad():
        wav = model.decode(c).audio_values[0, 0].numpy()
    pcm16 = np.clip(wav * 32768, -32768, 32767).astype(np.int16)
    with wave.open(str(outdir / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm16.tobytes())
    print(f"{name}: {len(wav) / 24000:.2f}s reconstructed")
