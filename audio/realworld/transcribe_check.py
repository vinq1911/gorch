"""Transcribe WAVs with local faster-whisper-small and report the text.

Used two ways by the real-world e2e work:
1. sanity-check the Azure-generated digit WAVs contain the right word;
2. grade Mimi-token round-trip reconstructions (production test) — see
   roundtrip_decode.py.

Usage: python transcribe_check.py <wav-or-dir> [...]
Prints one line per file: <path>\t<transcript>
Run with an env that has faster_whisper (miniconda base).
"""
import sys
from pathlib import Path

from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

paths = []
for a in sys.argv[1:]:
    p = Path(a)
    paths.extend(sorted(p.glob("*.wav")) if p.is_dir() else [p])

for p in paths:
    segments, _info = model.transcribe(str(p), language="en", beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    print(f"{p}\t{text}")
