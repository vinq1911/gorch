"""Generate real-world spoken-digit WAVs with Azure OpenAI gpt-realtime.

Synthesizes each digit word ("zero".."nine") in several voices via the
realtime voice API (24 kHz PCM16 — Mimi's native rate) and writes
mono WAVs to audio/testdata/realworld/{digit}_{voice}.wav. These are
the inputs for the real-world ingestion/production e2e tests
(e2e/mimi_realworld_test.go).

Azure specifics (resource lastbotus2-sandbox, verified 2026-08):
- wss://<resource>.openai.azure.com/openai/realtime?api-version=2025-04-01-preview&deployment=gpt-realtime
- auth header `api-key`, FLAT session.update shape (the GA nested
  audio{} shape is rejected on this resource).

Usage:
    python audio/realworld/generate_azure_speech.py [outdir]
Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_URI_OPENAI in the env.
"""
import asyncio
import base64
import json
import os
import struct
import sys
import wave
from pathlib import Path

import websockets

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
VOICES = ["alloy", "echo", "shimmer"]
SR = 24000


def wss_url() -> str:
    base = os.environ["AZURE_OPENAI_URI_OPENAI"]
    host = base.replace("https://", "").rstrip("/")
    return f"wss://{host}/openai/realtime?api-version=2025-04-01-preview&deployment=gpt-realtime"


def instructions_for(word: str, attempt: int) -> str:
    """Phrasing variants — the content filter occasionally false-positives
    on terse single-word prompts, so later retries reword the request."""
    variants = [
        f"Say exactly the single word '{word}' once, clearly, "
        "with no other words, sounds, or punctuation.",
        f"Read this digit aloud exactly once: {word}",
        f"You are recording a clean voice sample for a speech dataset. "
        f"Pronounce the English number word '{word}' one time.",
        f"Count aloud starting and ending at {DIGITS.index(word)}. "
        "That is: speak just that one number word.",
    ]
    return variants[attempt % len(variants)]


async def synth(ws, voice: str, word: str, attempt: int = 0) -> bytes:
    await ws.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["audio", "text"],
            "voice": voice,
            "output_audio_format": "pcm16",
            "instructions": instructions_for(word, attempt),
        },
    }))
    pcm = bytearray()
    while True:
        msg = json.loads(await ws.recv())
        t = msg.get("type", "")
        if t == "response.audio.delta":
            pcm.extend(base64.b64decode(msg["delta"]))
        elif t == "response.done":
            status = msg.get("response", {}).get("status")
            if status != "completed":
                raise RuntimeError(f"response status {status}: {json.dumps(msg)[:400]}")
            return bytes(pcm)
        elif t == "error":
            raise RuntimeError(json.dumps(msg)[:400])


def write_wav(path: Path, pcm16: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16)


def trim_silence(pcm16: bytes, thresh: int = 300) -> bytes:
    """Trim leading/trailing near-silence, keep 100 ms margins."""
    n = len(pcm16) // 2
    samples = struct.unpack(f"<{n}h", pcm16[: n * 2])
    idx = [i for i, s in enumerate(samples) if abs(s) > thresh]
    if not idx:
        return pcm16
    margin = SR // 10
    lo, hi = max(0, idx[0] - margin), min(n, idx[-1] + margin)
    return struct.pack(f"<{hi - lo}h", *samples[lo:hi])


async def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "audio/testdata/realworld")
    outdir.mkdir(parents=True, exist_ok=True)
    headers = {"api-key": os.environ["AZURE_OPENAI_API_KEY"]}
    for voice in VOICES:
        async with websockets.connect(wss_url(), additional_headers=headers, max_size=1 << 24) as ws:
            # wait for session.created, then pin the session config
            while json.loads(await ws.recv()).get("type") != "session.created":
                pass
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": voice,
                    "output_audio_format": "pcm16",
                    "turn_detection": None,
                },
            }))
            for word in DIGITS:
                path = outdir / f"{word}_{voice}.wav"
                if path.exists():
                    print(f"{path}  (exists, skipped)")
                    continue
                for attempt in range(8):
                    try:
                        pcm = trim_silence(await synth(ws, voice, word, attempt))
                        n = len(pcm) // 2
                        samples = struct.unpack(f"<{n}h", pcm[: n * 2])
                        peak = max(abs(s) for s in samples) / 32768 if n else 0.0
                        if n < SR // 5 or peak < 0.05:
                            # "completed" responses occasionally carry silent
                            # audio — treat as failure and retry
                            raise RuntimeError(f"silent/short audio (n={n}, peak={peak:.3f})")
                        break
                    except RuntimeError as e:
                        # content_filter false positives and silent responses
                        # both happen; retry with reworded instructions
                        print(f"  retry {attempt + 1} for {word}/{voice}: {str(e)[:120]}")
                        if attempt == 7:
                            raise
                        await asyncio.sleep(1.5)
                write_wav(path, pcm)
                print(f"{path}  {len(pcm) // 2 / SR:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
