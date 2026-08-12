#!/bin/bash
# convert_librispeech.sh — LibriSpeech FLAC → WAV for cmd/voicedata
# (plan 0008 §4.2: "LibriSpeech ships FLAC → offline ffmpeg convert
# step (script, documented; Go stays WAV-only)").
#
# Converts the first NSPK speakers of train-clean-100 — speaker ids
# sorted NUMERICALLY ascending, the deterministic subset order that
# cmd/voicedata -corpus librispeech consumes — to mono 16-bit PCM WAV
# at the NATIVE 16 kHz rate. The 16 kHz → 24 kHz conversion is NOT
# done here: cmd/voicedata runs audio.Resample (the scipy-parity
# polyphase resampler) so both corpora (LibriSpeech 16 k, LJSpeech
# 22.05 k) go through the same audited resampling path into Mimi.
#
# Requires ffmpeg (checked below). If ffmpeg is unavailable, the
# equivalent fallback is the ace_step python env with soundfile:
#   python -c "import soundfile as sf, sys; d, r = sf.read(sys.argv[1]); sf.write(sys.argv[2], d, r, subtype='PCM_16')" in.flac out.wav
#
# Usage: convert_librispeech.sh [SRC] [DST] [NSPK] [JOBS]
#   SRC  default ~/speech-corpora/LibriSpeech/train-clean-100
#   DST  default ~/speech-corpora/LibriSpeech-wav16k
#   NSPK default 100 (first 100 numeric speaker ids; ~40 h raw audio,
#        comfortably covering the 10 h kept subset + held-out eval
#        speakers after the ≤9.5 s clip cap)
#   JOBS default 8 parallel ffmpeg processes
#
# Idempotent: existing non-empty outputs are skipped.
set -euo pipefail

SRC="${1:-$HOME/speech-corpora/LibriSpeech/train-clean-100}"
DST="${2:-$HOME/speech-corpora/LibriSpeech-wav16k}"
NSPK="${3:-100}"
JOBS="${4:-8}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found; install it or use the soundfile fallback documented above" >&2
    exit 1
fi

mkdir -p "$DST"

# shellcheck disable=SC2012
ls "$SRC" | grep -E '^[0-9]+$' | sort -n | head -n "$NSPK" | while read -r spk; do
    find "$SRC/$spk" -name '*.flac' | sort
done > "$DST/.filelist"

n_total=$(wc -l < "$DST/.filelist" | tr -d ' ')
echo "converting $n_total flac files from first $NSPK speakers ($JOBS jobs)"

export SRC DST
xargs -P "$JOBS" -n 1 sh -c '
    f="$1"
    rel="${f#"$SRC"/}"
    out="$DST/${rel%.flac}.wav"
    mkdir -p "$(dirname "$out")"
    if [ ! -s "$out" ]; then
        ffmpeg -loglevel error -y -i "$f" -ac 1 -sample_fmt s16 "$out" </dev/null
    fi
' convert < "$DST/.filelist"

echo "done: $(find "$DST" -name '*.wav' | wc -l | tr -d ' ') wav files in $DST"
