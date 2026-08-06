#!/bin/bash
# Curation loop for the Azure-generated digit WAVs: generate missing
# clips, transcribe with local faster-whisper, delete any clip whose
# transcript is not exactly the expected digit (word or numeral), and
# repeat. Every surviving clip is ASR-confirmed ground truth.
#
# Env pythons: ace_step has websockets (generator), miniconda base has
# faster_whisper (transcriber).
set -u
GEN=/opt/homebrew/Caskroom/miniconda/base/envs/ace_step/bin/python
ASR=/opt/homebrew/Caskroom/miniconda/base/bin/python
DIR=audio/testdata/realworld
HERE="$(cd "$(dirname "$0")" && pwd)"

for round in 1 2 3 4 5 6; do
  echo "== round $round: generate missing"
  "$GEN" "$HERE/generate_azure_speech.py" "$DIR" 2>&1 | grep -v "exists, skipped" | tail -5
  echo "== round $round: transcribe + curate"
  "$ASR" "$HERE/transcribe_check.py" "$DIR" 2>/dev/null | "$GEN" "$HERE/curate_filter.py"
  n=$(ls "$DIR"/*.wav 2>/dev/null | wc -l | tr -d ' ')
  echo "== round $round: $n/30 confirmed"
  [ "$n" = "30" ] && break
done
ls "$DIR"/*.wav | wc -l
