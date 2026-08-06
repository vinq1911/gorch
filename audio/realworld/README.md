# Real-world speech tests for the native gorch-Mimi pipeline

Two e2e tests (`e2e/mimi_realworld_test.go`) prove the native Go
pipeline on speech that shares nothing with the FSDD training set — no
common speakers, microphones, or recording chain. The test clips are
spoken digits synthesized by **Azure OpenAI gpt-realtime** voices
(alloy, echo, shimmer) at Mimi's native 24 kHz, and every committed
clip is **ASR-verified** by faster-whisper before acceptance.

## Test 1 — ingestion (`TestMimiRealWorldIngestion`)

Azure-voice WAV → pure-Go `ReadWAV → Resample (24k→8k→24k domain
match) → mimi.Encode → mean+std pool` → gorch digit head trained only
on FSDD embeddings.

**Result: 100% (30/30)** — an FSDD-trained classifier generalizes to
modern TTS voices it has never heard, through the fully native
pipeline. Gate: ≥80%.

## Test 2 — production (`TestMimiRealWorldTokenProduction`)

gorch encodes each clip into **discrete Mimi tokens** (8 codebooks —
the currency Moshi-style speech LMs emit when *producing* speech).
The committed evidence chain proves those exact tokens are valid,
intelligible speech:

1. Go writes `audio/testdata/realworld/tokens.safetensors`
   (`GORCH_MIMI_WRITE_TOKENS=1`).
2. `roundtrip_decode.py` decodes the tokens with the reference Mimi
   decoder back into waveforms.
3. faster-whisper transcribes the reconstructions; `verdict.py` grades
   them (homophone-aware — ruling confirmed by a GPT-5.6 judge) into
   `roundtrip_transcripts.tsv`.
4. The Go test re-derives all tokens from scratch, requires an **exact
   integer match** with the committed tokens, and asserts the recorded
   intelligibility.

**Result: 30/30 (100%)** token reconstructions re-transcribed as the
correct digit. Gate: ≥80%.

## Pipeline scripts

| Script | Env | Purpose |
|---|---|---|
| `generate_azure_speech.py` | ace_step (websockets) | gpt-realtime TTS → 24 kHz WAVs; retries content-filter false positives and silent responses |
| `transcribe_check.py` | miniconda base (faster_whisper) | transcribe WAVs |
| `curate_filter.py` / `curate.sh` | both | generate → verify → reject → regenerate loop until every clip is ASR-confirmed |
| `roundtrip_decode.py` | ace_step (transformers) | Go tokens → reference Mimi decoder → WAVs |
| `verdict.py` | any | grade round-trip transcripts into the committed TSV |

Regeneration needs `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_URI_OPENAI`
(gpt-realtime deployment) — the committed WAVs, tokens, and verdicts
make the Go tests self-contained without them.
