# Real-world speech tests for the native gorch-Mimi pipeline

Three e2e tests (`e2e/mimi_realworld_test.go`) prove the native Go
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
The Go test re-derives all tokens from scratch and requires an **exact
integer match** with the committed
`audio/testdata/realworld/tokens.safetensors`
(written once with `GORCH_MIMI_WRITE_TOKENS=1`). Intelligibility of
those tokens is proven by the NATIVE round trip (Test 3); this test
additionally asserts the committed `roundtrip_transcripts.tsv` — the
same tokens decoded by the **reference** (transformers) Mimi decoder
via `roundtrip_decode.py` and re-transcribed — as an independent
cross-check through a decoder gorch shares no code with.

**Result: 30/30 (100%)** reference-decoded reconstructions
re-transcribed as the correct digit. Gate: ≥80%.

## Test 3 — native round trip (`TestMimiRealWorldNativeRoundtrip`)

The production path is now **fully native**: tokens → audio runs in Go
(plan `doc/plans/0007-mimi-native-decoder.md`). The committed evidence
chain:

1. The native decoder (`mimi.LoadFull` → `Decoder.Decode`) decodes the
   committed 30-clip tokens to 24 kHz waveforms;
   `GORCH_MIMI_WRITE_DECODED=1` writes them to
   `audio/testdata/realworld/native_roundtrip/` (committed — the
   audible artifact).
2. faster-whisper transcribes the native reconstructions; `verdict.py`
   grades them (homophone-aware — ruling confirmed by a GPT-5.6 judge)
   into `native_roundtrip_transcripts.tsv`.
3. The normal run re-decodes every clip natively, asserts the exact
   1920·T sample-length property, the committed 30-clip verdict
   coverage, and ≥40 dB SNR against the 3 committed HF reference
   decodes (`rw_*_dec_wav` fixtures) — measured 118.6–122.8 dB.

**Result: 30/30 (100%)** native reconstructions re-transcribed as the
correct digit (gate ≥80%). Native decode of all 30 clips: ~1.0 s
total (~35 ms/clip).

## Pipeline scripts

| Script | Env | Purpose |
|---|---|---|
| `generate_azure_speech.py` | ace_step (websockets) | gpt-realtime TTS → 24 kHz WAVs; retries content-filter false positives and silent responses |
| `transcribe_check.py` | miniconda base (faster_whisper) | transcribe WAVs — verification tooling for curation and round-trip grading |
| `curate_filter.py` / `curate.sh` | both | generate → verify → reject → regenerate loop until every clip is ASR-confirmed |
| `roundtrip_decode.py` | ace_step (transformers) | independent cross-check: Go tokens → **reference** Mimi decoder → WAVs (production decode is native Go) |
| `verdict.py` | any | grade round-trip transcripts into the committed TSVs |

Regeneration needs `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_URI_OPENAI`
(gpt-realtime deployment) — the committed WAVs, tokens, and verdicts
make the Go tests self-contained without them.
