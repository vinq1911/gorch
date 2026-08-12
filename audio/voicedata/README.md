# voicedata — Stage-A corpus artifacts (plan 0008 §4.2/§4.3, M2a)

Companion artifacts for `cmd/voicedata`, the corpus→Mimi-token shard
pipeline. The shards themselves are large (~35 MB) and are NOT
committed; they live in `~/speech-corpora/shards/stageA/` and are
exactly reproducible from the committed tooling:

```
audio/voicedata/convert_librispeech.sh          # FLAC → 16 kHz WAV (ffmpeg), all 251 speakers
go run ./cmd/voicedata -corpus librispeech -out ~/speech-corpora/shards/stageA
go run ./cmd/voicedata -corpus ljspeech    -out ~/speech-corpora/shards/stageA
go run ./cmd/voicedata -corpus textreplay  -out ~/speech-corpora/shards/stageA
```

Determinism: LibriSpeech selection order is speaker-numeric /
chapter-numeric / utterance-lexical with the 12 highest-numbered
speakers reserved for eval; LJSpeech splits metadata.csv order into
even-index train / odd-index held-out; TEXT-replay prompts are a fixed
list (`cmd/voicedata/prompts.go`) answered greedily by the frozen base
model. Every knob that shapes the shards is echoed into the
`*_stats.json` files here.

## manifests/

| file | contents |
|---|---|
| `listen_train_manifest.tsv` | every LISTEN training clip: utt id, seconds, Mimi frames, sample tokens, transcript |
| `listen_eval_manifest.tsv` | 50 held-out ≤8 s utterances (reserved speakers — dev-clean substitute) for the §4.4 LISTEN gate |
| `listen_stats.json` | selection config + drop tallies + token totals |
| `speak_train_manifest.tsv` | every SPEAK training clip: LJ id, seconds, frames, sample tokens, normalized text |
| `speak_eval_manifest.tsv` | 50 held-out LJSpeech sentences for the §4.4 SPEAK gate |
| `speak_stats.json` | split config + token totals |
| `text_replay_manifest.tsv` | prompt + greedy base-model answer for each TEXT sample |
| `text_replay_stats.json` | generation config + token totals |
| `smoke_run/` | M2a smoke-training evidence (losses.tsv + config) |

Known deviations from plan §4.1 numbers, measured on the real corpora:

- The ≤9.5 s LISTEN clip cap keeps only ~9% of train-clean-100 *hours*
  (the corpus mean is 12.7 s and long clips carry most of the mass), so
  the "10 h subset" saturates at ~7.7 h kept over the whole corpus.
  The plan's "keeps ~70% of clips" estimate was wrong; the LISTEN v0
  hours knob is bounded above by ~7.7 h until long-clip splitting
  (with re-aligned transcripts) exists.
- LibriSpeech dev-clean is not on disk; the LISTEN eval manifest draws
  from 12 reserved train-clean-100 speakers instead (never seen in
  training).
