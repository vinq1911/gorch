//go:build darwin && e2e

package e2e

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/audio"
	"github.com/vinq1911/gorch/audio/mimi"
	"github.com/vinq1911/gorch/model"
)

// The real-world clips are spoken digits synthesized by Azure OpenAI
// gpt-realtime voices (alloy/echo/shimmer) at Mimi's native 24 kHz and
// ASR-verified by faster-whisper before being committed — a speech
// source completely unrelated to FSDD's speakers, microphones, and
// 8 kHz recording chain. See audio/realworld/.
const realworldDir = "../audio/testdata/realworld"

var digitWords = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}

func realworldClips(t *testing.T) []string {
	paths, err := filepath.Glob(filepath.Join(realworldDir, "*.wav"))
	if err != nil || len(paths) == 0 {
		t.Skipf("no real-world clips in %s — run audio/realworld/curate.sh", realworldDir)
	}
	sort.Strings(paths)
	return paths
}

func digitOf(t *testing.T, path string) int {
	word := strings.SplitN(filepath.Base(path), "_", 2)[0]
	for d, w := range digitWords {
		if w == word {
			return d
		}
	}
	t.Fatalf("unrecognized digit word in %s", path)
	return -1
}

// TestMimiRealWorldIngestion proves speech INGESTION on out-of-domain,
// real-world audio: digits spoken by Azure TTS voices are pushed
// through the fully native Go pipeline (WAV → domain-matching
// resample → Mimi encode → pool) and classified by a gorch head
// trained only on FSDD. The 24 kHz clips are band-limited through an
// 8 kHz hop (Resample 24k→8k→24k) so they match the training domain's
// telephone-band recording chain.
func TestMimiRealWorldIngestion(t *testing.T) {
	clips := realworldClips(t)

	fsddPath := envDefault("GORCH_MIMI_FSDD", "../audio/fsdd_mimi.safetensors")
	if _, err := os.Stat(fsddPath); err != nil {
		t.Skipf("FSDD embeddings not found at %s — run audio/export_fsdd_mimi.py", fsddPath)
	}
	mimiPath := envDefault("MIMI_MODEL", defaultMimiCheckpoint)
	if _, err := os.Stat(mimiPath); err != nil {
		t.Skipf("Mimi checkpoint not found at %s", mimiPath)
	}

	enc, err := mimi.Load(mimiPath)
	if err != nil {
		t.Fatalf("load Mimi: %v", err)
	}

	// Embed the real-world clips with the native pipeline.
	var xs []float32
	var ys []int
	for _, p := range clips {
		w, err := audio.ReadWAV(p)
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		pcm := w.Mono()
		if w.SampleRate != 24000 {
			pcm = audio.Resample(pcm, w.SampleRate, 24000)
		}
		// Match FSDD's 8 kHz-band domain.
		pcm = audio.Resample(audio.Resample(pcm, 24000, 8000), 8000, 24000)
		latent := enc.Encode(pcm)
		xs = append(xs, poolNative(latent)...)
		ys = append(ys, digitOf(t, p))
	}
	dim := len(xs) / len(clips)

	// Train the digit head on the full FSDD train split (Python- or
	// Go-computed embeddings — parity between the two is already
	// proven by TestMimiNativeFSDD).
	sf, err := model.LoadSafetensors(fsddPath)
	if err != nil {
		t.Fatalf("load FSDD embeddings: %v", err)
	}
	trainSet := &embDataset{x: sf.Tensors["train_x"], y: sf.Tensors["train_y"], dim: dim}
	testSet := &embDataset{
		x:   g.NewTensor(xs, len(clips), dim),
		y:   g.NewTensor(intsToF32(ys), len(clips), 1),
		dim: dim,
	}
	res := trainHead(t, trainSet, testSet, 10, 30)

	t.Logf("real-world ingestion: %d Azure-voice clips, accuracy %.1f%% (%d/%d)",
		len(clips), res.accuracy, res.correct, res.total)
	if res.accuracy < 80.0 {
		t.Fatalf("real-world accuracy %.1f%% below 80%% gate", res.accuracy)
	}
}

func intsToF32(xs []int) []float32 {
	out := make([]float32, len(xs))
	for i, v := range xs {
		out[i] = float32(v)
	}
	return out
}

// TestMimiRealWorldTokenProduction proves speech PRODUCTION in the
// token domain: gorch encodes each real-world clip into discrete Mimi
// tokens (8 codebooks — Moshi's speech-generation currency). This test
// re-derives the tokens from scratch and requires an exact match with
// the committed tokens. The primary intelligibility evidence for those
// tokens is now fully NATIVE (plan 0007 D4): the Go decoder turns the
// committed tokens back into audio and faster-whisper re-transcribes
// the reconstructions — asserted by TestMimiRealWorldNativeRoundtrip
// via audio/testdata/realworld/native_roundtrip_transcripts.tsv. The
// Python reference decode (audio/realworld/roundtrip_decode.py →
// roundtrip_transcripts.tsv) stays committed as an INDEPENDENT
// cross-check that the same tokens are intelligible through a decoder
// implementation gorch shares no code with; this test still reads it
// as that secondary check.
func TestMimiRealWorldTokenProduction(t *testing.T) {
	clips := realworldClips(t)
	mimiPath := envDefault("MIMI_MODEL", defaultMimiCheckpoint)
	if _, err := os.Stat(mimiPath); err != nil {
		t.Skipf("Mimi checkpoint not found at %s", mimiPath)
	}
	enc, err := mimi.Load(mimiPath)
	if err != nil {
		t.Fatalf("load Mimi: %v", err)
	}
	q, err := mimi.LoadQuantizer(mimiPath)
	if err != nil {
		t.Fatalf("load quantizer: %v", err)
	}

	const numQuantizers = 8
	tokens := map[string]*g.Tensor{}
	for _, p := range clips {
		w, err := audio.ReadWAV(p)
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		latent := enc.Encode(w.Mono())
		codes := q.Encode(latent, numQuantizers)
		flat := make([]float32, 0, len(codes)*len(codes[0]))
		for _, level := range codes {
			for _, c := range level {
				flat = append(flat, float32(c))
			}
		}
		tokens[strings.TrimSuffix(filepath.Base(p), ".wav")] =
			g.NewTensor(flat, len(codes), len(codes[0]))
	}

	goldenPath := filepath.Join(realworldDir, "tokens.safetensors")
	if os.Getenv("GORCH_MIMI_WRITE_TOKENS") == "1" {
		if err := model.SaveSafetensors(goldenPath, tokens); err != nil {
			t.Fatalf("write tokens: %v", err)
		}
		t.Skipf("tokens written to %s — run audio/realworld/roundtrip_decode.py, commit outputs, then re-run", goldenPath)
	}
	golden, err := model.LoadSafetensors(goldenPath)
	if err != nil {
		t.Skipf("golden tokens missing (%v) — run with GORCH_MIMI_WRITE_TOKENS=1 first", err)
	}

	// 1. Go token production is deterministic and matches the tokens
	//    whose decodability is on record.
	for name, got := range tokens {
		ref, ok := golden.Tensors[name]
		if !ok {
			t.Fatalf("golden tokens missing entry %s", name)
		}
		if !shapesEqual(got.Shape(), ref.Shape()) {
			t.Fatalf("%s: token shape %v vs golden %v", name, got.Shape(), ref.Shape())
		}
		for i, v := range got.Data() {
			if v != ref.Data()[i] {
				t.Fatalf("%s: token mismatch at %d: %v vs %v", name, i, v, ref.Data()[i])
			}
		}
	}
	t.Logf("token production: %d clips × %d codebooks — exact match with committed tokens", len(clips), numQuantizers)

	// 2. The committed Python-reference round-trip evidence — the
	//    independent cross-check (the native evidence lives in
	//    TestMimiRealWorldNativeRoundtrip).
	total, okCount := readVerdicts(t, "roundtrip_transcripts.tsv", nil)
	if total != len(clips) {
		t.Fatalf("round-trip evidence covers %d clips, want %d", total, len(clips))
	}
	frac := float64(okCount) / float64(total) * 100
	t.Logf("reference-decoder cross-check: %d/%d clips (%.0f%%) re-transcribed correctly after decode", okCount, total, frac)
	if frac < 80.0 {
		t.Fatalf("round-trip intelligibility %.0f%% below 80%% gate", frac)
	}
}

// readVerdicts parses a committed verdict TSV (verdict.py output:
// "<clip>\t<transcript>\t<OK|MISS>", comment lines start with #) and
// returns (total, ok) counts. When covered is non-nil, every data
// line's clip name is recorded into it.
func readVerdicts(t *testing.T, name string, covered map[string]bool) (total, okCount int) {
	verdicts, err := os.ReadFile(filepath.Join(realworldDir, name))
	if err != nil {
		t.Fatalf("round-trip evidence %s missing: %v", name, err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(verdicts)), "\n") {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		total++
		if covered != nil {
			covered[strings.SplitN(line, "\t", 2)[0]] = true
		}
		if strings.HasSuffix(line, "\tOK") {
			okCount++
		}
	}
	return total, okCount
}

// TestMimiRealWorldNativeRoundtrip closes the production loop entirely
// in Go (plan 0007 §7): the committed 30-clip Mimi tokens — proven
// byte-exact reproductions of gorch's encoder output by
// TestMimiRealWorldTokenProduction — are decoded back to 24 kHz audio
// by the NATIVE decoder (mimi.LoadFull → Decoder.Decode; tokens→audio
// with zero Python). The committed evidence chain then shows the
// native reconstructions are intelligible speech:
//
//  1. GORCH_MIMI_WRITE_DECODED=1 writes the 30 native reconstructions
//     to audio/testdata/realworld/native_roundtrip/.
//  2. faster-whisper transcribes them and verdict.py grades the
//     transcripts (homophone-aware) into the committed
//     native_roundtrip_transcripts.tsv.
//  3. The normal run re-decodes every clip natively, asserts the exact
//     1920·T length property, asserts the committed verdicts cover all
//     30 clips with ≥80% OK (expected 30/30), and — for the 3 clips
//     with committed HF reference decodes (rw_*_dec_wav in
//     audio/testdata/mimi_decoder_fixtures.safetensors) — asserts the
//     native waveform matches the reference at ≥40 dB SNR, tying the
//     audio whisper heard to the golden-verified decoder output.
func TestMimiRealWorldNativeRoundtrip(t *testing.T) {
	clips := realworldClips(t)
	mimiPath := envDefault("MIMI_MODEL", defaultMimiCheckpoint)
	if _, err := os.Stat(mimiPath); err != nil {
		t.Skipf("Mimi checkpoint not found at %s", mimiPath)
	}
	golden, err := model.LoadSafetensors(filepath.Join(realworldDir, "tokens.safetensors"))
	if err != nil {
		t.Skipf("committed tokens missing (%v) — run TestMimiRealWorldTokenProduction with GORCH_MIMI_WRITE_TOKENS=1 first", err)
	}
	_, q, dec, err := mimi.LoadFull(mimiPath)
	if err != nil {
		t.Fatalf("load Mimi: %v", err)
	}

	// Decode every committed clip natively: tokens → 24 kHz waveform.
	names := append([]string(nil), golden.Names...)
	sort.Strings(names)
	if len(names) != len(clips) {
		t.Fatalf("committed tokens cover %d clips, want %d", len(names), len(clips))
	}
	wavs := map[string][]float32{}
	start := time.Now()
	for _, name := range names {
		codes := tokenCodes(t, golden, name)
		wav := dec.Decode(q, codes)
		if want := 1920 * len(codes[0]); len(wav) != want {
			t.Fatalf("%s: native decode produced %d samples, want 1920·T = %d", name, len(wav), want)
		}
		wavs[name] = wav
	}
	decodeTime := time.Since(start)
	t.Logf("native decode: %d clips in %v (%.1f ms/clip)",
		len(names), decodeTime.Round(time.Millisecond), float64(decodeTime.Milliseconds())/float64(len(names)))

	nativeDir := filepath.Join(realworldDir, "native_roundtrip")
	if os.Getenv("GORCH_MIMI_WRITE_DECODED") == "1" {
		if err := os.MkdirAll(nativeDir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", nativeDir, err)
		}
		for _, name := range names {
			if err := audio.WriteWAV(filepath.Join(nativeDir, name+".wav"), 24000, wavs[name]); err != nil {
				t.Fatalf("write %s: %v", name, err)
			}
		}
		t.Skipf("native reconstructions written to %s — transcribe with "+
			"audio/realworld/transcribe_check.py, grade with verdict.py into "+
			"audio/testdata/realworld/native_roundtrip_transcripts.tsv, commit, then re-run", nativeDir)
	}

	// 1. The committed whisper verdicts for the NATIVE reconstructions
	//    must cover exactly the committed clips at ≥80% OK.
	covered := map[string]bool{}
	total, okCount := readVerdicts(t, "native_roundtrip_transcripts.tsv", covered)
	if total != len(names) || len(covered) != len(names) {
		t.Fatalf("native round-trip evidence covers %d clips (%d unique), want %d", total, len(covered), len(names))
	}
	for _, name := range names {
		if !covered[name] {
			t.Fatalf("native round-trip evidence missing clip %s", name)
		}
	}
	frac := float64(okCount) / float64(total) * 100
	t.Logf("native round-trip intelligibility: %d/%d clips (%.0f%%) re-transcribed correctly after native decode", okCount, total, frac)
	if frac < 80.0 {
		t.Fatalf("native round-trip intelligibility %.0f%% below 80%% gate", frac)
	}

	// 2. Anchor the native reconstructions to the golden-verified HF
	//    reference decodes for the 3 representative clips.
	fixtures, err := model.LoadSafetensors("../audio/testdata/mimi_decoder_fixtures.safetensors")
	if err != nil {
		t.Fatalf("load decoder fixtures: %v", err)
	}
	for _, clip := range []string{"zero_alloy", "five_echo", "nine_shimmer"} {
		ref, ok := fixtures.Tensors["rw_"+clip+"_dec_wav"]
		if !ok {
			t.Fatalf("decoder fixtures missing rw_%s_dec_wav", clip)
		}
		snr := snrDB(wavs[clip], ref.Data())
		t.Logf("%s: native vs HF reference decode SNR %.1f dB", clip, snr)
		if snr < 40 {
			t.Fatalf("%s: SNR %.1f dB below 40 dB gate", clip, snr)
		}
	}
}

// tokenCodes converts a committed (numQuantizers, T) token tensor into
// the [][]int layout Quantizer/Decoder consume, checking integrality.
func tokenCodes(t *testing.T, sf *model.SafetensorsFile, name string) [][]int {
	tt, ok := sf.Tensors[name]
	if !ok {
		t.Fatalf("tokens missing entry %s", name)
	}
	shape := tt.Shape()
	if len(shape) != 2 {
		t.Fatalf("%s: token shape %v, want 2-D", name, shape)
	}
	K, T := shape[0], shape[1]
	data := tt.Data()
	codes := make([][]int, K)
	for k := 0; k < K; k++ {
		codes[k] = make([]int, T)
		for i := 0; i < T; i++ {
			v := data[k*T+i]
			c := int(v)
			if float32(c) != v || c < 0 || c >= 2048 {
				t.Fatalf("%s[%d][%d] = %v is not a valid Mimi code", name, k, i, v)
			}
			codes[k][i] = c
		}
	}
	return codes
}

// snrDB computes 10·log10(Σref² / Σ(got−ref)²) over the common prefix
// of the two waveforms (trimmed to min length, plan 0007 risk 6).
func snrDB(got, ref []float32) float64 {
	n := len(got)
	if len(ref) < n {
		n = len(ref)
	}
	var sig, noise float64
	for i := 0; i < n; i++ {
		d := float64(got[i]) - float64(ref[i])
		sig += float64(ref[i]) * float64(ref[i])
		noise += d * d
	}
	if noise == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(sig/noise)
}

func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
