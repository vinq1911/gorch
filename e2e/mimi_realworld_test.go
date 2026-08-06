//go:build darwin && e2e

package e2e

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

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
// tokens (8 codebooks — Moshi's speech-generation currency), and the
// committed evidence chain shows those exact tokens decode back to
// intelligible audio: audio/realworld/roundtrip_decode.py fed the
// committed tokens through the reference Mimi decoder and
// faster-whisper re-transcribed the reconstructions
// (audio/testdata/realworld/roundtrip_transcripts.tsv). This test
// re-derives the tokens from scratch and requires an exact match with
// the committed tokens, plus a passing intelligibility verdict count.
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

	// 2. The committed round-trip evidence for those tokens.
	verdicts, err := os.ReadFile(filepath.Join(realworldDir, "roundtrip_transcripts.tsv"))
	if err != nil {
		t.Fatalf("round-trip evidence missing: %v — run audio/realworld/roundtrip_decode.py", err)
	}
	total, okCount := 0, 0
	for _, line := range strings.Split(strings.TrimSpace(string(verdicts)), "\n") {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		total++
		if strings.HasSuffix(line, "\tOK") {
			okCount++
		}
	}
	if total != len(clips) {
		t.Fatalf("round-trip evidence covers %d clips, want %d", total, len(clips))
	}
	frac := float64(okCount) / float64(total) * 100
	t.Logf("token round-trip intelligibility: %d/%d clips (%.0f%%) re-transcribed correctly after decode", okCount, total, frac)
	if frac < 80.0 {
		t.Fatalf("round-trip intelligibility %.0f%% below 80%% gate", frac)
	}
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
