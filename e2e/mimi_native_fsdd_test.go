//go:build darwin && e2e

package e2e

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/audio"
	"github.com/vinq1911/gorch/audio/mimi"
	"github.com/vinq1911/gorch/model"
)

// Machine-local defaults (same convention as audio/mimi/seanet_test.go's
// defaultCheckpoint); override with the env vars.
const (
	// FSDD_DIR: directory with the 3000 {digit}_{speaker}_{index}.wav
	// recordings of the Free Spoken Digit Dataset (8 kHz PCM16).
	defaultFSDDDir = "/private/tmp/claude-501/-Users-tuomas-gorch--claude-worktrees-audio-classifier-mimi-gorch-82b2f9/26560a03-b831-4143-ba9e-975d1f17a875/scratchpad/fsdd/recordings"
	// MIMI_MODEL: kyutai/mimi model.safetensors from the HF cache.
	defaultMimiCheckpoint = "/Users/tuomas/.cache/huggingface/hub/models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7/model.safetensors"
)

// Mixed feature-parity tolerance vs the Python exporter:
// |go - py| <= parityAbsTol + parityRelTol*|py|. The Go pipeline is
// float64-exact vs scipy's float64 resampler, but the Python reference
// ran scipy's float32 fast path and PyTorch f32 kernels with different
// summation orders; that ~1e-7-scale input perturbation grows through
// 14 convs + 8 attention layers, so a few-1e-4 difference on the
// pooled features is the expected floor (the encoder itself is
// golden-verified at ~1e-6 vs HF on bit-identical input). Measured on
// the full 3000-clip set: max abs diff 8.2e-5 on an idle machine
// (where repeated runs are bit-identical), up to 5.4e-4 with heavy
// concurrent CPU load (Accelerate BLAS threading varies its reduction
// splits). The gate below keeps ~2x headroom over the worst observed.
const (
	parityAbsTol = 1e-3
	parityRelTol = 2e-3
)

func envDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// poolNative mirrors export_fsdd_mimi.py::pool in Go: mean+std over
// time, population std (unbiased=False), (T, C) -> (2C,). T == 1
// yields zero std (VarAxis with n=1, denom n gives exactly 0).
func poolNative(latent *g.Tensor) []float32 {
	c := latent.Shape()[1]
	out := make([]float32, 2*c)
	copy(out, g.MeanAxis(latent, 0).Data())
	for i, v := range g.VarAxis(latent, 0, false).Data() {
		out[c+i] = float32(math.Sqrt(float64(v)))
	}
	return out
}

// TestMimiNativeFSDD is the plan 0006 Phase 6 acceptance gate: the FSDD
// classifier results of TestMimiFSDD reproduced with embeddings computed
// entirely in Go — WAV read → 8k→24k resample → native Mimi encode →
// mean+std pool — no Python anywhere in the pipeline. When the
// Python-exported reference features are available it also asserts
// per-element feature parity against them.
func TestMimiNativeFSDD(t *testing.T) {
	fsddDir := envDefault("FSDD_DIR", defaultFSDDDir)
	if _, err := os.Stat(fsddDir); err != nil {
		t.Skipf("FSDD recordings not available at %s (set FSDD_DIR): %v", fsddDir, err)
	}
	mimiPath := envDefault("MIMI_MODEL", defaultMimiCheckpoint)
	if _, err := os.Stat(mimiPath); err != nil {
		t.Skipf("mimi checkpoint not available at %s (set MIMI_MODEL): %v", mimiPath, err)
	}

	enc, err := mimi.Load(mimiPath)
	if err != nil {
		t.Fatalf("load mimi encoder: %v", err)
	}

	files, err := filepath.Glob(filepath.Join(fsddDir, "*.wav"))
	if err != nil {
		t.Fatalf("glob %s: %v", fsddDir, err)
	}
	sort.Strings(files) // exporter order: sorted filename walk
	if len(files) == 0 {
		t.Fatalf("no wav files in %s", fsddDir)
	}

	// Speaker label = index into the lexicographically sorted speaker
	// set, exactly like the exporter.
	speakerSet := map[string]bool{}
	for _, f := range files {
		parts := strings.Split(strings.TrimSuffix(filepath.Base(f), ".wav"), "_")
		if len(parts) != 3 {
			t.Fatalf("unexpected FSDD filename %s", f)
		}
		speakerSet[parts[1]] = true
	}
	speakers := make([]string, 0, len(speakerSet))
	for s := range speakerSet {
		speakers = append(speakers, s)
	}
	sort.Strings(speakers)
	speakerIdx := map[string]int{}
	for i, s := range speakers {
		speakerIdx[s] = i
	}
	t.Logf("%d clips, speakers %v", len(files), speakers)

	// Feature extraction, entirely in Go. Rows are appended to their
	// split in file order — the same layout as the exporter's
	// train_x/test_x, so parity can compare row-for-row.
	var trX, teX, trY, teY, trS, teS []float32
	dim := 0
	start := time.Now()
	for i, f := range files {
		parts := strings.Split(strings.TrimSuffix(filepath.Base(f), ".wav"), "_")
		digit, _ := strconv.Atoi(parts[0])
		take, _ := strconv.Atoi(parts[2])

		w, err := audio.ReadWAV(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		if w.SampleRate != 8000 {
			t.Fatalf("%s: expected 8000 Hz, got %d", f, w.SampleRate)
		}
		pcm := audio.Resample(w.Mono(), 8000, 24000)
		feat := poolNative(enc.Encode(pcm))
		dim = len(feat)

		if take < 5 { // conventional FSDD test split
			teX = append(teX, feat...)
			teY = append(teY, float32(digit))
			teS = append(teS, float32(speakerIdx[parts[1]]))
		} else {
			trX = append(trX, feat...)
			trY = append(trY, float32(digit))
			trS = append(trS, float32(speakerIdx[parts[1]]))
		}
		if (i+1)%500 == 0 {
			t.Logf("  encoded %d/%d (%.0f ms/clip)", i+1, len(files),
				float64(time.Since(start).Milliseconds())/float64(i+1))
		}
	}
	extractTime := time.Since(start)
	t.Logf("native feature extraction: %d clips, dim %d, total %v, avg %v/clip",
		len(files), dim, extractTime.Round(time.Millisecond),
		(extractTime / time.Duration(len(files))).Round(time.Microsecond))

	// Feature parity vs the Python-exported reference, when present.
	refPath := os.Getenv("GORCH_MIMI_FSDD_REF")
	if refPath == "" {
		refPath = "../audio/fsdd_mimi.safetensors"
	}
	if _, err := os.Stat(refPath); err != nil {
		t.Logf("python reference features not found at %s (set GORCH_MIMI_FSDD_REF) — skipping parity check", refPath)
	} else {
		sf, err := model.LoadSafetensors(refPath)
		if err != nil {
			t.Fatalf("load reference features: %v", err)
		}
		checkParity(t, "train", trX, trY, trS, sf.Tensors["train_x"], sf.Tensors["train_y"], sf.Tensors["train_spk"], dim)
		checkParity(t, "test", teX, teY, teS, sf.Tensors["test_x"], sf.Tensors["test_y"], sf.Tensors["test_spk"], dim)
	}

	// The three classifier heads, reusing the TestMimiFSDD harness.
	trainX := g.NewTensor(trX, len(trY), dim)
	testX := g.NewTensor(teX, len(teY), dim)

	t.Log("digit head (spoken content, 10 classes):")
	digit := trainHead(t,
		&embDataset{x: trainX, y: g.NewTensor(trY, len(trY), 1), dim: dim},
		&embDataset{x: testX, y: g.NewTensor(teY, len(teY), 1), dim: dim},
		10, 30)
	t.Logf("digit accuracy: %.2f%% (%d/%d)", digit.accuracy, digit.correct, digit.total)

	t.Log("speaker head (paralinguistic, 6 classes):")
	speaker := trainHead(t,
		&embDataset{x: trainX, y: g.NewTensor(trS, len(trS), 1), dim: dim},
		&embDataset{x: testX, y: g.NewTensor(teS, len(teS), 1), dim: dim},
		len(speakers), 30)
	t.Logf("speaker accuracy: %.2f%% (%d/%d)", speaker.accuracy, speaker.correct, speaker.total)

	// Speaker-independent digit head: train on 5 voices, test on the
	// fully held-out 6th (speaker index 5), over all 3000 clips.
	heldOut := float32(5)
	allX := append(append([]float32{}, trX...), teX...)
	allY := append(append([]float32{}, trY...), teY...)
	allS := append(append([]float32{}, trS...), teS...)
	var siTrX, siTeX, siTrY, siTeY []float32
	for i, spk := range allS {
		row := allX[i*dim : (i+1)*dim]
		if spk == heldOut {
			siTeX = append(siTeX, row...)
			siTeY = append(siTeY, allY[i])
		} else {
			siTrX = append(siTrX, row...)
			siTrY = append(siTrY, allY[i])
		}
	}
	t.Logf("speaker-independent digit head (train %d clips / 5 voices, test %d clips / 1 unseen voice):",
		len(siTrY), len(siTeY))
	si := trainHead(t,
		&embDataset{x: g.NewTensor(siTrX, len(siTrY), dim), y: g.NewTensor(siTrY, len(siTrY), 1), dim: dim},
		&embDataset{x: g.NewTensor(siTeX, len(siTeY), dim), y: g.NewTensor(siTeY, len(siTeY), 1), dim: dim},
		10, 30)
	t.Logf("speaker-independent digit accuracy: %.2f%% (%d/%d)", si.accuracy, si.correct, si.total)

	if digit.accuracy < 99.0 {
		t.Errorf("digit accuracy %.2f%% below 99%% gate", digit.accuracy)
	}
	if speaker.accuracy < 99.0 {
		t.Errorf("speaker accuracy %.2f%% below 99%% gate", speaker.accuracy)
	}
	if si.accuracy < 96.0 {
		t.Errorf("speaker-independent accuracy %.2f%% below 96%% gate", si.accuracy)
	}
}

// checkParity compares Go-computed features and labels against the
// Python exporter's, element-wise over the full split, with the mixed
// tolerance |go - py| <= parityAbsTol + parityRelTol*|py|.
func checkParity(t *testing.T, split string, goX, goY, goS []float32, refX, refY, refS *g.Tensor, dim int) {
	t.Helper()
	if refX == nil || refY == nil || refS == nil {
		t.Fatalf("%s: reference file missing tensors", split)
	}
	n := len(goY)
	if rs := refX.Shape(); rs[0] != n || rs[1] != dim {
		t.Fatalf("%s: reference x shape %v, want [%d %d]", split, rs, n, dim)
	}
	for i := 0; i < n; i++ {
		if goY[i] != refY.Data()[i] || goS[i] != refS.Data()[i] {
			t.Fatalf("%s clip %d: label mismatch go (y=%v spk=%v) vs ref (y=%v spk=%v) — split/order drift",
				split, i, goY[i], goS[i], refY.Data()[i], refS.Data()[i])
		}
	}
	var maxAbs, maxExcess float64
	violations, worstClip := 0, -1
	ref := refX.Data()
	for i, b := range ref {
		d := math.Abs(float64(goX[i]) - float64(b))
		if d > maxAbs {
			maxAbs = d
			worstClip = i / dim
		}
		tol := parityAbsTol + parityRelTol*math.Abs(float64(b))
		if r := d / tol; r > maxExcess {
			maxExcess = r
		}
		if d > tol {
			violations++
		}
	}
	t.Logf("%s feature parity vs Python (%d clips x %d dims): max abs diff %.3g (clip %d, digit %v spk %v), max |diff|/tol %.3f (tol = %.0e + %.0e|ref|)",
		split, n, dim, maxAbs, worstClip, goY[worstClip], goS[worstClip], maxExcess, parityAbsTol, parityRelTol)
	if violations > 0 {
		t.Errorf("%s: %d/%d elements exceed parity tolerance (max ratio %.2f)", split, violations, len(ref), maxExcess)
	}
}
