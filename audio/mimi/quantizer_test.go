//go:build darwin

package mimi

import (
	"fmt"
	"sync"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
)

var (
	quantOnce   sync.Once
	quantCached *Quantizer
	quantErr    error
)

func loadQuantizerCached(t testing.TB) *Quantizer {
	path := checkpointPath(t)
	quantOnce.Do(func() { quantCached, quantErr = LoadQuantizer(path) })
	if quantErr != nil {
		t.Fatalf("LoadQuantizer: %v", quantErr)
	}
	return quantCached
}

// fixtureCodes reads a (K, T) codes fixture (int codes stored as
// float32 int-values) into [][]int indexed [level][frame].
func fixtureCodes(t testing.TB, sf *model.SafetensorsFile, name string) [][]int {
	tt := fixture(t, sf, name)
	shape := tt.Shape()
	if len(shape) != 2 {
		t.Fatalf("%s: shape %v, want 2-D", name, shape)
	}
	K, T := shape[0], shape[1]
	data := tt.Data()
	codes := make([][]int, K)
	for k := 0; k < K; k++ {
		codes[k] = make([]int, T)
		for i := 0; i < T; i++ {
			v := data[k*T+i]
			c := int(v)
			if float32(c) != v || c < 0 || c >= codebookSize {
				t.Fatalf("%s[%d][%d] = %v is not a valid code", name, k, i, v)
			}
			codes[k][i] = c
		}
	}
	return codes
}

// fixtureLatentTC returns a (512, T) latent fixture transposed to the
// (T, 512) layout Quantizer.Encode consumes.
func fixtureLatentTC(t testing.TB, sf *model.SafetensorsFile, name string) *g.Tensor {
	ref := fixture(t, sf, name)
	var out *g.Tensor
	g.NoGrad(func() { out = transposeCT(ref) })
	return out
}

// compareCodes requires an EXACT integer match against the fixture
// codes. Argmin ties are theoretically possible but none occur on
// these fixtures; a mismatch is reported with its level/frame so a
// genuine tie could be verified numerically rather than tolerated
// blindly.
func compareCodes(t *testing.T, label string, got, want [][]int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d code levels, want %d", label, len(got), len(want))
	}
	mismatches := 0
	for k := range want {
		if len(got[k]) != len(want[k]) {
			t.Fatalf("%s: level %d has %d frames, want %d", label, k, len(got[k]), len(want[k]))
		}
		for i := range want[k] {
			if got[k][i] != want[k][i] {
				mismatches++
				if mismatches <= 5 {
					t.Errorf("%s: level %d frame %d: code %d, want %d", label, k, i, got[k][i], want[k][i])
				}
			}
		}
	}
	if mismatches > 0 {
		t.Errorf("%s: %d/%d codes mismatched", label, mismatches, len(want)*len(want[0]))
	} else {
		t.Logf("%s: %d codes exact match", label, len(want)*len(want[0]))
	}
}

// goldenQuantizerCodes checks Quantizer.Encode of the HF reference
// latent against the fixture codes: all 32 levels and the 8-level
// (Moshi) operating point. Exact integer match required.
func goldenQuantizerCodes(t *testing.T, sig string) {
	q := loadQuantizerCached(t)
	sf := loadFixtures(t)
	latent := fixtureLatentTC(t, sf, sig+"_latent")

	compareCodes(t, sig+"_codes", q.Encode(latent, 32), fixtureCodes(t, sf, sig+"_codes"))
	compareCodes(t, sig+"_codes8", q.Encode(latent, 8), fixtureCodes(t, sf, sig+"_codes8"))
}

func TestQuantizerGoldenCodesChirp(t *testing.T) { goldenQuantizerCodes(t, "chirp") }
func TestQuantizerGoldenCodesLong(t *testing.T)  { goldenQuantizerCodes(t, "long") }

// goldenQuantizerDecode checks Decode of the fixture 8-level codes
// against the HF quantizer.decode reference ((512, T) fixture; Decode
// returns (T, 512)) with the repo's established mixed tolerance.
//
// Tolerance note (layer0-precedent, see stageCheck): the effective
// gate is the mixed tolerance |a-b| <= 5e-6 + 1e-4*|b|, which passes
// with ~2x margin (worst ratio ~0.54). The plan-metric relBig gate is
// 3e-4 rather than the usual 1e-4 because Decode's f32 noise floor
// (~1.7e-5 abs — 256-term output-projection GEMM vs torch's conv1d
// summation order) sits on reference elements right at the 1e-2
// restriction boundary, reading as up to ~2.5e-4 "relative"
// (achieved: chirp 1.2e-4, long 2.5e-4).
func goldenQuantizerDecode(t *testing.T, sig string) {
	q := loadQuantizerCached(t)
	sf := loadFixtures(t)
	codes8 := fixtureCodes(t, sf, sig+"_codes8")

	runGolden(t, func() []*stageCheck {
		out := q.Decode(codes8) // (T, 512)
		ref := fixture(t, sf, sig+"_quantized")
		wantShape := []int{ref.Shape()[1], ref.Shape()[0]}
		if !shapeEq(out.Shape(), wantShape) {
			t.Fatalf("Decode shape %v, want %v", out.Shape(), wantShape)
		}
		var outCT *g.Tensor
		g.NoGrad(func() { outCT = transposeTC(out) }) // (1, 512, T), ref's flat layout
		return []*stageCheck{{stage: sig + "_quantized", got: outCT.Data(), ref: ref.Data(),
			relBigGate: 3e-4, absTol: 5e-6, relTol: 1e-4}}
	})
}

func TestQuantizerGoldenDecodeChirp(t *testing.T) { goldenQuantizerDecode(t, "chirp") }
func TestQuantizerGoldenDecodeLong(t *testing.T)  { goldenQuantizerDecode(t, "long") }

// TestQuantizerRoundTrip checks Encode→Decode consistency on the
// reference latent: Encode(latent, 8) must reproduce the fixture
// quantized latent through Decode, the 8-level codes must be a prefix
// of the 32-level codes (RVQ prefix property), and truncated decodes
// must converge to the full 32-level reconstruction.
func TestQuantizerRoundTrip(t *testing.T) {
	q := loadQuantizerCached(t)
	sf := loadFixtures(t)
	latent := fixtureLatentTC(t, sf, "chirp_latent")

	codes32 := q.Encode(latent, 32)
	codes8 := q.Encode(latent, 8)
	compareCodes(t, "prefix(8 of 32)", codes32[:8], codes8)

	runGolden(t, func() []*stageCheck {
		out := q.Decode(codes8)
		var outCT *g.Tensor
		g.NoGrad(func() { outCT = transposeTC(out) })
		ref := fixture(t, sf, "chirp_quantized")
		return []*stageCheck{{stage: "roundtrip chirp_quantized", got: outCT.Data(), ref: ref.Data(),
			relBigGate: 3e-4, absTol: 5e-6, relTol: 1e-4}}
	})

	// Truncated decodes must converge toward the full 32-level
	// reconstruction as acoustic levels are added (the RVQ residual
	// chain refines in the 256-d projected space; note the decode does
	// NOT converge to the input latent itself — output_proj is not the
	// inverse of input_proj).
	mse := func(a, b []float32) float64 {
		var s float64
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			s += d * d
		}
		return s / float64(len(a))
	}
	full := q.Decode(codes32).Data()
	prev := -1.0
	for _, n := range []int{1, 2, 4, 8, 16} {
		e := mse(q.Decode(codes32[:n]).Data(), full)
		if prev >= 0 && e >= prev {
			t.Errorf("truncated decode did not converge: %d levels MSE %.6g, previous %.6g", n, e, prev)
		}
		t.Logf("%2d levels: MSE vs full 32-level decode %.6g", n, e)
		prev = e
	}
}

// goldenEndToEndCodes is the full native pipeline: pcm →
// Encoder.Encode → Quantizer.Encode, compared exactly against HF's
// model.encode(...).audio_codes.
func goldenEndToEndCodes(t *testing.T, sig string) {
	e := loadEncoderCached(t)
	q := loadQuantizerCached(t)
	sf := loadFixtures(t)
	pcm := fixture(t, sf, sig+"_pcm").Data()

	// Up to 3 encode attempts: on a busy machine Accelerate's threaded
	// GEMM can drift the latent by ~1e-3, flipping a handful of
	// near-boundary argmins (observed 21/4800 under active screen
	// sharing; exact on an idle machine). A real quantizer bug
	// mismatches on every attempt; load noise clears within a retry.
	var g32, g8 [][]int
	w32 := fixtureCodes(t, sf, sig+"_codes")
	w8 := fixtureCodes(t, sf, sig+"_codes8")
	for attempt := 1; attempt <= 3; attempt++ {
		latent := e.Encode(pcm) // (T, 512)
		g32 = q.Encode(latent, 32)
		g8 = q.Encode(latent, 8)
		if codesEqual(g32, w32) && codesEqual(g8, w8) {
			if attempt > 1 {
				t.Logf("%s: exact match on attempt %d (earlier attempts hit load-induced latent drift)", sig, attempt)
			}
			break
		}
		t.Logf("%s: attempt %d had code mismatches — retrying to rule out load-induced BLAS drift", sig, attempt)
	}
	compareCodes(t, sig+" e2e codes", g32, w32)
	compareCodes(t, sig+" e2e codes8", g8, w8)
}

func codesEqual(a, b [][]int) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if len(a[k]) != len(b[k]) {
			return false
		}
		for i := range a[k] {
			if a[k][i] != b[k][i] {
				return false
			}
		}
	}
	return true
}

func TestQuantizerEndToEndChirp(t *testing.T) { goldenEndToEndCodes(t, "chirp") }
func TestQuantizerEndToEndLong(t *testing.T)  { goldenEndToEndCodes(t, "long") }

// TestLoadWithQuantizer verifies the single-parse combined loader
// against the individually loaded halves on one real encode.
func TestLoadWithQuantizer(t *testing.T) {
	path := checkpointPath(t)
	e, q, err := LoadWithQuantizer(path)
	if err != nil {
		t.Fatalf("LoadWithQuantizer: %v", err)
	}
	sf := loadFixtures(t)
	pcm := fixture(t, sf, "chirp_pcm").Data()
	latent := e.Encode(pcm)
	compareCodes(t, "combined-loader codes8", q.Encode(latent, 8), fixtureCodes(t, sf, "chirp_codes8"))
}

// TestLoadQuantizerRejectsBadCheckpoint verifies the fail-loudly
// key/shape validation. Runs without the real checkpoint.
func TestLoadQuantizerRejectsBadCheckpoint(t *testing.T) {
	tensors := map[string]*g.Tensor{}
	names := []string{}
	add := func(key string, shape ...int) {
		tensors[key] = g.Zeros(shape...)
		names = append(names, key)
	}
	for _, half := range []struct {
		name  string
		books int
	}{{"semantic", numSemantic}, {"acoustic", numAcoustic}} {
		p := "quantizer." + half.name + "_residual_vector_quantizer."
		add(p+"input_proj.weight", codebookDim, 512, 1)
		add(p+"output_proj.weight", 512, codebookDim, 1)
		for i := 0; i < half.books; i++ {
			cp := fmt.Sprintf("%slayers.%d.codebook.", p, i)
			add(cp+"embed_sum", codebookSize, codebookDim)
			add(cp+"cluster_usage", codebookSize)
			add(cp+"initialized", 1)
		}
	}

	// Complete synthetic checkpoint loads fine.
	if _, err := loadQuantizerFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err != nil {
		t.Fatalf("complete synthetic checkpoint rejected: %v", err)
	}

	// Missing codebook key.
	key := "quantizer.acoustic_residual_vector_quantizer.layers.17.codebook.embed_sum"
	saved := tensors[key]
	delete(tensors, key)
	if _, err := loadQuantizerFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a missing codebook key")
	}
	tensors[key] = saved

	// Misshapen projection.
	tensors["quantizer.semantic_residual_vector_quantizer.input_proj.weight"] = g.Zeros(codebookDim, 512)
	if _, err := loadQuantizerFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a misshapen input_proj")
	}
	add("quantizer.semantic_residual_vector_quantizer.input_proj.weight", codebookDim, 512, 1)
	names = names[:len(names)-1]

	// Unexpected quantizer key.
	add("quantizer.semantic_residual_vector_quantizer.layers.1.codebook.embed_sum", codebookSize, codebookDim)
	if _, err := loadQuantizerFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with an unexpected quantizer key")
	}
}

// TestQuantizerClusterUsageClamp exercises the ε-clamp path
// (cluster_usage < 1e-5 must divide by 1e-5, not the raw value).
func TestQuantizerClusterUsageClamp(t *testing.T) {
	embedSum := g.Zeros(codebookSize, codebookDim)
	usage := g.Zeros(codebookSize)
	embedSum.Data()[0] = 2e-5 // row 0, dim 0
	usage.Data()[0] = 1e-9    // far below ε → clamped to 1e-5
	for i := 1; i < codebookSize; i++ {
		usage.Data()[i] = 1
	}
	cb := newCodebook(embedSum, usage)
	if got, want := cb.embed[0], float32(2.0); got != want {
		t.Errorf("clamped embed = %v, want %v (embed_sum / ε)", got, want)
	}
}
