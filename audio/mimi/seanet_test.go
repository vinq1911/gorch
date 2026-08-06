//go:build darwin

package mimi

import (
	"math"
	"math/rand"
	"os"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
)

// defaultCheckpoint is the local HF cache path of kyutai/mimi
// model.safetensors; override with the MIMI_MODEL env var.
const defaultCheckpoint = "/Users/tuomas/.cache/huggingface/hub/models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7/model.safetensors"

func checkpointPath(t testing.TB) string {
	path := os.Getenv("MIMI_MODEL")
	if path == "" {
		path = defaultCheckpoint
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("mimi checkpoint not available at %s (set MIMI_MODEL): %v", path, err)
	}
	return path
}

func loadFixtures(t testing.TB) *model.SafetensorsFile {
	sf, err := model.LoadSafetensors("../testdata/mimi_fixtures.safetensors")
	if err != nil {
		t.Fatalf("load fixtures: %v", err)
	}
	return sf
}

func fixture(t testing.TB, sf *model.SafetensorsFile, name string) *g.Tensor {
	tt, ok := sf.Tensors[name]
	if !ok {
		t.Fatalf("fixture tensor %q missing", name)
	}
	return tt
}

func maxAbsErr(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("maxAbsErr: length mismatch")
	}
	var worst float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > worst {
			worst = d
		}
	}
	return worst
}

// TestSEANetShapes runs without the checkpoint: random weights, random
// input, verify the output geometry (960× downsampling with
// ceil-cascade length semantics).
func TestSEANetShapes(t *testing.T) {
	s := NewSEANet(DefaultConfig())
	if got := len(s.Convs); got != 14 {
		t.Fatalf("conv count = %d, want 14", got)
	}

	rng := rand.New(rand.NewSource(42))
	for _, tc := range []struct{ L, wantT int }{
		{1920, 2},   // one 80 ms chunk
		{48000, 50}, // 2 s, stride-aligned
		{1000, 2},   // awkward length: exercises extra right padding
		{960, 1},
	} {
		data := make([]float32, tc.L)
		for i := range data {
			data[i] = float32(rng.NormFloat64())
		}
		var out *g.Tensor
		g.NoGrad(func() {
			out = s.Forward(g.NewTensor(data, 1, 1, tc.L))
		})
		want := []int{1, s.Cfg.HiddenSize, tc.wantT}
		if !shapeEq(out.Shape(), want) {
			t.Errorf("L=%d: output shape %v, want %v", tc.L, out.Shape(), want)
		}
	}
}

// TestFuseWeightNorm unit-tests the g/v fusion fallback with synthetic
// tensors (the real checkpoint stores plain fused weights, so the
// fallback is not exercised by the golden tests).
func TestFuseWeightNorm(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	outC, inC, k := 3, 4, 5
	v := make([]float32, outC*inC*k)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	gv := []float32{1.5, -0.25, 3.0}

	vt := g.NewTensor(v, outC, inC, k)
	gt := g.NewTensor(gv, outC, 1, 1) // PyTorch stores (outC,1,1)
	fused := fuseWeightNorm(gt, vt)

	if !shapeEq(fused.Shape(), []int{outC, inC, k}) {
		t.Fatalf("fused shape %v, want [%d %d %d]", fused.Shape(), outC, inC, k)
	}
	row := inC * k
	for o := 0; o < outC; o++ {
		var ss float64
		for i := o * row; i < (o+1)*row; i++ {
			ss += float64(v[i]) * float64(v[i])
		}
		norm := math.Sqrt(ss)
		for i := o * row; i < (o+1)*row; i++ {
			want := float64(gv[o]) * float64(v[i]) / norm
			if math.Abs(float64(fused.Data()[i])-want) > 1e-6 {
				t.Fatalf("fused[%d] = %v, want %v", i, fused.Data()[i], want)
			}
		}
	}

	// The loader's convWeight must find both weight-norm key styles
	// when the plain fused key is absent.
	for _, keys := range [][2]string{
		{"x.conv.weight_g", "x.conv.weight_v"},
		{"x.conv.parametrizations.weight.original0", "x.conv.parametrizations.weight.original1"},
	} {
		tensors := map[string]*g.Tensor{keys[0]: gt, keys[1]: vt}
		w, consumed, ok := convWeight(tensors, "x.conv")
		if !ok {
			t.Fatalf("convWeight did not find fallback keys %v", keys)
		}
		if len(consumed) != 2 || consumed[0] != keys[0] || consumed[1] != keys[1] {
			t.Fatalf("convWeight consumed %v, want %v", consumed, keys)
		}
		if maxAbsErr(w.Data(), fused.Data()) != 0 {
			t.Fatalf("convWeight fallback fusion differs from direct fusion")
		}
	}

	// Plain fused key wins over nothing being present.
	if _, _, ok := convWeight(map[string]*g.Tensor{}, "x.conv"); ok {
		t.Fatal("convWeight found a weight in an empty map")
	}
}

// TestLoadSEANetRejectsBadCheckpoint verifies the loader fails loudly
// on missing and unexpected keys.
func TestLoadSEANetRejectsBadCheckpoint(t *testing.T) {
	s := NewSEANet(DefaultConfig())
	tensors := map[string]*g.Tensor{}
	names := []string{}
	for i, prefix := range hfConvPrefixes(s) {
		conv := s.Convs[i]
		wk, bk := prefix+".weight", prefix+".bias"
		tensors[wk] = g.Zeros(conv.Weight.Shape()...)
		tensors[bk] = g.Zeros(conv.Bias.Shape()...)
		names = append(names, wk, bk)
	}

	// Complete synthetic checkpoint loads fine.
	if _, err := loadSEANetFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err != nil {
		t.Fatalf("complete synthetic checkpoint rejected: %v", err)
	}

	// Missing key.
	delete(tensors, "encoder.layers.6.conv.bias")
	_, err := loadSEANetFrom(&model.SafetensorsFile{Tensors: tensors, Names: names})
	if err == nil {
		t.Fatal("loader accepted checkpoint with a missing key")
	}
	tensors["encoder.layers.6.conv.bias"] = g.Zeros(256)

	// Unexpected encoder key.
	tensors["encoder.layers.2.conv.weight"] = g.Zeros(1)
	badNames := append(append([]string{}, names...), "encoder.layers.2.conv.weight")
	if _, err := loadSEANetFrom(&model.SafetensorsFile{Tensors: tensors, Names: badNames}); err == nil {
		t.Fatal("loader accepted checkpoint with an unexpected encoder key")
	}
	delete(tensors, "encoder.layers.2.conv.weight")

	// Wrong shape.
	tensors["encoder.layers.0.conv.weight"] = g.Zeros(64, 1, 5)
	if _, err := loadSEANetFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a misshapen tensor")
	}
}

// goldenSEANet runs the real checkpoint on a PCM fixture and compares
// the offline SEANet output against the transformers reference.
//
// Tolerance note (deviation from the literal plan §5.2 gate): the plan
// budget "max |a−b|/(|b|+1e-5) ≤ 1e-4 over ALL elements" is not
// achievable in f32 — the fixture reference is itself f32 torch, and
// f32 summation noise on near-zero reference entries (|b| ~ 1e-6,
// |a−b| ~ 5e-8 when intermediate activations are O(1)) caps the
// metric at ~5e-3 for any implementation that is not bit-identical to
// PyTorch. Measured here: max abs err 2.3–2.7e-6 against |b| up to
// ~1.04, median elementwise rel err ~1.4e-6. The test therefore
// asserts:
//  1. the plan metric ≤ 1e-4 restricted to |b| ≥ 1e-2 (where it is
//     meaningful), and
//  2. a strict mixed tolerance |a−b| ≤ 5e-6 + 1e-4·|b| over all
//     elements (any real bug — wrong ELU placement, pad off-by-one,
//     swapped weights — produces errors orders of magnitude larger),
//
// and logs the unrestricted plan metric for the record.
func goldenSEANet(t *testing.T, pcmName, refName string) {
	s, err := LoadSEANet(checkpointPath(t))
	if err != nil {
		t.Fatalf("LoadSEANet: %v", err)
	}
	sf := loadFixtures(t)
	pcm := fixture(t, sf, pcmName)
	ref := fixture(t, sf, refName)

	var out *g.Tensor
	g.NoGrad(func() {
		out = s.Forward(g.NewTensor(pcm.Data(), 1, 1, len(pcm.Data())))
	})

	wantShape := append([]int{1}, ref.Shape()...)
	if !shapeEq(out.Shape(), wantShape) {
		t.Fatalf("output shape %v, want %v", out.Shape(), wantShape)
	}

	a, b := out.Data(), ref.Data()
	var relAll, relBig, mixedWorst float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		ab := math.Abs(float64(b[i]))
		if rel := d / (ab + 1e-5); rel > relAll {
			relAll = rel
		}
		if ab >= 1e-2 {
			if rel := d / (ab + 1e-5); rel > relBig {
				relBig = rel
			}
		}
		if m := d / (5e-6 + 1e-4*ab); m > mixedWorst {
			mixedWorst = m
		}
	}
	t.Logf("%s vs %s: plan metric max rel err %.3g (all), %.3g (|ref|>=1e-2); worst mixed-tolerance ratio %.3g",
		pcmName, refName, relAll, relBig, mixedWorst)
	if relBig > 1e-4 {
		t.Fatalf("max relative error %.3g > 1e-4 budget on |ref| >= 1e-2", relBig)
	}
	if mixedWorst > 1 {
		t.Fatalf("mixed tolerance violated: max |a-b|/(5e-6 + 1e-4*|b|) = %.3g > 1", mixedWorst)
	}
}

func TestSEANetGoldenChirp(t *testing.T) { goldenSEANet(t, "chirp_pcm", "chirp_seanet") }
func TestSEANetGoldenLong(t *testing.T)  { goldenSEANet(t, "long_pcm", "long_seanet") }

// TestSEANetStreamingMatchesOffline feeds chirp_pcm as 25 × 1920-sample
// chunks through ForwardStream and requires the concatenated output to
// match the offline Forward within 1e-5 absolute (expected ~exact:
// same conv summation order, chunks stride-aligned so extra padding
// never fires). Uses requireClose's recompute-once discipline: under
// concurrent CPU load Accelerate's threaded GEMM can split reductions
// differently between the two passes, producing transient ~1e-4-scale
// blips that vanish on recompute.
func TestSEANetStreamingMatchesOffline(t *testing.T) {
	s, err := LoadSEANet(checkpointPath(t))
	if err != nil {
		t.Fatalf("LoadSEANet: %v", err)
	}
	sf := loadFixtures(t)
	pcm := fixture(t, sf, "chirp_pcm").Data()

	const chunkLen = 1920 // 80 ms at 24 kHz → 2 output frames
	if len(pcm)%chunkLen != 0 {
		t.Fatalf("fixture length %d not a multiple of %d", len(pcm), chunkLen)
	}
	nChunks := len(pcm) / chunkLen

	requireClose(t, "seanet streaming vs offline", 1e-5, func() (got, ref []float32) {
		var offline *g.Tensor
		g.NoGrad(func() {
			offline = s.Forward(g.NewTensor(pcm, 1, 1, len(pcm)))
		})
		C := offline.Shape()[1]
		T := offline.Shape()[2]

		streamed := make([]float32, C*T)
		states := s.NewStreamStates()
		framesPerChunk := T / nChunks
		g.NoGrad(func() {
			for c := 0; c < nChunks; c++ {
				chunk := g.NewTensor(pcm[c*chunkLen:(c+1)*chunkLen], 1, 1, chunkLen)
				out := s.ForwardStream(chunk, states)
				if !shapeEq(out.Shape(), []int{1, C, framesPerChunk}) {
					t.Fatalf("chunk %d: output shape %v, want [1 %d %d]", c, out.Shape(), C, framesPerChunk)
				}
				for ch := 0; ch < C; ch++ {
					for f := 0; f < framesPerChunk; f++ {
						streamed[ch*T+c*framesPerChunk+f] = out.Data()[ch*framesPerChunk+f]
					}
				}
			}
		})
		return streamed, offline.Data()
	})
}

// BenchmarkSEANetChirp measures the offline SEANet forward on a 2 s
// 24 kHz clip (48000 samples → 50 frames).
func BenchmarkSEANetChirp(b *testing.B) {
	s, err := LoadSEANet(checkpointPath(b))
	if err != nil {
		b.Fatalf("LoadSEANet: %v", err)
	}
	sf := loadFixtures(b)
	pcm := fixture(b, sf, "chirp_pcm").Data()
	x := g.NewTensor(pcm, 1, 1, len(pcm))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.NoGrad(func() {
			s.Forward(x)
		})
	}
}
