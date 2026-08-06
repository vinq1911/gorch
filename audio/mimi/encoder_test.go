//go:build darwin

package mimi

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
)

var (
	encOnce   sync.Once
	encCached *Encoder
	encErr    error
)

// loadEncoderCached loads the full encoder from the real checkpoint
// once per test binary (the 385 MB safetensors parse is the slow part).
func loadEncoderCached(t testing.TB) *Encoder {
	path := checkpointPath(t)
	encOnce.Do(func() { encCached, encErr = Load(path) })
	if encErr != nil {
		t.Fatalf("Load: %v", encErr)
	}
	return encCached
}

// stageCheck is one golden comparison: a computed stage against a
// fixture reference with the P2 precedent gates (see goldenSEANet's
// tolerance note):
//  1. plan metric |a−b|/(|b|+1e-5) ≤ relBigGate restricted to
//     |b| ≥ 1e-2 (1e-4 for most stages; the layer0 stages use 2.5e-4
//     because f32 noise of ~2e-6 abs on a reference element right at
//     the 1e-2 restriction boundary reads as ~1.2e-4 "relative"),
//  2. mixed tolerance |a−b| ≤ absTol + relTol·|b| over all elements.
type stageCheck struct {
	stage          string
	got, ref       []float32
	relBigGate     float64
	absTol, relTol float64
	maxAbs, relAll float64
	relBig, mixed  float64
	failure        string
}

func (c *stageCheck) evaluate() bool {
	c.maxAbs, c.relAll, c.relBig, c.mixed = 0, 0, 0, 0
	for i := range c.got {
		d := math.Abs(float64(c.got[i]) - float64(c.ref[i]))
		ab := math.Abs(float64(c.ref[i]))
		if d > c.maxAbs {
			c.maxAbs = d
		}
		if rel := d / (ab + 1e-5); rel > c.relAll {
			c.relAll = rel
		}
		if ab >= 1e-2 {
			if rel := d / (ab + 1e-5); rel > c.relBig {
				c.relBig = rel
			}
		}
		if m := d / (c.absTol + c.relTol*ab); m > c.mixed {
			c.mixed = m
		}
	}
	c.failure = ""
	if c.relBig > c.relBigGate {
		c.failure = fmt.Sprintf("%s: max relative error %.3g > %.0e budget on |ref| >= 1e-2", c.stage, c.relBig, c.relBigGate)
	}
	if c.mixed > 1 {
		c.failure += fmt.Sprintf("; %s: mixed tolerance violated: max |a-b|/(%.0e + %.0e*|b|) = %.3g > 1",
			c.stage, c.absTol, c.relTol, c.mixed)
	}
	return c.failure == ""
}

func (c *stageCheck) log(t *testing.T) {
	t.Helper()
	t.Logf("%s: max abs err %.3g; plan metric max rel err %.3g (all), %.3g (|ref|>=1e-2); worst ratio %.3g vs |a-b| <= %.0e + %.0e*|b|",
		c.stage, c.maxAbs, c.relAll, c.relBig, c.mixed, c.absTol, c.relTol)
}

// runGolden evaluates the checks produced by compute; on a gate
// failure it recomputes ONCE and requires the failure to reproduce.
// Rationale: results are bit-identical run to run in isolation (also
// under GOGC=1), but under heavy parallel load Apple Accelerate's
// threaded GEMM occasionally splits work differently and the changed
// f32 summation order showed up as a spurious ~1.7e-5 max-abs blip
// (~1 in 15 loaded runs). A transient scheduling difference does not
// repeat deterministically; a real regression fails both attempts.
func runGolden(t *testing.T, compute func() []*stageCheck) {
	t.Helper()
	checks := compute()
	ok := true
	for _, c := range checks {
		if !c.evaluate() {
			ok = false
		}
		c.log(t)
	}
	if ok {
		return
	}
	t.Logf("gate failure — recomputing once to rule out load-induced BLAS nondeterminism")
	checks = compute()
	for _, c := range checks {
		pass := c.evaluate()
		c.log(t)
		if !pass {
			t.Error(c.failure)
		}
	}
}

// goldenTransformer runs SEANet + the 8 transformer layers from the
// real checkpoint on a PCM fixture and checks the post-layer0 and
// post-transformer stages. window <= 0 is plain causal (the HF offline
// reference the default Encode must match); window = 250 is the strict
// sliding-window variant recorded in the *_win fixtures. layer0Ref may
// be empty (no windowed layer0 fixture exists).
func goldenTransformer(t *testing.T, sig string, window int, layer0Ref, transRef string) {
	e := loadEncoderCached(t)
	sf := loadFixtures(t)
	pcm := fixture(t, sf, sig+"_pcm").Data()

	runGolden(t, func() []*stageCheck {
		var h0, hT *g.Tensor
		g.NoGrad(func() {
			x := e.SEANet.Forward(g.NewTensor(pcm, 1, 1, len(pcm)))
			h := transposeCT(x)
			h0 = e.Layers[0].Forward(h, e.Rope, window)
			h = h0
			for _, l := range e.Layers[1:] {
				h = l.Forward(h, e.Rope, window)
			}
			hT = h
		})

		var checks []*stageCheck
		if layer0Ref != "" {
			ref := fixture(t, sf, layer0Ref)
			if !shapeEq(h0.Shape(), ref.Shape()) {
				t.Fatalf("layer0 shape %v, want %v", h0.Shape(), ref.Shape())
			}
			checks = append(checks, &stageCheck{stage: layer0Ref, got: h0.Data(), ref: ref.Data(),
				relBigGate: 2.5e-4, absTol: 5e-6, relTol: 1e-4})
		}
		ref := fixture(t, sf, transRef)
		if !shapeEq(hT.Shape(), ref.Shape()) {
			t.Fatalf("transformer shape %v, want %v", hT.Shape(), ref.Shape())
		}
		return append(checks, &stageCheck{stage: transRef, got: hT.Data(), ref: ref.Data(),
			relBigGate: 1e-4, absTol: 5e-6, relTol: 1e-4})
	})
}

func TestTransformerGoldenChirp(t *testing.T) {
	goldenTransformer(t, "chirp", 0, "chirp_layer0", "chirp_transformer")
}

func TestTransformerGoldenLong(t *testing.T) {
	goldenTransformer(t, "long", 0, "long_layer0", "long_transformer")
}

func TestTransformerGoldenLongWindowed(t *testing.T) {
	goldenTransformer(t, "long", DefaultConfig().SlidingWindow, "", "long_transformer_win")
}

// goldenLatent checks the full Encode/EncodeWindowed output against
// the fixture latent ((512, T12.5); Encode returns (T12.5, 512)).
func goldenLatent(t *testing.T, sig string, window int, latentRef string) {
	e := loadEncoderCached(t)
	sf := loadFixtures(t)
	pcm := fixture(t, sf, sig+"_pcm").Data()

	runGolden(t, func() []*stageCheck {
		var out *g.Tensor
		if window > 0 {
			out = e.EncodeWindowed(pcm)
		} else {
			out = e.Encode(pcm)
		}

		ref := fixture(t, sf, latentRef) // (512, T)
		wantShape := []int{ref.Shape()[1], ref.Shape()[0]}
		if !shapeEq(out.Shape(), wantShape) {
			t.Fatalf("latent shape %v, want %v", out.Shape(), wantShape)
		}
		var outCT *g.Tensor
		g.NoGrad(func() { outCT = transposeTC(out) }) // (1, 512, T), same flat layout as ref
		return []*stageCheck{{stage: latentRef, got: outCT.Data(), ref: ref.Data(),
			relBigGate: 1e-4, absTol: 5e-6, relTol: 1e-4}}
	})
}

func TestEncoderGoldenChirp(t *testing.T) { goldenLatent(t, "chirp", 0, "chirp_latent") }
func TestEncoderGoldenLong(t *testing.T)  { goldenLatent(t, "long", 0, "long_latent") }
func TestEncoderGoldenLongWindowed(t *testing.T) {
	goldenLatent(t, "long", DefaultConfig().SlidingWindow, "long_latent_win")
}

// TestEncoderPooledChirp verifies the classifier-facing pooled vector:
// mean+std pooling (VarAxis unbiased=false) of the chirp latent
// against the Python exporter's chirp_pooled fixture.
func TestEncoderPooledChirp(t *testing.T) {
	e := loadEncoderCached(t)
	sf := loadFixtures(t)
	pcm := fixture(t, sf, "chirp_pcm").Data()

	runGolden(t, func() []*stageCheck {
		latent := e.Encode(pcm) // (T, 512)
		dim := latent.Shape()[1]
		pooled := make([]float32, 2*dim)
		g.NoGrad(func() {
			mean := g.MeanAxis(latent, 0)
			vr := g.VarAxis(latent, 0, false)
			copy(pooled, mean.Data())
			for i, v := range vr.Data() {
				pooled[dim+i] = float32(math.Sqrt(float64(v)))
			}
		})

		ref := fixture(t, sf, "chirp_pooled")
		return []*stageCheck{{stage: "chirp_pooled", got: pooled, ref: ref.Data(),
			relBigGate: 1e-4, absTol: 5e-6, relTol: 1e-4}}
	})
}

// TestLoadEncoderRejectsBadCheckpoint verifies the full-encoder loader
// fails loudly on missing, unexpected and misshapen transformer /
// downsample keys. Runs without the real checkpoint.
func TestLoadEncoderRejectsBadCheckpoint(t *testing.T) {
	s := NewSEANet(DefaultConfig())
	tensors := map[string]*g.Tensor{}
	names := []string{}
	add := func(key string, shape ...int) {
		tensors[key] = g.Zeros(shape...)
		names = append(names, key)
	}
	for i, prefix := range hfConvPrefixes(s) {
		conv := s.Convs[i]
		add(prefix+".weight", conv.Weight.Shape()...)
		add(prefix+".bias", conv.Bias.Shape()...)
	}
	for i := 0; i < 8; i++ {
		p := fmt.Sprintf("encoder_transformer.layers.%d.", i)
		add(p+"self_attn.q_proj.weight", 512, 512)
		add(p+"self_attn.k_proj.weight", 512, 512)
		add(p+"self_attn.v_proj.weight", 512, 512)
		add(p+"self_attn.o_proj.weight", 512, 512)
		add(p+"mlp.fc1.weight", 2048, 512)
		add(p+"mlp.fc2.weight", 512, 2048)
		add(p+"input_layernorm.weight", 512)
		add(p+"input_layernorm.bias", 512)
		add(p+"post_attention_layernorm.weight", 512)
		add(p+"post_attention_layernorm.bias", 512)
		add(p+"self_attn_layer_scale.scale", 512)
		add(p+"mlp_layer_scale.scale", 512)
	}
	add("downsample.conv.weight", 512, 512, 4)

	// Complete synthetic checkpoint loads fine.
	if _, err := loadEncoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err != nil {
		t.Fatalf("complete synthetic checkpoint rejected: %v", err)
	}

	// Missing transformer key.
	saved := tensors["encoder_transformer.layers.3.mlp.fc1.weight"]
	delete(tensors, "encoder_transformer.layers.3.mlp.fc1.weight")
	if _, err := loadEncoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a missing transformer key")
	}
	tensors["encoder_transformer.layers.3.mlp.fc1.weight"] = saved

	// Unexpected transformer key.
	add("encoder_transformer.norm.weight", 512)
	if _, err := loadEncoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with an unexpected transformer key")
	}
	delete(tensors, "encoder_transformer.norm.weight")
	names = names[:len(names)-1]

	// Misshapen downsample weight.
	tensors["downsample.conv.weight"] = g.Zeros(512, 512, 3)
	if _, err := loadEncoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a misshapen downsample weight")
	}
}

// TestEncoderShapes runs without the checkpoint: random weights,
// verify the end-to-end 1920×-downsampling geometry.
func TestEncoderShapes(t *testing.T) {
	e := NewEncoder(DefaultConfig())
	rng := rand.New(rand.NewSource(9))
	for _, tc := range []struct{ L, wantT int }{
		{1920, 1},   // one 80 ms chunk → one 12.5 Hz frame
		{48000, 25}, // 2 s
	} {
		pcm := make([]float32, tc.L)
		for i := range pcm {
			pcm[i] = float32(rng.NormFloat64())
		}
		out := e.Encode(pcm)
		if !shapeEq(out.Shape(), []int{tc.wantT, 512}) {
			t.Errorf("L=%d: latent shape %v, want [%d 512]", tc.L, out.Shape(), tc.wantT)
		}
	}
}

// BenchmarkMimiEncode10s measures the full offline Encode on 10 s of
// 24 kHz audio (240000 samples → 125 latent frames). Python baseline:
// 334 ms.
func BenchmarkMimiEncode10s(b *testing.B) {
	e := loadEncoderCached(b)
	rng := rand.New(rand.NewSource(1))
	pcm := make([]float32, 10*24000)
	for i := range pcm {
		pcm[i] = float32(0.5 * rng.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Encode(pcm)
	}
}
