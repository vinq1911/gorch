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
	"github.com/vinq1911/gorch/nn"
)

var (
	fullOnce  sync.Once
	fullEnc   *Encoder
	fullQuant *Quantizer
	fullDec   *Decoder
	fullErr   error
)

// loadFullCached loads encoder+quantizer+decoder from the real
// checkpoint once per test binary via LoadFull, so the decoder goldens
// also exercise LoadFull's 350-tensor coverage assertion.
func loadFullCached(t testing.TB) (*Quantizer, *Decoder) {
	path := checkpointPath(t)
	fullOnce.Do(func() { fullEnc, fullQuant, fullDec, fullErr = LoadFull(path) })
	if fullErr != nil {
		t.Fatalf("LoadFull: %v", fullErr)
	}
	return fullQuant, fullDec
}

func loadDecoderFixtures(t testing.TB) *model.SafetensorsFile {
	sf, err := model.LoadSafetensors("../testdata/mimi_decoder_fixtures.safetensors")
	if err != nil {
		t.Fatalf("load decoder fixtures: %v", err)
	}
	return sf
}

// goldenCheck generalizes the stageCheck evaluate/log contract so the
// waveform two-part gate (wavCheck) can share one runner with the
// per-stage checks.
type goldenCheck interface {
	evaluate() bool
	log(t *testing.T)
	fail() string
}

func (c *stageCheck) fail() string { return c.failure }

// Waveform two-part gate (plan 0007 §5.3). Both parts enforced:
//  1. per-clip SNR >= 60 dB — f32-parity implementations land
//     80–100 dB while masked bugs sit at −20…−40 dB;
//  2. sample-level mixed |a−b| <= wavAbsTol + wavRelTol·|b| — catches
//     isolated spikes the SNR averages away.
//
// Calibration (D2, 2026-08-07, M4, three separate runs per clip with
// the busy-machine min discipline — the minimum is the algorithmic
// floor; observed values were bit-stable across runs):
//
//	clip                 SNR (dB)   max|a−b|
//	chirp_dec_wav        121.0      1.91e-6
//	long_dec_wav         119.8      1.11e-5
//	long_dec_wav_win     119.5      1.16e-5
//	rw_zero_alloy        122.8      8.34e-7
//	rw_five_echo         118.6      1.58e-6
//	rw_nine_shimmer      118.8      2.24e-6
//
// Secondary-gate constants are ~10× the measured worst max-abs floor
// (1.16e-5, on a [-1,1]-scale waveform, so absTol carries it): absTol
// 1e-4, relTol 1e-3. The measured SNR floor (118.6 dB) keeps ~59 dB
// headroom over the plan-fixed 60 dB primary gate.
const (
	wavMinSNR = 60.0
	wavAbsTol = 1e-4
	wavRelTol = 1e-3
)

// wavCheck is one waveform comparison under the two-part gate.
type wavCheck struct {
	stage    string
	got, ref []float32
	minSNR   float64
	absTol   float64
	relTol   float64

	snr, maxAbs, mixed float64
	failure            string
}

func (c *wavCheck) evaluate() bool {
	var sumRef, sumErr float64
	c.maxAbs, c.mixed = 0, 0
	for i := range c.got {
		r := float64(c.ref[i])
		d := float64(c.got[i]) - r
		sumRef += r * r
		sumErr += d * d
		ad := math.Abs(d)
		if ad > c.maxAbs {
			c.maxAbs = ad
		}
		if m := ad / (c.absTol + c.relTol*math.Abs(r)); m > c.mixed {
			c.mixed = m
		}
	}
	if sumErr == 0 {
		c.snr = math.Inf(1)
	} else {
		c.snr = 10 * math.Log10(sumRef/sumErr)
	}
	c.failure = ""
	if c.snr < c.minSNR {
		c.failure = fmt.Sprintf("%s: SNR %.1f dB < %.0f dB gate", c.stage, c.snr, c.minSNR)
	}
	if c.mixed > 1 {
		c.failure += fmt.Sprintf("; %s: mixed tolerance violated: max |a-b|/(%.0e + %.0e*|b|) = %.3g > 1",
			c.stage, c.absTol, c.relTol, c.mixed)
	}
	return c.failure == ""
}

func (c *wavCheck) log(t *testing.T) {
	t.Helper()
	t.Logf("%s: SNR %.1f dB (gate >= %.0f); max abs err %.3g; worst ratio %.3g vs |a-b| <= %.0e + %.0e*|b|",
		c.stage, c.snr, c.minSNR, c.maxAbs, c.mixed, c.absTol, c.relTol)
}

func (c *wavCheck) fail() string { return c.failure }

// runGoldenChecks evaluates the checks produced by compute with the
// hardened retry discipline (requireClose precedent, stream_test.go):
// up to 4 attempts, passing as soon as one attempt clears every gate.
// Under heavy parallel load Accelerate's threaded GEMM reorders f32
// reductions differently per run; a genuine algorithmic bug produces a
// stable error floor that fails every attempt, while transient
// scheduling noise eventually yields a quiet run.
func runGoldenChecks(t *testing.T, compute func() []goldenCheck) {
	t.Helper()
	const attempts = 4
	var checks []goldenCheck
	for i := 0; i < attempts; i++ {
		checks = compute()
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
		if i < attempts-1 {
			t.Logf("gate failure on attempt %d — recomputing to rule out load-induced BLAS nondeterminism", i+1)
		}
	}
	for _, c := range checks {
		if msg := c.fail(); msg != "" {
			t.Error(msg)
		}
	}
}

// decStages holds the decoder pipeline intermediates matching the D0
// fixture stages (mirrors the encoder goldens' manual staging so a
// failure attributes to one component).
type decStages struct {
	upsampled   *g.Tensor // (1, 512, 2T)  after the depthwise upsample
	layer0      *g.Tensor // (2T, 512)     after decoder_transformer layer 0
	transformer *g.Tensor // (2T, 512)     decoder_transformer output
	seanet0     *g.Tensor // (1, 1024, 2T) after decoder.layers.0
	stage1      *g.Tensor // (1, 512, 16T) after decoder.layers.3
	stage2      *g.Tensor // (1, 256, 96T) after decoder.layers.6
}

// computeDecoderStages runs the same pipeline as Decoder.decode but
// keeps the fixture-stage intermediates.
func computeDecoderStages(d *Decoder, latent *g.Tensor, window int) *decStages {
	st := &decStages{}
	g.NoGrad(func() {
		x := transposeTC(latent)
		st.upsampled = d.Upsample.Forward(x)
		h := transposeCT(st.upsampled)
		st.layer0 = d.Layers[0].Forward(h, d.Rope, window)
		h = st.layer0
		for _, l := range d.Layers[1:] {
			h = l.Forward(h, d.Rope, window)
		}
		st.transformer = h
		y := d.SEANet.Init.Forward(transposeTC(h))
		st.seanet0 = y
		for i := range d.SEANet.Ups {
			y = g.ELU(y)
			y = d.SEANet.Ups[i].Forward(y)
			y = resnetForward(y, d.SEANet.Res[i][0], d.SEANet.Res[i][1])
			switch i {
			case 0:
				st.stage1 = y
			case 1:
				st.stage2 = y
			}
		}
	})
	return st
}

// goStageLatent returns the Go-quantized (T, 512) latent for sig: the
// decoder stage input is Quantizer.Decode of the fixture codes (exact
// vs HF, quantizer goldens), so stage errors attribute to decoder code
// alone.
func goStageLatent(t testing.TB, q *Quantizer, sig string) *g.Tensor {
	sf := loadFixtures(t)
	return q.Decode(fixtureCodes(t, sf, sig+"_codes8"))
}

// requireFlatLen fails unless got (flattened) matches the ref fixture
// element count; stage tensors compare flat because the fixture drops
// the leading batch dim ((512, T) vs Go's (1, 512, T)).
func requireFlatLen(t *testing.T, stage string, got *g.Tensor, ref *g.Tensor) {
	t.Helper()
	n := 1
	for _, s := range got.Shape() {
		n *= s
	}
	if n != len(ref.Data()) {
		t.Fatalf("%s: %d elements (shape %v), fixture has %d", stage, n, got.Shape(), len(ref.Data()))
	}
}

// Per-stage gates, CALIBRATED (D2, 2026-08-07, M4; the plan §5.3
// values were provisional). The decoder's stage-input latent is the Go
// Quantizer.Decode of exact fixture codes, but that latent itself
// already differs from HF's by up to relBig 1.2e-4 (chirp) / 2.5e-4
// (long) — pure f32 summation-order noise in the 2048-sum RVQ output
// projection GEMM on values up to |32| — so the plan's 1e-4-class
// per-stage gates are unreachable from this side of the quantizer.
// Measured floors (bit-stable across 4 in-run attempts × 3 separate
// runs — a deterministic algorithmic floor, not load noise; the
// ~120 dB final-waveform SNR pins the pipeline itself as correct):
//
//	stage                     relBig floor   mixed floor (vs listed tol)
//	chirp_dec_upsampled       4.18e-5        0.084
//	long_dec_upsampled        4.54e-5        0.131
//	chirp_dec_layer0          4.59e-5        0.144
//	chirp_dec_transformer     2.39e-4        0.15
//	long_dec_transformer(_win) 6.39e-4       0.48
//	chirp_dec_seanet0         3.78e-4        0.18
//	long_dec_seanet0          9.09e-4        0.67
//	chirp_dec_stage1          2.60e-3        0.54
//	chirp_dec_stage2          1.54e-3        0.45
//
// Gates sit ~4–10× above the floors; real decoder bugs (swapped taps,
// wrong trim, double bias) produce relative errors >=1e-1 and fail by
// orders of magnitude.
const (
	upsampledRelBig, upsampledAbsTol, upsampledRelTol       = 1e-4, 5e-6, 1e-4 // plan value holds
	layer0RelBig, layer0AbsTol, layer0RelTol                = 2.5e-4, 5e-6, 1e-4
	transformerRelBig, transformerAbsTol, transformerRelTol = 2.5e-3, 2e-5, 4e-4
	seanet0RelBig, seanet0AbsTol, seanet0RelTol             = 5e-3, 5e-5, 1e-3
	stage12RelBig, stage12AbsTol, stage12RelTol             = 1e-2, 5e-5, 2.5e-3
)

// goldenDecoderStages checks the per-stage decoder pipeline for sig
// against the D0 fixtures. layer0/stage1/stage2 fixtures exist for
// chirp only (D0 size trim, plan 0007 §5.1).
func goldenDecoderStages(t *testing.T, sig string, withChirpOnly bool) {
	q, d := loadFullCached(t)
	dsf := loadDecoderFixtures(t)
	latent := goStageLatent(t, q, sig)

	runGoldenChecks(t, func() []goldenCheck {
		st := computeDecoderStages(d, latent, 0)

		mk := func(stage string, got *g.Tensor, relBig, absTol, relTol float64) goldenCheck {
			ref := fixture(t, dsf, stage)
			requireFlatLen(t, stage, got, ref)
			return &stageCheck{stage: stage, got: got.Data(), ref: ref.Data(),
				relBigGate: relBig, absTol: absTol, relTol: relTol}
		}
		checks := []goldenCheck{
			mk(sig+"_dec_upsampled", st.upsampled, upsampledRelBig, upsampledAbsTol, upsampledRelTol),
			mk(sig+"_dec_transformer", st.transformer, transformerRelBig, transformerAbsTol, transformerRelTol),
			mk(sig+"_dec_seanet0", st.seanet0, seanet0RelBig, seanet0AbsTol, seanet0RelTol),
		}
		if withChirpOnly {
			checks = append(checks,
				mk(sig+"_dec_layer0", st.layer0, layer0RelBig, layer0AbsTol, layer0RelTol),
				mk(sig+"_dec_stage1", st.stage1, stage12RelBig, stage12AbsTol, stage12RelTol),
				mk(sig+"_dec_stage2", st.stage2, stage12RelBig, stage12AbsTol, stage12RelTol))
		}
		return checks
	})
}

func TestDecoderGoldenStagesChirp(t *testing.T) { goldenDecoderStages(t, "chirp", true) }
func TestDecoderGoldenStagesLong(t *testing.T)  { goldenDecoderStages(t, "long", false) }

// goldenDecoderWav checks the public DecodeLatent/DecodeLatentWindowed
// waveform for sig against the fixture under the two-part gate.
func goldenDecoderWav(t *testing.T, sig string, windowed bool, wavRef string) {
	q, d := loadFullCached(t)
	dsf := loadDecoderFixtures(t)
	latent := goStageLatent(t, q, sig)
	T := latent.Shape()[0]

	runGoldenChecks(t, func() []goldenCheck {
		var wav []float32
		if windowed {
			wav = d.DecodeLatentWindowed(latent)
		} else {
			wav = d.DecodeLatent(latent)
		}
		if len(wav) != 1920*T {
			t.Fatalf("%s: %d samples, want 1920*%d = %d", wavRef, len(wav), T, 1920*T)
		}
		ref := fixture(t, dsf, wavRef)
		if len(ref.Data()) != len(wav) {
			t.Fatalf("%s: fixture has %d samples, decode produced %d", wavRef, len(ref.Data()), len(wav))
		}
		return []goldenCheck{&wavCheck{stage: wavRef, got: wav, ref: ref.Data(),
			minSNR: wavMinSNR, absTol: wavAbsTol, relTol: wavRelTol}}
	})
}

func TestDecoderGoldenWavChirp(t *testing.T) { goldenDecoderWav(t, "chirp", false, "chirp_dec_wav") }
func TestDecoderGoldenWavLong(t *testing.T)  { goldenDecoderWav(t, "long", false, "long_dec_wav") }

// TestDecoderGoldenLongWindowed covers the strict 250-frame window
// variant on the 300-frame long signal (the only fixture where the
// window binds): the transformer stage under the explicit window mask
// and the full DecodeLatentWindowed waveform.
func TestDecoderGoldenLongWindowed(t *testing.T) {
	q, d := loadFullCached(t)
	dsf := loadDecoderFixtures(t)
	latent := goStageLatent(t, q, "long")

	runGoldenChecks(t, func() []goldenCheck {
		st := computeDecoderStages(d, latent, d.Cfg.SlidingWindow)
		ref := fixture(t, dsf, "long_dec_transformer_win")
		requireFlatLen(t, "long_dec_transformer_win", st.transformer, ref)
		return []goldenCheck{&stageCheck{stage: "long_dec_transformer_win",
			got: st.transformer.Data(), ref: ref.Data(),
			relBigGate: transformerRelBig, absTol: transformerAbsTol, relTol: transformerRelTol}}
	})
	goldenDecoderWav(t, "long", true, "long_dec_wav_win")
}

// TestDecoderGoldenRealWorld decodes the 3 representative real-world
// clips from their committed tokens (the exact tensors
// roundtrip_decode.py fed to whisper) and compares against the HF
// reference decodes under the same two-part gate.
func TestDecoderGoldenRealWorld(t *testing.T) {
	q, d := loadFullCached(t)
	dsf := loadDecoderFixtures(t)
	tok, err := model.LoadSafetensors("../testdata/realworld/tokens.safetensors")
	if err != nil {
		t.Fatalf("load realworld tokens: %v", err)
	}
	for _, clip := range []string{"zero_alloy", "five_echo", "nine_shimmer"} {
		codes := fixtureCodes(t, tok, clip)
		refName := "rw_" + clip + "_dec_wav"
		ref := fixture(t, dsf, refName)
		runGoldenChecks(t, func() []goldenCheck {
			wav := d.Decode(q, codes)
			if want := 1920 * len(codes[0]); len(wav) != want || len(ref.Data()) != want {
				t.Fatalf("%s: decode %d samples, fixture %d, want 1920*T = %d",
					refName, len(wav), len(ref.Data()), want)
			}
			return []goldenCheck{&wavCheck{stage: refName, got: wav, ref: ref.Data(),
				minSNR: wavMinSNR, absTol: wavAbsTol, relTol: wavRelTol}}
		})
	}
}

// TestDecoderLengthSemantics runs without the checkpoint: random
// weights, random latents, verify the exact len == 1920·T property
// (plan 0007 §0.2.5).
func TestDecoderLengthSemantics(t *testing.T) {
	d := NewDecoder(DefaultConfig())
	rng := rand.New(rand.NewSource(17))
	for _, T := range []int{1, 25, 150} {
		data := make([]float32, T*512)
		for i := range data {
			data[i] = float32(0.5 * rng.NormFloat64())
		}
		out := d.DecodeLatent(g.NewTensor(data, T, 512))
		if len(out) != 1920*T {
			t.Errorf("T=%d: %d samples, want %d", T, len(out), 1920*T)
		}
	}
}

// TestDecoderRandomLatentSmoke: a random latent through the real
// checkpoint must produce finite, bounded output (real latents give
// |wav| <= 1-ish; an unnormalized blowup or NaN propagation shows up
// immediately).
func TestDecoderRandomLatentSmoke(t *testing.T) {
	_, d := loadFullCached(t)
	rng := rand.New(rand.NewSource(23))
	data := make([]float32, 25*512)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	out := d.DecodeLatent(g.NewTensor(data, 25, 512))
	if len(out) != 1920*25 {
		t.Fatalf("%d samples, want %d", len(out), 1920*25)
	}
	var maxAbs float64
	for i, v := range out {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("sample %d is %v", i, v)
		}
		if a := math.Abs(f); a > maxAbs {
			maxAbs = a
		}
	}
	// A random N(0,1) latent is far outside the quantizer's output
	// manifold; the decoded "waveform" still stays O(1)-bounded.
	if maxAbs > 100 {
		t.Fatalf("max |sample| = %.3g, want <= 100", maxAbs)
	}
	t.Logf("random-latent decode: max |sample| = %.3g", maxAbs)
}

// TestLoadDecoderRejectsBadCheckpoint verifies the decoder loader
// fails loudly on missing, misshapen and unexpected keys across all
// three decoder-side families. Runs without the real checkpoint.
func TestLoadDecoderRejectsBadCheckpoint(t *testing.T) {
	d := NewDecoder(DefaultConfig())
	tensors := map[string]*g.Tensor{}
	names := []string{}
	add := func(key string, shape ...int) {
		tensors[key] = g.Zeros(shape...)
		names = append(names, key)
	}
	addConv := func(prefix string, conv *nn.CausalConv1d) {
		add(prefix+".weight", conv.Weight.Shape()...)
		add(prefix+".bias", conv.Bias.Shape()...)
	}

	add("upsample.conv.weight", 512, 1, 4)
	for i := 0; i < 8; i++ {
		p := fmt.Sprintf("decoder_transformer.layers.%d.", i)
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
	addConv("decoder.layers.0.conv", d.SEANet.Init)
	for s := range d.SEANet.Ups {
		up := d.SEANet.Ups[s]
		add(fmt.Sprintf("decoder.layers.%d.conv.weight", 2+3*s), up.Weight.Shape()...)
		add(fmt.Sprintf("decoder.layers.%d.conv.bias", 2+3*s), up.Bias.Shape()...)
		blk := fmt.Sprintf("decoder.layers.%d.block.", 3+3*s)
		addConv(blk+"1.conv", d.SEANet.Res[s][0])
		addConv(blk+"3.conv", d.SEANet.Res[s][1])
	}
	addConv("decoder.layers.14.conv", d.SEANet.Final)

	// Complete synthetic checkpoint loads fine (125 keys, matching
	// audio/testdata/mimi_decoder_keys.txt).
	if len(names) != 125 {
		t.Fatalf("synthetic checkpoint has %d keys, manifest has 125", len(names))
	}
	if _, err := loadDecoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err != nil {
		t.Fatalf("complete synthetic checkpoint rejected: %v", err)
	}

	// Missing SEANet ConvTranspose bias.
	saved := tensors["decoder.layers.5.conv.bias"]
	delete(tensors, "decoder.layers.5.conv.bias")
	if _, err := loadDecoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a missing ConvTranspose bias")
	}
	tensors["decoder.layers.5.conv.bias"] = saved

	// Missing decoder-transformer key.
	saved = tensors["decoder_transformer.layers.6.mlp.fc2.weight"]
	delete(tensors, "decoder_transformer.layers.6.mlp.fc2.weight")
	if _, err := loadDecoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a missing decoder-transformer key")
	}
	tensors["decoder_transformer.layers.6.mlp.fc2.weight"] = saved

	// Transposed (Conv1d-layout) ConvTranspose weight — the highest-risk
	// mistake (plan 0007 §8 risk 1) must be caught by shape validation.
	saved = tensors["decoder.layers.2.conv.weight"]
	tensors["decoder.layers.2.conv.weight"] = g.Zeros(512, 1024, 16)
	if _, err := loadDecoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted a transposed ConvTranspose weight layout")
	}
	tensors["decoder.layers.2.conv.weight"] = saved

	// Unexpected decoder-side key.
	add("decoder_transformer.norm.weight", 512)
	if _, err := loadDecoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with an unexpected decoder-transformer key")
	}
	delete(tensors, "decoder_transformer.norm.weight")
	names = names[:len(names)-1]

	// Misshapen upsample weight.
	tensors["upsample.conv.weight"] = g.Zeros(512, 1, 3)
	if _, err := loadDecoderFrom(&model.SafetensorsFile{Tensors: tensors, Names: names}); err == nil {
		t.Fatal("loader accepted checkpoint with a misshapen upsample weight")
	}
}

// TestLoadFullRejectsStrayKeys: LoadFull must reject checkpoints with
// keys outside every loader's families (the coverage assertion runs
// before the sub-loaders, so no full synthetic checkpoint is needed).
func TestLoadFullRejectsStrayKeys(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/stray.safetensors"
	if err := model.SaveSafetensors(path, map[string]*g.Tensor{"lm_head.weight": g.Zeros(2, 2)}); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	if _, _, _, err := LoadFull(path); err == nil {
		t.Fatal("LoadFull accepted a checkpoint with a stray key family")
	}
}

// BenchmarkMimiDecode10s measures the full offline decode of 125
// tokens (10 s of audio → 240000 samples): the chirp fixture's 8-level
// codes tiled 5× for a deterministic input. Python baseline (plan 0007
// §7, D0 2026-08-07, transformers 4.57.1 / torch 2.9.1, CPU, M4):
// 466.9 ms best-of-3 for 150 tokens ≈ 389 ms per 125 tokens; target
// >=5× faster (<93 ms).
//
// Measured (D2, 2026-08-07, M4): 285.6 ms min-of-3 — 1.36× the
// like-for-like Python baseline, NOT the 5× target. Profiling shows
// the decode is GEMM-bound (~137 ms inside Accelerate Sgemm per
// decode; SEANet 204 ms — resnet convs 90 ms + ConvT 83 ms — and
// transformer 111 ms), matching the offline encoder's precedent
// (BenchmarkMimiEncode10s: 333 ms vs Python 334 ms on this machine).
// Reaching 5× needs op-level re-engineering (threaded im2col/col2im,
// fused elementwise, or Metal), out of D2 scope — see plan §8 risk 8.
func BenchmarkMimiDecode10s(b *testing.B) {
	q, d := loadFullCached(b)
	sf := loadFixtures(b)
	base := fixtureCodes(b, sf, "chirp_codes8")
	codes := make([][]int, len(base))
	for k := range base {
		for r := 0; r < 5; r++ {
			codes[k] = append(codes[k], base[k]...)
		}
	}
	if len(codes[0]) != 125 {
		b.Fatalf("tiled codes have %d frames, want 125", len(codes[0]))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Decode(q, codes)
	}
}
