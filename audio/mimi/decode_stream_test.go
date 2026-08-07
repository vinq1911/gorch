//go:build darwin

package mimi

import (
	"math/rand"
	"testing"

	g "github.com/vinq1911/gorch"
)

// pushAll feeds every token column of codes ((numQuantizers, T)) through
// a fresh DecodeStream and returns the concatenated waveform (1920·T
// samples).
func pushAll(t testing.TB, q *Quantizer, d *Decoder, codes [][]int) []float32 {
	t.Helper()
	s := d.NewStream(q)
	T := len(codes[0])
	out := make([]float32, 0, T*ChunkSamples)
	col := make([]int, len(codes))
	for f := 0; f < T; f++ {
		for level := range codes {
			col[level] = codes[level][f]
		}
		chunk := s.Push(col)
		if len(chunk) != ChunkSamples {
			t.Fatalf("Push returned %d samples, want %d", len(chunk), ChunkSamples)
		}
		out = append(out, chunk...)
	}
	return out
}

// TestDecodeStreamMatchesOfflineChirp: 25 tokens (50 transformer frames
// < window 250) — concatenated Push output must match both
// DecodeLatentWindowed and plain DecodeLatent (identical below the
// window) within 1e-5, using the requireClose min-over-4-attempts
// busy-machine discipline from stream_test.go.
func TestDecodeStreamMatchesOfflineChirp(t *testing.T) {
	q, d := loadFullCached(t)
	sf := loadFixtures(t)
	codes := fixtureCodes(t, sf, "chirp_codes8")
	latent := q.Decode(codes)

	requireClose(t, "chirp decode stream vs DecodeLatentWindowed", 1e-5, func() ([]float32, []float32) {
		return pushAll(t, q, d, codes), d.DecodeLatentWindowed(latent)
	})
	requireClose(t, "chirp decode stream vs DecodeLatent", 1e-5, func() ([]float32, []float32) {
		return pushAll(t, q, d, codes), d.DecodeLatent(latent)
	})
}

// TestDecodeStreamMatchesOfflineLong: 150 tokens (300 transformer
// frames > window 250) — the WindowKV-eviction binding test. The
// reference is DecodeLatentWindowed (plain DecodeLatent diverges past
// 250 frames, the same caveat as the encoder).
func TestDecodeStreamMatchesOfflineLong(t *testing.T) {
	q, d := loadFullCached(t)
	sf := loadFixtures(t)
	codes := fixtureCodes(t, sf, "long_codes8")
	latent := q.Decode(codes)

	// Tolerance 1e-4, matching the encoder's long streaming test: the
	// decoder has no LayerNorm after the transformer, so Accelerate's
	// load-dependent f32 reduction ordering between the chunked and
	// full-sequence GEMMs is amplified through ~19 conv layers (waveform
	// scale is [-1, 1], so 1e-4 is a -80 dB-class discrepancy). Real
	// streaming-state bugs (bias double-count, tail off-by-one, KV
	// eviction errors) produce >=1e-2 errors and fail by orders of
	// magnitude.
	requireClose(t, "long decode stream vs DecodeLatentWindowed", 1e-4, func() ([]float32, []float32) {
		return pushAll(t, q, d, codes), d.DecodeLatentWindowed(latent)
	})
}

// TestDecodeStreamGoldenLongWindowed compares the concatenated Push
// output for the 150-token long signal directly against the HF
// long_dec_wav_win fixture under the D2-calibrated two-part waveform
// gate (SNR >= 60 dB + mixed sample tolerance) — the streamed decoder
// vs the external reference, not just Go-vs-Go.
func TestDecodeStreamGoldenLongWindowed(t *testing.T) {
	q, d := loadFullCached(t)
	sf := loadFixtures(t)
	dsf := loadDecoderFixtures(t)
	codes := fixtureCodes(t, sf, "long_codes8")
	ref := fixture(t, dsf, "long_dec_wav_win")

	runGoldenChecks(t, func() []goldenCheck {
		wav := pushAll(t, q, d, codes)
		if len(wav) != len(ref.Data()) {
			t.Fatalf("streamed %d samples, fixture has %d", len(wav), len(ref.Data()))
		}
		return []goldenCheck{&wavCheck{stage: "stream long_dec_wav_win", got: wav, ref: ref.Data(),
			minSNR: wavMinSNR, absTol: wavAbsTol, relTol: wavRelTol}}
	})
}

// TestDecodeStreamPushLatentMatchesPush: PushLatent on the same
// column's Quantizer.Decode output must be bit-identical to Push
// (Push is defined as PushLatent ∘ Decode; this pins the equivalence
// through the public API on real weights).
func TestDecodeStreamPushLatentMatchesPush(t *testing.T) {
	q, d := loadFullCached(t)
	sf := loadFixtures(t)
	codes := fixtureCodes(t, sf, "chirp_codes8")

	sc := d.NewStream(q)
	sl := d.NewStream(nil) // PushLatent needs no quantizer
	col := make([]int, len(codes))
	for f := 0; f < 5; f++ {
		for level := range codes {
			col[level] = codes[level][f]
		}
		got := sc.Push(col)
		cols := make([][]int, len(col))
		for i, c := range col {
			cols[i] = []int{c}
		}
		want := sl.PushLatent(q.Decode(cols))
		if d := maxAbsErr(got, want); d != 0 {
			t.Fatalf("frame %d: Push vs PushLatent max abs diff %g, want bit-exact", f, d)
		}
	}
}

// TestDecodeStreamResetReuse: a Reset DecodeStream must reproduce a
// fresh session bit-for-bit. Runs on random weights via PushLatent (no
// checkpoint or quantizer needed), mirroring TestStreamResetReuse.
func TestDecodeStreamResetReuse(t *testing.T) {
	d := NewDecoder(DefaultConfig())
	rng := rand.New(rand.NewSource(11))
	latents := make([]*g.Tensor, 3)
	for i := range latents {
		data := make([]float32, 512)
		for j := range data {
			data[j] = float32(0.5 * rng.NormFloat64())
		}
		latents[i] = g.NewTensor(data, 1, 512)
	}

	s := d.NewStream(nil)
	run := func() []float32 {
		var out []float32
		for _, l := range latents {
			out = append(out, s.PushLatent(l)...)
		}
		return out
	}
	first := run()
	if s.Pos() != 6 {
		t.Fatalf("Pos() = %d after 3 tokens, want 6", s.Pos())
	}
	s.Reset()
	if s.Pos() != 0 {
		t.Fatalf("Pos() = %d after Reset, want 0", s.Pos())
	}
	requireClose(t, "decode stream reset reuse", 0, func() ([]float32, []float32) {
		s.Reset()
		return run(), first
	})
}

// TestDecodeStreamPushValidation: Push must reject a missing quantizer,
// level counts outside [1, 32], and out-of-range codes; PushLatent must
// reject non-(1, 512) latents.
func TestDecodeStreamPushValidation(t *testing.T) {
	d := NewDecoder(DefaultConfig())

	// No quantizer: Push is unusable, PushLatent still works.
	s := d.NewStream(nil)
	expectPanic(t, "requires a Quantizer", func() { s.Push([]int{1, 2, 3}) })

	expectPanic(t, "one token frame per push", func() { s.PushLatent(g.Zeros(2, 512)) })
	expectPanic(t, "one token frame per push", func() { s.PushLatent(g.Zeros(1, 256)) })

	// With a quantizer (random-weight decoder still fine: validation
	// happens before any math).
	q, _ := loadFullCached(t)
	s = d.NewStream(q)
	expectPanic(t, "code levels", func() { s.Push(nil) })
	expectPanic(t, "code levels", func() { s.Push(make([]int, 33)) })
	expectPanic(t, "out of range", func() { s.Push([]int{2048}) })
}

// TestDecodeStreamPositionCap: a session is capped at MaxPositions
// (8000) transformer frames = 320 s, mirroring the encoder Stream; the
// last in-range token succeeds, the next panics, and Reset clears the
// cap.
func TestDecodeStreamPositionCap(t *testing.T) {
	d := NewDecoder(DefaultConfig())
	s := d.NewStream(nil)
	latent := g.Zeros(1, 512)
	s.pos = d.Cfg.MaxPositions - 2
	s.PushLatent(latent) // frames 7998-7999: still in range
	if s.Pos() != d.Cfg.MaxPositions {
		t.Fatalf("Pos() = %d, want %d", s.Pos(), d.Cfg.MaxPositions)
	}
	expectPanic(t, "320 s", func() { s.PushLatent(latent) })
	s.Reset()
	s.PushLatent(latent) // fresh session works again
}

// BenchmarkMimiDecodeStreamChunk measures steady-state per-Push
// latency: prime a session with 50 tokens so every conv tail, left
// context and KV buffer is warm, then time Push. Target <10 ms per
// 80 ms chunk (Python KV-only per-token baseline: 33.95 ms mean, plan
// 0007 §7).
func BenchmarkMimiDecodeStreamChunk(b *testing.B) {
	q, d := loadFullCached(b)
	sf := loadFixtures(b)
	codes := fixtureCodes(b, sf, "chirp_codes8")
	col := make([]int, len(codes))
	for level := range codes {
		col[level] = codes[level][0]
	}
	s := d.NewStream(q)
	prime := func() {
		for i := 0; i < 50; i++ {
			s.Push(col)
		}
	}
	prime()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if s.Pos()+2 > d.Cfg.MaxPositions {
			b.StopTimer()
			s.Reset()
			prime()
			b.StartTimer()
		}
		s.Push(col)
	}
}
