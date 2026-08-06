//go:build darwin

package mimi

import (
	"math/rand"
	"strings"
	"testing"

	g "github.com/vinq1911/gorch"
)

// streamAll pushes pcm through a fresh Stream in 80 ms chunks and
// returns the concatenated latent frames as a flat (T, 512) buffer.
// pcm length must be a multiple of ChunkSamples.
func streamAll(t testing.TB, e *Encoder, pcm []float32) []float32 {
	t.Helper()
	if len(pcm)%ChunkSamples != 0 {
		t.Fatalf("streamAll: pcm length %d not a multiple of %d", len(pcm), ChunkSamples)
	}
	s := e.NewStream()
	out := make([]float32, 0, len(pcm)/ChunkSamples*e.Cfg.HiddenSize)
	for off := 0; off < len(pcm); off += ChunkSamples {
		fr := s.Push(pcm[off : off+ChunkSamples])
		if !shapeEq(fr.Shape(), []int{1, e.Cfg.HiddenSize}) {
			t.Fatalf("Push returned shape %v, want [1 %d]", fr.Shape(), e.Cfg.HiddenSize)
		}
		out = append(out, fr.Data()...)
	}
	return out
}

// requireClose checks max |got-ref| <= tol with the recompute-once
// precedent from encoder_test.go's runGolden: under heavy parallel
// load Accelerate's threaded GEMM can split work differently, so a
// transient over-tolerance blip must reproduce on an immediate
// recompute before it counts as a failure.
func requireClose(t *testing.T, name string, tol float64, compute func() (got, ref []float32)) float64 {
	t.Helper()
	got, ref := compute()
	if len(got) != len(ref) {
		t.Fatalf("%s: length mismatch %d vs %d", name, len(got), len(ref))
	}
	d := maxAbsErr(got, ref)
	if d <= tol {
		t.Logf("%s: max abs diff %.3g (tol %.0e)", name, d, tol)
		return d
	}
	t.Logf("%s: max abs diff %.3g > %.0e — recomputing once to rule out load-induced BLAS nondeterminism", name, d, tol)
	got, ref = compute()
	d = maxAbsErr(got, ref)
	t.Logf("%s: recomputed max abs diff %.3g (tol %.0e)", name, d, tol)
	if d > tol {
		t.Errorf("%s: max abs diff %.3g > %.0e", name, d, tol)
	}
	return d
}

// transposeRef flattens a (C, T) fixture tensor to frame-major (T, C)
// for comparison with concatenated Push outputs.
func transposeRef(ref *g.Tensor) []float32 {
	C, T := ref.Shape()[0], ref.Shape()[1]
	out := make([]float32, C*T)
	data := ref.Data()
	for c := 0; c < C; c++ {
		for tt := 0; tt < T; tt++ {
			out[tt*C+c] = data[c*T+tt]
		}
	}
	return out
}

// TestStreamMatchesOfflineChirp: 2 s chirp (48000 samples = 25 chunks,
// 50 transformer frames < window 250) — streamed output must match
// both EncodeWindowed(250) and plain Encode (identical below the
// window) within 1e-5, and HF's own streaming fixture within 1e-3.
func TestStreamMatchesOfflineChirp(t *testing.T) {
	e := loadEncoderCached(t)
	sf := loadFixtures(t)
	pcm := fixture(t, sf, "chirp_pcm").Data()

	requireClose(t, "chirp stream vs EncodeWindowed", 1e-5, func() ([]float32, []float32) {
		return streamAll(t, e, pcm), e.EncodeWindowed(pcm).Data()
	})
	requireClose(t, "chirp stream vs Encode", 1e-5, func() ([]float32, []float32) {
		return streamAll(t, e, pcm), e.Encode(pcm).Data()
	})
	// Cross-check vs the HF streaming reference (window never binds at
	// 50 frames, so HF's 251-effective-key eviction is equivalent).
	requireClose(t, "chirp stream vs HF chirp_stream_latent", 1e-3, func() ([]float32, []float32) {
		return streamAll(t, e, pcm), transposeRef(fixture(t, sf, "chirp_stream_latent"))
	})
}

// TestStreamMatchesOfflineLong: 12 s (288000 samples = 150 chunks, 300
// transformer frames > window 250) — validates WindowKV eviction + the
// position mask against the masked offline EncodeWindowed path. NOTE:
// the HF long_stream_latent fixture is NOT a bit-equality target past
// latent frame 125 (HF streaming keeps 249 past keys with no window
// mask = 251 effective keys vs our strict 250); the Go-vs-Go
// comparison here is the correctness claim.
func TestStreamMatchesOfflineLong(t *testing.T) {
	e := loadEncoderCached(t)
	sf := loadFixtures(t)
	pcm := fixture(t, sf, "long_pcm").Data()

	requireClose(t, "long stream vs EncodeWindowed", 1e-5, func() ([]float32, []float32) {
		return streamAll(t, e, pcm), e.EncodeWindowed(pcm).Data()
	})
}

// TestStreamResetReuse: a Reset Stream must reproduce a fresh session
// bit-for-bit. Runs on random weights (no checkpoint needed).
func TestStreamResetReuse(t *testing.T) {
	e := NewEncoder(DefaultConfig())
	rng := rand.New(rand.NewSource(11))
	pcm := make([]float32, 3*ChunkSamples)
	for i := range pcm {
		pcm[i] = float32(0.5 * rng.NormFloat64())
	}

	s := e.NewStream()
	run := func() []float32 {
		var out []float32
		for off := 0; off < len(pcm); off += ChunkSamples {
			out = append(out, s.Push(pcm[off:off+ChunkSamples]).Data()...)
		}
		return out
	}
	first := run()
	if s.Pos() != 6 {
		t.Fatalf("Pos() = %d after 3 chunks, want 6", s.Pos())
	}
	s.Reset()
	if s.Pos() != 0 {
		t.Fatalf("Pos() = %d after Reset, want 0", s.Pos())
	}
	requireClose(t, "reset reuse", 0, func() ([]float32, []float32) {
		s.Reset()
		return run(), first
	})
}

func expectPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic containing %q, got none", want)
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, want) {
			t.Fatalf("panic %v, want message containing %q", r, want)
		}
	}()
	f()
}

// TestStreamPushWrongSize: Push must reject any chunk size other than
// exactly 1920 samples.
func TestStreamPushWrongSize(t *testing.T) {
	e := NewEncoder(DefaultConfig())
	s := e.NewStream()
	for _, n := range []int{0, 960, 1919, 1921, 3840} {
		expectPanic(t, "exactly 1920 samples", func() { s.Push(make([]float32, n)) })
	}
}

// TestStreamPositionCap: a session is capped at MaxPositions (8000)
// transformer frames = 320 s, mirroring HF's max_position_embeddings;
// the last in-range chunk succeeds, the next panics, and Reset clears
// the cap.
func TestStreamPositionCap(t *testing.T) {
	e := NewEncoder(DefaultConfig())
	s := e.NewStream()
	chunk := make([]float32, ChunkSamples)
	s.pos = e.Cfg.MaxPositions - 2
	s.Push(chunk) // frames 7998-7999: still in range
	if s.Pos() != e.Cfg.MaxPositions {
		t.Fatalf("Pos() = %d, want %d", s.Pos(), e.Cfg.MaxPositions)
	}
	expectPanic(t, "320 s", func() { s.Push(chunk) })
	s.Reset()
	s.Push(chunk) // fresh session works again
}

// TestWindowKVEviction unit-tests the ring semantics with a tiny
// window: retention of window-1 past rows and absolute positions.
func TestWindowKVEviction(t *testing.T) {
	const nH, hD, window = 2, 4, 3
	c := NewWindowKV(nH, hD, window)
	mk := func(base float32, S int) *g.Tensor {
		data := make([]float32, nH*S*hD)
		for i := range data {
			data[i] = base + float32(i)
		}
		return g.NewTensor(data, nH, S, hD)
	}
	c.Append(mk(0, 2), mk(100, 2), 0) // rows 0,1
	if c.Len() != 2 {
		t.Fatalf("Len = %d, want 2", c.Len())
	}
	c.Append(mk(200, 2), mk(300, 2), 2) // keep window-1=2 past + 2 new
	if c.Len() != 4 {
		t.Fatalf("Len = %d, want 4", c.Len())
	}
	c.Append(mk(400, 2), mk(500, 2), 4) // evict down to 2 past + 2 new
	if c.Len() != 4 {
		t.Fatalf("Len = %d, want 4", c.Len())
	}
	wantPos := []int{2, 3, 4, 5}
	for i, p := range c.positions {
		if p != wantPos[i] {
			t.Fatalf("positions = %v, want %v", c.positions, wantPos)
		}
	}
	// Head 0 of k: rows for positions 2,3 came from the (200-base)
	// append, rows 4,5 from the (400-base) append.
	want := []float32{200, 201, 202, 203, 204, 205, 206, 207, 400, 401, 402, 403, 404, 405, 406, 407}
	kData := c.kT.Data()
	for i, w := range want {
		if kData[i] != w {
			t.Fatalf("k[%d] = %v, want %v (head 0 block %v)", i, kData[i], w, kData[:16])
		}
	}
}

// BenchmarkMimiStreamChunk measures steady-state per-chunk latency:
// prime a session with 50 chunks so every conv cache and KV buffer is
// warm, then time Push. Target <10 ms per 80 ms chunk (Python
// baseline: 43 ms).
func BenchmarkMimiStreamChunk(b *testing.B) {
	e := loadEncoderCached(b)
	rng := rand.New(rand.NewSource(3))
	chunk := make([]float32, ChunkSamples)
	for i := range chunk {
		chunk[i] = float32(0.5 * rng.NormFloat64())
	}
	s := e.NewStream()
	prime := func() {
		for i := 0; i < 50; i++ {
			s.Push(chunk)
		}
	}
	prime()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if s.Pos()+2 > e.Cfg.MaxPositions {
			b.StopTimer()
			s.Reset()
			prime()
			b.StartTimer()
		}
		s.Push(chunk)
	}
}
