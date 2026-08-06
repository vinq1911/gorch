//go:build darwin

package mimi

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// ChunkSamples is the streaming chunk size: 80 ms at 24 kHz. One chunk
// is exactly 2 SEANet frames (stride product 960) → 2 transformer
// tokens → 1 downsampled latent frame at 12.5 Hz, and because 1920 is
// a multiple of every cumulative stride, no conv ever needs extra
// right padding during streaming (plan §0.2 "Streaming chunk math").
const ChunkSamples = 1920

// Stream is the incremental Mimi encoder (plan §4.5). It carries the
// per-conv left-context caches for the SEANet stack and the downsample
// conv, one WindowKV per transformer layer, and the absolute 25 Hz
// frame position used for RoPE. Feed it consecutive 80 ms chunks of a
// single audio session; outputs concatenate to EncodeWindowed's result
// on the same PCM (and to plain Encode for sessions under 250 frames /
// 10 s, where the sliding window never binds).
//
// A Stream is not safe for concurrent use.
type Stream struct {
	enc          *Encoder
	seanetStates []*nn.Conv1dStream
	dsState      *nn.Conv1dStream
	kv           [8]*WindowKV
	pos          int // absolute 25 Hz frame position (RoPE index)
	frames       int // transformer frames per chunk (2)
}

// NewStream creates a fresh streaming session over the encoder's
// weights. Streams share the (frozen) encoder; each carries only its
// own state, so multiple sessions can run over one Encoder — though a
// single Stream must not be used concurrently.
func (e *Encoder) NewStream() *Stream {
	stride := 1
	for _, r := range e.Cfg.UpsamplingRatios {
		stride *= r
	}
	s := &Stream{
		enc:          e,
		seanetStates: e.SEANet.NewStreamStates(),
		dsState:      &nn.Conv1dStream{},
		frames:       ChunkSamples / stride,
	}
	for i := range s.kv {
		s.kv[i] = NewWindowKV(e.Cfg.NumHeads, e.Cfg.HeadDim, e.Cfg.SlidingWindow)
	}
	return s
}

// Pos returns the absolute 25 Hz frame position of the next chunk.
func (s *Stream) Pos() int { return s.pos }

// Push encodes one 80 ms chunk (exactly ChunkSamples = 1920 samples of
// 24 kHz mono PCM) and returns the resulting (1, 512) latent frame at
// 12.5 Hz. Panics if the chunk size is wrong, or once the session
// reaches MaxPositions (8000) transformer frames = 320 s — the same
// hard limit as HF's max_position_embeddings; start a new session (or
// call Reset) to continue past it.
func (s *Stream) Push(chunk []float32) *g.Tensor {
	if len(chunk) != ChunkSamples {
		panic(fmt.Sprintf("mimi: Stream.Push needs exactly %d samples (80 ms at 24 kHz), got %d", ChunkSamples, len(chunk)))
	}
	if s.pos+s.frames > s.enc.Cfg.MaxPositions {
		panic(fmt.Sprintf("mimi: Stream session exceeds %d frames (320 s); Reset or start a new Stream", s.enc.Cfg.MaxPositions))
	}
	var out *g.Tensor
	g.NoGrad(func() {
		x := s.enc.SEANet.ForwardStream(g.NewTensor(chunk, 1, 1, len(chunk)), s.seanetStates) // (1, 512, 2)
		h := transposeCT(x)                                                                   // (2, 512)
		for i, l := range s.enc.Layers {
			h = l.ForwardCached(h, s.enc.Rope, s.kv[i], s.pos)
		}
		out = transposeCT(s.enc.Downsample.ForwardStream(transposeTC(h), s.dsState)) // (1, 512)
	})
	s.pos += s.frames
	return out
}

// Reset clears all streaming state (conv left contexts, KV caches,
// position) so the Stream can encode a new session; results after
// Reset are identical to those of a brand-new Stream.
func (s *Stream) Reset() {
	for _, st := range s.seanetStates {
		st.Reset()
	}
	s.dsState.Reset()
	for _, c := range s.kv {
		c.Reset()
	}
	s.pos = 0
}
