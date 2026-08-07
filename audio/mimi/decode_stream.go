//go:build darwin

package mimi

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// DecodeStream is the incremental Mimi decoder (plan 0007 §4): one
// token column in, 80 ms of audio out. It mirrors the encoder Stream
// structurally — per-conv streaming caches, one WindowKV per
// transformer layer, and the absolute 25 Hz frame position for RoPE —
// with the conv caches split by kind: an overlap-add tail
// (ConvT1dStream) per transposed conv and a left-context cache
// (Conv1dStream) per plain causal conv. Feed it consecutive token
// columns of a single audio session; outputs concatenate to
// DecodeLatentWindowed's result on the same codes (and to plain
// DecodeLatent for sessions under 125 tokens / 10 s, where the sliding
// window never binds).
//
// A DecodeStream is not safe for concurrent use.
type DecodeStream struct {
	dec *Decoder
	q   *Quantizer // may be nil: PushLatent-only sessions

	upState *nn.ConvT1dStream // 12.5→25 Hz depthwise upsample (tail 2)
	kv      [8]*WindowKV
	pos     int // absolute 25 Hz frame position (RoPE index)

	// SEANet decoder streaming state, in layer order.
	initState *nn.Conv1dStream       // decoder.layers.0 (left ctx 6)
	upStates  [4]*nn.ConvT1dStream   // ConvT tails 8/6/5/4
	resStates [4][2]*nn.Conv1dStream // resnet convs (left ctx 2, 0)
	finState  *nn.Conv1dStream       // decoder.layers.14 (left ctx 2)

	// Scratch for the per-push single-column code matrix, so Push
	// allocates no per-call [][]int (plan risk 9).
	cols [][]int
}

// NewStream creates a fresh streaming session over the decoder's
// weights. q supplies the codes→latent stage for Push; it may be nil
// when only PushLatent will be used. Streams share the (frozen)
// decoder; each carries only its own state, so multiple sessions can
// run over one Decoder — though a single DecodeStream must not be used
// concurrently.
func (d *Decoder) NewStream(q *Quantizer) *DecodeStream {
	s := &DecodeStream{
		dec:       d,
		q:         q,
		upState:   &nn.ConvT1dStream{},
		initState: &nn.Conv1dStream{},
		finState:  &nn.Conv1dStream{},
	}
	for i := range s.kv {
		s.kv[i] = NewWindowKV(d.Cfg.NumHeads, d.Cfg.HeadDim, d.Cfg.SlidingWindow)
	}
	for i := range s.upStates {
		s.upStates[i] = &nn.ConvT1dStream{}
		s.resStates[i][0] = &nn.Conv1dStream{}
		s.resStates[i][1] = &nn.Conv1dStream{}
	}
	return s
}

// Pos returns the absolute 25 Hz frame position of the next token.
func (s *DecodeStream) Pos() int { return s.pos }

// Push decodes one token column — codes[i] is the level-i codebook
// index of a single 12.5 Hz frame, as produced by Quantizer.Encode
// (Moshi emits 8 levels) — and returns exactly ChunkSamples = 1920
// samples (80 ms of 24 kHz mono PCM). Panics if the stream was created
// without a Quantizer, if the level count is outside [1, 32], if any
// code is out of range, or once the session reaches MaxPositions
// (8000) transformer frames = 320 s (same cap as the encoder Stream);
// Reset or start a new DecodeStream to continue past it.
func (s *DecodeStream) Push(codes []int) []float32 {
	if s.q == nil {
		panic("mimi: DecodeStream.Push requires a Quantizer; create the stream with NewStream(q) or use PushLatent")
	}
	if len(codes) < 1 || len(codes) > s.q.NumCodebooks() {
		panic(fmt.Sprintf("mimi: DecodeStream.Push got %d code levels, want 1..%d (one code per level of a single token)",
			len(codes), s.q.NumCodebooks()))
	}
	if s.cols == nil {
		s.cols = make([][]int, 0, s.q.NumCodebooks())
	}
	s.cols = s.cols[:0]
	for _, c := range codes {
		s.cols = append(s.cols, []int{c})
	}
	return s.PushLatent(s.q.Decode(s.cols))
}

// PushLatent is the codes-free variant of Push: it decodes one (1, 512)
// quantized-latent frame (a single row of Quantizer.Decode output) into
// exactly 1920 samples. Push(codes) is PushLatent(q.Decode(column)).
func (s *DecodeStream) PushLatent(latent *g.Tensor) []float32 {
	shape := latent.Shape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] != s.dec.Cfg.HiddenSize {
		panic(fmt.Sprintf("mimi: DecodeStream.PushLatent latent shape %v, want (1, %d) — one token frame per push",
			shape, s.dec.Cfg.HiddenSize))
	}
	if s.pos+2 > s.dec.Cfg.MaxPositions {
		panic(fmt.Sprintf("mimi: DecodeStream session exceeds %d frames (320 s); Reset or start a new DecodeStream",
			s.dec.Cfg.MaxPositions))
	}
	var out []float32
	g.NoGrad(func() {
		x := transposeTC(latent)                       // (1, 512, 1)
		x = s.dec.Upsample.ForwardStream(x, s.upState) // (1, 512, 2) — 25 Hz
		h := transposeCT(x)                            // (2, 512)
		for i, l := range s.dec.Layers {
			h = l.ForwardCached(h, s.dec.Rope, s.kv[i], s.pos)
		}
		out = s.seanetStream(transposeTC(h)).Data() // (1, 1, 1920)
	})
	s.pos += 2
	return out
}

// seanetStream runs one (1, 512, 2) chunk through the SEANet decoder
// stack with the stream's caches, mirroring SEANetDecoder.Forward
// op-for-op: Init → per stage [ELU → ConvT → resnet] → ELU → Final.
func (s *DecodeStream) seanetStream(x *g.Tensor) *g.Tensor {
	d := s.dec.SEANet
	x = d.Init.ForwardStream(x, s.initState)
	for i := range d.Ups {
		x = g.ELU(x)
		x = d.Ups[i].ForwardStream(x, s.upStates[i])
		h := g.ELU(x)
		h = d.Res[i][0].ForwardStream(h, s.resStates[i][0])
		h = g.ELU(h)
		h = d.Res[i][1].ForwardStream(h, s.resStates[i][1])
		x = g.Add(x, h)
	}
	x = g.ELU(x)
	return d.Final.ForwardStream(x, s.finState)
}

// Reset clears all streaming state (upsample and SEANet conv caches,
// KV caches, position) so the DecodeStream can decode a new session;
// results after Reset are identical to those of a brand-new stream.
func (s *DecodeStream) Reset() {
	s.upState.Reset()
	for _, c := range s.kv {
		c.Reset()
	}
	s.initState.Reset()
	for i := range s.upStates {
		s.upStates[i].Reset()
		s.resStates[i][0].Reset()
		s.resStates[i][1].Reset()
	}
	s.finState.Reset()
	s.pos = 0
}
