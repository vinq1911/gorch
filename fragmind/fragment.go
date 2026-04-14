//go:build darwin

// Package fragmind implements pipeline-parallel inference by splitting
// transformer blocks across multiple processes or machines.
//
// A "fragment" is a contiguous slice of transformer blocks. Fragments
// communicate by passing activation tensors over the network (TCP).
// This enables running models too large for one machine, or distributing
// compute for lower latency.
//
// Architecture:
//
//	Machine 1 (Fragment 0): Embedding + Blocks[0:6] → send activations
//	Machine 2 (Fragment 1): Blocks[6:12] + LM Head  → return logits
package fragmind

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// Fragment represents a slice of a GPT model that runs on one machine.
type Fragment struct {
	ID         int
	Blocks     []*nn.TransformerBlock
	HasEmbed   bool       // true if this fragment handles embedding
	HasLMHead  bool       // true if this fragment handles final norm + LM head
	TokenEmbed *nn.Embedding
	PosEmbed   *nn.Embedding
	FinalNorm  *nn.LayerNorm
	LMHead     *nn.Linear
	VocabSize  int
	Dim        int
}

// SplitGPT splits a GPT model into N fragments for pipeline parallelism.
// Fragment 0 gets the embeddings, fragment N-1 gets the LM head.
func SplitGPT(model *nn.GPT, numFragments int) []*Fragment {
	if numFragments < 1 {
		panic("need at least 1 fragment")
	}
	if numFragments > model.NumLayers {
		panic("more fragments than layers")
	}

	fragments := make([]*Fragment, numFragments)
	layersPerFrag := model.NumLayers / numFragments
	remainder := model.NumLayers % numFragments

	blockIdx := 0
	for i := 0; i < numFragments; i++ {
		n := layersPerFrag
		if i < remainder {
			n++ // distribute remainder layers to earlier fragments
		}

		frag := &Fragment{
			ID:     i,
			Blocks: model.Blocks[blockIdx : blockIdx+n],
			Dim:    model.Dim,
		}

		if i == 0 {
			frag.HasEmbed = true
			frag.TokenEmbed = model.TokenEmbed
			frag.PosEmbed = model.PosEmbed
		}
		if i == numFragments-1 {
			frag.HasLMHead = true
			frag.FinalNorm = model.FinalNorm
			frag.LMHead = model.LMHead
			frag.VocabSize = model.VocabSize
		}

		fragments = append(fragments[:i], frag)
		blockIdx += n
	}
	return fragments
}

// Forward runs this fragment's portion of the model.
// Input: either token IDs (if HasEmbed) or activation tensor (seq, dim).
// Output: activation tensor (seq, dim) or logits (seq, vocab) if HasLMHead.
func (f *Fragment) Forward(x *g.Tensor, tokenIDs []int, seqLen int) *g.Tensor {
	if f.HasEmbed {
		// Compute embeddings
		tokEmb := f.TokenEmbed.Forward(tokenIDs)
		posIDs := make([]int, seqLen)
		for i := range posIDs {
			posIDs[i] = i
		}
		posEmb := f.PosEmbed.Forward(posIDs)
		x = g.Add(tokEmb, posEmb)
	}

	// Run transformer blocks
	for _, block := range f.Blocks {
		x = block.Forward(x, seqLen)
	}

	if f.HasLMHead {
		x = f.FinalNorm.Forward(x)
		x = f.LMHead.Forward(x)
	}

	return x
}

// ---------- Network transport ----------

// SerializeTensor writes a tensor to a writer in a simple binary format.
// Format: ndim(4) shape(ndim*4) data(n*4)
func SerializeTensor(w io.Writer, t *g.Tensor) error {
	shape := t.Shape()
	if err := binary.Write(w, binary.LittleEndian, int32(len(shape))); err != nil {
		return err
	}
	for _, s := range shape {
		if err := binary.Write(w, binary.LittleEndian, int32(s)); err != nil {
			return err
		}
	}
	for _, v := range t.Data() {
		if err := binary.Write(w, binary.LittleEndian, math.Float32bits(v)); err != nil {
			return err
		}
	}
	return nil
}

// DeserializeTensor reads a tensor from a reader.
func DeserializeTensor(r io.Reader) (*g.Tensor, error) {
	var ndim int32
	if err := binary.Read(r, binary.LittleEndian, &ndim); err != nil {
		return nil, err
	}

	shape := make([]int, ndim)
	size := 1
	for i := 0; i < int(ndim); i++ {
		var s int32
		if err := binary.Read(r, binary.LittleEndian, &s); err != nil {
			return nil, err
		}
		shape[i] = int(s)
		size *= int(s)
	}

	data := make([]float32, size)
	for i := 0; i < size; i++ {
		var bits uint32
		if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
			return nil, err
		}
		data[i] = math.Float32frombits(bits)
	}

	return g.NewTensor(data, shape...), nil
}

// FragmentServer serves a fragment over TCP.
// Receives activation tensors, runs forward, sends results back.
type FragmentServer struct {
	Fragment *Fragment
	Addr     string
	listener net.Listener
}

// NewFragmentServer creates a server for a fragment.
func NewFragmentServer(frag *Fragment, addr string) *FragmentServer {
	return &FragmentServer{Fragment: frag, Addr: addr}
}

// Start begins listening for incoming activation tensors.
func (s *FragmentServer) Start() error {
	ln, err := net.Listen("tcp", s.Addr)
	if err != nil {
		return err
	}
	s.listener = ln
	s.Addr = ln.Addr().String() // update to actual bound address
	fmt.Printf("[Fragment %d] Listening on %s (%d blocks)\n", s.Fragment.ID, s.Addr, len(s.Fragment.Blocks))

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return // server stopped
			}
			go s.handleConn(conn)
		}
	}()
	return nil
}

func (s *FragmentServer) handleConn(conn net.Conn) {
	defer conn.Close()

	// Read sequence length
	var seqLen int32
	if err := binary.Read(conn, binary.LittleEndian, &seqLen); err != nil {
		return
	}

	// Read activation tensor
	activation, err := DeserializeTensor(conn)
	if err != nil {
		return
	}

	// Run forward
	result := s.Fragment.Forward(activation, nil, int(seqLen))

	// Send result back
	SerializeTensor(conn, result)
}

// Stop stops the server.
func (s *FragmentServer) Stop() {
	if s.listener != nil {
		s.listener.Close()
	}
}

// FragmentClient sends activations to a remote fragment server.
type FragmentClient struct {
	Addr string
}

// Forward sends an activation tensor to the remote fragment and returns the result.
func (c *FragmentClient) Forward(activation *g.Tensor, seqLen int) (*g.Tensor, time.Duration, error) {
	conn, err := net.Dial("tcp", c.Addr)
	if err != nil {
		return nil, 0, err
	}
	defer conn.Close()

	start := time.Now()

	// Send sequence length
	binary.Write(conn, binary.LittleEndian, int32(seqLen))

	// Send activation
	if err := SerializeTensor(conn, activation); err != nil {
		return nil, 0, fmt.Errorf("send: %w", err)
	}

	// Read result
	result, err := DeserializeTensor(conn)
	if err != nil {
		return nil, 0, fmt.Errorf("recv: %w", err)
	}

	return result, time.Since(start), nil
}

// PipelineInfer runs inference across a pipeline of local fragments.
// This simulates distributed inference without actual network transport.
func PipelineInfer(fragments []*Fragment, tokenIDs []int) *g.Tensor {
	seqLen := len(tokenIDs)
	var x *g.Tensor

	for _, frag := range fragments {
		if frag.HasEmbed {
			x = frag.Forward(nil, tokenIDs, seqLen)
		} else {
			x = frag.Forward(x, nil, seqLen)
		}
	}
	return x
}

// PipelineInferNetwork runs inference across fragments connected via TCP.
// Fragment 0 runs locally (has embeddings), others are remote.
func PipelineInferNetwork(localFrag *Fragment, remoteAddrs []string, tokenIDs []int) (*g.Tensor, []time.Duration, error) {
	seqLen := len(tokenIDs)

	// Run local fragment (embedding + first blocks)
	x := localFrag.Forward(nil, tokenIDs, seqLen)

	var durations []time.Duration

	// Send through remote fragments
	for _, addr := range remoteAddrs {
		client := &FragmentClient{Addr: addr}
		result, dur, err := client.Forward(x, seqLen)
		if err != nil {
			return nil, nil, fmt.Errorf("remote %s: %w", addr, err)
		}
		x = result
		durations = append(durations, dur)
	}

	return x, durations, nil
}
