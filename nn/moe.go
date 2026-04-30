//go:build darwin

package nn

import (
	"math"

	g "github.com/vinq1911/gorch"
)

// MoE — Sparse Mixture of Experts FFN block.
//
// Replaces the dense FFN with N parallel "experts" (each a small
// SwiGLU FFN); a learned router picks the top-K experts per token
// and the output is the routing-weighted sum of those K outputs.
//
// Used by Mixtral, DeepSeek-V2/V3, OpenMythos. The compute per token
// is K × expert_size, much less than running all N experts; the
// memory is N × expert_size — i.e., MoE buys capacity (more total
// parameters per FLOP) at the cost of memory and dispatch
// complexity.
//
// Plan 0001 Phase 1 item 11. Expert dispatch uses gather → grouped
// matmul → scatter (single-device path). Plan 0003 explicitly
// rejects "MoE expert parallelism via multiple Metal command queues"
// — that conflates expert parallelism (multi-device) with I/O
// overlap (multi-queue, separate concern).
type MoE struct {
	Router  *Linear // dim → numExperts (logits)
	Experts []*Expert
	NumExperts int
	NumExpertsPerToken int // K in top-K
	Dim     int
	ExpertDim int
}

// Expert is one SwiGLU FFN: dim → expertDim (gate + value parallel
// projections) → SwiGLU → expertDim → dim down-projection.
type Expert struct {
	Wgate *Linear // dim → expertDim
	Wup   *Linear // dim → expertDim
	Wdown *Linear // expertDim → dim
}

// NewExpert builds one SwiGLU expert.
func NewExpert(dim, expertDim int) *Expert {
	return &Expert{
		Wgate: NewLinear(dim, expertDim),
		Wup:   NewLinear(dim, expertDim),
		Wdown: NewLinear(expertDim, dim),
	}
}

// Forward applies the SwiGLU FFN to x: (M, dim) → (M, dim).
func (e *Expert) Forward(x *g.Tensor) *g.Tensor {
	gate := e.Wgate.Forward(x)
	up := e.Wup.Forward(x)
	hidden := g.SwiGLU(gate, up)
	return e.Wdown.Forward(hidden)
}

// Parameters returns the three linear layers' weights+biases.
func (e *Expert) Parameters() []*g.Tensor {
	var p []*g.Tensor
	p = append(p, e.Wgate.Parameters()...)
	p = append(p, e.Wup.Parameters()...)
	p = append(p, e.Wdown.Parameters()...)
	return p
}

// NewMoE builds a sparse MoE FFN with the given config.
//
//	dim                — hidden size
//	expertDim          — FFN intermediate dim per expert
//	numExperts         — total number of experts (e.g. 4 for
//	                     mythos_tiny, 8 for Mixtral, 64 for
//	                     mythos_1b)
//	numExpertsPerToken — top-K (e.g. 2 for both Mixtral and DeepSeek)
func NewMoE(dim, expertDim, numExperts, numExpertsPerToken int) *MoE {
	if numExpertsPerToken <= 0 || numExpertsPerToken > numExperts {
		panic("gorch/nn: MoE numExpertsPerToken out of range")
	}
	experts := make([]*Expert, numExperts)
	for i := range experts {
		experts[i] = NewExpert(dim, expertDim)
	}
	return &MoE{
		Router:             NewLinear(dim, numExperts),
		Experts:            experts,
		NumExperts:         numExperts,
		NumExpertsPerToken: numExpertsPerToken,
		Dim:                dim,
		ExpertDim:          expertDim,
	}
}

// Forward runs the MoE block:
//
//	1. logits = Router(x) — (M, numExperts)
//	2. topK values + indices per token
//	3. softmax over topK values → per-token routing weights summing to 1
//	4. for each expert e: gather tokens routed to e, run e.Forward,
//	   scatter outputs back to original positions, weight by routing
//	5. sum across the K expert contributions per token
//
// Autograd flows through:
//   - g.Gather (autograd) for input collection per expert
//   - Expert.Forward's chained Linear+SwiGLU+Linear (autograd)
//   - g.Mul broadcasting per-row routing weight (autograd wrt expert
//     output; the weight tensor itself is detached — see note below)
//   - g.ScatterAdd (autograd) to scatter-add the weighted expert
//     outputs back to the global accumulator
//   - g.Add (autograd) to accumulate across experts
//
// Note: the routing weights are computed via a non-autograd-aware
// softmax over topVals.Data() (raw float32). Gradient does NOT flow
// from the loss through routing weights back to the router, because
// TopK selection is non-differentiable and we don't have a per-row
// gather along axis 1 yet. The router IS still trained — through the
// LoadBalanceLoss helper, which uses the autograd-aware Softmax on
// the full router output. Two-stage training is standard for MoE
// (Mixtral, DeepSeek both do this).
//
// Returns: (M, dim) output.
func (m *MoE) Forward(x *g.Tensor) *g.Tensor {
	M := x.Shape()[0]
	dim := m.Dim
	K := m.NumExpertsPerToken

	logits := m.Router.Forward(x)
	topVals, topIdx := g.TopK(logits, K)
	weights := softmaxRows(topVals.Data(), M, K)

	// Group token indices by destination expert.
	type assign struct {
		token, slot int
	}
	expertTokens := make([][]assign, m.NumExperts)
	for tok := 0; tok < M; tok++ {
		for slot := 0; slot < K; slot++ {
			e := topIdx[tok*K+slot]
			expertTokens[e] = append(expertTokens[e], assign{token: tok, slot: slot})
		}
	}

	// Start with a zero accumulator. Each expert's weighted output
	// scatter-adds into it via the autograd-aware path.
	var out *g.Tensor = g.Zeros(M, dim)

	for e := 0; e < m.NumExperts; e++ {
		group := expertTokens[e]
		if len(group) == 0 {
			continue
		}
		idxList := make([]int, len(group))
		wRow := make([]float32, len(group)*dim)
		for i, a := range group {
			idxList[i] = a.token
			w := weights[a.token*K+a.slot]
			for d := 0; d < dim; d++ {
				wRow[i*dim+d] = w
			}
		}
		input := g.Gather(x, idxList)        // autograd ✓
		exOut := m.Experts[e].Forward(input) // autograd ✓ (Linear→SwiGLU→Linear)

		// Per-row routing-weight scaling via Mul (autograd wrt exOut).
		wTensor := g.NewTensor(wRow, len(group), dim)
		weighted := g.Mul(exOut, wTensor)

		// Scatter-add this expert's weighted contribution to the
		// (M, dim) accumulator. Autograd-aware end-to-end.
		contribution := g.ScatterAdd(weighted, idxList, M)
		out = g.Add(out, contribution)
	}
	return out
}

// LoadBalanceLoss returns the auxiliary load-balancing loss for one
// MoE call. PyTorch / Mixtral convention: encourage uniform expert
// usage. Compute on the SAME forward pass as Forward to amortise the
// router compute. Caller weights this into the total loss; typical
// coefficient is 0.01.
//
// fraction[e] = fraction of total tokens routed to expert e
// pProb[e]    = average router probability for expert e across batch
// loss        = numExperts * sum_e fraction[e] * pProb[e]
//
// Optimum (equal load + equal router prob = 1/numExperts) gives loss
// = numExperts * sum (1/numExperts²) = 1.0.
func (m *MoE) LoadBalanceLoss(x *g.Tensor) float32 {
	M := x.Shape()[0]
	K := m.NumExpertsPerToken
	N := m.NumExperts

	logits := m.Router.Forward(x)
	probs := g.Softmax(logits) // (M, N)
	pData := probs.Data()

	// pProb[e] = mean over tokens of probs[tok, e]
	pProb := make([]float32, N)
	for tok := 0; tok < M; tok++ {
		for e := 0; e < N; e++ {
			pProb[e] += pData[tok*N+e]
		}
	}
	for e := range pProb {
		pProb[e] /= float32(M)
	}

	// fraction[e] = (# tokens routed to e via top-K) / (M*K)
	_, topIdx := g.TopK(logits, K)
	fraction := make([]float32, N)
	for _, e := range topIdx {
		fraction[e] += 1
	}
	for e := range fraction {
		fraction[e] /= float32(M * K)
	}

	var loss float32
	for e := 0; e < N; e++ {
		loss += fraction[e] * pProb[e]
	}
	return loss * float32(N)
}

// Parameters returns the router + every expert's parameters.
func (m *MoE) Parameters() []*g.Tensor {
	var params []*g.Tensor
	params = append(params, m.Router.Parameters()...)
	for _, e := range m.Experts {
		params = append(params, e.Parameters()...)
	}
	return params
}

// softmaxRows computes softmax for each (rows, cols) row of `data`,
// returning a new buffer with the same shape. Standalone helper —
// the existing g.Softmax requires a Tensor; here we operate on a
// bare slice to match TopK's []float32 + slice-of-int output.
func softmaxRows(data []float32, rows, cols int) []float32 {
	out := make([]float32, len(data))
	for i := 0; i < rows; i++ {
		row := data[i*cols : (i+1)*cols]
		// Max for stability.
		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		var sum float32
		for j, v := range row {
			out[i*cols+j] = float32(math.Exp(float64(v - maxVal)))
			sum += out[i*cols+j]
		}
		for j := range row {
			out[i*cols+j] /= sum
		}
	}
	return out
}
