//go:build darwin

package mimi

import (
	"fmt"
	"sort"
	"strings"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/accelerate"
	"github.com/vinq1911/gorch/model"
)

// Quantizer geometry of the kyutai/mimi checkpoint (plan 0006 §0.2,
// verified against transformers' modeling_mimi.py): a split residual
// vector quantizer whose semantic half is a single codebook (distilled
// from WavLM) and whose acoustic half is a 31-level residual chain.
// Both halves carry their own bias-free 1×1 input/output projections
// 512↔256 and quantize the SAME pre-quantizer latent; the final
// quantized latent is the sum of the two output projections.
const (
	codebookSize = 2048
	codebookDim  = 256
	numSemantic  = 1
	numAcoustic  = 31

	// clusterUsageEps mirrors MimiEuclideanCodebook's epsilon: the
	// embedding matrix is not stored directly but reconstructed as
	// embed = embed_sum / cluster_usage.clamp(min=1e-5)  (per row).
	clusterUsageEps = 1e-5
)

// codebook is one EMA Euclidean codebook with the embedding matrix
// precomputed at load time from embed_sum / clamp(cluster_usage, ε).
type codebook struct {
	embed  []float32 // (codebookSize, codebookDim), row-major
	sqNorm []float32 // (codebookSize,) — per-row ‖e‖² for the argmin
}

func newCodebook(embedSum, clusterUsage *g.Tensor) *codebook {
	cb := &codebook{
		embed:  make([]float32, codebookSize*codebookDim),
		sqNorm: make([]float32, codebookSize),
	}
	es, cu := embedSum.Data(), clusterUsage.Data()
	for i := 0; i < codebookSize; i++ {
		usage := cu[i]
		if usage < clusterUsageEps {
			usage = clusterUsageEps
		}
		row := cb.embed[i*codebookDim : (i+1)*codebookDim]
		var ss float64
		for j := range row {
			e := es[i*codebookDim+j] / usage
			row[j] = e
			ss += float64(e) * float64(e)
		}
		cb.sqNorm[i] = float32(ss)
	}
	return cb
}

// rvq is one residual vector quantizer half (semantic or acoustic):
// bias-free 1×1 projections around a chain of Euclidean codebooks.
type rvq struct {
	inProj  []float32 // (codebookDim, dim): input_proj conv weight, no bias
	outProj []float32 // (dim, codebookDim): output_proj conv weight, no bias
	books   []*codebook
	dim     int
}

// encode projects x (T, dim) into codebook space and runs the residual
// argmin chain over the first n codebooks, returning (n, T) codes.
func (q *rvq) encode(x []float32, T, n int) [][]int {
	residual := make([]float32, T*codebookDim)
	accelerate.SgemmTransB(T, codebookDim, q.dim, 1, x, q.inProj, 0, residual)

	dots := make([]float32, T*codebookSize)
	codes := make([][]int, n)
	for level := 0; level < n; level++ {
		cb := q.books[level]
		// dists² = ‖r‖² − 2 r·e + ‖e‖²; ‖r‖² is constant per frame, so
		// argmin_j (‖e_j‖² − 2 r·e_j) selects the nearest codeword.
		accelerate.SgemmTransB(T, codebookSize, codebookDim, 1, residual, cb.embed, 0, dots)
		lc := make([]int, T)
		for t := 0; t < T; t++ {
			row := dots[t*codebookSize : (t+1)*codebookSize]
			best, bestD := 0, cb.sqNorm[0]-2*row[0]
			for j := 1; j < codebookSize; j++ {
				if d := cb.sqNorm[j] - 2*row[j]; d < bestD {
					best, bestD = j, d
				}
			}
			lc[t] = best
			e := cb.embed[best*codebookDim : (best+1)*codebookDim]
			r := residual[t*codebookDim : (t+1)*codebookDim]
			for j := range r {
				r[j] -= e[j]
			}
		}
		codes[level] = lc
	}
	return codes
}

// decode sums the codebook embeddings selected by codes ((n, T),
// level-major) and applies the output projection, returning (T, dim)
// appended into out (which must be zero-initialized or hold a partial
// sum: the projection is accumulated with beta=1).
func (q *rvq) decode(codes [][]int, T int, out []float32) {
	sum := make([]float32, T*codebookDim)
	for level, lc := range codes {
		cb := q.books[level]
		for t, c := range lc {
			e := cb.embed[c*codebookDim : (c+1)*codebookDim]
			s := sum[t*codebookDim : (t+1)*codebookDim]
			for j := range s {
				s[j] += e[j]
			}
		}
	}
	accelerate.SgemmTransB(T, q.dim, codebookDim, 1, sum, q.outProj, 1, out)
}

// Quantizer is the Mimi split residual vector quantizer: it turns the
// (T, 512) pre-quantizer latent produced by Encoder.Encode into
// discrete RVQ codes matching HF's model.encode(...).audio_codes, and
// codes back into the (T, 512) quantized latent (the decoder-side
// input).
type Quantizer struct {
	semantic *rvq // 1 codebook
	acoustic *rvq // 31 codebooks, residual chain
}

// NumCodebooks returns the total number of codebooks (32).
func (q *Quantizer) NumCodebooks() int { return numSemantic + numAcoustic }

// CodebookSize returns the per-codebook vocabulary size (2048).
func (q *Quantizer) CodebookSize() int { return codebookSize }

// Encode quantizes the (T, 512) pre-quantizer latent into
// numQuantizers codebook levels and returns codes indexed
// [level][frame] (shape (numQuantizers, T), matching HF audio_codes
// layout). Level 0 is the semantic codebook; levels 1.. are the
// acoustic residual chain over the same latent. numQuantizers must be
// in [1, 32]; Moshi uses 8, HF's default is all 32. Codes are
// prefix-consistent: Encode(x, 8) equals the first 8 rows of
// Encode(x, 32).
func (q *Quantizer) Encode(latent *g.Tensor, numQuantizers int) [][]int {
	shape := latent.Shape()
	if len(shape) != 2 || shape[1] != q.semantic.dim {
		panic(fmt.Sprintf("mimi: Quantizer.Encode latent shape %v, want (T, %d)", shape, q.semantic.dim))
	}
	if numQuantizers < numSemantic || numQuantizers > q.NumCodebooks() {
		panic(fmt.Sprintf("mimi: Quantizer.Encode numQuantizers = %d, want %d..%d",
			numQuantizers, numSemantic, q.NumCodebooks()))
	}
	T := shape[0]
	x := latent.Data()
	codes := q.semantic.encode(x, T, numSemantic)
	if numQuantizers > numSemantic {
		codes = append(codes, q.acoustic.encode(x, T, numQuantizers-numSemantic)...)
	}
	return codes
}

// Decode maps codes (shape (numQuantizers, T), as returned by Encode)
// back to the (T, 512) quantized latent: the sum of the semantic and
// acoustic output projections, exactly HF's quantizer.decode.
func (q *Quantizer) Decode(codes [][]int) *g.Tensor {
	n := len(codes)
	if n < numSemantic || n > q.NumCodebooks() {
		panic(fmt.Sprintf("mimi: Quantizer.Decode got %d code levels, want %d..%d",
			n, numSemantic, q.NumCodebooks()))
	}
	T := len(codes[0])
	for level, lc := range codes {
		if len(lc) != T {
			panic(fmt.Sprintf("mimi: Quantizer.Decode level %d has %d frames, level 0 has %d", level, len(lc), T))
		}
		for t, c := range lc {
			if c < 0 || c >= codebookSize {
				panic(fmt.Sprintf("mimi: Quantizer.Decode code %d out of range [0, %d) at level %d frame %d",
					c, codebookSize, level, t))
			}
		}
	}
	out := make([]float32, T*q.semantic.dim)
	q.semantic.decode(codes[:numSemantic], T, out)
	if n > numSemantic {
		q.acoustic.decode(codes[numSemantic:], T, out)
	}
	return g.NewTensor(out, T, q.semantic.dim)
}

// LoadQuantizer loads the split RVQ quantizer from a kyutai/mimi
// model.safetensors checkpoint, with the same fail-loudly key/shape
// validation as Load. Codebook embedding matrices are precomputed from
// embed_sum / clamp(cluster_usage, 1e-5) at load time.
func LoadQuantizer(path string) (*Quantizer, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}
	return loadQuantizerFrom(sf)
}

// LoadWithQuantizer loads both the pre-quantizer Encoder and the
// Quantizer from one checkpoint parse (the safetensors read dominates
// load time). Load's behavior is unchanged for callers that only need
// the continuous latent.
func LoadWithQuantizer(path string) (*Encoder, *Quantizer, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, nil, err
	}
	e, err := loadEncoderFrom(sf)
	if err != nil {
		return nil, nil, err
	}
	q, err := loadQuantizerFrom(sf)
	if err != nil {
		return nil, nil, err
	}
	return e, q, nil
}

func loadQuantizerFrom(sf *model.SafetensorsFile) (*Quantizer, error) {
	dim := DefaultConfig().HiddenSize

	consumed := map[string]bool{}
	var problems []string

	take := func(key string, want []int, assign func(*g.Tensor)) {
		t, ok := sf.Tensors[key]
		if !ok {
			problems = append(problems, "missing: "+key)
			return
		}
		consumed[key] = true
		if !shapeEq(t.Shape(), want) {
			problems = append(problems, fmt.Sprintf("shape: %s is %v, want %v", key, t.Shape(), want))
			return
		}
		assign(t)
	}

	loadRVQ := func(name string, numBooks int) *rvq {
		q := &rvq{dim: dim, books: make([]*codebook, numBooks)}
		p := "quantizer." + name + "_residual_vector_quantizer."
		// The 1×1 conv weights are (out, in, 1); the trailing kernel dim
		// is dropped for the flat (out, in) matmul view.
		take(p+"input_proj.weight", []int{codebookDim, dim, 1}, func(t *g.Tensor) { q.inProj = t.Data() })
		take(p+"output_proj.weight", []int{dim, codebookDim, 1}, func(t *g.Tensor) { q.outProj = t.Data() })
		for i := 0; i < numBooks; i++ {
			cp := fmt.Sprintf("%slayers.%d.codebook.", p, i)
			var embedSum, clusterUsage *g.Tensor
			take(cp+"embed_sum", []int{codebookSize, codebookDim}, func(t *g.Tensor) { embedSum = t })
			take(cp+"cluster_usage", []int{codebookSize}, func(t *g.Tensor) { clusterUsage = t })
			// EMA-init flag; carries no inference information but must
			// be accounted for by the fail-loudly key sweep.
			take(cp+"initialized", []int{1}, func(*g.Tensor) {})
			if embedSum != nil && clusterUsage != nil {
				q.books[i] = newCodebook(embedSum, clusterUsage)
			}
		}
		return q
	}

	quant := &Quantizer{
		semantic: loadRVQ("semantic", numSemantic),
		acoustic: loadRVQ("acoustic", numAcoustic),
	}

	// Every quantizer.* key must have been consumed.
	var unexpected []string
	for _, name := range sf.Names {
		if strings.HasPrefix(name, "quantizer.") && !consumed[name] {
			unexpected = append(unexpected, name)
		}
	}
	sort.Strings(unexpected)
	for _, name := range unexpected {
		problems = append(problems, "unexpected: "+name)
	}

	if len(problems) > 0 {
		return nil, fmt.Errorf("mimi: quantizer checkpoint mismatch (%d problems):\n  %s",
			len(problems), strings.Join(problems, "\n  "))
	}
	return quant, nil
}
