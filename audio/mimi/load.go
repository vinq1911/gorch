//go:build darwin

package mimi

import (
	"fmt"
	"math"
	"sort"
	"strings"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
)

// hfConvPrefixes returns the HF checkpoint key prefix (without
// ".weight"/".bias") for each conv in SEANet.Convs order, per the plan
// §4.6 name map: encoder.layers.{0,3,6,9,12,14}.conv and
// encoder.layers.{1,4,7,10}.block.{1,3}.conv.
func hfConvPrefixes(s *SEANet) []string {
	prefixes := make([]string, 0, len(s.Convs))
	prefixes = append(prefixes, "encoder.layers.0.conv")
	for i := 0; i < s.stages(); i++ {
		blockIdx := 1 + 3*i
		downIdx := 3 + 3*i
		prefixes = append(prefixes,
			fmt.Sprintf("encoder.layers.%d.block.1.conv", blockIdx),
			fmt.Sprintf("encoder.layers.%d.block.3.conv", blockIdx),
			fmt.Sprintf("encoder.layers.%d.conv", downIdx))
	}
	prefixes = append(prefixes, fmt.Sprintf("encoder.layers.%d.conv", 2+3*s.stages()))
	return prefixes
}

// LoadSEANet loads the SEANet encoder weights from a kyutai/mimi
// model.safetensors checkpoint. It verifies every expected key exists
// with the expected shape and fails listing all missing, misshapen,
// and unexpected keys. Keys under decoder.*, quantizer.*,
// encoder_transformer.*, downsample.* (and the other non-encoder
// families) are ignored until later phases.
func LoadSEANet(path string) (*SEANet, error) {
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}
	return loadSEANetFrom(sf)
}

func loadSEANetFrom(sf *model.SafetensorsFile) (*SEANet, error) {
	s := NewSEANet(DefaultConfig())
	prefixes := hfConvPrefixes(s)

	consumed := map[string]bool{}
	var problems []string

	for i, prefix := range prefixes {
		conv := s.Convs[i]

		w, keys, ok := convWeight(sf.Tensors, prefix)
		if !ok {
			problems = append(problems, fmt.Sprintf("missing: %s.weight (no weight-norm g/v fallback keys either)", prefix))
		} else {
			for _, k := range keys {
				consumed[k] = true
			}
			if !shapeEq(w.Shape(), conv.Weight.Shape()) {
				problems = append(problems, fmt.Sprintf("shape: %s.weight is %v, want %v", prefix, w.Shape(), conv.Weight.Shape()))
			} else {
				conv.Weight = w
			}
		}

		biasKey := prefix + ".bias"
		b, ok := sf.Tensors[biasKey]
		if !ok {
			problems = append(problems, "missing: "+biasKey)
		} else {
			consumed[biasKey] = true
			if !shapeEq(b.Shape(), conv.Bias.Shape()) {
				problems = append(problems, fmt.Sprintf("shape: %s is %v, want %v", biasKey, b.Shape(), conv.Bias.Shape()))
			} else {
				conv.Bias = b
			}
		}
	}

	// Every encoder.* key must have been consumed; other families are
	// deliberately ignored for now (transformer/downsample arrive in
	// Phase 3, quantizer in Phase 7, decoder never).
	var unexpected []string
	for _, name := range sf.Names {
		if strings.HasPrefix(name, "encoder.") && !consumed[name] {
			unexpected = append(unexpected, name)
		}
	}
	sort.Strings(unexpected)
	for _, name := range unexpected {
		problems = append(problems, "unexpected: "+name)
	}

	if len(problems) > 0 {
		return nil, fmt.Errorf("mimi: SEANet checkpoint mismatch (%d problems):\n  %s",
			len(problems), strings.Join(problems, "\n  "))
	}
	return s, nil
}

// convWeight fetches a conv weight by prefix. Primary path: the plain
// fused "<prefix>.weight" (what kyutai/mimi ships). Fallback: fuse
// weight-norm parameter pairs — "<prefix>.weight_g"/"weight_v" or
// "<prefix>.parametrizations.weight.original0/original1" — as
// w = g · v/‖v‖ with the norm over dims (1,2) per output channel.
// Returns the weight, the checkpoint keys consumed, and whether any
// style was found.
func convWeight(tensors map[string]*g.Tensor, prefix string) (*g.Tensor, []string, bool) {
	if w, ok := tensors[prefix+".weight"]; ok {
		return w, []string{prefix + ".weight"}, true
	}
	for _, pair := range [][2]string{
		{prefix + ".weight_g", prefix + ".weight_v"},
		{prefix + ".parametrizations.weight.original0", prefix + ".parametrizations.weight.original1"},
	} {
		gt, ok1 := tensors[pair[0]]
		vt, ok2 := tensors[pair[1]]
		if ok1 && ok2 {
			return fuseWeightNorm(gt, vt), []string{pair[0], pair[1]}, true
		}
	}
	return nil, nil, false
}

// fuseWeightNorm computes w[o,:,:] = g[o] * v[o,:,:] / ‖v[o,:,:]‖₂.
// gt may be (outC,), (outC,1) or (outC,1,1) — PyTorch stores
// (outC,1,1); only its outC leading dimension is used. vt is
// (outC, inC, k).
func fuseWeightNorm(gt, vt *g.Tensor) *g.Tensor {
	vShape := vt.Shape()
	if len(vShape) != 3 {
		panic(fmt.Sprintf("mimi: fuseWeightNorm expects 3-D v, got %v", vShape))
	}
	outC := vShape[0]
	if len(gt.Data()) != outC {
		panic(fmt.Sprintf("mimi: fuseWeightNorm g has %d elements, want %d", len(gt.Data()), outC))
	}
	row := vShape[1] * vShape[2]
	v := vt.Data()
	out := make([]float32, len(v))
	for o := 0; o < outC; o++ {
		var ss float64
		for i := o * row; i < (o+1)*row; i++ {
			ss += float64(v[i]) * float64(v[i])
		}
		scale := float32(float64(gt.Data()[o]) / math.Sqrt(ss))
		for i := o * row; i < (o+1)*row; i++ {
			out[i] = v[i] * scale
		}
	}
	return g.NewTensor(out, vShape...)
}

func shapeEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
