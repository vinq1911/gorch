//go:build darwin

// External test package: model imports gorch, so this fixture test
// (which loads safetensors through model) cannot live in package gorch.
package gorch_test

import (
	"fmt"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
)

// TestConvTranspose1dMatchesPyTorchFixtures checks
// ConvTranspose1dForward bit-exactly against the six D0 layout-pinning
// cases torch.conv_transpose1d generated
// (audio/testdata/mimi_decoder_fixtures.safetensors, plan 0007 risk
// #1: weight layout (inC, outC/groups, k), out length (L-1)*stride+k).
// All six cases match bit-exactly, GEMM path included — torch rounds
// each product before the scatter-add and so do we (the depthwise loop
// needs an explicit float32 conversion to stop Go's arm64 FMA fusion;
// see convTranspose1dDepthwise). If a future BLAS/toolchain change
// introduces f32 summation-order differences here, the agreed fallback
// gate is <= 1e-6 — but relax only after confirming the diff is pure
// rounding, not layout.
func TestConvTranspose1dMatchesPyTorchFixtures(t *testing.T) {
	sf, err := model.LoadSafetensors("audio/testdata/mimi_decoder_fixtures.safetensors")
	if err != nil {
		t.Fatalf("load decoder fixtures (regenerate with audio/export_mimi_fixtures.py): %v", err)
	}
	get := func(name string) *g.Tensor {
		tt, ok := sf.Tensors[name]
		if !ok {
			t.Fatalf("missing fixture tensor %q", name)
		}
		return tt
	}

	manifest := get("ct_manifest").Data() // rows: [kernel, stride, groups, has_bias]
	for i := 0; i < 6; i++ {
		k := int(manifest[i*4+0])
		stride := int(manifest[i*4+1])
		groups := int(manifest[i*4+2])
		hasBias := manifest[i*4+3] != 0
		t.Run(fmt.Sprintf("ct_%d_k%ds%dg%d", i, k, stride, groups), func(t *testing.T) {
			in := get(fmt.Sprintf("ct_%d_in", i))
			w := get(fmt.Sprintf("ct_%d_w", i))
			want := get(fmt.Sprintf("ct_%d_out", i))
			var b *g.Tensor
			if hasBias {
				b = get(fmt.Sprintf("ct_%d_b", i))
			}
			if w.Shape()[2] != k {
				t.Fatalf("manifest kernel %d != weight shape %v", k, w.Shape())
			}

			got := g.ConvTranspose1dForward(in, w, b, stride, groups)

			ws, gs := want.Shape(), got.Shape()
			if len(gs) != 3 || gs[0] != ws[0] || gs[1] != ws[1] || gs[2] != ws[2] {
				t.Fatalf("output shape %v, want %v", gs, ws)
			}
			gd, wd := got.Data(), want.Data()
			mismatches := 0
			for j := range wd {
				if gd[j] != wd[j] {
					if mismatches < 5 {
						t.Errorf("out[%d] = %v, want %v (diff %g)", j, gd[j], wd[j], gd[j]-wd[j])
					}
					mismatches++
				}
			}
			if mismatches > 0 {
				t.Errorf("%d/%d values differ from torch.conv_transpose1d (want bit-exact)", mismatches, len(wd))
			}
		})
	}
}
