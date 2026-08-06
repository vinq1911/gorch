//go:build darwin

package nn

import (
	"math"
	"testing"

	g "github.com/vinq1911/gorch"
)

// TestRoPEPositionZeroIsIdentity: at position 0, all θ are 0, so
// cos=1 and sin=0. Apply must return x unchanged.
func TestRoPEPositionZeroIsIdentity(t *testing.T) {
	r := NewRoPE(8, 16, 10000, RopeLlama)
	x := g.RandN(1, 1, 8) // (outer=1, seq=1, headDim=8) — only position 0
	y := r.Apply(x, 0)
	for i := range x.Data() {
		if math.Abs(float64(x.Data()[i]-y.Data()[i])) > 1e-6 {
			t.Fatalf("[%d] pos-0 changed value: x=%g y=%g", i, x.Data()[i], y.Data()[i])
		}
	}
}

// TestRoPENormPreserving: rotation preserves the L2 norm of each
// (x_i, x_{i+half}) pair. Verifies the math.
func TestRoPENormPreserving(t *testing.T) {
	const headDim = 16
	r := NewRoPE(headDim, 8, 10000, RopeLlama)
	x := g.RandN(2, 4, headDim) // (B=2, seq=4, headDim)
	y := r.Apply(x, 0)
	half := headDim / 2

	xData := x.Data()
	yData := y.Data()
	for i := 0; i < 2*4; i++ {
		off := i * headDim
		for j := 0; j < half; j++ {
			a := xData[off+j]
			b := xData[off+half+j]
			a2 := yData[off+j]
			b2 := yData[off+half+j]
			normIn := float64(a)*float64(a) + float64(b)*float64(b)
			normOut := float64(a2)*float64(a2) + float64(b2)*float64(b2)
			if math.Abs(normIn-normOut) > 1e-4*math.Max(normIn, 1) {
				t.Fatalf("row %d pair %d: norm changed from %g to %g", i, j, normIn, normOut)
			}
		}
	}
}

// TestRoPEStartPosEquivalence: applying RoPE to a (seq=2) batch with
// startPos=3 must give the same result as positions 3 and 4 from a
// fresh table. This is the load-bearing case for KV-cache decoding.
func TestRoPEStartPosEquivalence(t *testing.T) {
	const headDim = 8
	r := NewRoPE(headDim, 16, 10000, RopeLlama)

	// Build (1, 5, headDim) full sequence; apply RoPE with startPos=0.
	full := g.RandN(1, 5, headDim)
	rotFull := r.Apply(full, 0)

	// Now take rows [3..5] of full and apply with startPos=3.
	tail := g.NewTensor(full.Data()[3*headDim:], 1, 2, headDim)
	rotTail := r.Apply(tail, 3)

	// rotTail[s, :] should equal rotFull[3+s, :].
	for s := 0; s < 2; s++ {
		for d := 0; d < headDim; d++ {
			want := rotFull.At(0, 3+s, d)
			got := rotTail.At(0, s, d)
			if math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("[%d,%d] startPos slice differs: got %g want %g", s, d, got, want)
			}
		}
	}
}

// TestRoPEBackwardMatchesNumerical: numerical-vs-analytical for both
// RoPE styles.
func TestRoPEBackwardMatchesNumerical(t *testing.T) {
	for _, style := range []RopeStyle{RopeLlama, RopeGPTNeoX} {
		t.Run([]string{"Llama", "GPTNeoX"}[style], func(t *testing.T) {
			const headDim = 8
			r := NewRoPE(headDim, 16, 10000, style)
			x := g.RandN(1, 4, headDim).SetRequiresGrad(true)

			loss := g.Sum(r.Apply(x, 0))
			loss.Backward()
			dxAnalytic := append([]float32{}, x.Grad().Data()...)

			const h = 1e-3
			for i := range x.Data() {
				orig := x.Data()[i]
				x.Data()[i] = orig + h
				yPlus := g.Sum(r.Apply(x, 0)).Data()[0]
				x.Data()[i] = orig - h
				yMinus := g.Sum(r.Apply(x, 0)).Data()[0]
				x.Data()[i] = orig
				num := (yPlus - yMinus) / (2 * h)
				if math.Abs(float64(dxAnalytic[i]-num)) > 1e-2 {
					t.Fatalf("[%d] analytic=%g numeric=%g", i, dxAnalytic[i], num)
				}
			}
		})
	}
}

// TestRoPEDifferentPositionsDiffer: applying RoPE at position 0 vs
// position 5 to the same query vector must produce different output.
// Otherwise positional encoding isn't actually working.
func TestRoPEDifferentPositionsDiffer(t *testing.T) {
	r := NewRoPE(8, 16, 10000, RopeLlama)
	x := g.RandN(1, 1, 8)
	y0 := r.Apply(x, 0)
	y5 := r.Apply(x, 5)
	differs := false
	for i := range y0.Data() {
		if math.Abs(float64(y0.Data()[i]-y5.Data()[i])) > 1e-3 {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("RoPE at pos 0 vs pos 5 produced identical output")
	}
}

func TestRoPEValidatesInput(t *testing.T) {
	cases := []struct {
		name      string
		setup     func()
	}{
		{"odd headDim", func() { NewRoPE(7, 16, 10000, RopeLlama) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatal("expected panic")
				}
			}()
			tc.setup()
		})
	}

	// Apply-time validation: headDim mismatch.
	r := NewRoPE(8, 16, 10000, RopeLlama)
	t.Run("apply with wrong headDim", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic on headDim mismatch")
			}
		}()
		bad := g.RandN(1, 4, 6) // headDim=6 vs RoPE expects 8
		r.Apply(bad, 0)
	})

	// startPos+seqLen > maxSeq.
	t.Run("position out of range", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic on out-of-range pos")
			}
		}()
		x := g.RandN(1, 4, 8)
		r.Apply(x, 14) // 14+4=18 > maxSeq=16
	})
}

// TestRoPEStyleDiffers: Llama vs GPT-NeoX layouts produce different
// outputs given the same input — proves the two paths aren't aliased.
func TestRoPEStyleDiffers(t *testing.T) {
	x := g.RandN(1, 4, 8)
	rL := NewRoPE(8, 16, 10000, RopeLlama)
	rN := NewRoPE(8, 16, 10000, RopeGPTNeoX)
	yL := rL.Apply(x, 0)
	yN := rN.Apply(x, 0)
	differs := false
	for i := range yL.Data() {
		if math.Abs(float64(yL.Data()[i]-yN.Data()[i])) > 1e-3 {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("Llama and GPT-NeoX RoPE styles produced identical output")
	}
}
