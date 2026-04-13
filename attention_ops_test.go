//go:build darwin

package gorch

import (
	"math"
	"testing"
)

func TestReshapeOp(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := ReshapeOp(a, 3, 2)
	if b.Shape()[0] != 3 || b.Shape()[1] != 2 {
		t.Fatalf("shape = %v, want [3,2]", b.Shape())
	}
	// Data is shared
	if b.Data()[0] != 1 || b.Data()[5] != 6 {
		t.Fatal("data not preserved")
	}
}

func TestReshapeOpBackward(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	a.SetRequiresGrad(true)
	b := ReshapeOp(a, 6)
	loss := Sum(b)
	loss.Backward()
	if a.Grad() == nil {
		t.Fatal("grad is nil")
	}
	// Grad shape should match original
	gs := a.Grad().Shape()
	if gs[0] != 2 || gs[1] != 3 {
		t.Fatalf("grad shape = %v, want [2,3]", gs)
	}
}

func TestTranspose2D(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := Transpose2D(a)
	if b.Shape()[0] != 3 || b.Shape()[1] != 2 {
		t.Fatalf("shape = %v, want [3,2]", b.Shape())
	}
	// [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
	want := []float32{1, 4, 2, 5, 3, 6}
	for i, w := range want {
		if b.Data()[i] != w {
			t.Fatalf("b[%d] = %f, want %f", i, b.Data()[i], w)
		}
	}
}

func TestTranspose2DBackward(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	a.SetRequiresGrad(true)
	b := Transpose2D(a)
	loss := Sum(b)
	loss.Backward()
	// All gradients should be 1
	for i, v := range a.Grad().Data() {
		if !approx(v, 1) {
			t.Fatalf("grad[%d] = %f, want 1", i, v)
		}
	}
}

func TestAddBias(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	bias := NewTensor([]float32{10, 20, 30}, 3)
	c := AddBias(a, bias)

	want := []float32{11, 22, 33, 14, 25, 36}
	for i, w := range want {
		if c.Data()[i] != w {
			t.Fatalf("c[%d] = %f, want %f", i, c.Data()[i], w)
		}
	}
}

func TestAddBiasBackward(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	a.SetRequiresGrad(true)
	bias := NewTensor([]float32{10, 20, 30}, 3)
	bias.SetRequiresGrad(true)

	c := AddBias(a, bias)
	loss := Sum(c)
	loss.Backward()

	// bias grad = sum over rows = [1+1, 1+1, 1+1] = [2, 2, 2]
	for i, v := range bias.Grad().Data() {
		if !approx(v, 2) {
			t.Fatalf("bias.grad[%d] = %f, want 2", i, v)
		}
	}
}

func TestMaskFill(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	mask := CausalMask(3)
	// mask should be: F F F / F F F / ... no wait, upper triangular
	// mask[0*3+1]=true, mask[0*3+2]=true, mask[1*3+2]=true
	b := MaskFill(a, mask, float32(math.Inf(-1)))

	// Diagonal and below should be unchanged
	if b.Data()[0] != 1 || b.Data()[3] != 4 || b.Data()[4] != 5 {
		t.Fatal("non-masked values changed")
	}
	// Above diagonal should be -inf
	if !math.IsInf(float64(b.Data()[1]), -1) {
		t.Fatalf("b[0,1] = %f, want -inf", b.Data()[1])
	}
}

func TestCausalMask(t *testing.T) {
	mask := CausalMask(4)
	// 4x4: upper triangle should be true
	// Row 0: F T T T
	// Row 1: F F T T
	// Row 2: F F F T
	// Row 3: F F F F
	if mask[0] != false || mask[1] != true || mask[2] != true || mask[3] != true {
		t.Fatal("row 0 wrong")
	}
	if mask[4] != false || mask[5] != false || mask[6] != true || mask[7] != true {
		t.Fatal("row 1 wrong")
	}
	if mask[12] != false || mask[13] != false || mask[14] != false || mask[15] != false {
		t.Fatal("row 3 wrong")
	}
}

func TestEmbeddingLookup(t *testing.T) {
	// Vocab of 5, dim 3
	weight := NewTensor([]float32{
		0, 1, 2, // word 0
		3, 4, 5, // word 1
		6, 7, 8, // word 2
		9, 10, 11, // word 3
		12, 13, 14, // word 4
	}, 5, 3)

	out := EmbeddingLookup(weight, []int{2, 0, 4})
	// Should be: [6,7,8, 0,1,2, 12,13,14]
	want := []float32{6, 7, 8, 0, 1, 2, 12, 13, 14}
	for i, w := range want {
		if out.Data()[i] != w {
			t.Fatalf("out[%d] = %f, want %f", i, out.Data()[i], w)
		}
	}
	if out.Shape()[0] != 3 || out.Shape()[1] != 3 {
		t.Fatalf("shape = %v, want [3,3]", out.Shape())
	}
}

func TestEmbeddingLookupBackward(t *testing.T) {
	weight := NewTensor([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8}, 3, 3)
	weight.SetRequiresGrad(true)

	out := EmbeddingLookup(weight, []int{0, 2, 0}) // word 0 used twice
	loss := Sum(out)
	loss.Backward()

	// Word 0 gradient should be 2 (used twice), word 2 should be 1, word 1 should be 0
	g := weight.Grad().Data()
	if !approx(g[0], 2) || !approx(g[1], 2) || !approx(g[2], 2) {
		t.Fatalf("word 0 grad = [%f,%f,%f], want [2,2,2]", g[0], g[1], g[2])
	}
	if !approx(g[3], 0) {
		t.Fatalf("word 1 grad = %f, want 0", g[3])
	}
	if !approx(g[6], 1) {
		t.Fatalf("word 2 grad = %f, want 1", g[6])
	}
}

func TestScaledMatMul(t *testing.T) {
	a := NewTensor([]float32{1, 0, 0, 1}, 2, 2) // identity
	b := NewTensor([]float32{4, 0, 0, 4}, 2, 2)
	out := ScaledMatMul(a, b, 4.0) // divide by sqrt(4)=2

	// identity @ diag(4) / 2 = diag(2)
	want := []float32{2, 0, 0, 2}
	for i, w := range want {
		if !approx(out.Data()[i], w) {
			t.Fatalf("out[%d] = %f, want %f", i, out.Data()[i], w)
		}
	}
}
