//go:build darwin

package gorch

import (
	"testing"
)

func TestBroadcastScalarAdd(t *testing.T) {
	// (3,) + scalar(1,) = (3,)
	a := NewTensor([]float32{1, 2, 3}, 3)
	b := ScalarTensor(10)
	c := AddB(a, b)

	want := []float32{11, 12, 13}
	for i, w := range want {
		if !approx(c.data[i], w) {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestBroadcastMatrixScalarMul(t *testing.T) {
	// (2,3) * scalar(1,) = (2,3)
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := ScalarTensor(2)
	c := MulB(a, b)

	want := []float32{2, 4, 6, 8, 10, 12}
	for i, w := range want {
		if !approx(c.data[i], w) {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestBroadcastMatrixVectorAdd(t *testing.T) {
	// (2,3) + (3,) = (2,3) — add vector to each row
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := NewTensor([]float32{10, 20, 30}, 3)
	c := AddB(a, b)

	want := []float32{11, 22, 33, 14, 25, 36}
	for i, w := range want {
		if !approx(c.data[i], w) {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestBroadcastColumnAdd(t *testing.T) {
	// (2,3) + (2,1) = (2,3) — add column vector to each column
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := NewTensor([]float32{10, 20}, 2, 1)
	c := AddB(a, b)

	want := []float32{11, 12, 13, 24, 25, 26}
	for i, w := range want {
		if !approx(c.data[i], w) {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestBroadcastSameShape(t *testing.T) {
	// Same shape should use fast path
	a := NewTensor([]float32{1, 2, 3}, 3)
	b := NewTensor([]float32{4, 5, 6}, 3)
	c := AddB(a, b)
	want := []float32{5, 7, 9}
	for i, w := range want {
		if c.data[i] != w {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestBroadcastBackward(t *testing.T) {
	// (2,3) + (3,) — gradient of b should be sum over batch dim
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	a.SetRequiresGrad(true)
	b := NewTensor([]float32{10, 20, 30}, 3)
	b.SetRequiresGrad(true)

	c := AddB(a, b)
	loss := Sum(c)
	loss.Backward()

	// a.grad should be all 1s (same shape)
	for i, v := range a.Grad().Data() {
		if !approx(v, 1) {
			t.Fatalf("a.grad[%d] = %f, want 1", i, v)
		}
	}
	// b.grad should be [2, 2, 2] (summed over 2 rows)
	for i, v := range b.Grad().Data() {
		if !approx(v, 2) {
			t.Fatalf("b.grad[%d] = %f, want 2", i, v)
		}
	}
}

func TestBroadcastScalarMulBackward(t *testing.T) {
	a := NewTensor([]float32{2, 3, 4}, 3)
	a.SetRequiresGrad(true)
	s := ScalarTensor(5)
	s.SetRequiresGrad(true)

	c := MulB(a, s)
	loss := Sum(c)
	loss.Backward()

	// d(sum(a*5))/da = [5, 5, 5]
	for i, v := range a.Grad().Data() {
		if !approx(v, 5) {
			t.Fatalf("a.grad[%d] = %f, want 5", i, v)
		}
	}
	// d(sum(a*s))/ds = sum(a) = 9
	if !approx(s.Grad().Data()[0], 9) {
		t.Fatalf("s.grad = %f, want 9", s.Grad().Data()[0])
	}
}

func TestBroadcastSubDiv(t *testing.T) {
	a := NewTensor([]float32{10, 20, 30}, 3)
	b := ScalarTensor(5)
	c := SubB(a, b)
	want := []float32{5, 15, 25}
	for i, w := range want {
		if !approx(c.data[i], w) {
			t.Fatalf("sub[%d] = %f, want %f", i, c.data[i], w)
		}
	}

	d := DivB(a, b)
	want2 := []float32{2, 4, 6}
	for i, w := range want2 {
		if !approx(d.data[i], w) {
			t.Fatalf("div[%d] = %f, want %f", i, d.data[i], w)
		}
	}
}
