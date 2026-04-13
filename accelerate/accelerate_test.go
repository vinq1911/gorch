//go:build darwin

package accelerate

import (
	"math"
	"testing"
)

const eps = 1e-4

func approx(a, b float32) bool {
	return math.Abs(float64(a-b)) < float64(eps)
}

func TestSgemm(t *testing.T) {
	// [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
	A := []float32{1, 2, 3, 4}
	B := []float32{5, 6, 7, 8}
	C := make([]float32, 4)
	Sgemm(2, 2, 2, 1.0, A, B, 0.0, C)
	want := []float32{19, 22, 43, 50}
	for i, w := range want {
		if !approx(C[i], w) {
			t.Fatalf("C[%d] = %f, want %f", i, C[i], w)
		}
	}
}

func TestVAdd(t *testing.T) {
	A := []float32{1, 2, 3}
	B := []float32{4, 5, 6}
	C := make([]float32, 3)
	VAdd(A, B, C)
	want := []float32{5, 7, 9}
	for i, w := range want {
		if C[i] != w {
			t.Fatalf("C[%d] = %f, want %f", i, C[i], w)
		}
	}
}

func TestVSub(t *testing.T) {
	A := []float32{10, 20, 30}
	B := []float32{1, 2, 3}
	C := make([]float32, 3)
	VSub(A, B, C)
	want := []float32{9, 18, 27}
	for i, w := range want {
		if C[i] != w {
			t.Fatalf("C[%d] = %f, want %f", i, C[i], w)
		}
	}
}

func TestVMul(t *testing.T) {
	A := []float32{2, 3, 4}
	B := []float32{5, 6, 7}
	C := make([]float32, 3)
	VMul(A, B, C)
	want := []float32{10, 18, 28}
	for i, w := range want {
		if C[i] != w {
			t.Fatalf("C[%d] = %f, want %f", i, C[i], w)
		}
	}
}

func TestVDiv(t *testing.T) {
	A := []float32{10, 20, 30}
	B := []float32{2, 4, 5}
	C := make([]float32, 3)
	VDiv(A, B, C)
	want := []float32{5, 5, 6}
	for i, w := range want {
		if C[i] != w {
			t.Fatalf("C[%d] = %f, want %f", i, C[i], w)
		}
	}
}

func TestSum(t *testing.T) {
	A := []float32{1, 2, 3, 4}
	s := Sum(A)
	if !approx(s, 10) {
		t.Fatalf("sum = %f, want 10", s)
	}
}

func TestReLU(t *testing.T) {
	A := []float32{-2, -1, 0, 1, 2}
	C := make([]float32, 5)
	ReLU(A, C)
	want := []float32{0, 0, 0, 1, 2}
	for i, w := range want {
		if C[i] != w {
			t.Fatalf("C[%d] = %f, want %f", i, C[i], w)
		}
	}
}

func TestSigmoid(t *testing.T) {
	A := []float32{0}
	C := make([]float32, 1)
	Sigmoid(A, C)
	if !approx(C[0], 0.5) {
		t.Fatalf("sigmoid(0) = %f, want 0.5", C[0])
	}
}

func TestTanh(t *testing.T) {
	A := []float32{0}
	C := make([]float32, 1)
	Tanh(A, C)
	if !approx(C[0], 0) {
		t.Fatalf("tanh(0) = %f, want 0", C[0])
	}
}

func TestExp(t *testing.T) {
	A := []float32{0, 1}
	C := make([]float32, 2)
	Exp(A, C)
	if !approx(C[0], 1) || !approx(C[1], float32(math.E)) {
		t.Fatalf("exp = %v, want [1, e]", C)
	}
}

// ---------- Benchmarks ----------

func BenchmarkSgemm128(b *testing.B) {
	const n = 128
	A := make([]float32, n*n)
	B := make([]float32, n*n)
	C := make([]float32, n*n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sgemm(n, n, n, 1.0, A, B, 0.0, C)
	}
}

func BenchmarkSgemm512(b *testing.B) {
	const n = 512
	A := make([]float32, n*n)
	B := make([]float32, n*n)
	C := make([]float32, n*n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sgemm(n, n, n, 1.0, A, B, 0.0, C)
	}
}

func BenchmarkNaiveMatmul512(b *testing.B) {
	const n = 512
	A := make([]float32, n*n)
	B := make([]float32, n*n)
	C := make([]float32, n*n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for ii := 0; ii < n; ii++ {
			for jj := 0; jj < n; jj++ {
				var s float32
				for kk := 0; kk < n; kk++ {
					s += A[ii*n+kk] * B[kk*n+jj]
				}
				C[ii*n+jj] = s
			}
		}
	}
}

func BenchmarkVAdd1M(b *testing.B) {
	const n = 1 << 20
	A := make([]float32, n)
	B := make([]float32, n)
	C := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VAdd(A, B, C)
	}
}

func BenchmarkNaiveVAdd1M(b *testing.B) {
	const n = 1 << 20
	A := make([]float32, n)
	B := make([]float32, n)
	C := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < n; j++ {
			C[j] = A[j] + B[j]
		}
	}
}

func BenchmarkSigmoid100K(b *testing.B) {
	const n = 100_000
	A := make([]float32, n)
	C := make([]float32, n)
	for i := range A {
		A[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sigmoid(A, C)
	}
}
