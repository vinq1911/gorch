//go:build darwin

package metal

import (
	"math"
	"testing"
)

// Metal shader: element-wise vector addition.
const addKernel = `
#include <metal_stdlib>
using namespace metal;

kernel void vec_add(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    C[id] = A[id] + B[id];
}
`

func TestVectorAdd(t *testing.T) {
	dev, err := NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Release()

	queue := dev.NewCommandQueue()
	defer queue.Release()

	pipe, err := dev.CompileKernel(addKernel, "vec_add")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Release()

	const n = 1024
	bufA := dev.NewBuffer(n * 4)
	bufB := dev.NewBuffer(n * 4)
	bufC := dev.NewBuffer(n * 4)
	defer bufA.Release()
	defer bufB.Release()
	defer bufC.Release()

	// Write input data directly into unified memory.
	a := bufA.FloatSlice()
	b := bufB.FloatSlice()
	for i := 0; i < n; i++ {
		a[i] = float32(i)
		b[i] = float32(i * 2)
	}

	// Dispatch GPU kernel.
	queue.Dispatch1D(pipe, []*Buffer{bufA, bufB, bufC}, n)

	// Read results from the same unified memory — no copy.
	c := bufC.FloatSlice()
	for i := 0; i < n; i++ {
		want := float32(i + i*2)
		if c[i] != want {
			t.Fatalf("c[%d] = %f, want %f", i, c[i], want)
		}
	}
}

func TestMPSMatMul(t *testing.T) {
	dev, err := NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Release()

	queue := dev.NewCommandQueue()
	defer queue.Release()

	// C = A @ B where A is 2x3, B is 3x2, C is 2x2.
	const M, K, N = 2, 3, 2

	bufA := dev.NewBuffer(M * K * 4)
	bufB := dev.NewBuffer(K * N * 4)
	bufC := dev.NewBuffer(M * N * 4)
	defer bufA.Release()
	defer bufB.Release()
	defer bufC.Release()

	// A = [[1,2,3],[4,5,6]]
	a := bufA.FloatSlice()
	a[0], a[1], a[2] = 1, 2, 3
	a[3], a[4], a[5] = 4, 5, 6

	// B = [[7,8],[9,10],[11,12]]
	b := bufB.FloatSlice()
	b[0], b[1] = 7, 8
	b[2], b[3] = 9, 10
	b[4], b[5] = 11, 12

	queue.MatMul(bufA, bufB, bufC, M, N, K)

	c := bufC.FloatSlice()
	// Expected: [[58,64],[139,154]]
	want := []float32{58, 64, 139, 154}
	for i, w := range want {
		if math.Abs(float64(c[i]-w)) > 1e-3 {
			t.Fatalf("c[%d] = %f, want %f", i, c[i], w)
		}
	}
}

func BenchmarkVectorAdd(b *testing.B) {
	dev, _ := NewDevice()
	defer dev.Release()
	queue := dev.NewCommandQueue()
	defer queue.Release()
	pipe, _ := dev.CompileKernel(addKernel, "vec_add")
	defer pipe.Release()

	const n = 1 << 20 // 1M elements
	bufA := dev.NewBuffer(n * 4)
	bufB := dev.NewBuffer(n * 4)
	bufC := dev.NewBuffer(n * 4)
	defer bufA.Release()
	defer bufB.Release()
	defer bufC.Release()

	bufs := []*Buffer{bufA, bufB, bufC}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		queue.Dispatch1D(pipe, bufs, n)
	}
}

func BenchmarkMPSMatMul(b *testing.B) {
	dev, _ := NewDevice()
	defer dev.Release()
	queue := dev.NewCommandQueue()
	defer queue.Release()

	const M, K, N = 512, 512, 512
	bufA := dev.NewBuffer(M * K * 4)
	bufB := dev.NewBuffer(K * N * 4)
	bufC := dev.NewBuffer(M * N * 4)
	defer bufA.Release()
	defer bufB.Release()
	defer bufC.Release()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		queue.MatMul(bufA, bufB, bufC, M, N, K)
	}
}
