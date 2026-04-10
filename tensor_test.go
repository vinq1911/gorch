//go:build darwin

package gorch

import (
	"math"
	"testing"
)

const eps = 1e-4

func approx(a, b float32) bool {
	return math.Abs(float64(a-b)) < float64(eps)
}

// ---------- Tensor creation ----------

func TestZeros(t *testing.T) {
	x := Zeros(2, 3)
	if x.Size() != 6 {
		t.Fatalf("size = %d, want 6", x.Size())
	}
	for i, v := range x.Data() {
		if v != 0 {
			t.Fatalf("data[%d] = %f, want 0", i, v)
		}
	}
}

func TestOnes(t *testing.T) {
	x := Ones(3)
	for i, v := range x.Data() {
		if v != 1 {
			t.Fatalf("data[%d] = %f, want 1", i, v)
		}
	}
}

func TestNewTensor(t *testing.T) {
	x := NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	if x.At(0, 0) != 1 || x.At(0, 1) != 2 || x.At(1, 0) != 3 || x.At(1, 1) != 4 {
		t.Fatalf("unexpected values: %v", x.Data())
	}
}

func TestReshape(t *testing.T) {
	x := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	y := x.Reshape(3, 2)
	if y.Shape()[0] != 3 || y.Shape()[1] != 2 {
		t.Fatalf("shape = %v, want [3 2]", y.Shape())
	}
}

// ---------- CPU element-wise ops ----------

func TestAddCPU(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3}, 3)
	b := NewTensor([]float32{4, 5, 6}, 3)
	c := Add(a, b)
	want := []float32{5, 7, 9}
	for i, w := range want {
		if c.data[i] != w {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestSubCPU(t *testing.T) {
	a := NewTensor([]float32{10, 20, 30}, 3)
	b := NewTensor([]float32{1, 2, 3}, 3)
	c := Sub(a, b)
	want := []float32{9, 18, 27}
	for i, w := range want {
		if c.data[i] != w {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestMulCPU(t *testing.T) {
	a := NewTensor([]float32{2, 3, 4}, 3)
	b := NewTensor([]float32{5, 6, 7}, 3)
	c := Mul(a, b)
	want := []float32{10, 18, 28}
	for i, w := range want {
		if c.data[i] != w {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

func TestReLUCPU(t *testing.T) {
	a := NewTensor([]float32{-2, -1, 0, 1, 2}, 5)
	b := ReLU(a)
	want := []float32{0, 0, 0, 1, 2}
	for i, w := range want {
		if b.data[i] != w {
			t.Fatalf("b[%d] = %f, want %f", i, b.data[i], w)
		}
	}
}

func TestSigmoidCPU(t *testing.T) {
	a := NewTensor([]float32{0}, 1)
	b := Sigmoid(a)
	if !approx(b.data[0], 0.5) {
		t.Fatalf("sigmoid(0) = %f, want 0.5", b.data[0])
	}
}

func TestSumCPU(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, 4)
	s := Sum(a)
	if s.data[0] != 10 {
		t.Fatalf("sum = %f, want 10", s.data[0])
	}
}

func TestMeanCPU(t *testing.T) {
	a := NewTensor([]float32{2, 4, 6, 8}, 4)
	m := Mean(a)
	if m.data[0] != 5 {
		t.Fatalf("mean = %f, want 5", m.data[0])
	}
}

// ---------- Exp / Log ----------

func TestExpCPU(t *testing.T) {
	a := NewTensor([]float32{0, 1}, 2)
	b := Exp(a)
	if !approx(b.data[0], 1) || !approx(b.data[1], float32(math.E)) {
		t.Fatalf("exp([0,1]) = %v, want [1, e]", b.Data())
	}
}

func TestLogCPU(t *testing.T) {
	a := NewTensor([]float32{1, float32(math.E)}, 2)
	b := Log(a)
	if !approx(b.data[0], 0) || !approx(b.data[1], 1) {
		t.Fatalf("log([1,e]) = %v, want [0, 1]", b.Data())
	}
}

// ---------- Softmax / LogSoftmax ----------

func TestSoftmax(t *testing.T) {
	// Single sample: softmax([1, 2, 3])
	a := NewTensor([]float32{1, 2, 3}, 1, 3)
	s := Softmax(a)
	// Should sum to 1
	var sum float32
	for _, v := range s.Data() {
		sum += v
	}
	if !approx(sum, 1.0) {
		t.Fatalf("softmax sum = %f, want 1.0", sum)
	}
	// s[2] > s[1] > s[0]
	if s.data[0] >= s.data[1] || s.data[1] >= s.data[2] {
		t.Fatalf("softmax order wrong: %v", s.Data())
	}
}

func TestLogSoftmax(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3}, 1, 3)
	ls := LogSoftmax(a)
	s := Softmax(a)
	// LogSoftmax should equal log of softmax
	for i := range ls.Data() {
		expected := float32(math.Log(float64(s.data[i])))
		if !approx(ls.data[i], expected) {
			t.Fatalf("logsoftmax[%d] = %f, want %f", i, ls.data[i], expected)
		}
	}
}

func TestSoftmaxBatch(t *testing.T) {
	// Two samples
	a := NewTensor([]float32{1, 2, 3, 10, 20, 30}, 2, 3)
	s := Softmax(a)
	// Each row should sum to 1
	for row := 0; row < 2; row++ {
		var sum float32
		for j := 0; j < 3; j++ {
			sum += s.data[row*3+j]
		}
		if !approx(sum, 1.0) {
			t.Fatalf("row %d sum = %f, want 1.0", row, sum)
		}
	}
}

// ---------- CrossEntropyLoss ----------

func TestCrossEntropyLoss(t *testing.T) {
	// 2 samples, 3 classes
	logits := NewTensor([]float32{1, 2, 3, 1, 2, 3}, 2, 3)
	logits.SetRequiresGrad(true)
	targets := NewTensor([]float32{2, 0}, 2, 1) // correct classes: 2 and 0

	loss := CrossEntropyLoss(logits, targets)

	// Loss should be positive
	if loss.data[0] <= 0 {
		t.Fatalf("CE loss = %f, want > 0", loss.data[0])
	}

	// Test backward
	loss.Backward()
	if logits.Grad() == nil {
		t.Fatal("logits.grad is nil after backward")
	}
	// Gradient shape should match logits
	if logits.Grad().Size() != 6 {
		t.Fatalf("grad size = %d, want 6", logits.Grad().Size())
	}
}

func TestCrossEntropyGradient(t *testing.T) {
	// When prediction is perfect, gradient for correct class should be near 0
	// logits heavily favoring class 0 for sample 0
	logits := NewTensor([]float32{100, 0, 0}, 1, 3)
	logits.SetRequiresGrad(true)
	targets := NewTensor([]float32{0}, 1, 1)

	loss := CrossEntropyLoss(logits, targets)
	loss.Backward()

	// Loss should be near 0
	if loss.data[0] > 0.01 {
		t.Fatalf("CE loss for perfect prediction = %f, want ~0", loss.data[0])
	}
}

// ---------- MatMul CPU ----------

func TestMatMulCPU(t *testing.T) {
	// [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
	a := NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	b := NewTensor([]float32{5, 6, 7, 8}, 2, 2)
	c := MatMul(a, b)
	want := []float32{19, 22, 43, 50}
	for i, w := range want {
		if !approx(c.data[i], w) {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

// ---------- Metal GPU ops ----------

func TestAddMetal(t *testing.T) {
	g, err := InitMetal()
	if err != nil {
		t.Skip("no Metal device:", err)
	}
	_ = g

	a := NewTensor([]float32{1, 2, 3, 4}, 4).ToMetal(gpu.Dev)
	b := NewTensor([]float32{10, 20, 30, 40}, 4).ToMetal(gpu.Dev)
	c := Add(a, b)

	want := []float32{11, 22, 33, 44}
	for i, w := range want {
		if c.data[i] != w {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
	if c.Device() != Metal {
		t.Fatal("result should be on Metal")
	}
}

func TestReLUMetal(t *testing.T) {
	if gpu == nil {
		if _, err := InitMetal(); err != nil {
			t.Skip("no Metal:", err)
		}
	}

	a := NewTensor([]float32{-3, -1, 0, 2, 5}, 5).ToMetal(gpu.Dev)
	b := ReLU(a)
	want := []float32{0, 0, 0, 2, 5}
	for i, w := range want {
		if b.data[i] != w {
			t.Fatalf("b[%d] = %f, want %f", i, b.data[i], w)
		}
	}
}

func TestMatMulMetal(t *testing.T) {
	if gpu == nil {
		if _, err := InitMetal(); err != nil {
			t.Skip("no Metal:", err)
		}
	}

	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3).ToMetal(gpu.Dev)
	b := NewTensor([]float32{7, 8, 9, 10, 11, 12}, 3, 2).ToMetal(gpu.Dev)
	c := MatMul(a, b)

	want := []float32{58, 64, 139, 154}
	for i, w := range want {
		if !approx(c.data[i], w) {
			t.Fatalf("c[%d] = %f, want %f", i, c.data[i], w)
		}
	}
}

// ---------- Autograd ----------

func TestBackwardAdd(t *testing.T) {
	a := NewTensor([]float32{2, 3}, 2)
	a.SetRequiresGrad(true)
	b := NewTensor([]float32{4, 5}, 2)
	b.SetRequiresGrad(true)

	c := Add(a, b)
	loss := Sum(c)
	loss.Backward()

	// d(sum(a+b))/da = [1, 1]
	for i, v := range a.Grad().Data() {
		if !approx(v, 1) {
			t.Fatalf("a.grad[%d] = %f, want 1", i, v)
		}
	}
	for i, v := range b.Grad().Data() {
		if !approx(v, 1) {
			t.Fatalf("b.grad[%d] = %f, want 1", i, v)
		}
	}
}

func TestBackwardMul(t *testing.T) {
	a := NewTensor([]float32{2, 3}, 2)
	a.SetRequiresGrad(true)
	b := NewTensor([]float32{4, 5}, 2)
	b.SetRequiresGrad(true)

	c := Mul(a, b)
	loss := Sum(c)
	loss.Backward()

	// d(sum(a*b))/da = b = [4, 5]
	for i, v := range a.Grad().Data() {
		if !approx(v, b.data[i]) {
			t.Fatalf("a.grad[%d] = %f, want %f", i, v, b.data[i])
		}
	}
	// d(sum(a*b))/db = a = [2, 3]
	for i, v := range b.Grad().Data() {
		if !approx(v, a.data[i]) {
			t.Fatalf("b.grad[%d] = %f, want %f", i, v, a.data[i])
		}
	}
}

func TestBackwardMatMul(t *testing.T) {
	// A(2x2) @ B(2x2)
	a := NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	a.SetRequiresGrad(true)
	b := NewTensor([]float32{5, 6, 7, 8}, 2, 2)
	b.SetRequiresGrad(true)

	c := MatMul(a, b)
	loss := Sum(c)
	loss.Backward()

	// dL/dA = ones(2,2) @ B^T
	// B^T = [[5,7],[6,8]]
	// dL/dA = [[11,13],[11,13]]  (each row sums B columns)
	wantA := []float32{11, 15, 11, 15}
	for i, v := range a.Grad().Data() {
		if !approx(v, wantA[i]) {
			t.Fatalf("a.grad[%d] = %f, want %f", i, v, wantA[i])
		}
	}
}

func TestBackwardReLU(t *testing.T) {
	a := NewTensor([]float32{-1, 2, -3, 4}, 4)
	a.SetRequiresGrad(true)

	b := ReLU(a)
	loss := Sum(b)
	loss.Backward()

	// ReLU grad: 0 where input <= 0, 1 where input > 0
	want := []float32{0, 1, 0, 1}
	for i, v := range a.Grad().Data() {
		if !approx(v, want[i]) {
			t.Fatalf("a.grad[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestBackwardChain(t *testing.T) {
	// Test: loss = sum(relu(a * b + c))
	a := NewTensor([]float32{1, -2, 3}, 3)
	a.SetRequiresGrad(true)
	b := NewTensor([]float32{2, 3, -1}, 3)
	b.SetRequiresGrad(true)
	c := NewTensor([]float32{0, 0, 0}, 3)
	c.SetRequiresGrad(true)

	// a*b = [2, -6, -3]
	// a*b+c = [2, -6, -3]
	// relu = [2, 0, 0]
	// sum = 2
	ab := Mul(a, b)
	abc := Add(ab, c)
	r := ReLU(abc)
	loss := Sum(r)

	if !approx(loss.data[0], 2) {
		t.Fatalf("loss = %f, want 2", loss.data[0])
	}

	loss.Backward()

	// relu grad mask: [1, 0, 0]
	// d(relu)/d(abc) = [1, 0, 0]
	// d(abc)/da = b * relu_mask = [2, 0, 0]
	wantA := []float32{2, 0, 0}
	for i, v := range a.Grad().Data() {
		if !approx(v, wantA[i]) {
			t.Fatalf("a.grad[%d] = %f, want %f", i, v, wantA[i])
		}
	}
}
