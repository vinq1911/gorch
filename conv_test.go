//go:build darwin

package gorch

import (
	"testing"
)

func TestIm2col(t *testing.T) {
	// 1 channel, 3x3 input, 2x2 kernel, stride=1, pad=0
	// Input:
	// 1 2 3
	// 4 5 6
	// 7 8 9
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	// Output should be 4 columns (2x2 output), each with 4 values (1*2*2 kernel)
	// col shape: (C*kH*kW, outH*outW) = (4, 4)
	col := make([]float32, 4*4)
	im2col(input, 1, 3, 3, 2, 2, 1, 0, col)

	// Expected columns (each column is a 2x2 patch):
	// patch(0,0): 1,2,4,5  patch(0,1): 2,3,5,6  patch(1,0): 4,5,7,8  patch(1,1): 5,6,8,9
	// But im2col stores by kernel element across spatial positions:
	// Row 0 (ky=0,kx=0): 1,2,4,5
	// Row 1 (ky=0,kx=1): 2,3,5,6
	// Row 2 (ky=1,kx=0): 4,5,7,8
	// Row 3 (ky=1,kx=1): 5,6,8,9
	want := []float32{
		1, 2, 4, 5,
		2, 3, 5, 6,
		4, 5, 7, 8,
		5, 6, 8, 9,
	}
	for i, w := range want {
		if col[i] != w {
			t.Fatalf("col[%d] = %f, want %f", i, col[i], w)
		}
	}
}

func TestConv2dForwardBasic(t *testing.T) {
	// 1 sample, 1 channel, 3x3 input, 1 filter of 2x2, stride=1, pad=0
	input := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)

	// Weight: all ones (1 output channel, 1 input channel, 2x2)
	weight := NewTensor([]float32{1, 1, 1, 1}, 1, 1, 2, 2)

	out := Conv2dForward(input, weight, nil, 1, 0)

	// Output should be 2x2:
	// (1+2+4+5)=12  (2+3+5+6)=16
	// (4+5+7+8)=24  (5+6+8+9)=28
	if out.shape[0] != 1 || out.shape[1] != 1 || out.shape[2] != 2 || out.shape[3] != 2 {
		t.Fatalf("shape = %v, want [1,1,2,2]", out.shape)
	}
	want := []float32{12, 16, 24, 28}
	for i, w := range want {
		if !approx(out.data[i], w) {
			t.Fatalf("out[%d] = %f, want %f", i, out.data[i], w)
		}
	}
}

func TestConv2dForwardWithBias(t *testing.T) {
	input := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	weight := NewTensor([]float32{1, 1, 1, 1}, 1, 1, 2, 2)
	bias := NewTensor([]float32{10}, 1)

	out := Conv2dForward(input, weight, bias, 1, 0)

	// Same as above + 10
	want := []float32{22, 26, 34, 38}
	for i, w := range want {
		if !approx(out.data[i], w) {
			t.Fatalf("out[%d] = %f, want %f", i, out.data[i], w)
		}
	}
}

func TestConv2dForwardPadding(t *testing.T) {
	// 1 sample, 1 channel, 3x3 input, 3x3 kernel, stride=1, pad=1
	// Output should be same size: 3x3
	input := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	weight := Ones(1, 1, 3, 3) // all-ones 3x3 kernel

	out := Conv2dForward(input, weight, nil, 1, 1)

	if out.shape[2] != 3 || out.shape[3] != 3 {
		t.Fatalf("padded output shape = %v, want [1,1,3,3]", out.shape)
	}
	// Center pixel: sum of all = 45
	if !approx(out.data[4], 45) {
		t.Fatalf("center = %f, want 45", out.data[4])
	}
}

func TestConv2d1x1(t *testing.T) {
	// 1x1 conv should behave as per-pixel linear transform
	// 2 input channels, 3 output channels, 1x1 kernel
	input := NewTensor([]float32{
		// channel 0: 2x2
		1, 2, 3, 4,
		// channel 1: 2x2
		5, 6, 7, 8,
	}, 1, 2, 2, 2)

	// Weight: (3, 2, 1, 1) — 3 output channels, each takes 2 inputs
	weight := NewTensor([]float32{
		1, 0, // out0 = 1*in0 + 0*in1
		0, 1, // out1 = 0*in0 + 1*in1
		1, 1, // out2 = in0 + in1
	}, 3, 2, 1, 1)

	out := Conv2dForward(input, weight, nil, 1, 0)

	if out.shape[1] != 3 || out.shape[2] != 2 || out.shape[3] != 2 {
		t.Fatalf("shape = %v, want [1,3,2,2]", out.shape)
	}
	// out0 = input channel 0: [1,2,3,4]
	// out1 = input channel 1: [5,6,7,8]
	// out2 = sum: [6,8,10,12]
	wantOut2 := []float32{6, 8, 10, 12}
	out2 := out.data[8:12] // 3rd output channel
	for i, w := range wantOut2 {
		if !approx(out2[i], w) {
			t.Fatalf("out2[%d] = %f, want %f", i, out2[i], w)
		}
	}
}

func TestConv2dBackward(t *testing.T) {
	input := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	input.SetRequiresGrad(true)

	weight := NewTensor([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	weight.SetRequiresGrad(true)

	out := Conv2dForward(input, weight, nil, 1, 0)
	loss := Sum(out)
	loss.Backward()

	// Weight grad should exist and have correct shape
	if weight.Grad() == nil {
		t.Fatal("weight.grad is nil")
	}
	if weight.Grad().Size() != 4 {
		t.Fatalf("weight.grad size = %d, want 4", weight.Grad().Size())
	}

	// Input grad should exist
	if input.Grad() == nil {
		t.Fatal("input.grad is nil")
	}
	if input.Grad().Size() != 9 {
		t.Fatalf("input.grad size = %d, want 9", input.Grad().Size())
	}
}

func TestMaxPool2d(t *testing.T) {
	// 1 sample, 1 channel, 4x4 input, kernel=2, stride=2
	input := NewTensor([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4)

	out := MaxPool2dForward(input, 2, 2)

	if out.shape[2] != 2 || out.shape[3] != 2 {
		t.Fatalf("shape = %v, want [1,1,2,2]", out.shape)
	}
	// max of each 2x2 block: 6, 8, 14, 16
	want := []float32{6, 8, 14, 16}
	for i, w := range want {
		if out.data[i] != w {
			t.Fatalf("out[%d] = %f, want %f", i, out.data[i], w)
		}
	}
}

func TestMaxPool2dBackward(t *testing.T) {
	input := NewTensor([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4)
	input.SetRequiresGrad(true)

	out := MaxPool2dForward(input, 2, 2)
	loss := Sum(out)
	loss.Backward()

	grad := input.Grad().Data()
	// Gradient should be 1 at max positions (indices 5,7,13,15), 0 elsewhere
	for i, v := range grad {
		if i == 5 || i == 7 || i == 13 || i == 15 {
			if !approx(v, 1) {
				t.Fatalf("grad[%d] = %f, want 1 (max position)", i, v)
			}
		} else {
			if !approx(v, 0) {
				t.Fatalf("grad[%d] = %f, want 0 (non-max position)", i, v)
			}
		}
	}
}

func TestFlatten(t *testing.T) {
	input := NewTensor(make([]float32, 2*3*4*4), 2, 3, 4, 4)
	out := FlattenForward(input)
	if out.shape[0] != 2 || out.shape[1] != 48 {
		t.Fatalf("shape = %v, want [2, 48]", out.shape)
	}
}

func TestFlattenBackward(t *testing.T) {
	input := Ones(2, 3, 4, 4)
	input.SetRequiresGrad(true)
	out := FlattenForward(input)
	loss := Sum(out)
	loss.Backward()

	if input.Grad() == nil {
		t.Fatal("input.grad is nil")
	}
	// Gradient shape should match original 4D shape
	gs := input.Grad().Shape()
	if gs[0] != 2 || gs[1] != 3 || gs[2] != 4 || gs[3] != 4 {
		t.Fatalf("grad shape = %v, want [2,3,4,4]", gs)
	}
}

func TestCNNPipeline(t *testing.T) {
	// Test full pipeline: Conv2d -> ReLU -> MaxPool -> Flatten -> Linear
	// 1 sample, 1 channel, 8x8 input
	input := Rand(1, 1, 8, 8)
	input.SetRequiresGrad(true)

	// Conv: 1->4 channels, 3x3, pad=1 => (1,4,8,8)
	w := RandN(4, 1, 3, 3)
	w.SetRequiresGrad(true)
	b := Zeros(4)
	b.SetRequiresGrad(true)

	conv := Conv2dForward(input, w, b, 1, 1)   // (1,4,8,8)
	act := ReLU(conv)                            // (1,4,8,8)
	pooled := MaxPool2dForward(act, 2, 2)        // (1,4,4,4)
	flat := FlattenForward(pooled)               // (1,64)

	if flat.shape[0] != 1 || flat.shape[1] != 64 {
		t.Fatalf("flat shape = %v, want [1,64]", flat.shape)
	}

	loss := Sum(flat)
	loss.Backward()

	if w.Grad() == nil {
		t.Fatal("conv weight grad is nil after backward")
	}
	if input.Grad() == nil {
		t.Fatal("input grad is nil after backward")
	}
}
