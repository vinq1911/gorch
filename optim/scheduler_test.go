//go:build darwin

package optim

import (
	"math"
	"testing"
)

func approx(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-3
}

func TestStepLR(t *testing.T) {
	var currentLR float32 = 0.1
	setLR := func(lr float32) { currentLR = lr }

	sched := NewStepLR(nil, 0.1, 5, 0.5, setLR)

	// Epochs 1-4: LR should stay at 0.1
	for i := 0; i < 4; i++ {
		sched.Step()
		if !approx(sched.GetLR(), 0.1) {
			t.Fatalf("epoch %d: LR = %f, want 0.1", i+1, sched.GetLR())
		}
	}

	// Epoch 5: LR should drop to 0.05
	sched.Step()
	if !approx(sched.GetLR(), 0.05) {
		t.Fatalf("epoch 5: LR = %f, want 0.05", sched.GetLR())
	}

	// Epoch 10: LR should drop to 0.025
	for i := 0; i < 5; i++ {
		sched.Step()
	}
	if !approx(sched.GetLR(), 0.025) {
		t.Fatalf("epoch 10: LR = %f, want 0.025", sched.GetLR())
	}

	if !approx(currentLR, sched.GetLR()) {
		t.Fatal("setLR was not called correctly")
	}
}

func TestCosineAnnealingLR(t *testing.T) {
	var currentLR float32
	setLR := func(lr float32) { currentLR = lr }

	sched := NewCosineAnnealingLR(nil, 0.1, 0.0, 100, setLR)

	// At start, LR should be near baseLR
	sched.Step()
	if sched.GetLR() < 0.09 {
		t.Fatalf("step 1: LR = %f, want near 0.1", sched.GetLR())
	}

	// At midpoint, LR should be ~0.05
	for i := 0; i < 49; i++ {
		sched.Step()
	}
	if !approx(sched.GetLR(), 0.05) {
		t.Fatalf("step 50: LR = %f, want ~0.05", sched.GetLR())
	}

	// At end, LR should be near minLR (0)
	for i := 0; i < 50; i++ {
		sched.Step()
	}
	if sched.GetLR() > 0.01 {
		t.Fatalf("step 100: LR = %f, want near 0.0", sched.GetLR())
	}

	if !approx(currentLR, sched.GetLR()) {
		t.Fatal("setLR callback not invoked")
	}
}

func TestWarmupCosineScheduler(t *testing.T) {
	var currentLR float32
	setLR := func(lr float32) { currentLR = lr }

	sched := NewWarmupCosineScheduler(0.1, 0.0, 10, 100, setLR)

	// During warmup (steps 1-10), LR should increase linearly
	sched.Step() // step 1
	if !approx(sched.GetLR(), 0.01) {
		t.Fatalf("step 1: LR = %f, want 0.01", sched.GetLR())
	}

	for i := 0; i < 4; i++ {
		sched.Step()
	}
	// step 5: 0.1 * 5/10 = 0.05
	if !approx(sched.GetLR(), 0.05) {
		t.Fatalf("step 5: LR = %f, want 0.05", sched.GetLR())
	}

	for i := 0; i < 5; i++ {
		sched.Step()
	}
	// step 10: 0.1 * 10/10 = 0.1 (peak)
	if !approx(sched.GetLR(), 0.1) {
		t.Fatalf("step 10: LR = %f, want 0.1", sched.GetLR())
	}

	// After warmup, should decay via cosine
	for i := 0; i < 90; i++ {
		sched.Step()
	}
	// step 100: should be near minLR
	if sched.GetLR() > 0.01 {
		t.Fatalf("step 100: LR = %f, want near 0", sched.GetLR())
	}

	_ = currentLR
}
