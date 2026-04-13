//go:build darwin

package optim

import "math"

// LRScheduler adjusts the learning rate during training.
type LRScheduler interface {
	Step()
	GetLR() float32
}

// StepLR decays the learning rate by gamma every stepSize epochs.
type StepLR struct {
	optimizer Optimizer
	baseLR    float32
	stepSize  int
	gamma     float32
	epoch     int
	currentLR float32
	setLR     func(float32) // function to set LR on the optimizer
}

// NewStepLR creates a step-decay learning rate scheduler.
// Decays LR by gamma every stepSize epochs.
func NewStepLR(opt Optimizer, baseLR float32, stepSize int, gamma float32, setLR func(float32)) *StepLR {
	return &StepLR{
		optimizer: opt,
		baseLR:    baseLR,
		stepSize:  stepSize,
		gamma:     gamma,
		currentLR: baseLR,
		setLR:     setLR,
	}
}

func (s *StepLR) Step() {
	s.epoch++
	newLR := s.baseLR * float32(math.Pow(float64(s.gamma), float64(s.epoch/s.stepSize)))
	s.currentLR = newLR
	s.setLR(newLR)
}

func (s *StepLR) GetLR() float32 { return s.currentLR }

// CosineAnnealingLR reduces the learning rate following a cosine curve.
type CosineAnnealingLR struct {
	optimizer Optimizer
	baseLR    float32
	minLR     float32
	totalEpochs int
	epoch     int
	currentLR float32
	setLR     func(float32)
}

// NewCosineAnnealingLR creates a cosine annealing scheduler.
// LR decays from baseLR to minLR over totalEpochs following cos(pi * epoch / totalEpochs).
func NewCosineAnnealingLR(opt Optimizer, baseLR, minLR float32, totalEpochs int, setLR func(float32)) *CosineAnnealingLR {
	return &CosineAnnealingLR{
		optimizer:   opt,
		baseLR:      baseLR,
		minLR:       minLR,
		totalEpochs: totalEpochs,
		currentLR:   baseLR,
		setLR:       setLR,
	}
}

func (c *CosineAnnealingLR) Step() {
	c.epoch++
	progress := float64(c.epoch) / float64(c.totalEpochs)
	if progress > 1 {
		progress = 1
	}
	cosVal := (1 + math.Cos(math.Pi*progress)) / 2
	c.currentLR = c.minLR + (c.baseLR-c.minLR)*float32(cosVal)
	c.setLR(c.currentLR)
}

func (c *CosineAnnealingLR) GetLR() float32 { return c.currentLR }

// WarmupCosineScheduler combines linear warmup with cosine decay.
type WarmupCosineScheduler struct {
	baseLR       float32
	minLR        float32
	warmupSteps  int
	totalSteps   int
	step_        int
	currentLR    float32
	setLR        func(float32)
}

// NewWarmupCosineScheduler creates a warmup + cosine decay scheduler.
func NewWarmupCosineScheduler(baseLR, minLR float32, warmupSteps, totalSteps int, setLR func(float32)) *WarmupCosineScheduler {
	return &WarmupCosineScheduler{
		baseLR:      baseLR,
		minLR:       minLR,
		warmupSteps: warmupSteps,
		totalSteps:  totalSteps,
		currentLR:   0, // starts at 0 during warmup
		setLR:       setLR,
	}
}

func (w *WarmupCosineScheduler) Step() {
	w.step_++
	if w.step_ <= w.warmupSteps {
		// Linear warmup
		w.currentLR = w.baseLR * float32(w.step_) / float32(w.warmupSteps)
	} else {
		// Cosine decay
		progress := float64(w.step_-w.warmupSteps) / float64(w.totalSteps-w.warmupSteps)
		if progress > 1 {
			progress = 1
		}
		cosVal := (1 + math.Cos(math.Pi*progress)) / 2
		w.currentLR = w.minLR + (w.baseLR-w.minLR)*float32(cosVal)
	}
	w.setLR(w.currentLR)
}

func (w *WarmupCosineScheduler) GetLR() float32 { return w.currentLR }
