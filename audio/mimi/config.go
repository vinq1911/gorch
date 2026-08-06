//go:build darwin

package mimi

// Config holds the kyutai/mimi encoder hyperparameters (plan
// doc/plans/0006-mimi-native-encoder.md §4.1). SEANet fields drive the
// conv stack built in seanet.go; the transformer fields are consumed
// from Phase 3 on.
type Config struct {
	// SEANet encoder.
	SampleRate         int   // input PCM rate, Hz
	NumFilters         int   // base channel count of the first conv
	KernelSize         int   // first conv kernel
	LastKernelSize     int   // final conv kernel
	ResidualKernelSize int   // resnet-block first-conv kernel
	UpsamplingRatios   []int // decoder order [8,6,5,4]; encoder iterates reversed
	DilationGrowthRate int   // dilation = rate**j per residual layer (always 1 here)
	NumResidualLayers  int   // resnet blocks per stage
	Compress           int   // resnet hidden = dim / Compress

	// Transformer (Phase 3).
	HiddenSize    int // model dim, also the SEANet output dimension
	NumLayers     int
	NumHeads      int
	HeadDim       int
	Intermediate  int
	RopeTheta     float32
	SlidingWindow int
	NormEps       float32
	MaxPositions  int
}

// DefaultConfig returns the kyutai/mimi checkpoint values verified in
// plan §0.2.
func DefaultConfig() Config {
	return Config{
		SampleRate:         24000,
		NumFilters:         64,
		KernelSize:         7,
		LastKernelSize:     3,
		ResidualKernelSize: 3,
		UpsamplingRatios:   []int{8, 6, 5, 4},
		DilationGrowthRate: 2,
		NumResidualLayers:  1,
		Compress:           2,

		HiddenSize:    512,
		NumLayers:     8,
		NumHeads:      8,
		HeadDim:       64,
		Intermediate:  2048,
		RopeTheta:     10000,
		SlidingWindow: 250,
		NormEps:       1e-5,
		MaxPositions:  8000,
	}
}
