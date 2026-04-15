//go:build darwin

// Package gorch is a PyTorch-like deep learning framework in Go, backed by Apple Metal.
package gorch

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/vinq1911/gorch/metal"
)

// DType represents the data type of tensor elements.
type DType int

const (
	Float32 DType = iota
)

// Device represents where tensor data lives.
type DeviceType int

const (
	CPU DeviceType = iota
	Metal
)

// Tensor is an N-dimensional array that can live on CPU or Metal GPU.
// When on Metal, the underlying data uses unified memory (zero-copy).
type Tensor struct {
	data  []float32     // CPU data or unified-memory slice (backed by Metal buffer)
	shape []int         // dimensions, e.g. [2, 3] for a 2x3 matrix
	buf   *metal.Buffer // non-nil when on Metal device

	// Autograd fields
	requiresGrad bool
	grad         *Tensor
	gradFn       *GradFn // backward function that produced this tensor
}

// GradFn records how a tensor was computed, enabling backward pass.
type GradFn struct {
	name     string
	inputs   []*Tensor
	backward func(grad *Tensor) []*Tensor // returns gradients for each input
}

// SetGradFn attaches a backward function to this tensor (used by nn package).
func (t *Tensor) SetGradFn(name string, inputs []*Tensor, backward func(grad *Tensor) []*Tensor) {
	t.gradFn = &GradFn{name: name, inputs: inputs, backward: backward}
}

// ---------- Creation ----------

// NewTensor creates a tensor from a flat data slice and shape.
func NewTensor(data []float32, shape ...int) *Tensor {
	n := numElements(shape)
	if len(data) != n {
		panic(fmt.Sprintf("gorch: data length %d does not match shape %v (need %d)", len(data), shape, n))
	}
	cp := make([]float32, n)
	copy(cp, data)
	return &Tensor{data: cp, shape: copyShape(shape)}
}

// Zeros creates a tensor filled with zeros.
func Zeros(shape ...int) *Tensor {
	return &Tensor{data: make([]float32, numElements(shape)), shape: copyShape(shape)}
}

// Ones creates a tensor filled with ones.
func Ones(shape ...int) *Tensor {
	n := numElements(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = 1
	}
	return &Tensor{data: data, shape: copyShape(shape)}
}

// Rand creates a tensor with uniform random values in [0, 1).
func Rand(shape ...int) *Tensor {
	n := numElements(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()
	}
	return &Tensor{data: data, shape: copyShape(shape)}
}

// RandN creates a tensor with standard normal random values.
func RandN(shape ...int) *Tensor {
	n := numElements(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(rand.NormFloat64())
	}
	return &Tensor{data: data, shape: copyShape(shape)}
}

// Full creates a tensor filled with a constant value.
func Full(val float32, shape ...int) *Tensor {
	n := numElements(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = val
	}
	return &Tensor{data: data, shape: copyShape(shape)}
}

// ---------- Properties ----------

// Shape returns a copy of the tensor's shape.
func (t *Tensor) Shape() []int { return copyShape(t.shape) }

// Size returns the total number of elements.
func (t *Tensor) Size() int { return numElements(t.shape) }

// Dim returns the number of dimensions.
func (t *Tensor) Dim() int { return len(t.shape) }

// Data returns the underlying float32 slice (shared, not copied).
func (t *Tensor) Data() []float32 { return t.data }

// Device returns CPU or Metal.
func (t *Tensor) Device() DeviceType {
	if t.buf != nil {
		return Metal
	}
	return CPU
}

// RequiresGrad returns whether this tensor tracks gradients.
func (t *Tensor) RequiresGrad() bool { return t.requiresGrad }

// SetRequiresGrad enables or disables gradient tracking.
func (t *Tensor) SetRequiresGrad(b bool) *Tensor {
	t.requiresGrad = b
	return t
}

// Grad returns the accumulated gradient, or nil.
func (t *Tensor) Grad() *Tensor { return t.grad }

// ZeroGrad resets the gradient to nil.
func (t *Tensor) ZeroGrad() { t.grad = nil }

// ---------- Device Transfer ----------

// ToMetal moves the tensor to Metal GPU using unified memory.
// If already on Metal, returns the same tensor.
func (t *Tensor) ToMetal(dev *metal.Device) *Tensor {
	if t.buf != nil {
		return t
	}
	buf := dev.NewBuffer(len(t.data) * 4)
	gpuSlice := buf.FloatSlice()
	copy(gpuSlice, t.data)
	t.data = gpuSlice // now backed by unified memory
	t.buf = buf
	return t
}

// ToCPU copies tensor data back to a regular Go slice.
// If already on CPU, returns the same tensor.
func (t *Tensor) ToCPU() *Tensor {
	if t.buf == nil {
		return t
	}
	cpuData := make([]float32, len(t.data))
	copy(cpuData, t.data)
	t.buf.Release()
	t.data = cpuData
	t.buf = nil
	return t
}

// MetalBuffer returns the underlying Metal buffer (nil if on CPU).
func (t *Tensor) MetalBuffer() *metal.Buffer { return t.buf }

// NewTensorOnMetal creates a tensor directly on Metal GPU.
// The data lives in unified memory from the start — no copy needed.
func NewTensorOnMetal(dev *metal.Device, data []float32, shape ...int) *Tensor {
	n := numElements(shape)
	if len(data) != n {
		panic("gorch: data length mismatch")
	}
	buf := dev.NewBuffer(n * 4)
	gpuSlice := buf.FloatSlice()
	copy(gpuSlice, data)
	return &Tensor{data: gpuSlice, shape: copyShape(shape), buf: buf}
}

// ZerosOnMetal creates a zero tensor directly on Metal GPU.
func ZerosOnMetal(dev *metal.Device, shape ...int) *Tensor {
	n := numElements(shape)
	buf := dev.NewBuffer(n * 4)
	return &Tensor{data: buf.FloatSlice(), shape: copyShape(shape), buf: buf}
}

// IsOnMetal returns true if this tensor has a Metal buffer.
func (t *Tensor) IsOnMetal() bool { return t.buf != nil }

// MetalDev returns the device from the GPU singleton (if initialized).
func MetalDev() *metal.Device {
	if gpu == nil {
		return nil
	}
	return gpu.Dev
}

// ---------- Indexing ----------

// At returns the value at the given indices.
func (t *Tensor) At(indices ...int) float32 {
	return t.data[t.flatIndex(indices)]
}

// Set sets the value at the given indices.
func (t *Tensor) Set(val float32, indices ...int) {
	t.data[t.flatIndex(indices)] = val
}

func (t *Tensor) flatIndex(indices []int) int {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("gorch: expected %d indices, got %d", len(t.shape), len(indices)))
	}
	idx := 0
	stride := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.shape[i]
	}
	return idx
}

// ---------- Reshape ----------

// Reshape returns a new tensor with the same data but different shape.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	n := numElements(shape)
	if n != t.Size() {
		panic(fmt.Sprintf("gorch: cannot reshape %v to %v", t.shape, shape))
	}
	return &Tensor{data: t.data, shape: copyShape(shape), buf: t.buf}
}

// ---------- Reshape / Transpose ----------

// ReshapeOp returns a new tensor with the same data but different shape, with autograd support.
func ReshapeOp(a *Tensor, shape ...int) *Tensor {
	n := numElements(shape)
	if n != a.Size() {
		panic(fmt.Sprintf("gorch: cannot reshape %v to %v", a.shape, shape))
	}
	out := &Tensor{data: a.data, shape: copyShape(shape), buf: a.buf}
	if a.requiresGrad {
		origShape := copyShape(a.shape)
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Reshape",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{&Tensor{data: grad.data, shape: origShape}}
			},
		}
	}
	return out
}

// Transpose2D swaps rows and columns of a 2-D tensor.
// Keeps data on Metal if input is on Metal.
func Transpose2D(a *Tensor) *Tensor {
	if a.Dim() != 2 {
		panic(fmt.Sprintf("gorch: Transpose2D requires 2-D tensor, got %d-D", a.Dim()))
	}
	M, N := a.shape[0], a.shape[1]
	out := Zeros(N, M)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			out.data[j*M+i] = a.data[i*N+j]
		}
	}
	if a.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Transpose2D",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{Transpose2D(grad)}
			},
		}
	}
	return out
}

// AddBias adds a (N,) or (1,N) bias to each row of a (M,N) matrix.
// Broadcasts bias across the first dimension.
func AddBias(a, bias *Tensor) *Tensor {
	if a.Dim() != 2 {
		panic("gorch: AddBias requires 2-D tensor for a")
	}
	M, N := a.shape[0], a.shape[1]
	bData := bias.data
	if len(bData) != N {
		panic(fmt.Sprintf("gorch: AddBias bias length %d != feature dim %d", len(bData), N))
	}

	out := Zeros(M, N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			out.data[i*N+j] = a.data[i*N+j] + bData[j]
		}
	}

	if a.requiresGrad || bias.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "AddBias",
			inputs: []*Tensor{a, bias},
			backward: func(grad *Tensor) []*Tensor {
				// dL/da = grad (same shape)
				// dL/dbias = sum over rows
				db := Zeros(N)
				for i := 0; i < M; i++ {
					for j := 0; j < N; j++ {
						db.data[j] += grad.data[i*N+j]
					}
				}
				return []*Tensor{grad, db}
			},
		}
	}
	return out
}

// ---------- Display ----------

func (t *Tensor) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "tensor(")
	if t.Size() <= 20 {
		b.WriteString(fmt.Sprintf("%v", t.data))
	} else {
		b.WriteString(fmt.Sprintf("[%v ... %v]", t.data[:3], t.data[len(t.data)-3:]))
	}
	fmt.Fprintf(&b, ", shape=%v", t.shape)
	if t.buf != nil {
		b.WriteString(", device=metal")
	}
	if t.requiresGrad {
		b.WriteString(", requires_grad=true")
	}
	b.WriteString(")")
	return b.String()
}

// ---------- helpers ----------

func numElements(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

func copyShape(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
