//go:build darwin

// Package gorch is a PyTorch-like deep learning framework in Go, backed by Apple Metal.
package gorch

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/vinq1911/gorch/metal"
)

// DType represents the data type of tensor elements.
type DType int

const (
	// Float32 is the default and only fully-supported dtype as of plan
	// 0002 PR 1. All ops (forward, backward, autograd) work on F32.
	Float32 DType = iota
	// BFloat16 is a 1-bit-sign + 8-bit-exponent + 7-bit-mantissa float
	// stored as []uint16 in the tensor. Same exponent range as F32 so
	// no loss-scaling is needed for training. THIS PR ships ONLY the
	// storage type and conversion helpers; no kernels operate on it
	// yet. Calling forward/backward ops on a BFloat16 tensor will
	// panic until the per-op dispatch lands (Plan 0002 PRs 4–7).
	BFloat16
)

// String renders a DType for diagnostics.
func (d DType) String() string {
	switch d {
	case Float32:
		return "F32"
	case BFloat16:
		return "BF16"
	default:
		return "DType(?)"
	}
}

// Device represents where tensor data lives.
type DeviceType int

const (
	CPU DeviceType = iota
	Metal
)

// Tensor is an N-dimensional array that can live on CPU or Metal GPU.
// When on Metal, the underlying data uses unified memory (zero-copy).
//
// As of plan 0002 PR 1, a Tensor holds either F32 data (in `data`)
// or BF16 data (in `data16`); exactly one of the two is non-nil and
// `dtype` records which. F32 remains the default for every existing
// constructor and op; BF16 ships with constructors and conversion
// only — actual ops on BF16 tensors arrive in subsequent PRs.
type Tensor struct {
	dtype  DType
	data   []float32     // F32 path: CPU or unified-memory slice
	data16 []uint16      // BF16 path: bf16-as-uint16 storage
	shape  []int
	buf    *metal.Buffer // non-nil when on Metal device

	// Autograd fields
	requiresGrad bool
	grad         *Tensor
	gradFn       *GradFn // backward function that produced this tensor
}

// Dtype returns the tensor's element type.
func (t *Tensor) Dtype() DType { return t.dtype }

// ---------- BF16 conversion helpers ----------
//
// Bfloat16 is a 32-bit IEEE 754 float with the lower 16 mantissa bits
// truncated. Conversion is a bit-shift and a round-to-nearest-even
// adjustment for the truncated bits. Same exponent range as F32 — no
// scaling needed.

// f32ToBF16 converts a float32 to a bfloat16 stored in a uint16.
// Round-to-nearest-even on the truncated bits matches the standard
// PyTorch / hardware behaviour for bf16 storage.
func f32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	// NaN handling: preserve quiet NaN bit pattern; flush sub-NaNs to NaN.
	if (bits>>23)&0xff == 0xff && bits&0x7fffff != 0 {
		return uint16((bits >> 16) | 0x40)
	}
	// Round-to-nearest-even on the truncated low 16 bits.
	rounding := uint32(0x7fff + ((bits >> 16) & 1))
	return uint16((bits + rounding) >> 16)
}

// bf16ToF32 reverses f32ToBF16. Lossless in the bf16 → f32 direction.
func bf16ToF32(v uint16) float32 {
	return math.Float32frombits(uint32(v) << 16)
}

// F32ToBF16Slice and BF16ToF32Slice are public conversion utilities
// for callers loading or saving bf16 weights. They allocate; pass a
// pre-sized output slice to avoid the allocation.

// F32ToBF16Slice converts each f32 to bf16 in a fresh slice.
func F32ToBF16Slice(in []float32) []uint16 {
	out := make([]uint16, len(in))
	for i, v := range in {
		out[i] = f32ToBF16(v)
	}
	return out
}

// BF16ToF32Slice converts each bf16 back to f32 in a fresh slice.
func BF16ToF32Slice(in []uint16) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = bf16ToF32(v)
	}
	return out
}

// NewTensorBF16 creates a tensor with bfloat16 storage. The caller
// supplies bf16 data (already-converted uint16 values); use
// F32ToBF16Slice to convert from f32 first if needed.
//
// Op support: as of plan 0002 PR 1, no op accepts a BF16 tensor.
// Calling forward/backward ops on it will panic until the per-op
// dispatch lands. The constructor + storage exist so safetensors
// loaders can preserve native bf16 from disk instead of upcasting
// to f32 (the existing waste documented in plan 0002).
func NewTensorBF16(data []uint16, shape ...int) *Tensor {
	n := numElements(shape)
	if len(data) != n {
		panic(fmt.Sprintf("gorch: bf16 data length %d does not match shape %v (need %d)", len(data), shape, n))
	}
	cp := make([]uint16, n)
	copy(cp, data)
	return &Tensor{dtype: BFloat16, data16: cp, shape: copyShape(shape)}
}

// ToF32 returns a fresh F32 tensor with the same logical values as t.
// If t is already F32, returns a deep copy (callers can mutate the
// result without affecting t). Used as the slow-path interop hook
// for callers that hold a BF16 tensor but need to call an op that
// hasn't gained BF16 dispatch yet.
func (t *Tensor) ToF32() *Tensor {
	switch t.dtype {
	case Float32:
		cp := make([]float32, len(t.data))
		copy(cp, t.data)
		return &Tensor{dtype: Float32, data: cp, shape: copyShape(t.shape)}
	case BFloat16:
		return &Tensor{dtype: Float32, data: BF16ToF32Slice(t.data16), shape: copyShape(t.shape)}
	default:
		panic("gorch: unknown dtype")
	}
}

// ToBF16 returns a fresh BF16 tensor with t's values rounded to bf16.
// If t is already BF16, returns a deep copy.
func (t *Tensor) ToBF16() *Tensor {
	switch t.dtype {
	case Float32:
		return &Tensor{dtype: BFloat16, data16: F32ToBF16Slice(t.data), shape: copyShape(t.shape)}
	case BFloat16:
		cp := make([]uint16, len(t.data16))
		copy(cp, t.data16)
		return &Tensor{dtype: BFloat16, data16: cp, shape: copyShape(t.shape)}
	default:
		panic("gorch: unknown dtype")
	}
}

// GradFn records how a tensor was computed, enabling backward pass.
type GradFn struct {
	name     string
	inputs   []*Tensor
	backward func(grad *Tensor) []*Tensor // returns gradients for each input
}

// SetGradFn attaches a backward function to this tensor (used by nn package).
// Inside a NoGrad scope this is a no-op — the autograd graph is not built,
// which keeps activations short-lived and dramatically reduces GC pressure
// during inference.
func (t *Tensor) SetGradFn(name string, inputs []*Tensor, backward func(grad *Tensor) []*Tensor) {
	if !GradEnabled() {
		return
	}
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

// SetRequiresGrad enables or disables gradient tracking. Inside a
// NoGrad scope, attempts to enable tracking are silently ignored;
// disabling always works.
func (t *Tensor) SetRequiresGrad(b bool) *Tensor {
	if b && !GradEnabled() {
		return t
	}
	t.requiresGrad = b
	return t
}

// Detach returns a new Tensor sharing the same underlying data
// (and Metal buffer, if any) as t but with requires_grad=false and
// no gradFn. Use it as a goroutine-local "no autograd" escape hatch
// when the process-global g.NoGrad scope isn't safe to use — e.g.,
// concurrent inference goroutines mixed with a training loop in
// another goroutine. Mutating the returned tensor's data mutates t's
// data too; treat Detach as "make a non-tracking handle to the same
// memory," not a copy. dtype (F32 or BF16) is preserved.
func (t *Tensor) Detach() *Tensor {
	return &Tensor{
		dtype:  t.dtype,
		data:   t.data,
		data16: t.data16,
		shape:  copyShape(t.shape),
		buf:    t.buf,
	}
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
// The returned tensor preserves autograd: when the source has
// requires_grad=true, the reshape's backward is "reshape grad back to
// the original shape." This matches PyTorch's tensor.reshape — the
// no-autograd variant was a bug that broke autograd through every
// multi-head attention reshape.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	n := numElements(shape)
	if n != t.Size() {
		panic(fmt.Sprintf("gorch: cannot reshape %v to %v", t.shape, shape))
	}
	out := &Tensor{
		dtype:  t.dtype,
		data:   t.data,
		data16: t.data16,
		shape:  copyShape(shape),
		buf:    t.buf,
	}
	if GradEnabled() && t.requiresGrad {
		origShape := copyShape(t.shape)
		dtype := t.dtype
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Reshape",
			inputs: []*Tensor{t},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{&Tensor{
					dtype:  dtype,
					data:   grad.data,
					data16: grad.data16,
					shape:  origShape,
				}}
			},
		}
	}
	return out
}

// ---------- Reshape / Transpose ----------

// ReshapeOp returns a new tensor with the same data but different shape, with autograd support.
func ReshapeOp(a *Tensor, shape ...int) *Tensor {
	n := numElements(shape)
	if n != a.Size() {
		panic(fmt.Sprintf("gorch: cannot reshape %v to %v", a.shape, shape))
	}
	out := &Tensor{
		dtype:  a.dtype,
		data:   a.data,
		data16: a.data16,
		shape:  copyShape(shape),
		buf:    a.buf,
	}
	if a.requiresGrad {
		origShape := copyShape(a.shape)
		dtype := a.dtype
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "Reshape",
			inputs: []*Tensor{a},
			backward: func(grad *Tensor) []*Tensor {
				return []*Tensor{&Tensor{
					dtype:  dtype,
					data:   grad.data,
					data16: grad.data16,
					shape:  origShape,
				}}
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
	if a.dtype == BFloat16 {
		return downcastToBF16(Transpose2D(promoteToF32(a)))
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
	requireSameDtype(a, bias, "AddBias")
	if a.dtype == BFloat16 {
		return downcastToBF16(AddBias(promoteToF32(a), promoteToF32(bias)))
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
	// Render via the active storage so bf16 tensors don't crash on a
	// nil .data slice. Materialise as f32 for display only.
	display := t.data
	if t.dtype == BFloat16 {
		display = BF16ToF32Slice(t.data16)
	}
	if t.Size() <= 20 {
		b.WriteString(fmt.Sprintf("%v", display))
	} else {
		b.WriteString(fmt.Sprintf("[%v ... %v]", display[:3], display[len(display)-3:]))
	}
	fmt.Fprintf(&b, ", shape=%v", t.shape)
	if t.dtype != Float32 {
		fmt.Fprintf(&b, ", dtype=%s", t.dtype)
	}
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
