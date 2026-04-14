//go:build darwin

package gorch

import "fmt"

// Broadcasting rules (NumPy-compatible):
// 1. If tensors have different numbers of dims, pad the shorter shape with 1s on the left
// 2. Dimensions are compatible if they are equal or one of them is 1
// 3. The output shape uses the max of each dimension

// broadcastShapes returns the broadcast output shape, or panics if incompatible.
func broadcastShapes(a, b []int) []int {
	maxDim := len(a)
	if len(b) > maxDim {
		maxDim = len(b)
	}

	// Pad shorter shape with 1s on the left
	pa := padShape(a, maxDim)
	pb := padShape(b, maxDim)

	out := make([]int, maxDim)
	for i := 0; i < maxDim; i++ {
		if pa[i] == pb[i] {
			out[i] = pa[i]
		} else if pa[i] == 1 {
			out[i] = pb[i]
		} else if pb[i] == 1 {
			out[i] = pa[i]
		} else {
			panic(fmt.Sprintf("gorch: broadcast incompatible shapes %v and %v at dim %d", a, b, i))
		}
	}
	return out
}

func padShape(s []int, n int) []int {
	if len(s) == n {
		return s
	}
	out := make([]int, n)
	offset := n - len(s)
	for i := 0; i < offset; i++ {
		out[i] = 1
	}
	copy(out[offset:], s)
	return out
}

// broadcastIndex maps a flat index in the output shape to a flat index in
// the (possibly smaller) source shape.
func broadcastIndex(outIdx int, outShape, srcShape []int) int {
	ndim := len(outShape)
	ps := padShape(srcShape, ndim)

	idx := 0
	stride := 1
	for d := ndim - 1; d >= 0; d-- {
		outSize := outShape[d]
		coord := (outIdx / product(outShape[d+1:])) % outSize

		// If source dim is 1, this coord maps to 0 (broadcast)
		srcCoord := coord
		if ps[d] == 1 {
			srcCoord = 0
		}
		idx += srcCoord * stride
		if ps[d] > 1 {
			stride *= ps[d]
		}
	}
	return idx
}

func product(s []int) int {
	p := 1
	for _, v := range s {
		p *= v
	}
	return p
}

// broadcastData expands src data to match outShape using broadcasting.
func broadcastData(srcData []float32, srcShape, outShape []int) []float32 {
	outSize := numElements(outShape)
	out := make([]float32, outSize)
	for i := 0; i < outSize; i++ {
		out[i] = srcData[broadcastIndex(i, outShape, srcShape)]
	}
	return out
}

// reduceBroadcastGrad sums the gradient along broadcast dimensions to match
// the original (smaller) shape. This is the backward pass for broadcasting.
func reduceBroadcastGrad(gradData []float32, gradShape, origShape []int) []float32 {
	origSize := numElements(origShape)
	out := make([]float32, origSize)
	outSize := numElements(gradShape)
	for i := 0; i < outSize; i++ {
		srcIdx := broadcastIndex(i, gradShape, origShape)
		out[srcIdx] += gradData[i]
	}
	return out
}

// ---------- Broadcast-aware ops ----------

// AddB returns a + b with broadcasting support.
// Handles: scalar + tensor, vector + matrix, etc.
func AddB(a, b *Tensor) *Tensor {
	// Fast path: same shape
	if sameShape(a.shape, b.shape) {
		return Add(a, b)
	}

	outShape := broadcastShapes(a.shape, b.shape)
	outSize := numElements(outShape)

	aData := broadcastData(a.data, a.shape, outShape)
	bData := broadcastData(b.data, b.shape, outShape)

	outData := make([]float32, outSize)
	for i := range outData {
		outData[i] = aData[i] + bData[i]
	}
	out := NewTensor(outData, outShape...)

	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name:   "AddB",
			inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				ga := NewTensor(reduceBroadcastGrad(grad.data, grad.shape, a.shape), a.shape...)
				gb := NewTensor(reduceBroadcastGrad(grad.data, grad.shape, b.shape), b.shape...)
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}

// SubB returns a - b with broadcasting support.
func SubB(a, b *Tensor) *Tensor {
	if sameShape(a.shape, b.shape) {
		return Sub(a, b)
	}

	outShape := broadcastShapes(a.shape, b.shape)
	outSize := numElements(outShape)
	aData := broadcastData(a.data, a.shape, outShape)
	bData := broadcastData(b.data, b.shape, outShape)

	outData := make([]float32, outSize)
	for i := range outData {
		outData[i] = aData[i] - bData[i]
	}
	out := NewTensor(outData, outShape...)

	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name: "SubB", inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				ga := NewTensor(reduceBroadcastGrad(grad.data, grad.shape, a.shape), a.shape...)
				negGrad := make([]float32, len(grad.data))
				for i, v := range grad.data {
					negGrad[i] = -v
				}
				gb := NewTensor(reduceBroadcastGrad(negGrad, grad.shape, b.shape), b.shape...)
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}

// MulB returns a * b with broadcasting support.
func MulB(a, b *Tensor) *Tensor {
	if sameShape(a.shape, b.shape) {
		return Mul(a, b)
	}

	outShape := broadcastShapes(a.shape, b.shape)
	outSize := numElements(outShape)
	aData := broadcastData(a.data, a.shape, outShape)
	bData := broadcastData(b.data, b.shape, outShape)

	outData := make([]float32, outSize)
	for i := range outData {
		outData[i] = aData[i] * bData[i]
	}
	out := NewTensor(outData, outShape...)

	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name: "MulB", inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				// d(a*b)/da = b*grad, reduced to a's shape
				gaData := make([]float32, outSize)
				gbData := make([]float32, outSize)
				for i := range gaData {
					gaData[i] = grad.data[i] * bData[i]
					gbData[i] = grad.data[i] * aData[i]
				}
				ga := NewTensor(reduceBroadcastGrad(gaData, outShape, a.shape), a.shape...)
				gb := NewTensor(reduceBroadcastGrad(gbData, outShape, b.shape), b.shape...)
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}

// DivB returns a / b with broadcasting support.
func DivB(a, b *Tensor) *Tensor {
	if sameShape(a.shape, b.shape) {
		return Div(a, b)
	}

	outShape := broadcastShapes(a.shape, b.shape)
	outSize := numElements(outShape)
	aData := broadcastData(a.data, a.shape, outShape)
	bData := broadcastData(b.data, b.shape, outShape)

	outData := make([]float32, outSize)
	for i := range outData {
		outData[i] = aData[i] / bData[i]
	}
	out := NewTensor(outData, outShape...)

	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.gradFn = &GradFn{
			name: "DivB", inputs: []*Tensor{a, b},
			backward: func(grad *Tensor) []*Tensor {
				gaData := make([]float32, outSize)
				gbData := make([]float32, outSize)
				for i := range gaData {
					gaData[i] = grad.data[i] / bData[i]
					gbData[i] = -grad.data[i] * aData[i] / (bData[i] * bData[i])
				}
				ga := NewTensor(reduceBroadcastGrad(gaData, outShape, a.shape), a.shape...)
				gb := NewTensor(reduceBroadcastGrad(gbData, outShape, b.shape), b.shape...)
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}

// ScalarTensor creates a scalar (1-element) tensor for use with broadcasting ops.
func ScalarTensor(val float32) *Tensor {
	return NewTensor([]float32{val}, 1)
}
