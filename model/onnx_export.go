//go:build darwin

package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
)

// ONNX TensorProto.DataType values we use. From onnx.proto3 — only
// the ones we actually emit.
const (
	onnxDTypeFloat = 1
	onnxDTypeInt64 = 7
)

// AttributeProto.AttributeType
const (
	onnxAttrFloat   = 1
	onnxAttrInt     = 2
	onnxAttrString  = 3
	onnxAttrInts    = 7
)

// ONNX IR version we target. 7 corresponds to opset 14, which covers
// every op we emit (Gemm, Relu, Sigmoid, Tanh, Softmax, Conv,
// MaxPool, Flatten).
const onnxIRVersion = 7
const onnxOpset = 14

// ExportSequentialToONNX serialises a Sequential of supported
// layers to an ONNX model file at path. inputShape is the static
// shape of the input tensor — typically (batch, features) for MLPs
// or (batch, channels, H, W) for CNNs. A symbolic batch dimension
// is emitted as -1 so downstream tools handle dynamic batch.
//
// Supported layers:
//   - nn.Linear            → Gemm (with transB=1)
//   - nn.ReLUModule        → Relu
//   - nn.SigmoidModule     → Sigmoid
//   - nn.TanhModule        → Tanh
//   - nn.Conv2d            → Conv
//   - nn.MaxPool2d         → MaxPool
//   - nn.Flatten           → Flatten (axis=1)
//
// Returns an error if a layer type is unsupported.
func ExportSequentialToONNX(seq *nn.Sequential, inputShape []int, path string) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("onnx: inputShape is required")
	}

	gr := newONNXGraph(inputShape)
	if err := gr.addLayers(seq.Layers); err != nil {
		return err
	}
	bytes, err := gr.encode()
	if err != nil {
		return err
	}
	return os.WriteFile(path, bytes, 0644)
}

// ExportLinearToONNX is a convenience wrapper for a single Linear
// layer (for tests and minimal models).
func ExportLinearToONNX(l *nn.Linear, batchSize int, path string) error {
	inFeat := l.Weight.Shape()[1]
	return ExportSequentialToONNX(nn.NewSequential(l), []int{batchSize, inFeat}, path)
}

// onnxGraph is a builder accumulating nodes and initializers as we
// walk the Sequential.
type onnxGraph struct {
	nodes        []onnxNode
	initializers []onnxTensor
	inputShape   []int
	inputName    string
	outputName   string // name of last produced activation
	outputShape  []int  // shape of last produced activation (for output ValueInfo)
	nextID       int
}

type onnxNode struct {
	opType  string
	name    string
	inputs  []string
	outputs []string
	attrs   []onnxAttr
}

type onnxAttr struct {
	name string
	kind int
	f    float32
	i    int64
	s    string
	ints []int64
}

type onnxTensor struct {
	name  string
	dims  []int64
	dtype int32
	raw   []byte
}

func newONNXGraph(inputShape []int) *onnxGraph {
	return &onnxGraph{
		inputShape:  inputShape,
		inputName:   "input",
		outputName:  "input",
		outputShape: append([]int(nil), inputShape...),
	}
}

func (gr *onnxGraph) freshName(prefix string) string {
	gr.nextID++
	return fmt.Sprintf("%s_%d", prefix, gr.nextID)
}

func (gr *onnxGraph) addInitializerFloat32(name string, t *g.Tensor, dims []int64) {
	gr.initializers = append(gr.initializers, onnxTensor{
		name:  name,
		dims:  dims,
		dtype: onnxDTypeFloat,
		raw:   floatsToLE(t.Data()),
	})
}

func (gr *onnxGraph) addLayers(layers []nn.Module) error {
	for _, layer := range layers {
		if err := gr.addLayer(layer); err != nil {
			return err
		}
	}
	return nil
}

func (gr *onnxGraph) addLayer(layer nn.Module) error {
	switch l := layer.(type) {
	case *nn.Linear:
		return gr.addLinear(l)
	case *nn.ReLUModule:
		return gr.addElementwise("Relu", "relu")
	case *nn.SigmoidModule:
		return gr.addElementwise("Sigmoid", "sigmoid")
	case *nn.TanhModule:
		return gr.addElementwise("Tanh", "tanh")
	case *nn.Conv2d:
		return gr.addConv2d(l)
	case *nn.MaxPool2d:
		return gr.addMaxPool2d(l)
	case *nn.Flatten:
		return gr.addFlatten()
	default:
		return fmt.Errorf("onnx: unsupported layer type %T", layer)
	}
}

func (gr *onnxGraph) addLinear(l *nn.Linear) error {
	wShape := l.Weight.Shape()
	if len(wShape) != 2 {
		return fmt.Errorf("onnx: Linear weight must be 2-D, got %v", wShape)
	}
	out := wShape[0]
	in := wShape[1]
	if len(gr.outputShape) != 2 || gr.outputShape[1] != in {
		return fmt.Errorf("onnx: Linear input shape %v incompatible with weight (out=%d, in=%d)",
			gr.outputShape, out, in)
	}

	wName := gr.freshName("W")
	bName := gr.freshName("B")
	outName := gr.freshName("y")

	// Gemm: y = alpha * (A @ B^T) + beta * C, transB=1.
	// A=input, B=weight (out, in), C=bias (out,)
	// gorch's bias is shape (1, out); reshape to (out,) for ONNX convention.
	gr.addInitializerFloat32(wName, l.Weight, []int64{int64(out), int64(in)})
	biasFlat := flattenTo1D(l.Bias)
	gr.addInitializerFloat32(bName, biasFlat, []int64{int64(out)})

	gr.nodes = append(gr.nodes, onnxNode{
		opType: "Gemm",
		name:   gr.freshName("Gemm"),
		inputs: []string{gr.outputName, wName, bName},
		outputs: []string{outName},
		attrs: []onnxAttr{
			{name: "alpha", kind: onnxAttrFloat, f: 1.0},
			{name: "beta", kind: onnxAttrFloat, f: 1.0},
			{name: "transA", kind: onnxAttrInt, i: 0},
			{name: "transB", kind: onnxAttrInt, i: 1},
		},
	})
	gr.outputName = outName
	gr.outputShape = []int{gr.outputShape[0], out}
	return nil
}

func (gr *onnxGraph) addElementwise(opType, namePrefix string) error {
	outName := gr.freshName(namePrefix)
	gr.nodes = append(gr.nodes, onnxNode{
		opType:  opType,
		name:    gr.freshName(opType),
		inputs:  []string{gr.outputName},
		outputs: []string{outName},
	})
	gr.outputName = outName
	// shape unchanged
	return nil
}

func (gr *onnxGraph) addConv2d(c *nn.Conv2d) error {
	if len(gr.outputShape) != 4 {
		return fmt.Errorf("onnx: Conv2d expects 4-D input, got %v", gr.outputShape)
	}
	wShape := c.Weight.Shape() // (outC, inC, kH, kW)
	outC := wShape[0]
	kH := wShape[2]
	kW := wShape[3]

	wName := gr.freshName("W")
	bName := gr.freshName("B")
	outName := gr.freshName("conv")

	gr.addInitializerFloat32(wName, c.Weight,
		[]int64{int64(wShape[0]), int64(wShape[1]), int64(wShape[2]), int64(wShape[3])})
	gr.addInitializerFloat32(bName, c.Bias, []int64{int64(outC)})

	gr.nodes = append(gr.nodes, onnxNode{
		opType:  "Conv",
		name:    gr.freshName("Conv"),
		inputs:  []string{gr.outputName, wName, bName},
		outputs: []string{outName},
		attrs: []onnxAttr{
			{name: "kernel_shape", kind: onnxAttrInts, ints: []int64{int64(kH), int64(kW)}},
			{name: "strides", kind: onnxAttrInts, ints: []int64{int64(c.Stride), int64(c.Stride)}},
			{name: "pads", kind: onnxAttrInts, ints: []int64{
				int64(c.Padding), int64(c.Padding), int64(c.Padding), int64(c.Padding),
			}},
		},
	})

	// Compute output shape: ((H + 2*pad - kH)/stride) + 1
	inH := gr.outputShape[2]
	inW := gr.outputShape[3]
	outH := (inH+2*c.Padding-kH)/c.Stride + 1
	outW := (inW+2*c.Padding-kW)/c.Stride + 1
	gr.outputName = outName
	gr.outputShape = []int{gr.outputShape[0], outC, outH, outW}
	return nil
}

func (gr *onnxGraph) addMaxPool2d(m *nn.MaxPool2d) error {
	if len(gr.outputShape) != 4 {
		return fmt.Errorf("onnx: MaxPool2d expects 4-D input, got %v", gr.outputShape)
	}
	outName := gr.freshName("pool")
	gr.nodes = append(gr.nodes, onnxNode{
		opType:  "MaxPool",
		name:    gr.freshName("MaxPool"),
		inputs:  []string{gr.outputName},
		outputs: []string{outName},
		attrs: []onnxAttr{
			{name: "kernel_shape", kind: onnxAttrInts, ints: []int64{int64(m.KernelSize), int64(m.KernelSize)}},
			{name: "strides", kind: onnxAttrInts, ints: []int64{int64(m.Stride), int64(m.Stride)}},
		},
	})
	inH := gr.outputShape[2]
	inW := gr.outputShape[3]
	outH := (inH-m.KernelSize)/m.Stride + 1
	outW := (inW-m.KernelSize)/m.Stride + 1
	gr.outputName = outName
	gr.outputShape = []int{gr.outputShape[0], gr.outputShape[1], outH, outW}
	return nil
}

func (gr *onnxGraph) addFlatten() error {
	if len(gr.outputShape) < 2 {
		return fmt.Errorf("onnx: Flatten expects rank ≥ 2 input, got %v", gr.outputShape)
	}
	outName := gr.freshName("flat")
	gr.nodes = append(gr.nodes, onnxNode{
		opType:  "Flatten",
		name:    gr.freshName("Flatten"),
		inputs:  []string{gr.outputName},
		outputs: []string{outName},
		attrs: []onnxAttr{
			{name: "axis", kind: onnxAttrInt, i: 1},
		},
	})
	prod := 1
	for i := 1; i < len(gr.outputShape); i++ {
		prod *= gr.outputShape[i]
	}
	gr.outputName = outName
	gr.outputShape = []int{gr.outputShape[0], prod}
	return nil
}

// ---------- Encoding ----------

func (gr *onnxGraph) encode() ([]byte, error) {
	var model pbBuf

	// ModelProto fields:
	//   1: int64 ir_version
	//   2: string producer_name
	//   3: string producer_version
	//   7: GraphProto graph
	//   8: repeated OperatorSetIdProto opset_import
	model.PutInt64(1, onnxIRVersion)
	model.PutString(2, "gorch")
	model.PutString(3, "0.1")

	// opset_import — empty domain means the default ai.onnx ops.
	model.PutMessage(8, func(m *pbBuf) {
		m.PutString(1, "") // domain
		m.PutInt64(2, onnxOpset)
	})

	// GraphProto: field 7
	model.PutMessage(7, func(graph *pbBuf) {
		gr.encodeGraph(graph)
	})

	return model.Bytes(), nil
}

func (gr *onnxGraph) encodeGraph(graph *pbBuf) {
	graph.PutString(2, "gorch_graph") // name

	// nodes — field 1
	for _, n := range gr.nodes {
		graph.PutMessage(1, func(node *pbBuf) {
			for _, in := range n.inputs {
				node.PutString(1, in)
			}
			for _, out := range n.outputs {
				node.PutString(2, out)
			}
			node.PutString(3, n.name)
			node.PutString(4, n.opType)
			for _, a := range n.attrs {
				node.PutMessage(5, func(attr *pbBuf) {
					encodeAttribute(attr, a)
				})
			}
		})
	}

	// initializer — field 5
	for _, t := range gr.initializers {
		graph.PutMessage(5, func(tp *pbBuf) {
			tp.PutRepeatedInt64Unpacked(1, t.dims)
			tp.PutInt32(2, t.dtype)
			tp.PutString(8, t.name)
			tp.PutBytes(9, t.raw)
		})
	}

	// input ValueInfo — field 11
	graph.PutMessage(11, func(vi *pbBuf) {
		encodeValueInfo(vi, gr.inputName, gr.inputShape, true)
	})
	// output ValueInfo — field 12
	graph.PutMessage(12, func(vi *pbBuf) {
		encodeValueInfo(vi, gr.outputName, gr.outputShape, true)
	})
}

func encodeAttribute(buf *pbBuf, a onnxAttr) {
	// AttributeProto fields:
	//   1: string name
	//   20: AttributeType type
	//   2: float f
	//   3: int64 i
	//   4: bytes s
	//   8: repeated int64 ints
	buf.PutString(1, a.name)
	buf.PutInt32(20, int32(a.kind))
	switch a.kind {
	case onnxAttrFloat:
		buf.PutFloat32(2, a.f)
	case onnxAttrInt:
		buf.PutInt64(3, a.i)
	case onnxAttrString:
		buf.PutBytes(4, []byte(a.s))
	case onnxAttrInts:
		// ONNX uses unpacked encoding for repeated ints in attrs.
		buf.PutRepeatedInt64Unpacked(8, a.ints)
	}
}

func encodeValueInfo(buf *pbBuf, name string, shape []int, dynamicBatch bool) {
	buf.PutString(1, name)
	buf.PutMessage(2, func(typ *pbBuf) {
		typ.PutMessage(1, func(tt *pbBuf) {
			// TypeProto.Tensor: elem_type=1 (FLOAT)
			tt.PutInt32(1, onnxDTypeFloat)
			tt.PutMessage(2, func(sh *pbBuf) {
				for i, d := range shape {
					sh.PutMessage(1, func(dim *pbBuf) {
						if dynamicBatch && i == 0 {
							// Symbolic batch dim — use dim_param (field 2).
							dim.PutString(2, "batch")
						} else {
							dim.PutInt64(1, int64(d))
						}
					})
				}
			})
		})
	})
}

// ---------- Helpers ----------

func floatsToLE(data []float32) []byte {
	out := make([]byte, 4*len(data))
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(v))
	}
	return out
}

// flattenTo1D returns a 1-D copy of a tensor. gorch stores Linear
// bias as (1, out); ONNX wants (out,).
func flattenTo1D(t *g.Tensor) *g.Tensor {
	d := t.Data()
	flat := make([]float32, len(d))
	copy(flat, d)
	return g.NewTensor(flat, len(flat))
}
