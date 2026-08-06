//go:build darwin

package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	g "github.com/vinq1911/gorch"
)

// ONNXFile is a partial parse of an ONNX model. We only decode the
// pieces gorch can act on today — initializer tensors keyed by name
// — so users can load weights from any ONNX producer without
// implementing the full graph spec. Op nodes are recorded as a flat
// list for inspection but not executed.
type ONNXFile struct {
	Tensors  map[string]*g.Tensor // initializer name → tensor
	Names    []string             // initializer names in file order
	Nodes    []ONNXNodeInfo       // graph node summary (for inspection)
	IRVer    int64
	Producer string
}

// ONNXNodeInfo is a lightweight summary of a graph node for
// inspection. We don't reconstruct execution from these.
type ONNXNodeInfo struct {
	OpType string
	Name   string
	Inputs []string
	Output []string
}

// LoadONNX reads an ONNX file and returns its initializer tensors.
// Supports float32 (raw_data or float_data), int64 (raw_data),
// and the exporter we just wrote. Other dtypes are returned as
// errors so silent dtype mismatches don't propagate downstream.
func LoadONNX(path string) (*ONNXFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := &ONNXFile{
		Tensors: make(map[string]*g.Tensor),
	}
	if err := parseModelProto(data, out); err != nil {
		return nil, fmt.Errorf("onnx: %w", err)
	}
	return out, nil
}

// parseModelProto walks a ModelProto message.
func parseModelProto(b []byte, out *ONNXFile) error {
	r := newPBReader(b)
	for !r.eof() {
		field, wire, err := r.readField()
		if err != nil {
			return err
		}
		switch field {
		case 1: // ir_version (int64 varint)
			v, err := r.readVarint()
			if err != nil {
				return err
			}
			out.IRVer = int64(v)
		case 2: // producer_name (string)
			s, err := r.readString()
			if err != nil {
				return err
			}
			out.Producer = s
		case 7: // graph (GraphProto)
			gb, err := r.readBytes()
			if err != nil {
				return err
			}
			if err := parseGraphProto(gb, out); err != nil {
				return err
			}
		default:
			if err := r.skipField(wire); err != nil {
				return err
			}
		}
	}
	return nil
}

func parseGraphProto(b []byte, out *ONNXFile) error {
	r := newPBReader(b)
	for !r.eof() {
		field, wire, err := r.readField()
		if err != nil {
			return err
		}
		switch field {
		case 1: // node
			nb, err := r.readBytes()
			if err != nil {
				return err
			}
			n, err := parseNodeProto(nb)
			if err != nil {
				return err
			}
			out.Nodes = append(out.Nodes, n)
		case 5: // initializer (TensorProto)
			tb, err := r.readBytes()
			if err != nil {
				return err
			}
			name, t, err := parseTensorProto(tb)
			if err != nil {
				return err
			}
			if _, exists := out.Tensors[name]; !exists {
				out.Names = append(out.Names, name)
			}
			out.Tensors[name] = t
		default:
			if err := r.skipField(wire); err != nil {
				return err
			}
		}
	}
	return nil
}

func parseNodeProto(b []byte) (ONNXNodeInfo, error) {
	r := newPBReader(b)
	var info ONNXNodeInfo
	for !r.eof() {
		field, wire, err := r.readField()
		if err != nil {
			return info, err
		}
		switch field {
		case 1:
			s, err := r.readString()
			if err != nil {
				return info, err
			}
			info.Inputs = append(info.Inputs, s)
		case 2:
			s, err := r.readString()
			if err != nil {
				return info, err
			}
			info.Output = append(info.Output, s)
		case 3:
			s, err := r.readString()
			if err != nil {
				return info, err
			}
			info.Name = s
		case 4:
			s, err := r.readString()
			if err != nil {
				return info, err
			}
			info.OpType = s
		default:
			if err := r.skipField(wire); err != nil {
				return info, err
			}
		}
	}
	return info, nil
}

func parseTensorProto(b []byte) (string, *g.Tensor, error) {
	r := newPBReader(b)
	var (
		name    string
		dims    []int
		dtype   int32
		raw     []byte
		floats  []float32 // from packed float_data field 4
		hasData bool
	)
	for !r.eof() {
		field, wire, err := r.readField()
		if err != nil {
			return "", nil, err
		}
		switch field {
		case 1: // dims (repeated int64, unpacked or packed)
			if wire == wireVarint {
				v, err := r.readVarint()
				if err != nil {
					return "", nil, err
				}
				dims = append(dims, int(v))
			} else if wire == wireLenDelim {
				// packed
				inner, err := r.readBytes()
				if err != nil {
					return "", nil, err
				}
				ir := newPBReader(inner)
				for !ir.eof() {
					v, err := ir.readVarint()
					if err != nil {
						return "", nil, err
					}
					dims = append(dims, int(v))
				}
			} else {
				if err := r.skipField(wire); err != nil {
					return "", nil, err
				}
			}
		case 2: // data_type (int32)
			v, err := r.readVarint()
			if err != nil {
				return "", nil, err
			}
			dtype = int32(v)
		case 4: // float_data (packed float32 in the proto3 form, or repeated)
			if wire != wireLenDelim {
				if err := r.skipField(wire); err != nil {
					return "", nil, err
				}
				continue
			}
			fb, err := r.readBytes()
			if err != nil {
				return "", nil, err
			}
			if len(fb)%4 != 0 {
				return "", nil, fmt.Errorf("float_data length %d not divisible by 4", len(fb))
			}
			n := len(fb) / 4
			floats = make([]float32, n)
			for i := 0; i < n; i++ {
				floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(fb[i*4 : i*4+4]))
			}
			hasData = true
		case 8: // name
			s, err := r.readString()
			if err != nil {
				return "", nil, err
			}
			name = s
		case 9: // raw_data (bytes, little-endian)
			rb, err := r.readBytes()
			if err != nil {
				return "", nil, err
			}
			raw = rb
			hasData = true
		default:
			if err := r.skipField(wire); err != nil {
				return "", nil, err
			}
		}
	}
	if !hasData {
		return "", nil, fmt.Errorf("tensor %q has no data", name)
	}

	switch dtype {
	case onnxDTypeFloat:
		var data []float32
		if len(floats) > 0 {
			data = floats
		} else {
			if len(raw)%4 != 0 {
				return "", nil, fmt.Errorf("raw_data length %d not divisible by 4 for float", len(raw))
			}
			n := len(raw) / 4
			data = make([]float32, n)
			for i := 0; i < n; i++ {
				data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
			}
		}
		// Validate against dims.
		expected := 1
		for _, d := range dims {
			expected *= d
		}
		if expected != len(data) {
			return "", nil, fmt.Errorf("tensor %q: dims product %d != data %d", name, expected, len(data))
		}
		if len(dims) == 0 {
			dims = []int{len(data)}
		}
		return name, g.NewTensor(data, dims...), nil
	default:
		return "", nil, fmt.Errorf("tensor %q has unsupported dtype %d", name, dtype)
	}
}
