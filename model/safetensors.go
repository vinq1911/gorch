//go:build darwin

// Package model provides weight loading and model serialization for gorch.
package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	g "github.com/vinq1911/gorch"
)

// SafetensorsHeader represents the metadata for one tensor in a safetensors file.
type SafetensorsHeader struct {
	DType   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets [2]int  `json:"data_offsets"`
}

// SafetensorsFile represents a loaded safetensors file.
type SafetensorsFile struct {
	Tensors map[string]*g.Tensor
	Names   []string // ordered tensor names
}

// LoadSafetensors loads a .safetensors file and returns all tensors.
//
// Safetensors format:
//   - 8 bytes: little-endian uint64 header length
//   - N bytes: JSON header mapping tensor name → {dtype, shape, data_offsets}
//   - Remaining: raw tensor data
//
// Supports F32, F16 (converted to F32), and BF16 (converted to F32).
func LoadSafetensors(path string) (*SafetensorsFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read safetensors: %w", err)
	}

	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors file too small")
	}

	// Parse header length
	headerLen := binary.LittleEndian.Uint64(data[:8])
	if 8+headerLen > uint64(len(data)) {
		return nil, fmt.Errorf("header length %d exceeds file size %d", headerLen, len(data))
	}

	// Parse JSON header
	headerJSON := data[8 : 8+headerLen]
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &rawHeader); err != nil {
		return nil, fmt.Errorf("parse safetensors header: %w", err)
	}

	dataStart := 8 + int(headerLen)
	result := &SafetensorsFile{
		Tensors: make(map[string]*g.Tensor),
	}

	for name, raw := range rawHeader {
		// Skip metadata key
		if name == "__metadata__" {
			continue
		}

		var hdr SafetensorsHeader
		if err := json.Unmarshal(raw, &hdr); err != nil {
			return nil, fmt.Errorf("parse tensor %q header: %w", name, err)
		}

		tensorData := data[dataStart+hdr.Offsets[0] : dataStart+hdr.Offsets[1]]

		var floats []float32
		switch hdr.DType {
		case "F32":
			floats = decodeF32(tensorData)
		case "F16":
			floats = decodeF16(tensorData)
		case "BF16":
			floats = decodeBF16(tensorData)
		default:
			return nil, fmt.Errorf("unsupported dtype %q for tensor %q", hdr.DType, name)
		}

		tensor := g.NewTensor(floats, hdr.Shape...)
		result.Tensors[name] = tensor
		result.Names = append(result.Names, name)
	}

	sort.Strings(result.Names)
	return result, nil
}

func decodeF32(data []byte) []float32 {
	n := len(data) / 4
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		result[i] = math.Float32frombits(bits)
	}
	return result
}

func decodeF16(data []byte) []float32 {
	n := len(data) / 2
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		h := binary.LittleEndian.Uint16(data[i*2 : i*2+2])
		result[i] = float16ToFloat32(h)
	}
	return result
}

func decodeBF16(data []byte) []float32 {
	n := len(data) / 2
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bf := binary.LittleEndian.Uint16(data[i*2 : i*2+2])
		// BF16 is just the upper 16 bits of F32
		result[i] = math.Float32frombits(uint32(bf) << 16)
	}
	return result
}

// float16ToFloat32 converts an IEEE 754 half-precision float to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff

	var f uint32
	switch {
	case exp == 0:
		if frac == 0 {
			f = sign << 31 // +-zero
		} else {
			// Denormalized
			exp = 127 - 14
			for frac&0x400 == 0 {
				frac <<= 1
				exp--
			}
			frac &= 0x3ff
			f = (sign << 31) | ((exp) << 23) | (frac << 13)
		}
	case exp == 0x1f:
		f = (sign << 31) | (0xff << 23) | (frac << 13) // Inf/NaN
	default:
		f = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
	}
	return math.Float32frombits(f)
}

// SaveSafetensors saves tensors to a .safetensors file.
func SaveSafetensors(path string, tensors map[string]*g.Tensor) error {
	// Build header and compute offsets
	header := make(map[string]SafetensorsHeader)
	var names []string
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	offset := 0
	for _, name := range names {
		t := tensors[name]
		size := t.Size() * 4 // F32
		header[name] = SafetensorsHeader{
			DType:   "F32",
			Shape:   t.Shape(),
			Offsets: [2]int{offset, offset + size},
		}
		offset += size
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("marshal header: %w", err)
	}

	// Write file
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	// Header length (8 bytes LE)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		return err
	}

	// Header JSON
	if _, err := f.Write(headerJSON); err != nil {
		return err
	}

	// Tensor data (F32, little-endian)
	for _, name := range names {
		t := tensors[name]
		for _, v := range t.Data() {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			if _, err := f.Write(buf[:]); err != nil {
				return err
			}
		}
	}

	return nil
}

// LoadModelWeights loads a safetensors file and maps weights to a named parameter map.
// nameMap maps safetensors tensor names to model parameter indices.
func LoadModelWeights(path string, params []*g.Tensor, nameMap map[string]int) error {
	sf, err := LoadSafetensors(path)
	if err != nil {
		return err
	}

	loaded := 0
	for stName, paramIdx := range nameMap {
		tensor, ok := sf.Tensors[stName]
		if !ok {
			return fmt.Errorf("tensor %q not found in safetensors file", stName)
		}
		if paramIdx >= len(params) {
			return fmt.Errorf("parameter index %d out of range for %q", paramIdx, stName)
		}
		target := params[paramIdx]
		if target.Size() != tensor.Size() {
			return fmt.Errorf("size mismatch for %q: model has %d, file has %d",
				stName, target.Size(), tensor.Size())
		}
		copy(target.Data(), tensor.Data())
		loaded++
	}

	fmt.Printf("Loaded %d/%d tensors from %s\n", loaded, len(nameMap), path)
	return nil
}

// SaveModelWeights saves model parameters to a safetensors file.
func SaveModelWeights(path string, params []*g.Tensor, nameMap map[int]string) error {
	tensors := make(map[string]*g.Tensor)
	for idx, name := range nameMap {
		if idx >= len(params) {
			return fmt.Errorf("parameter index %d out of range for %q", idx, name)
		}
		tensors[name] = params[idx]
	}
	return SaveSafetensors(path, tensors)
}
