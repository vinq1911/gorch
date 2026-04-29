//go:build darwin

package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Hand-rolled protobuf encoder/decoder restricted to what ONNX needs.
// Wire format is documented at
// https://protobuf.dev/programming-guides/encoding/.
//
// Supported wire types here: 0 (varint), 2 (length-delimited), 5 (fixed32).
// Field tag = (fieldNumber << 3) | wireType.
//
// We only implement the slice of ONNX (onnx.proto3) that GPT-2-shaped
// nets need — no protoc, no reflection, no third-party deps.

const (
	wireVarint    = 0
	wireFixed64   = 1
	wireLenDelim  = 2
	wireFixed32   = 5
)

// pbBuf is an append-only protobuf encoder.
type pbBuf struct {
	b []byte
}

// Bytes returns the encoded buffer.
func (p *pbBuf) Bytes() []byte { return p.b }

// putVarint appends a varint-encoded uint64.
func (p *pbBuf) putVarint(v uint64) {
	for v >= 0x80 {
		p.b = append(p.b, byte(v)|0x80)
		v >>= 7
	}
	p.b = append(p.b, byte(v))
}

// putTag appends a field tag.
func (p *pbBuf) putTag(field, wireType int) {
	p.putVarint(uint64(field)<<3 | uint64(wireType))
}

// PutInt64 writes a varint-encoded int64 field (zigzag NOT used — ONNX
// uses regular int64, so negative numbers blow up to 10 bytes; that's OK).
func (p *pbBuf) PutInt64(field int, v int64) {
	if v == 0 {
		return // protobuf default value, omit for compactness
	}
	p.putTag(field, wireVarint)
	p.putVarint(uint64(v))
}

// PutInt32 writes a varint-encoded int32 field.
func (p *pbBuf) PutInt32(field int, v int32) {
	if v == 0 {
		return
	}
	p.putTag(field, wireVarint)
	p.putVarint(uint64(int64(v)))
}

// PutString writes a length-delimited string.
func (p *pbBuf) PutString(field int, s string) {
	if s == "" {
		return
	}
	p.putTag(field, wireLenDelim)
	p.putVarint(uint64(len(s)))
	p.b = append(p.b, s...)
}

// PutBytes writes a length-delimited bytes field.
func (p *pbBuf) PutBytes(field int, b []byte) {
	if len(b) == 0 {
		return
	}
	p.putTag(field, wireLenDelim)
	p.putVarint(uint64(len(b)))
	p.b = append(p.b, b...)
}

// PutMessage writes a nested message: builds it via fn into a temp
// buffer, then emits length-prefixed.
func (p *pbBuf) PutMessage(field int, fn func(m *pbBuf)) {
	var inner pbBuf
	fn(&inner)
	if len(inner.b) == 0 {
		return
	}
	p.putTag(field, wireLenDelim)
	p.putVarint(uint64(len(inner.b)))
	p.b = append(p.b, inner.b...)
}

// PutFloat32 writes a fixed32 float field.
func (p *pbBuf) PutFloat32(field int, v float32) {
	if v == 0 {
		return
	}
	p.putTag(field, wireFixed32)
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], floatBits(v))
	p.b = append(p.b, buf[:]...)
}

// PutPackedInt64 writes a repeated int64 field as a packed varint
// length-delimited group.
func (p *pbBuf) PutPackedInt64(field int, vs []int64) {
	if len(vs) == 0 {
		return
	}
	// Compute body length first.
	var inner pbBuf
	for _, v := range vs {
		inner.putVarint(uint64(v))
	}
	p.putTag(field, wireLenDelim)
	p.putVarint(uint64(len(inner.b)))
	p.b = append(p.b, inner.b...)
}

// PutRepeatedInt64Unpacked writes int64 repeated as one varint field
// per element. ONNX uses this for `dims` in TensorProto.
func (p *pbBuf) PutRepeatedInt64Unpacked(field int, vs []int64) {
	for _, v := range vs {
		p.putTag(field, wireVarint)
		p.putVarint(uint64(v))
	}
}

func floatBits(f float32) uint32 { return math.Float32bits(f) }

// ---------- Decoder ----------

type pbReader struct {
	b   []byte
	pos int
}

func newPBReader(b []byte) *pbReader { return &pbReader{b: b} }

func (r *pbReader) readVarint() (uint64, error) {
	var v uint64
	var shift uint
	for {
		if r.pos >= len(r.b) {
			return 0, errors.New("pb: truncated varint")
		}
		c := r.b[r.pos]
		r.pos++
		v |= uint64(c&0x7f) << shift
		if c < 0x80 {
			return v, nil
		}
		shift += 7
		if shift >= 64 {
			return 0, errors.New("pb: varint overflow")
		}
	}
}

// readField returns the field number and wire type.
func (r *pbReader) readField() (int, int, error) {
	tag, err := r.readVarint()
	if err != nil {
		return 0, 0, err
	}
	return int(tag >> 3), int(tag & 7), nil
}

func (r *pbReader) readBytes() ([]byte, error) {
	n, err := r.readVarint()
	if err != nil {
		return nil, err
	}
	if uint64(r.pos)+n > uint64(len(r.b)) {
		return nil, fmt.Errorf("pb: truncated bytes (need %d, have %d)", n, len(r.b)-r.pos)
	}
	out := r.b[r.pos : r.pos+int(n)]
	r.pos += int(n)
	return out, nil
}

func (r *pbReader) readString() (string, error) {
	b, err := r.readBytes()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// skipField advances past a field of the given wire type without
// interpreting it. Used to ignore fields we don't care about.
func (r *pbReader) skipField(wireType int) error {
	switch wireType {
	case wireVarint:
		_, err := r.readVarint()
		return err
	case wireFixed64:
		if r.pos+8 > len(r.b) {
			return errors.New("pb: truncated fixed64")
		}
		r.pos += 8
		return nil
	case wireLenDelim:
		_, err := r.readBytes()
		return err
	case wireFixed32:
		if r.pos+4 > len(r.b) {
			return errors.New("pb: truncated fixed32")
		}
		r.pos += 4
		return nil
	default:
		return fmt.Errorf("pb: unknown wire type %d", wireType)
	}
}

func (r *pbReader) eof() bool { return r.pos >= len(r.b) }
