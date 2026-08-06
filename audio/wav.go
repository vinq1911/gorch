//go:build darwin

// Package audio provides audio I/O and sample-rate conversion for the
// native Mimi encoder pipeline (doc/plans/0006-mimi-native-encoder.md).
package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// WAV holds decoded audio as interleaved float32 samples.
// Integer PCM is normalized to [-1, 1) by dividing by 2^(bits-1),
// matching python-soundfile's float32 reads (required for FSDD parity).
type WAV struct {
	SampleRate int
	Channels   int
	Samples    []float32 // interleaved: frame f, channel c at Samples[f*Channels+c]
}

// ReadWAV decodes the WAV file at path.
func ReadWAV(path string) (*WAV, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	w, err := ReadWAVReader(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return w, nil
}

// wavFormat is the decoded "fmt " chunk with any WAVE_FORMAT_EXTENSIBLE
// indirection already resolved: formatTag is 1 (integer PCM) or 3 (IEEE float).
type wavFormat struct {
	formatTag     uint16
	channels      int
	sampleRate    int
	bitsPerSample int
}

// ReadWAVReader decodes a WAV stream. It walks the RIFF chunk list,
// tolerating LIST/fact/JUNK and other unknown chunks before (or after)
// the data chunk, and supports format tags 1 (PCM int 16/24/32),
// 3 (IEEE float32/float64) and 0xFFFE (extensible; dispatched on the
// SubFormat GUID's leading format tag).
func ReadWAVReader(r io.ReadSeeker) (*WAV, error) {
	var hdr [12]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, fmt.Errorf("read RIFF header: %w", err)
	}
	if string(hdr[0:4]) != "RIFF" || string(hdr[8:12]) != "WAVE" {
		return nil, errors.New("not a RIFF/WAVE file")
	}

	var format *wavFormat
	dataOff := int64(-1)
	var dataSize int64

	for {
		var ch [8]byte
		if _, err := io.ReadFull(r, ch[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return nil, fmt.Errorf("read chunk header: %w", err)
		}
		id := string(ch[0:4])
		size := int64(binary.LittleEndian.Uint32(ch[4:8]))
		pos, err := r.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, err
		}

		switch id {
		case "fmt ":
			if size < 16 || size > 1<<16 {
				return nil, fmt.Errorf("implausible fmt chunk size %d", size)
			}
			buf := make([]byte, size)
			if _, err := io.ReadFull(r, buf); err != nil {
				return nil, fmt.Errorf("read fmt chunk: %w", err)
			}
			if format, err = parseFmt(buf); err != nil {
				return nil, err
			}
		case "data":
			dataOff, dataSize = pos, size
		}

		next := pos + size
		if size%2 == 1 {
			next++ // chunks are word-aligned; odd sizes carry a pad byte
		}
		if _, err := r.Seek(next, io.SeekStart); err != nil {
			return nil, fmt.Errorf("seek past chunk %q: %w", id, err)
		}
		if format != nil && dataOff >= 0 {
			break
		}
	}

	if format == nil {
		return nil, errors.New("missing fmt chunk")
	}
	if dataOff < 0 {
		return nil, errors.New("missing data chunk")
	}

	// Clamp the data size to the actual stream length (some writers store
	// a placeholder size, e.g. 0xFFFFFFFF, when streaming).
	end, err := r.Seek(0, io.SeekEnd)
	if err != nil {
		return nil, err
	}
	if dataOff+dataSize > end {
		dataSize = end - dataOff
	}

	raw := make([]byte, dataSize)
	if _, err := r.Seek(dataOff, io.SeekStart); err != nil {
		return nil, err
	}
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, fmt.Errorf("read data chunk: %w", err)
	}

	samples, err := decodeSamples(raw, format)
	if err != nil {
		return nil, err
	}
	return &WAV{
		SampleRate: format.sampleRate,
		Channels:   format.channels,
		Samples:    samples,
	}, nil
}

// Mono returns a single-channel view of the audio: a copy for mono input,
// the per-frame channel average otherwise.
func (w *WAV) Mono() []float32 {
	if w.Channels <= 1 {
		out := make([]float32, len(w.Samples))
		copy(out, w.Samples)
		return out
	}
	frames := len(w.Samples) / w.Channels
	out := make([]float32, frames)
	inv := 1 / float32(w.Channels)
	for f := 0; f < frames; f++ {
		var s float32
		for c := 0; c < w.Channels; c++ {
			s += w.Samples[f*w.Channels+c]
		}
		out[f] = s * inv
	}
	return out
}

func parseFmt(b []byte) (*wavFormat, error) {
	tag := binary.LittleEndian.Uint16(b[0:2])
	f := &wavFormat{
		channels:      int(binary.LittleEndian.Uint16(b[2:4])),
		sampleRate:    int(binary.LittleEndian.Uint32(b[4:8])),
		bitsPerSample: int(binary.LittleEndian.Uint16(b[14:16])),
	}
	if tag == 0xFFFE { // WAVE_FORMAT_EXTENSIBLE
		// 16-byte base + cbSize(2) + validBits(2) + channelMask(4) + GUID(16).
		if len(b) < 40 {
			return nil, fmt.Errorf("extensible fmt chunk too short: %d bytes", len(b))
		}
		validBits := int(binary.LittleEndian.Uint16(b[18:20]))
		if validBits != 0 && validBits != f.bitsPerSample {
			return nil, fmt.Errorf("unsupported extensible format: %d valid bits in %d-bit container",
				validBits, f.bitsPerSample)
		}
		// The SubFormat GUID leads with the underlying format tag.
		tag = binary.LittleEndian.Uint16(b[24:26])
	}
	switch tag {
	case 1, 3:
		f.formatTag = tag
	default:
		return nil, fmt.Errorf("unsupported WAV format tag 0x%04X", tag)
	}
	if f.channels <= 0 {
		return nil, fmt.Errorf("invalid channel count %d", f.channels)
	}
	if f.sampleRate <= 0 {
		return nil, fmt.Errorf("invalid sample rate %d", f.sampleRate)
	}
	return f, nil
}

func decodeSamples(raw []byte, f *wavFormat) ([]float32, error) {
	switch f.formatTag {
	case 1: // integer PCM
		switch f.bitsPerSample {
		case 16:
			n := len(raw) / 2
			out := make([]float32, n)
			for i := 0; i < n; i++ {
				v := int16(binary.LittleEndian.Uint16(raw[2*i:]))
				out[i] = float32(v) / 32768
			}
			return out, nil
		case 24:
			n := len(raw) / 3
			out := make([]float32, n)
			for i := 0; i < n; i++ {
				b := raw[3*i : 3*i+3]
				v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
				if v&0x800000 != 0 {
					v -= 1 << 24 // sign-extend
				}
				out[i] = float32(v) / 8388608
			}
			return out, nil
		case 32:
			n := len(raw) / 4
			out := make([]float32, n)
			for i := 0; i < n; i++ {
				v := int32(binary.LittleEndian.Uint32(raw[4*i:]))
				out[i] = float32(v) / 2147483648
			}
			return out, nil
		default:
			return nil, fmt.Errorf("unsupported PCM bit depth %d", f.bitsPerSample)
		}
	case 3: // IEEE float
		switch f.bitsPerSample {
		case 32:
			n := len(raw) / 4
			out := make([]float32, n)
			for i := 0; i < n; i++ {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
			}
			return out, nil
		case 64:
			n := len(raw) / 8
			out := make([]float32, n)
			for i := 0; i < n; i++ {
				out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:])))
			}
			return out, nil
		default:
			return nil, fmt.Errorf("unsupported float bit depth %d", f.bitsPerSample)
		}
	}
	return nil, fmt.Errorf("unsupported format tag %d", f.formatTag)
}
