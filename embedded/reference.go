// Package embedded contains the int8 inference reference and AVR test harness
// for gorch models exported to microcontroller targets.
//
// This file is the Go-side mirror of gm1_avr.h. Every arithmetic step here
// must produce the same byte as the C kernel given the same inputs, so the
// harness can compare firmware output against the reference without running
// any hardware.
//
// The package does NOT import gorch. Gorch is darwin-only today; the exporter
// that turns a trained *nn.Linear into the int8 format lives in a separate
// file behind a darwin build tag. The reference kernel and the harness stay
// portable so they run in CI on any platform with avr-gcc + simavr.
package embedded

// SatI8 saturates an int32 to int8 range. Mirrors gm1_sat_i8.
func SatI8(v int32) int8 {
	if v > 127 {
		return 127
	}
	if v < -128 {
		return -128
	}
	return int8(v)
}

// Requant performs fused requantization: saturate(((acc * M) + 2^(S-1)) >> S).
// Mirrors gm1_requant exactly (int64 intermediate, half-up rounding).
func Requant(acc int32, M int32, S uint8) int8 {
	t := int64(acc) * int64(M)
	var rounding int64
	if S > 0 {
		rounding = int64(1) << (S - 1)
	}
	r := (t + rounding) >> S
	if r > 127 {
		r = 127
	}
	if r < -128 {
		r = -128
	}
	return int8(r)
}

// LinearI8 implements the fused-requant linear layer. W is row-major
// [outDim][inDim]. Must match gm1_linear_i8 bit-for-bit.
func LinearI8(W []int8, B []int32, M int32, S uint8, inDim, outDim int, x, y []int8) {
	for o := 0; o < outDim; o++ {
		acc := B[o]
		row := W[o*inDim : o*inDim+inDim]
		for i := 0; i < inDim; i++ {
			acc += int32(row[i]) * int32(x[i])
		}
		y[o] = Requant(acc, M, S)
	}
}

// LinearI32 is the unquantized tail: used as the classifier layer so argmax
// operates on full-precision accumulators. Matches gm1_linear_i32.
func LinearI32(W []int8, B []int32, inDim, outDim int, x []int8, y []int32) {
	for o := 0; o < outDim; o++ {
		acc := B[o]
		row := W[o*inDim : o*inDim+inDim]
		for i := 0; i < inDim; i++ {
			acc += int32(row[i]) * int32(x[i])
		}
		y[o] = acc
	}
}

// ReluI8 clamps negatives to zero in place. Matches gm1_relu_i8.
func ReluI8(v []int8) {
	for i, x := range v {
		if x < 0 {
			v[i] = 0
		}
	}
}

// ArgmaxI32 returns the index of the largest element (ties go to lowest
// index). Matches gm1_argmax_i32.
func ArgmaxI32(v []int32) uint8 {
	best := 0
	bv := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bv {
			bv = v[i]
			best = i
		}
	}
	return uint8(best)
}
