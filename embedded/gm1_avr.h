// gm1_avr.h — int8 inference kernel for AVR (ATmega2560 and friends).
//
// Design:
//   - Weights live in flash via PROGMEM and are read with pgm_read_byte /
//     pgm_read_dword. On the host (no <avr/pgmspace.h>) the macros degrade to
//     plain dereferences so the same code can run as a Go-side reference.
//   - Activations are int8; accumulators are int32. Each Linear layer has a
//     fused requantization multiplier M (Q0.31) and right-shift S derived from
//     (w_scale * x_scale) / y_scale in the exporter.
//   - No malloc, no floats. Two caller-supplied scratch buffers.
//
// Layer contract:
//   gm1_linear_i8(W, B, M, S, in_dim, out_dim, x, y)
//     y[o] = sat_i8( ((B[o] + sum_i W[o*in+i] * x[i]) * M) >> S )
//   gm1_relu_i8(v, n): clamp negatives to 0.
//   gm1_argmax_i32(v, n): return index of max (ties: lowest index).

#ifndef GM1_AVR_H
#define GM1_AVR_H

#include <stdint.h>

#ifdef __AVR__
  #include <avr/pgmspace.h>
  #define GM1_RD_I8(p)  ((int8_t)pgm_read_byte(p))
  #define GM1_RD_I32(p) ((int32_t)pgm_read_dword(p))
  #define GM1_PROGMEM   PROGMEM
#else
  // Host build: weights are plain memory.
  #define GM1_RD_I8(p)  (*(const int8_t *)(p))
  #define GM1_RD_I32(p) (*(const int32_t *)(p))
  #define GM1_PROGMEM
#endif

static inline int8_t gm1_sat_i8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Fused requantize: round-half-up on the shifted value.
// We compute ((int64)acc * M + (1 << (S-1))) >> S  and saturate.
// This matches the TFLite-style MultiplyByQuantizedMultiplier closely enough
// for our purposes (symmetric, no zero-point) and is easy to mirror in Go.
static inline int8_t gm1_requant(int32_t acc, int32_t M, uint8_t S) {
    int64_t t = (int64_t)acc * (int64_t)M;
    int64_t rounding = (S > 0) ? ((int64_t)1 << (S - 1)) : 0;
    int64_t r = (t + rounding) >> S;
    if (r >  127)  r =  127;
    if (r < -128)  r = -128;
    return (int8_t)r;
}

// Linear layer: y = requant(W * x + B).
// W: row-major [out_dim][in_dim], stored in flash (int8).
// B: int32 bias, stored in flash (pre-added into the accumulator; same scale
//    as W*x, i.e. scale_w * scale_x).
// x, y: SRAM buffers, int8.
static inline void gm1_linear_i8(
    const int8_t  *W,    // PROGMEM on AVR
    const int32_t *B,    // PROGMEM on AVR
    int32_t M, uint8_t S,
    uint16_t in_dim, uint16_t out_dim,
    const int8_t *x, int8_t *y)
{
    for (uint16_t o = 0; o < out_dim; o++) {
        int32_t acc = GM1_RD_I32(&B[o]);
        // Cast to uint32_t so (o * in_dim) doesn't overflow on 16-bit int hosts.
        const int8_t *row = W + (uint32_t)o * (uint32_t)in_dim;
        for (uint16_t i = 0; i < in_dim; i++) {
            int8_t w = GM1_RD_I8(&row[i]);
            acc += (int16_t)w * (int16_t)x[i];
        }
        y[o] = gm1_requant(acc, M, S);
    }
}

// Same as gm1_linear_i8 but leaves the int32 accumulator unquantized; used as
// the final classifier layer so we can argmax without losing resolution.
static inline void gm1_linear_i32(
    const int8_t  *W, const int32_t *B,
    uint16_t in_dim, uint16_t out_dim,
    const int8_t *x, int32_t *y)
{
    for (uint16_t o = 0; o < out_dim; o++) {
        int32_t acc = GM1_RD_I32(&B[o]);
        const int8_t *row = W + (uint32_t)o * (uint32_t)in_dim;
        for (uint16_t i = 0; i < in_dim; i++) {
            int8_t w = GM1_RD_I8(&row[i]);
            acc += (int16_t)w * (int16_t)x[i];
        }
        y[o] = acc;
    }
}

static inline void gm1_relu_i8(int8_t *v, uint16_t n) {
    for (uint16_t i = 0; i < n; i++) if (v[i] < 0) v[i] = 0;
}

static inline uint8_t gm1_argmax_i32(const int32_t *v, uint16_t n) {
    uint8_t best = 0;
    int32_t bv = v[0];
    for (uint16_t i = 1; i < n; i++) {
        if (v[i] > bv) { bv = v[i]; best = (uint8_t)i; }
    }
    return best;
}

#endif // GM1_AVR_H
