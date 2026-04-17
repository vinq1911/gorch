// Test firmware for the gorch int8 inference kernel on ATmega2560.
//
// Topology is hard-coded here (LinearI8 -> ReLU -> LinearI32). The numeric
// tables (weights, biases, requant multipliers, input vector) come from
// tiny_model.h and tiny_input.h, which the Go harness regenerates before
// each build.
//
// Output: a single line "CLASS <n> LOGITS <l0> <l1> ...\n" emitted over
// UART0. The host-side runner hooks UART0_TX and pipes it to stdout.

#include <stdint.h>
#include <stdlib.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include "avr/avr_mcu_section.h"

AVR_MCU(F_CPU, "atmega2560");

#include "../gm1_avr.h"
#include "tiny_model.h"
#include "tiny_input.h"

// --- UART0 TX (9600 8N1 @ 16MHz) -------------------------------------------

static void uart_init(void) {
    UCSR0A = 0;
    UCSR0B = (1 << TXEN0);
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
    UBRR0H = 0;
    UBRR0L = 103;
}

static void uart_putc(char c) {
    while (!(UCSR0A & (1 << UDRE0))) {}
    UDR0 = (uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

// Signed decimal print for int32/int16, LSB last. Small and dependency-free.
static void uart_put_long(long v) {
    char buf[12];
    int i = 0;
    if (v < 0) { uart_putc('-'); v = -v; }
    if (v == 0) { uart_putc('0'); return; }
    while (v > 0) { buf[i++] = '0' + (v % 10); v /= 10; }
    while (i--) uart_putc(buf[i]);
}

// --- Inference fixture -----------------------------------------------------

static int8_t  buf_a[GM1_MAX_DIM];
static int8_t  buf_b[GM1_MAX_DIM];
static int32_t logits[GM1_NUM_CLASSES];

int main(void) {
    uart_init();

    for (uint16_t i = 0; i < GM1_INPUT_DIM; i++) {
        buf_a[i] = (int8_t)pgm_read_byte(&GM1_INPUT[i]);
    }

    gm1_linear_i8(GM1_W0, GM1_B0, GM1_M0, GM1_S0,
                  GM1_L0_IN, GM1_L0_OUT, buf_a, buf_b);
    gm1_relu_i8(buf_b, GM1_L0_OUT);
    gm1_linear_i32(GM1_W2, GM1_B2, GM1_L2_IN, GM1_L2_OUT, buf_b, logits);

    uint8_t cls = gm1_argmax_i32(logits, GM1_NUM_CLASSES);

    uart_puts("CLASS ");
    uart_put_long((long)cls);
    uart_puts(" LOGITS");
    for (uint16_t i = 0; i < GM1_NUM_CLASSES; i++) {
        uart_putc(' ');
        uart_put_long((long)logits[i]);
    }
    uart_putc('\n');

    for (;;) {}
    return 0;
}
