// Bare-metal ESP32-C3 firmware running on Espressif's QEMU fork.
// Loads the Iris classifier as int8 weights and runs inference via the
// gm1_avr.h kernel. Outputs "CLASS n LOGITS l0 l1 l2\n" over UART0.
//
// This is a REAL ESP32-C3 emulation — same CPU core (rv32imc), same
// memory map (IRAM at 0x40380000), same UART peripheral at 0x60000000 —
// not a generic RISC-V virt machine.

#include <stdint.h>

#include "../../gm1_avr.h"
#include "iris_model.h"
#include "iris_input.h"

// --- ESP32-C3 UART0 (Espressif custom peripheral, APB bus) -----------------
// The UART peripheral in QEMU auto-initializes, so we skip clock/baud
// setup. Transmit: write a byte to UART_FIFO_REG (offset 0). Poll
// UART_STATUS_REG bits [23:16] for TX FIFO count (< 128 means room).

#define UART0_BASE    0x60000000UL
#define UART0_FIFO    (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART0_STATUS  (*(volatile uint32_t *)(UART0_BASE + 0x1C))

static void uart_putc(char c) {
    while (((UART0_STATUS >> 16) & 0xFF) >= 126) {}
    UART0_FIFO = (uint32_t)(uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

static void uart_put_long(long v) {
    char buf[12];
    int i = 0;
    if (v < 0) { uart_putc('-'); v = -v; }
    if (v == 0) { uart_putc('0'); return; }
    while (v > 0) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i--) uart_putc(buf[i]);
}

// --- Inference -------------------------------------------------------------

static int8_t  buf_a[GM1_MAX_DIM];
static int8_t  buf_b[GM1_MAX_DIM];
static int32_t logits[GM1_NUM_CLASSES];

int main(void) {
    for (int i = 0; i < GM1_INPUT_DIM; i++) {
        buf_a[i] = GM1_INPUT[i];
    }

    gm1_linear_i8(GM1_W0, GM1_B0, GM1_M0, GM1_S0,
                  GM1_L0_IN, GM1_L0_OUT, buf_a, buf_b);
    gm1_relu_i8(buf_b, GM1_L0_OUT);
    gm1_linear_i32(GM1_W2, GM1_B2, GM1_L2_IN, GM1_L2_OUT, buf_b, logits);

    uint8_t cls = gm1_argmax_i32(logits, GM1_NUM_CLASSES);

    uart_puts("CLASS ");
    uart_put_long((long)cls);
    uart_puts(" LOGITS");
    for (int i = 0; i < GM1_NUM_CLASSES; i++) {
        uart_putc(' ');
        uart_put_long((long)logits[i]);
    }
    uart_putc('\n');

    // No SiFive test device on ESP32-C3; spin and let the harness kill QEMU.
    for (;;) {}
}
