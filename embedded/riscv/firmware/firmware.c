// Bare-metal RISC-V (rv32im) firmware for the gorch int8 inference kernel,
// targeting QEMU's `virt` machine. Represents ESP32-C3-class hardware (32-bit
// RISC-V, same word size, same-level of UART peripheral). The Espressif QEMU
// fork targeting a real ESP32-C3 SoC would use the same kernel and the same
// generated tables; only the firmware wrapper (startup, UART driver, exit
// method) differs.
//
// Output: a single line "CLASS <n> LOGITS <l0> <l1> <l2>\n" on the 16550
// UART at 0x10000000. QEMU's chardev pipes that to stdout when invoked
// with `-serial mon:stdio` or `-serial stdio`.

#include <stdint.h>

#include "../../gm1_avr.h"
#include "iris_model.h"
#include "iris_input.h"

// --- QEMU virt peripherals -------------------------------------------------

// NS16550A at 0x10000000. We only need the transmit holding register and the
// line status register (to poll for "transmit buffer empty").
#define UART0_BASE        0x10000000UL
#define UART_THR          (*(volatile uint8_t *)(UART0_BASE + 0x0))
#define UART_LSR          (*(volatile uint8_t *)(UART0_BASE + 0x5))
#define UART_LSR_TX_IDLE  (1 << 5)

// SiFive test device at 0x00100000. Writing 0x5555 shuts QEMU down with
// exit code 0; any higher bits become the (shifted) exit code. See
// hw/misc/sifive_test.c in QEMU source.
#define SIFIVE_TEST       (*(volatile uint32_t *)0x00100000UL)
#define SIFIVE_TEST_PASS  0x5555u

static void uart_putc(char c) {
    while (!(UART_LSR & UART_LSR_TX_IDLE)) {}
    UART_THR = (uint8_t)c;
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

    // Clean exit so the harness doesn't need a SIGTERM.
    SIFIVE_TEST = SIFIVE_TEST_PASS;
    for (;;) {}
}
