// Host-side runner that loads an AVR ELF via libsimavr, hooks UART0's
// TX IRQ, and prints each transmitted byte to stdout. The firmware is
// expected to use standard UART0 TX (check UDRE0, write UDR0). Exits
// when the simulated core halts, the cycle budget is exhausted, or the
// firmware emits a '\n' and then no new bytes for the grace period.
//
// Usage: avr_runner <firmware.elf>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sim_avr.h"
#include "sim_elf.h"
#include "sim_io.h"
#include "sim_irq.h"
#include "avr_uart.h"

static volatile int seen_newline_at = -1;
static long long current_cycle = 0;

static void uart_tx_hook(struct avr_irq_t *irq, uint32_t value, void *param) {
    (void)irq; (void)param;
    char c = (char)value;
    fputc(c, stdout);
    fflush(stdout);
    if (c == '\n' && seen_newline_at < 0) {
        seen_newline_at = (int)(current_cycle & 0x7fffffff);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <firmware.elf>\n", argv[0]);
        return 2;
    }

    elf_firmware_t fw;
    memset(&fw, 0, sizeof(fw));
    if (elf_read_firmware(argv[1], &fw) != 0) {
        fprintf(stderr, "runner: elf_read_firmware(%s) failed\n", argv[1]);
        return 2;
    }
    if (fw.mmcu[0] == 0) {
        fprintf(stderr, "runner: ELF has no .mmcu section (use AVR_MCU macro)\n");
        return 2;
    }

    avr_t *avr = avr_make_mcu_by_name(fw.mmcu);
    if (!avr) {
        fprintf(stderr, "runner: unknown MCU '%s'\n", fw.mmcu);
        return 3;
    }
    avr_init(avr);
    avr_load_firmware(avr, &fw);

    // Hook UART0 transmitter: every byte the firmware writes to UDR0 fires
    // UART_IRQ_OUTPUT on the '0' UART. Not available on all MCUs but works
    // on every ATmega we care about.
    struct avr_irq_t *tx_irq = avr_io_getirq(avr, AVR_IOCTL_UART_GETIRQ('0'), UART_IRQ_OUTPUT);
    if (!tx_irq) {
        fprintf(stderr, "runner: no UART0 on MCU '%s'\n", fw.mmcu);
        return 3;
    }
    avr_irq_register_notify(tx_irq, uart_tx_hook, NULL);

    // Drop AVR_UART_FLAG_POLL_SLEEP so the sim doesn't idle while the
    // firmware polls UDRE0. AVR_UART_FLAG_STDIO is off so transmitted
    // bytes go through our hook, not simavr's built-in line printer.
    uint32_t flags = 0;
    avr_ioctl(avr, AVR_IOCTL_UART_GET_FLAGS('0'), &flags);
    flags &= ~AVR_UART_FLAG_POLL_SLEEP;
    flags &= ~AVR_UART_FLAG_STDIO;
    avr_ioctl(avr, AVR_IOCTL_UART_SET_FLAGS('0'), &flags);

    // Run until halt, newline + grace period, or the cycle cap. The grace
    // is needed because the firmware typically prints a final line and
    // then spins.
    const long long max_cycles = 400000000LL;
    const long long grace_cycles = 200000LL;
    for (current_cycle = 0; current_cycle < max_cycles; current_cycle++) {
        int state = avr_run(avr);
        if (state == cpu_Done || state == cpu_Crashed) break;
        if (seen_newline_at >= 0 &&
            (current_cycle - seen_newline_at) > grace_cycles) {
            break;
        }
    }
    fflush(stdout);
    return 0;
}
