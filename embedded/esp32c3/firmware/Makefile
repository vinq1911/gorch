# Build the gorch int8 kernel + Iris fixture for bare-metal ESP32-C3.
# Uses Espressif's QEMU fork (qemu-system-riscv32 -machine esp32c3).
#
# The Go harness regenerates iris_model.h and iris_input.h before invoking
# make. Output: firmware.elf (QEMU loads ELF directly via -kernel).

CC      = riscv64-unknown-elf-gcc
# ESP32-C3 is rv32imc (base integer + multiply + compressed)
ARCH    = rv32imc
ABI     = ilp32

CFLAGS  = -march=$(ARCH) -mabi=$(ABI) \
          -Os -std=gnu11 -Wall -Wextra \
          -ffreestanding -fno-builtin -nostdlib \
          -ffunction-sections -fdata-sections

LDFLAGS = -march=$(ARCH) -mabi=$(ABI) \
          -nostdlib -Wl,--gc-sections \
          -T link.ld

OBJS = start.o firmware.o

all: firmware.elf

firmware.elf: $(OBJS) link.ld
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

start.o: start.S
	$(CC) $(CFLAGS) -c -o $@ $<

firmware.o: firmware.c ../../gm1_avr.h iris_model.h iris_input.h
	$(CC) $(CFLAGS) -c -o $@ firmware.c

clean:
	rm -f firmware.elf *.o

.PHONY: all clean
