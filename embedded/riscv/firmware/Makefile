# Build the gorch int8 kernel + Iris test fixture for bare-metal RISC-V
# (rv32im). The Go harness regenerates iris_model.h and iris_input.h
# before invoking make.
#
# Output: firmware.elf (QEMU loads ELF directly; no .bin needed).

CC      = riscv64-unknown-elf-gcc
ARCH    = rv32im
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
