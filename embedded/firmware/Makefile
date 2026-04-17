# Build the gorch test firmware for ATmega2560 under simavr.
#
# The harness regenerates tiny_model.h and tiny_input.h before invoking make.
# Output: firmware.elf (simavr reads ELF directly, no .hex needed).

MCU      ?= atmega2560
F_CPU    ?= 16000000UL
CC       = avr-gcc
OBJCOPY  = avr-objcopy
SIMAVR_INC ?= /usr/include/simavr

CFLAGS = -Os -std=gnu11 -Wall -Wextra -Wno-unused-parameter \
         -mmcu=$(MCU) -DF_CPU=$(F_CPU) \
         -ffunction-sections -fdata-sections \
         -I$(SIMAVR_INC)

# Keep the .mmcu section: simavr reads it from the ELF to discover the MCU
# model, clock, console register, etc. We pin it at a high address that
# simavr extracts but never programs into the AVR (LMA outside flash).
LDFLAGS = -mmcu=$(MCU) -Wl,--section-start=.mmcu=0x910000 -Wl,--undefined=_mmcu

SRC = firmware.c
ELF = firmware.elf

all: $(ELF)

$(ELF): $(SRC) tiny_model.h tiny_input.h ../gm1_avr.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC)

clean:
	rm -f $(ELF) *.o

.PHONY: all clean
