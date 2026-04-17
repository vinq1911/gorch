# Build the host-side libsimavr runner.
#
# Produces `avr_runner`, invoked by the Go harness to run firmware ELFs with
# console output piped to stdout.

CC     ?= cc
CFLAGS ?= -O2 -Wall -Wextra -I/usr/include/simavr
LDLIBS ?= -lsimavr -lelf

all: avr_runner

avr_runner: runner.c
	$(CC) $(CFLAGS) -o $@ runner.c $(LDLIBS)

clean:
	rm -f avr_runner

.PHONY: all clean
