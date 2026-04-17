# `embedded/riscv` — gorch on 32-bit RISC-V (ESP32-C3 class)

This is the sibling of `embedded/` (AVR / ATmega2560) for a **32-bit
RISC-V embedded target**, emulated under upstream QEMU's `virt` machine.
The same `gm1_avr.h` kernel runs here unchanged — it's already portable
to any target with an `int8`/`int32` integer unit.

## What runs here

A **trained Iris classifier**:

- 4 features → 8-unit hidden layer → 3 classes
- Trained in pure Go with plain SGD on a 90-sample balanced subset of
  Fisher's Iris dataset
- Post-training quantized to int8 using training-set calibration
- **98.9% accuracy** (89/90) on the full dataset, measured by running
  every sample through the QEMU firmware and comparing against ground
  truth

## Why RISC-V virt and not the Espressif ESP32 QEMU fork

Upstream QEMU's `virt` machine is:

- `apt`-installable (`qemu-system-misc`),
- representative of ESP32-C3-class hardware (same arch, same 32-bit
  word size, same level of peripheral abstraction we care about),
- CI-friendly (no ~50 MB vendor binary, no 2 GB ESP-IDF, no FreeRTOS).

The `gm1_avr.h` kernel, the Go reference, the `TinyModel` format, and
the harness pattern all transfer **unchanged** to an ESP32-C3 QEMU build.
Only the firmware wrapper (startup code, UART driver, exit method)
differs. The firmware under `firmware/` is small enough (~100 LOC) that
porting it to a real ESP32-C3 ESP-IDF project would take an afternoon.

## Pipeline

```
Iris dataset (iris/data.go, 90 samples)
          │
          ▼
Go SGD training (iris/train.go, float32)
          │
          ▼  + training-set calibration
Post-training int8 quantization (iris/train.go:Quantize)
          │
          ▼
embedded.TinyModel
  ├─ emits iris_model.h  (int8 weights + Q0.31 requant M/S)
  └─ emits iris_input.h  (the quantized test sample)
          │
          ▼
riscv64-unknown-elf-gcc (rv32im, -nostdlib, -ffreestanding)
          │
          ▼
firmware.elf, loaded at 0x80000000
          │
          ▼
qemu-system-riscv32 -machine virt -bios none
          │
          ▼  UART0 (16550 @ 0x10000000) → Go harness stdout
"CLASS <n> LOGITS <l0> <l1> <l2>\n"
```

## Running it

One-off deps:

```
apt-get install qemu-system-misc \
                gcc-riscv64-unknown-elf \
                picolibc-riscv64-unknown-elf
```

Tests:

```
go test -v ./embedded/riscv/
# ~3 seconds: 3 per-class runs, round-trip through QEMU

go test -v -run Accuracy ./embedded/riscv/
# ~30 seconds: all 90 Iris samples, accuracy assertion >= 0.90
```

The test `t.Skip`s cleanly if the RISC-V toolchain or QEMU isn't on
PATH, so CI on a minimal machine still runs the pure-Go tests in
`iris/`.

## Footprint

`riscv64-unknown-elf-objdump -h firmware.elf` on a real Iris build:

| Section  | Size   | Notes                              |
| -------- | ------ | ---------------------------------- |
| .text    | ~900 B | kernel + UART + main               |
| .rodata  | ~220 B | weights (32+8+24+3 bytes) + strings |
| .bss     | ~50 B  | ping-pong buffers + logits         |

Total firmware size well under 2 KB. ESP32-C3 has 400 KB of SRAM and
4 MB of flash; this leaves room for ~two-thousand-times-larger models.

## Scaling up

The same harness will validate:

- **MNIST MLP** (784→64→10): 51 KB weights, ~1.5 KB activations, still
  trivially inside ESP32-C3's budget.
- **Small CNNs** if `KindConv2d` is added to the kernel (im2col-on-the-
  fly to bound activation memory).
- **Quantization-aware training** once gorch's exporter is wired in —
  right now we do post-training quantization, which is a single pass
  over the training set.
