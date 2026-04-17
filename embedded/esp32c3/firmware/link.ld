/* ESP32-C3 IRAM linker script for Espressif QEMU.
   IRAM starts at 0x40380000 (384 KB). The kernel is loaded here by
   qemu-system-riscv32 -machine esp32c3 -kernel firmware.elf. */

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    IRAM (rwx) : ORIGIN = 0x40380000, LENGTH = 384K
}

SECTIONS
{
    .text : {
        *(.text.init)
        *(.text*)
    } > IRAM

    .rodata : {
        *(.rodata*)
        *(.srodata*)
    } > IRAM

    .data : {
        *(.data*)
        *(.sdata*)
    } > IRAM

    .bss : {
        *(.bss*)
        *(.sbss*)
        *(COMMON)
    } > IRAM

    . = ALIGN(16);
    . = . + 0x2000;
    _stack_top = .;
}
