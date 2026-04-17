/* Linker script for QEMU riscv32 -machine virt. DRAM starts at
   0x80000000; we only need a few KB for this firmware. */

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128K
}

SECTIONS
{
    .text : {
        *(.text.init)
        *(.text*)
    } > RAM

    .rodata : {
        *(.rodata*)
        *(.srodata*)
    } > RAM

    .data : {
        *(.data*)
        *(.sdata*)
    } > RAM

    .bss : {
        _bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        _bss_end = .;
    } > RAM

    . = ALIGN(16);
    . = . + 0x2000;
    _stack_top = .;
}
