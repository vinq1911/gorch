// Minimal RISC-V startup for QEMU virt. Sets the stack pointer and jumps
// to main. No .bss zeroing is needed because QEMU provides zeroed DRAM
// and we only have small .bss that's effectively zero-initialized by
// the loader.
    .section .text.init
    .globl _start
_start:
    la  sp, _stack_top
    call main
1:  j   1b
