//go:build darwin

// vmregions.h — a counter for the calling task's VM map entries.
//
// This is a diagnostic, not part of the Metal binding. It exists
// because the Metal shared-buffer churn in a training step was observed
// to grow the process's VM map region count by ~9000 per optimizer
// step with no plateau, while the physical footprint stayed flat (see
// doc/vm-region-growth.md). Attributing that growth needs a per-region
// count that is cheap enough to sample in a loop — `vmmap --summary`
// forks a process and walks the whole map, which is what made the old
// footprint guard pathological in the first place.

#ifndef GORCH_VMREGIONS_H
#define GORCH_VMREGIONS_H

#include <stdint.h>

// gorch_vm_region_count walks the calling task's VM map and returns the
// number of leaf (non-submap) entries.
//
// tagCounts, when non-NULL, receives a per-user-tag histogram: element
// i is the number of leaf entries whose VM user tag is i. Tags are the
// VM_MEMORY_* constants from <mach/vm_statistics.h> — notably
// VM_MEMORY_MALLOC* (1..11), VM_MEMORY_IOKIT (6) and
// VM_MEMORY_IOACCELERATOR (100). Tag 0 is untagged (VM_ALLOCATE, file
// mappings, the Go runtime's own reservations). nTags is the length of
// tagCounts; tags at or beyond it are counted only in the total.
//
// totalBytes, when non-NULL, receives the summed virtual size of the
// leaf entries.
//
// COST. One mach_vm_region_recurse trap per entry, each a lookup from
// the previous address, so the walk is O(n) traps. Measured on this
// project it stays sub-second into the low hundreds of thousands of
// entries, but it is emphatically not free — do not put it on a
// per-micro-step path. That mistake has been made once already.
uint64_t gorch_vm_region_count(uint32_t* tagCounts, uint32_t nTags,
                               uint64_t* totalBytes);

#endif // GORCH_VMREGIONS_H
