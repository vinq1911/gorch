//go:build darwin

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#include "vmregions.h"
*/
import "C"

// VMRegionTag names the VM user tags this package cares about. The
// values are the VM_MEMORY_* constants from <mach/vm_statistics.h>.
const (
	// VMTagNone is the untagged bucket: plain vm_allocate, file
	// mappings, and the Go runtime's own arena reservations.
	VMTagNone = 0
	// VMTagMallocSmall/Large/Huge are libmalloc's zones. Growth here
	// would mean the churn is in Go's cgo/C allocations, not Metal's.
	VMTagMallocSmall = 3
	VMTagMallocLarge = 4
	VMTagMallocHuge  = 5
	// VMTagIOKit is the generic IOKit mapping tag.
	VMTagIOKit = 6
	// VMTagIOAccelerator is what the Metal/AGX driver tags its buffer
	// mappings with. Growth here means the Metal allocator.
	VMTagIOAccelerator = 100
)

const vmTagLimit = 256

// VMRegionStats is a snapshot of the calling process's VM map.
type VMRegionStats struct {
	// Total is the number of leaf (non-submap) VM map entries. This is
	// the number `vmmap --summary` reports on its TOTAL line, modulo
	// vmmap's own coalescing of adjacent identical regions in the
	// display — the trend is what matters, not the absolute agreement.
	Total int
	// Bytes is the summed virtual size of those entries. Virtual, not
	// physical: a purged or never-touched region contributes its full
	// size here while costing no physical pages.
	Bytes int64
	// ByTag is a histogram of leaf entries by VM user tag, indexed by
	// the VMTag* constants. Length is always vmTagLimit.
	ByTag []int
}

// Tag returns the entry count for one VM user tag.
func (s VMRegionStats) Tag(tag int) int {
	if tag < 0 || tag >= len(s.ByTag) {
		return 0
	}
	return s.ByTag[tag]
}

// VMRegionSnapshot walks the calling task's VM map and returns the leaf
// entry count, total virtual bytes, and a per-tag histogram.
//
// This is a DIAGNOSTIC. The walk costs one mach trap per entry, so it
// is O(entries) — cheap at a few thousand, tens of milliseconds at a
// few hundred thousand. Never call it on a per-step path.
func VMRegionSnapshot() VMRegionStats {
	tags := make([]C.uint32_t, vmTagLimit)
	var bytes C.uint64_t
	n := C.gorch_vm_region_count(&tags[0], C.uint32_t(vmTagLimit), &bytes)

	byTag := make([]int, vmTagLimit)
	for i, v := range tags {
		byTag[i] = int(v)
	}
	return VMRegionStats{Total: int(n), Bytes: int64(bytes), ByTag: byTag}
}

// VMRegionCount is VMRegionSnapshot without the histogram.
func VMRegionCount() int { return VMRegionSnapshot().Total }
