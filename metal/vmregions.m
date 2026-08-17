//go:build darwin

#include "vmregions.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>

uint64_t gorch_vm_region_count(uint32_t* tagCounts, uint32_t nTags,
                               uint64_t* totalBytes) {
    mach_vm_address_t addr = 0;
    uint64_t n = 0;
    uint64_t bytes = 0;

    // depth is the RECURSION LIMIT, not a per-call cursor, and must live
    // outside the loop. Declaring it inside and re-zeroing it every
    // iteration makes the submap case (depth++; continue) spin forever
    // on the same address — the shared region is a submap, so every
    // process hits it immediately.
    natural_t depth = 0;

    for (;;) {
        mach_vm_size_t size = 0;
        vm_region_submap_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;

        kern_return_t kr = mach_vm_region_recurse(
            mach_task_self(), &addr, &size, &depth,
            (vm_region_recurse_info_t)&info, &count);
        if (kr != KERN_SUCCESS) break; // KERN_INVALID_ADDRESS == end of map

        if (info.is_submap) {
            // Descend rather than skip: the shared region is a submap,
            // and its entries are real entries in this task's map.
            // Raising the limit is enough — the address cursor stays put
            // and the next call resolves the submap's first leaf.
            depth++;
            continue;
        }
        n++;
        bytes += (uint64_t)size;
        if (tagCounts && info.user_tag < nTags) tagCounts[info.user_tag]++;

        // Guard against a zero-size entry wedging the walk.
        if (size == 0) break;
        addr += size;
    }

    if (totalBytes) *totalBytes = bytes;
    return n;
}
