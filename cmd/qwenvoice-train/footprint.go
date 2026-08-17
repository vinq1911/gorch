//go:build darwin

package main

/*
#include <mach/mach.h>
#include <mach/task_info.h>
#include <string.h>

// gorch_phys_footprint returns this process's PHYSICAL FOOTPRINT in
// bytes — the same quantity vmmap prints as "Physical footprint:" and
// the one macOS jetsam acts on — via a single task_info() trap.
//
// phys_footprint lives in TASK_VM_INFO's REV1 tail, so the returned
// count must be checked: an older kernel can succeed while returning a
// struct that stops short of the field.
static uint64_t gorch_phys_footprint(void) {
    task_vm_info_data_t info;
    memset(&info, 0, sizeof(info));
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                                 (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS || count < TASK_VM_INFO_REV1_COUNT) {
        return 0;
    }
    return (uint64_t)info.phys_footprint;
}
*/
import "C"

// physFootprintBytes reports the process's physical footprint, or 0 if
// the kernel would not supply it.
//
// WHY THIS EXISTS, AND WHY IT IS NOT `vmmap --summary` (2026-08-17).
// The footprint ceiling is the guard that stands between a too-large
// (accum, max-seq) and a jetsam event that takes the desktop down with
// it, so it is checked once per accumulation MICRO-step, plus twice
// more per optimizer step. It used to be read by forking `vmmap
// --summary $pid` and parsing the line.
//
// vmmap walks the target's entire VM region map. A training micro-step
// allocates and frees ~10,000 Metal buffers, and the region map
// fragments as it churns: five optimizer steps in, the trainer had
// 68,538 regions and ONE vmmap call took 11.0 s. At 4-6 calls per step
// that is 40-70 s of pure instrumentation on top of a ~12 s step —
// and it GROWS with process age, because the region count does.
//
// That is the whole of the "step time ramps 3x within a process"
// symptom (measured 2026-08-17): a 10-step Stage-A run ramped
// 33.9 -> 55.6 s/step while the summed wall time of the actual
// forward/backward micro-steps stayed flat at 2.4-8.0 s and tracked
// sequence length alone, and while the footprint it was measuring sat
// at 2048 -> 2253 MB. The compiled-MPSGraph cache — the prior suspect,
// which does grow +10 entries per micro-step — was exonerated
// separately: fixed-shape matmul time is flat from 1 to 600 cached
// graphs and a fresh compile is a constant ~10 ms (TestDTCacheSizeProbe
// in metal/). A fresh process looked fast for the same reason it looked
// thermal-innocent: it had a small region map, not a cool GPU.
//
// task_info(TASK_VM_INFO) returns the identical number the kernel hands
// vmmap, without the fork, the exec, the region walk or the text
// parsing — microseconds, and constant in the region count. The guard
// keeps its semantics and its cadence; only its price changes.
func physFootprintBytes() uint64 {
	return uint64(C.gorch_phys_footprint())
}
