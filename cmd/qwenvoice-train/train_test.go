//go:build darwin

package main

import (
	"math"
	"testing"
	"time"
)

// TestPhysFootprintMatchesVmmap pins the substitution that removed the
// per-step slowdown: task_info(TASK_VM_INFO)'s phys_footprint must be
// the same number `vmmap --summary` prints, or the memory ceiling now
// guards the wrong quantity — and that ceiling is what stands between a
// too-large config and a jetsam event that takes the desktop with it.
//
// The two are sampled at different instants in a live process, so they
// are compared with slack rather than for equality. vmmap also reports
// only ~2 significant figures ("2.2G"), which alone is ~5%.
//
// Scope, stated honestly: vmmap's "Physical footprint:" line very
// likely reports the same kernel counter this reads, so agreement is
// NOT an independent second measurement of the footprint. What this
// pins is that the replacement selects the right field, in the right
// units, and did not silently start reporting something else — which is
// the failure mode a wrong constant or a struct-layout change would
// produce. The independent evidence that the counter tracks reality is
// the ceiling firing on runs that were about to be jetsam-killed.
func TestPhysFootprintMatchesVmmap(t *testing.T) {
	got := physFootprintBytes()
	if got == 0 {
		t.Fatal("task_info returned no physical footprint — the ceiling would silently " +
			"fall back to the vmmap path it exists to replace")
	}
	gotMB := float64(got) / (1 << 20)
	wantMB := vmmapFootprintMB()
	if wantMB == 0 {
		t.Skip("vmmap unavailable")
	}
	if rel := math.Abs(gotMB-wantMB) / wantMB; rel > 0.10 {
		t.Fatalf("task_info footprint %.1f MB vs vmmap %.1f MB (%.1f%% apart) — these must "+
			"be the same quantity", gotMB, wantMB, rel*100)
	}
}

// TestPhysFootprintIsCheap is the regression that matters more than the
// value. The footprint guard runs once per accumulation micro-step and
// twice per optimizer step; when it forked vmmap, each call walked the
// process's VM region map, which grows as the Metal allocator churns
// (68,538 regions after five Stage-A steps -> 11.0 s per call). That
// instrument, not the model, was the 3x per-step ramp. Any future
// reader of the footprint must stay O(1).
func TestPhysFootprintIsCheap(t *testing.T) {
	physFootprintBytes() // warm
	start := time.Now()
	const n = 1000
	for i := 0; i < n; i++ {
		physFootprintBytes()
	}
	per := time.Since(start) / n
	if per > 100*time.Microsecond {
		t.Fatalf("physFootprintBytes costs %v per call — the footprint guard is on the "+
			"per-micro-step path and must not walk the VM map", per)
	}
	t.Logf("physFootprintBytes: %v per call", per)
}

func TestPadToBucket(t *testing.T) {
	for _, tc := range []struct {
		name        string
		in          []int
		bucket, max int
		wantLen     int
	}{
		{"disabled-zero", []int{1, 2, 3}, 0, 1024, 3},
		{"disabled-one", []int{1, 2, 3}, 1, 1024, 3},
		{"pads-up", []int{1, 2, 3}, 4, 1024, 4},
		{"already-on-boundary", []int{1, 2, 3, 4}, 4, 1024, 4},
		{"one-past-boundary", []int{1, 2, 3, 4, 5}, 4, 1024, 8},
		{"realistic", make([]int, 625), 128, 1024, 640},
		{"max-seq-boundary", make([]int, 1016), 128, 1024, 1024},
		{"never-exceeds-model-max", make([]int, 1000), 128, 1008, 1008},
		{"empty", nil, 128, 1024, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := append([]int(nil), tc.in...)
			out := padToBucket(in, tc.bucket, tc.max)
			if len(out) != tc.wantLen {
				t.Fatalf("len = %d, want %d", len(out), tc.wantLen)
			}
			// The caller's slice — which the dataset may share with
			// other samples — must not be mutated or extended in place.
			if len(in) != len(tc.in) {
				t.Fatalf("input slice length changed from %d to %d", len(tc.in), len(in))
			}
			for i := range in {
				if in[i] != tc.in[i] {
					t.Fatalf("input slice element %d mutated", i)
				}
			}
			// Real prefix preserved; padding is the last real token.
			for i := range in {
				if out[i] != in[i] {
					t.Fatalf("real token %d changed from %d to %d", i, in[i], out[i])
				}
			}
			if len(out) > len(in) {
				last := in[len(in)-1]
				for i := len(in); i < len(out); i++ {
					if out[i] != last {
						t.Fatalf("pad token %d is %d, want the last real token %d", i, out[i], last)
					}
				}
			}
		})
	}
}

// TestPadToBucketShapeConvergence is the actual point of bucketing:
// the set of distinct sequence lengths presented to the GPU must be
// bounded by max-seq/bucket, not by the number of distinct samples.
func TestPadToBucketShapeConvergence(t *testing.T) {
	seen := map[int]bool{}
	for n := 24; n <= 1016; n++ {
		seen[len(padToBucket(make([]int, n), 128, 1024))] = true
	}
	if len(seen) > 1024/128 {
		t.Fatalf("bucketing 993 distinct lengths still yields %d distinct padded lengths, "+
			"want <= %d", len(seen), 1024/128)
	}
}
