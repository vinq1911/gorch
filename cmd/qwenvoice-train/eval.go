//go:build darwin

package main

import (
	"fmt"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model/qwen"
)

// evalRegen runs the overfit-gate measurement (plan 0008 §3.6): mean
// supervised-token loss over every sample, and greedy regeneration of
// each sample's supervised target span with exact-match counting.
func evalRegen(c cliConfig) error {
	// Eval runs the CPU f32 path regardless of -accel: greedy span
	// regeneration is the KV-cached decode loop, whose per-token
	// single-row matmuls would hit the bf16 widen-per-call fallback.
	c.accel = "off"
	vm, err := loadVoiceModel(c)
	if err != nil {
		return err
	}
	ds, _, err := loadDataset(c)
	if err != nil {
		return err
	}

	ckptPath := c.resume
	if ckptPath == "auto" || ckptPath == "" {
		p, ok := qwen.LatestCheckpoint(c.outDir)
		if !ok {
			return fmt.Errorf("no checkpoint in %s", c.outDir)
		}
		ckptPath = p
	}
	meta, err := qwen.LoadCheckpoint(ckptPath, vm, nil)
	if err != nil {
		return err
	}
	fmt.Printf("eval: %s (step %d), %d samples\n", ckptPath, meta.Step, ds.Len())

	var lossSum float64
	exact := 0
	perTaskExact := map[string]int{}
	perTaskTotal := map[string]int{}
	for i := 0; i < ds.Len(); i++ {
		tokens, sup, task := ds.Get(i)
		perTaskTotal[task]++

		var lv float32
		g.NoGrad(func() {
			lv = vm.SupervisedLoss(tokens, sup).Data()[0]
		})
		lossSum += float64(lv)

		prompt, target := regenSplit(tokens, sup)
		maxNew := len(target)
		if maxNew > c.evalMaxNew {
			maxNew = c.evalMaxNew
		}
		gen := vm.GenerateGreedy(prompt, maxNew, nil)
		ok := len(gen) == len(target)
		match := 0
		for j := range gen {
			if j < len(target) && gen[j] == target[j] {
				match++
			} else {
				ok = false
			}
		}
		if ok {
			exact++
			perTaskExact[task]++
		}
		status := "MISS"
		if ok {
			status = "OK"
		}
		fmt.Printf("sample %3d [%s] loss %.4f regen %d/%d %s\n", i, task, lv, match, len(target), status)
	}

	fmt.Printf("\nmean supervised loss: %.4f over %d samples\n", lossSum/float64(ds.Len()), ds.Len())
	fmt.Printf("exact span regeneration: %d/%d", exact, ds.Len())
	for task, tot := range perTaskTotal {
		fmt.Printf("  %s %d/%d", task, perTaskExact[task], tot)
	}
	fmt.Println()
	gate1 := lossSum/float64(ds.Len()) < 0.1
	gate2 := exact*10 >= ds.Len()*9
	fmt.Printf("gate: loss<0.1 %v, exact≥90%% %v\n", gate1, gate2)
	return nil
}
