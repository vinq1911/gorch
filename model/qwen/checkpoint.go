//go:build darwin

package qwen

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/optim"
)

// Checkpoint/resume for the M1 trainer (plan 0008 §3.4): one
// safetensors file per checkpoint holding the trainable parameters
// (LoRA A/B + Ext rows) and the AdamW m/v moments, plus a JSON
// sidecar with the scalar training state. Keep-last-3 pruning.

// CheckpointMeta is the JSON sidecar: everything scalar a resumed run
// needs to reproduce the uninterrupted trajectory (the schedule is a
// pure function of Step; the dataset stream replays from seed+draws).
type CheckpointMeta struct {
	Step         int                `json:"step"`          // optimizer steps completed
	MicroStep    int64              `json:"micro_step"`    // sequences consumed
	AdamStep     int                `json:"adam_step"`     // AdamW timestep t
	BaseLR       float32            `json:"base_lr"`       // base LR at save time (sanity)
	DatasetSeed  int64              `json:"dataset_seed"`  // data.DatasetState
	DatasetDraws int64              `json:"dataset_draws"` //
	TaskRatios   map[string]float64 `json:"task_ratios,omitempty"`
}

// SaveCheckpoint writes ckpt-{step:06d}.safetensors + .json into dir
// atomically (temp + rename) and prunes to the newest keepLast
// checkpoints (keepLast ≤ 0 keeps everything). Returns the checkpoint
// path.
func SaveCheckpoint(dir string, vm *VoiceModel, opt *optim.AdamW, meta CheckpointMeta, keepLast int) (string, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	names, params := vm.TrainableParams()
	tensors := make(map[string]*g.Tensor, 3*len(names))
	for i, name := range names {
		tensors[name] = params[i]
	}
	adamStep, m, v := opt.StateTensors()
	if len(m) != len(params) {
		return "", fmt.Errorf("qwen: optimizer tracks %d params, model has %d trainable", len(m), len(params))
	}
	meta.AdamStep = adamStep
	for i, name := range names {
		tensors["adamw.m/"+name] = g.NewTensor(m[i], len(m[i]))
		tensors["adamw.v/"+name] = g.NewTensor(v[i], len(v[i]))
	}

	stem := filepath.Join(dir, fmt.Sprintf("ckpt-%06d", meta.Step))
	metaBytes, err := json.MarshalIndent(&meta, "", " ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(stem+".json.tmp", metaBytes, 0644); err != nil {
		return "", err
	}
	if err := model.SaveSafetensors(stem+".safetensors.tmp", tensors); err != nil {
		return "", err
	}
	// Rename the sidecar first: LatestCheckpoint requires BOTH files,
	// so a crash between the renames leaves no half-visible checkpoint.
	if err := os.Rename(stem+".json.tmp", stem+".json"); err != nil {
		return "", err
	}
	if err := os.Rename(stem+".safetensors.tmp", stem+".safetensors"); err != nil {
		return "", err
	}

	if keepLast > 0 {
		if all, err := listCheckpoints(dir); err == nil && len(all) > keepLast {
			for _, old := range all[:len(all)-keepLast] {
				os.Remove(old)
				os.Remove(sidecarFor(old))
			}
		}
	}
	return stem + ".safetensors", nil
}

// LatestCheckpoint returns the newest complete checkpoint in dir
// (highest step with both files present), or ok=false when none.
func LatestCheckpoint(dir string) (path string, ok bool) {
	all, err := listCheckpoints(dir)
	if err != nil || len(all) == 0 {
		return "", false
	}
	return all[len(all)-1], true
}

func sidecarFor(ckptPath string) string {
	return ckptPath[:len(ckptPath)-len(".safetensors")] + ".json"
}

// listCheckpoints returns complete checkpoints sorted oldest→newest.
func listCheckpoints(dir string) ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "ckpt-*.safetensors"))
	if err != nil {
		return nil, err
	}
	var out []string
	for _, m := range matches {
		if _, err := os.Stat(sidecarFor(m)); err == nil {
			out = append(out, m)
		}
	}
	sort.Strings(out)
	return out, nil
}

// LoadCheckpoint restores trainable parameters (and, when opt is
// non-nil, the AdamW moments + timestep) from a checkpoint written by
// SaveCheckpoint. Every trainable tensor and moment must be present
// with the exact size — a partial checkpoint is an error, not a warn.
func LoadCheckpoint(path string, vm *VoiceModel, opt *optim.AdamW) (CheckpointMeta, error) {
	var meta CheckpointMeta
	metaBytes, err := os.ReadFile(sidecarFor(path))
	if err != nil {
		return meta, err
	}
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return meta, fmt.Errorf("qwen: checkpoint sidecar %s: %w", sidecarFor(path), err)
	}
	sf, err := model.LoadSafetensors(path)
	if err != nil {
		return meta, err
	}

	names, params := vm.TrainableParams()
	take := func(name string, wantSize int) ([]float32, error) {
		t, ok := sf.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("qwen: checkpoint %s missing tensor %q", path, name)
		}
		if t.Size() != wantSize {
			return nil, fmt.Errorf("qwen: checkpoint tensor %q has %d elements, want %d", name, t.Size(), wantSize)
		}
		return t.Data(), nil
	}
	for i, name := range names {
		d, err := take(name, params[i].Size())
		if err != nil {
			return meta, err
		}
		copy(params[i].Data(), d)
	}
	if opt != nil {
		m := make([][]float32, len(names))
		v := make([][]float32, len(names))
		for i, name := range names {
			if m[i], err = take("adamw.m/"+name, params[i].Size()); err != nil {
				return meta, err
			}
			if v[i], err = take("adamw.v/"+name, params[i].Size()); err != nil {
				return meta, err
			}
		}
		if err := opt.LoadState(meta.AdamStep, m, v); err != nil {
			return meta, err
		}
	}
	return meta, nil
}
