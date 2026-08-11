//go:build darwin

package qwen

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

// tinyCfg is a fast random-weight config for training-mechanics tests.
func tinyCfg() Config {
	return Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumQueryHeads:    4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        64,
		MaxSeq:           128,
		RopeTheta:        10000,
		RMSNormEps:       1e-6,
		UseQKNorm:        true,
		TiedEmbeddings:   true,
	}
}

func tinyVoice(t *testing.T) *VoiceModel {
	t.Helper()
	return NewVoiceModel(New(tinyCfg()), VoiceConfig{LoRARank: 2, LoRAAlpha: 4, NumExt: 16})
}

// TestVoiceSupervisedLossMatchesFullReference — the gathered loss
// (Gather rows → head on gathered rows → CE) equals a full-sequence
// reference (head on ALL rows → gather logits → CE).
func TestVoiceSupervisedLossMatchesFullReference(t *testing.T) {
	vm := tinyVoice(t)
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2}
	supervised := []int{2, 3, 5, 6}

	loss := vm.SupervisedLoss(tokens, supervised)

	h := vm.ForwardHidden(tokens)
	fullLogits := vm.Embed.Logits(h)
	gathered := g.Gather(fullLogits, supervised)
	targets := make([]float32, len(supervised))
	for i, p := range supervised {
		targets[i] = float32(tokens[p+1])
	}
	ref := g.CrossEntropyLoss(gathered, g.NewTensor(targets, len(targets), 1))

	d := math.Abs(float64(loss.Data()[0]) - float64(ref.Data()[0]))
	if d > 1e-5 {
		t.Fatalf("gathered loss %v vs full reference %v (|Δ| %.3g)", loss.Data()[0], ref.Data()[0], d)
	}
}

// TestVoiceGradReachesOnlyTrainableState — after backward, every LoRA
// factor and Ext have gradients; every frozen base tensor has none.
func TestVoiceGradReachesOnlyTrainableState(t *testing.T) {
	vm := tinyVoice(t)
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2}
	loss := vm.SupervisedLoss(tokens, []int{1, 2, 3, 4, 5, 6})
	loss.Backward()

	_, params := vm.TrainableParams()
	for i, p := range params {
		if p.Grad() == nil {
			t.Fatalf("trainable param %d received no gradient", i)
		}
	}
	for i, p := range vm.Base.Parameters() {
		if p.Grad() != nil {
			t.Fatalf("frozen base param %d received a gradient", i)
		}
	}
}

// TestVoiceGenerateUsesAdapters — cached greedy generation runs over
// the extended vocab and is deterministic; and adapter state affects
// the logits (LoRA is live in the decode path).
func TestVoiceGenerateUsesAdapters(t *testing.T) {
	vm := tinyVoice(t)
	prompt := []int{1, 70, 5}

	out1 := vm.GenerateGreedy(prompt, 4, nil)
	out2 := vm.GenerateGreedy(prompt, 4, nil)
	if len(out1) != 4 || len(out2) != 4 {
		t.Fatalf("generated %d/%d tokens, want 4", len(out1), len(out2))
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatal("greedy generation not deterministic")
		}
		if out1[i] < 0 || out1[i] >= vm.Embed.VocabSize() {
			t.Fatalf("generated id %d outside extended vocab", out1[i])
		}
	}

	// Perturb one adapter's B hard; logits path must change.
	cache1 := vm.Base.NewCache()
	before := vm.ForwardCached(prompt, cache1).Data()[0]
	b := vm.Adapters()[0].B
	for i := range b.Data() {
		b.Data()[i] += 5
	}
	cache2 := vm.Base.NewCache()
	after := vm.ForwardCached(prompt, cache2).Data()[0]
	if before == after {
		t.Fatal("LoRA adapter change did not affect cached-decode logits")
	}
}

// trainSteps drives n identical optimizer steps on a fixed sample.
func trainSteps(vm *VoiceModel, opt *optim.AdamW, n int) []float32 {
	tokens := []int{1, 5, 70, 71, 3, 9, 64, 2, 40, 41}
	sup := []int{2, 3, 4, 5, 6, 7, 8}
	losses := make([]float32, n)
	for i := 0; i < n; i++ {
		loss := vm.SupervisedLoss(tokens, sup)
		loss.Backward()
		opt.Step()
		opt.ZeroGrad()
		losses[i] = loss.Data()[0]
	}
	return losses
}

func newOpt(vm *VoiceModel) *optim.AdamW {
	_, params := vm.TrainableParams()
	lora := params[:len(params)-1]
	ext := params[len(params)-1:]
	return optim.NewAdamWGroups([]optim.ParamGroup{
		{Params: lora, LR: 1e-3},
		{Params: ext, LR: 5e-3},
	}, 0)
}

// TestVoiceCheckpointResume — kill-and-resume: save at step 3,
// restore into a FRESH tiny model, continue — parameters and losses
// must track the uninterrupted run exactly (f32, same GEMM shapes).
func TestVoiceCheckpointResume(t *testing.T) {
	dir := t.TempDir()

	vmA := NewVoiceModel(New(tinyCfg()), VoiceConfig{LoRARank: 2, LoRAAlpha: 4, NumExt: 16})
	optA := newOpt(vmA)
	trainSteps(vmA, optA, 3)
	if _, err := SaveCheckpoint(dir, vmA, optA, CheckpointMeta{Step: 3, DatasetSeed: 42, DatasetDraws: 3}, 3); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}
	lossesA := trainSteps(vmA, optA, 4)

	// Fresh model with different random init; the frozen base is what
	// a real resume re-loads from the same pretrained checkpoint, so
	// copy it over explicitly. All TRAINABLE state (perturbed below)
	// must be overwritten by LoadCheckpoint.
	vmB := NewVoiceModel(New(tinyCfg()), VoiceConfig{LoRARank: 2, LoRAAlpha: 4, NumExt: 16})
	baseA, baseB := vmA.Base.Parameters(), vmB.Base.Parameters()
	for i := range baseA {
		copy(baseB[i].Data(), baseA[i].Data())
	}
	for _, p := range vmB.LoRAParams() {
		for i := range p.Data() {
			p.Data()[i] += 0.123
		}
	}
	optB := newOpt(vmB)
	path, ok := LatestCheckpoint(dir)
	if !ok {
		t.Fatal("no checkpoint found")
	}
	meta, err := LoadCheckpoint(path, vmB, optB)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	if meta.Step != 3 || meta.DatasetSeed != 42 || meta.DatasetDraws != 3 || meta.AdamStep != 3 {
		t.Fatalf("meta roundtrip: %+v", meta)
	}
	lossesB := trainSteps(vmB, optB, 4)

	for i := range lossesA {
		d := math.Abs(float64(lossesA[i]) - float64(lossesB[i]))
		if d > 1e-6 {
			t.Fatalf("resumed loss[%d] = %v vs uninterrupted %v (|Δ| %.3g)", i, lossesB[i], lossesA[i], d)
		}
	}
	nA, pA := vmA.TrainableParams()
	_, pB := vmB.TrainableParams()
	for i := range pA {
		for j := range pA[i].Data() {
			if pA[i].Data()[j] != pB[i].Data()[j] {
				t.Fatalf("param %s diverged after resume at element %d", nA[i], j)
			}
		}
	}
}

// TestVoiceCheckpointKeepLast — pruning keeps only the newest 3.
func TestVoiceCheckpointKeepLast(t *testing.T) {
	dir := t.TempDir()
	vm := tinyVoice(t)
	opt := newOpt(vm)
	trainSteps(vm, opt, 1)
	for step := 1; step <= 5; step++ {
		if _, err := SaveCheckpoint(dir, vm, opt, CheckpointMeta{Step: step}, 3); err != nil {
			t.Fatalf("SaveCheckpoint step %d: %v", step, err)
		}
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "ckpt-*.safetensors"))
	if len(matches) != 3 {
		t.Fatalf("expected 3 kept checkpoints, found %d: %v", len(matches), matches)
	}
	if _, err := os.Stat(filepath.Join(dir, "ckpt-000002.safetensors")); err == nil {
		t.Fatal("old checkpoint 2 not pruned")
	}
	latest, ok := LatestCheckpoint(dir)
	if !ok || filepath.Base(latest) != "ckpt-000005.safetensors" {
		t.Fatalf("LatestCheckpoint = %q, %v", latest, ok)
	}
	for _, m := range matches {
		if _, err := os.Stat(sidecarFor(m)); err != nil {
			t.Fatalf("sidecar missing for %s", m)
		}
	}
}

// TestIsDeeperLayerKey — the truncated-load knob's key filter.
func TestIsDeeperLayerKey(t *testing.T) {
	cases := []struct {
		key  string
		want bool
	}{
		{"model.layers.4.self_attn.q_proj.weight", true},
		{"model.layers.27.mlp.up_proj.weight", true},
		{"model.layers.3.mlp.up_proj.weight", false},
		{"model.layers.0.input_layernorm.weight", false},
		{"model.embed_tokens.weight", false},
		{"model.norm.weight", false},
	}
	for _, c := range cases {
		if got := isDeeperLayerKey(c.key, 4); got != c.want {
			t.Errorf("isDeeperLayerKey(%q, 4) = %v", c.key, got)
		}
	}
}
