//go:build darwin

package data

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeTestShard(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	var samples []ShardSample
	// 6 listen, 3 speak, 3 chain samples with recognisable payloads.
	mk := func(task string, tag, n int) ShardSample {
		toks := make([]int, n)
		for i := range toks {
			toks[i] = tag*1000 + i
		}
		return ShardSample{Tokens: toks, Task: task, Supervised: []Span{{Start: n / 2, End: n - 1}}}
	}
	for i := 0; i < 6; i++ {
		samples = append(samples, mk("listen", i, 10+i))
	}
	for i := 0; i < 3; i++ {
		samples = append(samples, mk("speak", 10+i, 12))
	}
	for i := 0; i < 3; i++ {
		samples = append(samples, mk("chain", 20+i, 30))
	}
	p := filepath.Join(dir, "test.bin")
	if err := WriteTokenShard(p, samples); err != nil {
		t.Fatalf("WriteTokenShard: %v", err)
	}
	return p
}

func TestTokenShardRoundtripAndGet(t *testing.T) {
	p := writeTestShard(t)
	ds, err := LoadTokenDataset([]string{p}, TokenDatasetConfig{Seed: 1})
	if err != nil {
		t.Fatalf("LoadTokenDataset: %v", err)
	}
	if ds.Len() != 12 {
		t.Fatalf("Len = %d, want 12", ds.Len())
	}
	if got := ds.Tasks(); !reflect.DeepEqual(got, []string{"chain", "listen", "speak"}) {
		t.Fatalf("Tasks = %v", got)
	}
	toks, sup, task := ds.Get(0)
	if task != "listen" || len(toks) != 10 || toks[0] != 0 || toks[9] != 9 {
		t.Fatalf("Get(0) = %v tokens task %s", toks, task)
	}
	if !reflect.DeepEqual(sup, []int{5, 6, 7, 8}) {
		t.Fatalf("Get(0) supervised = %v", sup)
	}
}

func TestTokenDatasetDeterminismAndRestore(t *testing.T) {
	p := writeTestShard(t)
	cfg := TokenDatasetConfig{Seed: 42, MaxTrainSeq: 20,
		TaskRatios: map[string]float64{"listen": 0.4, "speak": 0.4, "chain": 0.2}}

	load := func() *TokenDataset {
		ds, err := LoadTokenDataset([]string{p}, cfg)
		if err != nil {
			t.Fatalf("LoadTokenDataset: %v", err)
		}
		return ds
	}
	type draw struct {
		toks []int
		sup  []int
		task string
	}
	drawN := func(ds *TokenDataset, n int) []draw {
		out := make([]draw, n)
		for i := range out {
			tk, sp, ta := ds.Sample()
			out[i] = draw{tk, sp, ta}
		}
		return out
	}

	ds1 := load()
	seq1 := drawN(ds1, 60)
	ds2 := load()
	seq2 := drawN(ds2, 60)
	if !reflect.DeepEqual(seq1, seq2) {
		t.Fatal("same seed, different draw sequences")
	}

	// Task-ratio sanity over 60 draws.
	counts := map[string]int{}
	for _, d := range seq1 {
		counts[d.task]++
	}
	if counts["listen"] < 12 || counts["speak"] < 12 || counts["chain"] < 3 {
		t.Fatalf("task mix implausible for 0.4/0.4/0.2: %v", counts)
	}

	// Truncation: chain samples have 30 tokens, cap is 20; supervised
	// positions must stay < 19.
	for _, d := range seq1 {
		if d.task == "chain" {
			if len(d.toks) != 20 {
				t.Fatalf("chain sample not truncated: %d tokens", len(d.toks))
			}
			for _, s := range d.sup {
				if s >= 19 {
					t.Fatalf("supervised position %d has no target inside truncated length 20", s)
				}
			}
		}
	}

	// Kill-and-resume: state at draw 25, fresh dataset, Restore, and
	// the continuation must match the uninterrupted stream exactly.
	ds3 := load()
	drawN(ds3, 25)
	st := ds3.State()
	ds4 := load()
	drawN(ds4, 7) // desynchronise on purpose
	if err := ds4.Restore(st); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	rest := drawN(ds4, 35)
	if !reflect.DeepEqual(rest, seq1[25:60]) {
		t.Fatal("post-Restore draws diverge from the uninterrupted stream")
	}

	// Epoch bookkeeping advanced (6 listen samples, ≥12 listen draws).
	if ds1.Epoch("listen") < 1 {
		t.Fatalf("listen epoch = %d after %d listen draws of 6 samples", ds1.Epoch("listen"), counts["listen"])
	}
}

func TestTokenShardValidation(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.bin")
	err := WriteTokenShard(p, []ShardSample{{Tokens: []int{1, 2, 3}, Task: "x",
		Supervised: []Span{{Start: 2, End: 5}}}})
	if err == nil {
		t.Fatal("out-of-range supervised span accepted")
	}
	if err := WriteTokenShard(p, []ShardSample{{Tokens: []int{1, 2, 3}, Task: "x",
		Supervised: []Span{{Start: 0, End: 2}}}}); err != nil {
		t.Fatalf("valid shard rejected: %v", err)
	}
	// Corrupt index → load must fail loudly.
	if err := os.WriteFile(filepath.Join(dir, "bad.idx"), []byte("{"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTokenDataset([]string{p}, TokenDatasetConfig{}); err == nil {
		t.Fatal("corrupt .idx accepted")
	}
}
