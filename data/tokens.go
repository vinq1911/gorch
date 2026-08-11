//go:build darwin

package data

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
)

// Packed multi-task token loader (plan 0008 §3.3).
//
// On-disk format: pre-tokenized shards `{name}.bin` (uint32 LE token
// ids, all samples back-to-back) + `{name}.idx` (JSON): per-sample
// {offset, len, task, supervised spans}. Shards are built offline
// (cmd/voicedata, or the trainer's overfit-set builder) — training
// never tokenizes.

// Span is a half-open [Start, End) range of SEQUENCE POSITIONS whose
// next-token prediction is graded (the supervised-loss convention:
// position i supervised ⇒ the model is graded on predicting token
// i+1).
type Span struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// SampleInfo locates one sample inside a shard's .bin file.
type SampleInfo struct {
	Offset     int64  `json:"offset"` // token offset into the shard
	Len        int    `json:"len"`    // token count
	Task       string `json:"task"`   // e.g. "listen", "speak", "chain"
	Supervised []Span `json:"supervised"`
}

// shardIndex is the {name}.idx JSON payload.
type shardIndex struct {
	Samples []SampleInfo `json:"samples"`
}

// ShardSample is the in-memory form used when building shards.
type ShardSample struct {
	Tokens     []int
	Task       string
	Supervised []Span
}

// WriteTokenShard writes samples to binPath (uint32 LE ids) and the
// matching index to binPath with .bin swapped for .idx.
func WriteTokenShard(binPath string, samples []ShardSample) error {
	if !strings.HasSuffix(binPath, ".bin") {
		return fmt.Errorf("data: shard path %q must end in .bin", binPath)
	}
	idx := shardIndex{Samples: make([]SampleInfo, 0, len(samples))}
	var buf []byte
	var offset int64
	for si, s := range samples {
		if len(s.Tokens) < 2 {
			return fmt.Errorf("data: sample %d has %d tokens (< 2)", si, len(s.Tokens))
		}
		for _, sp := range s.Supervised {
			if sp.Start < 0 || sp.End > len(s.Tokens) || sp.Start >= sp.End {
				return fmt.Errorf("data: sample %d has invalid supervised span [%d,%d) for len %d",
					si, sp.Start, sp.End, len(s.Tokens))
			}
		}
		for _, tok := range s.Tokens {
			if tok < 0 {
				return fmt.Errorf("data: sample %d has negative token id %d", si, tok)
			}
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], uint32(tok))
			buf = append(buf, b[:]...)
		}
		idx.Samples = append(idx.Samples, SampleInfo{
			Offset: offset, Len: len(s.Tokens), Task: s.Task, Supervised: s.Supervised,
		})
		offset += int64(len(s.Tokens))
	}
	if err := os.WriteFile(binPath, buf, 0644); err != nil {
		return err
	}
	idxBytes, err := json.MarshalIndent(&idx, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(idxPathFor(binPath), idxBytes, 0644)
}

func idxPathFor(binPath string) string {
	return strings.TrimSuffix(binPath, ".bin") + ".idx"
}

// TokenDatasetConfig configures sampling.
type TokenDatasetConfig struct {
	// MaxTrainSeq truncates every sample to at most this many tokens
	// (supervised positions past the truncation are dropped). ≤0 = no cap.
	MaxTrainSeq int
	// TaskRatios weights task selection (need not sum to 1). Tasks
	// absent from the map get weight 0; a nil/empty map weights every
	// present task equally.
	TaskRatios map[string]float64
	// Seed drives all sampling; two datasets with equal shards, config
	// and seed produce identical draw sequences.
	Seed int64
}

// DatasetState is the resumable sampling position: re-seeding with
// Seed and replaying Draws draws restores the exact stream.
type DatasetState struct {
	Seed  int64 `json:"seed"`
	Draws int64 `json:"draws"`
}

// taskPool tracks one task's samples and its epoch shuffle state.
type taskPool struct {
	name    string
	indices []int // indices into ds.samples, insertion order
	order   []int // current epoch's shuffled order
	cursor  int
	epoch   int
}

// TokenDataset samples task-ratio-weighted training sequences from
// packed shards. Deterministic under seed; epoch bookkeeping supports
// exact resume via State/Restore.
type TokenDataset struct {
	tokens  []int32
	samples []SampleInfo
	pools   []*taskPool // sorted by task name for determinism
	weights []float64   // pool weights, aligned with pools
	cfg     TokenDatasetConfig
	rng     *rand.Rand
	draws   int64
}

// LoadTokenDataset loads one or more shards (paths to .bin files;
// each must have a sibling .idx).
func LoadTokenDataset(binPaths []string, cfg TokenDatasetConfig) (*TokenDataset, error) {
	if len(binPaths) == 0 {
		return nil, fmt.Errorf("data: no shards given")
	}
	ds := &TokenDataset{cfg: cfg, rng: rand.New(rand.NewSource(cfg.Seed))}
	for _, p := range binPaths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("data: %s size %d is not a multiple of 4", p, len(raw))
		}
		base := int64(len(ds.tokens))
		for i := 0; i+4 <= len(raw); i += 4 {
			ds.tokens = append(ds.tokens, int32(binary.LittleEndian.Uint32(raw[i:])))
		}
		idxBytes, err := os.ReadFile(idxPathFor(p))
		if err != nil {
			return nil, err
		}
		var idx shardIndex
		if err := json.Unmarshal(idxBytes, &idx); err != nil {
			return nil, fmt.Errorf("data: %s: %w", idxPathFor(p), err)
		}
		for _, s := range idx.Samples {
			if s.Offset < 0 || s.Len < 2 {
				return nil, fmt.Errorf("data: %s: invalid sample {offset %d, len %d}",
					idxPathFor(p), s.Offset, s.Len)
			}
			if s.Offset+int64(s.Len) > int64(len(raw)/4) {
				return nil, fmt.Errorf("data: %s: sample [%d, +%d) exceeds shard length %d",
					idxPathFor(p), s.Offset, s.Len, len(raw)/4)
			}
			s.Offset += base
			ds.samples = append(ds.samples, s)
		}
	}
	if len(ds.samples) == 0 {
		return nil, fmt.Errorf("data: shards contain no samples")
	}

	// Group by task, sorted for cross-run determinism.
	byTask := map[string][]int{}
	for i, s := range ds.samples {
		byTask[s.Task] = append(byTask[s.Task], i)
	}
	names := make([]string, 0, len(byTask))
	for n := range byTask {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		w := 1.0
		if len(cfg.TaskRatios) > 0 {
			w = cfg.TaskRatios[n]
		}
		if w <= 0 {
			continue
		}
		ds.pools = append(ds.pools, &taskPool{name: n, indices: byTask[n]})
		ds.weights = append(ds.weights, w)
	}
	if len(ds.pools) == 0 {
		return nil, fmt.Errorf("data: task ratios exclude every task present in the shards")
	}
	for _, p := range ds.pools {
		ds.reshuffle(p)
	}
	return ds, nil
}

// reshuffle starts a new epoch for pool p using the dataset RNG (so
// the shuffle stream is part of the deterministic draw sequence).
func (ds *TokenDataset) reshuffle(p *taskPool) {
	p.order = make([]int, len(p.indices))
	copy(p.order, p.indices)
	ds.rng.Shuffle(len(p.order), func(i, j int) { p.order[i], p.order[j] = p.order[j], p.order[i] })
	p.cursor = 0
}

// Len returns the total number of samples.
func (ds *TokenDataset) Len() int { return len(ds.samples) }

// Tasks returns the sampled task names in deterministic order.
func (ds *TokenDataset) Tasks() []string {
	out := make([]string, len(ds.pools))
	for i, p := range ds.pools {
		out[i] = p.name
	}
	return out
}

// Epoch returns how many full passes the named task has completed.
func (ds *TokenDataset) Epoch(task string) int {
	for _, p := range ds.pools {
		if p.name == task {
			return p.epoch
		}
	}
	return 0
}

// Sample draws one training sequence: task chosen by ratio weight,
// sample chosen by the task's epoch shuffle, tokens truncated to
// MaxTrainSeq, supervised = flat positions (each < len(tokens)-1)
// whose next-token prediction is graded.
func (ds *TokenDataset) Sample() (tokens []int, supervised []int, task string) {
	ds.draws++

	// Task pick by cumulative weight.
	var total float64
	for _, w := range ds.weights {
		total += w
	}
	r := ds.rng.Float64() * total
	pool := ds.pools[len(ds.pools)-1]
	for i, w := range ds.weights {
		if r < w {
			pool = ds.pools[i]
			break
		}
		r -= w
	}

	if pool.cursor >= len(pool.order) {
		pool.epoch++
		ds.reshuffle(pool)
	}
	info := ds.samples[pool.order[pool.cursor]]
	pool.cursor++

	n := info.Len
	if ds.cfg.MaxTrainSeq > 0 && n > ds.cfg.MaxTrainSeq {
		n = ds.cfg.MaxTrainSeq
	}
	tokens = make([]int, n)
	for i := 0; i < n; i++ {
		tokens[i] = int(ds.tokens[info.Offset+int64(i)])
	}
	for _, sp := range info.Supervised {
		for pos := sp.Start; pos < sp.End; pos++ {
			if pos < n-1 { // target pos+1 must exist after truncation
				supervised = append(supervised, pos)
			}
		}
	}
	return tokens, supervised, pool.name
}

// Get returns sample i in shard order (truncated to MaxTrainSeq, same
// supervised-position semantics as Sample) without touching the
// sampling state — the deterministic-iteration path used by eval
// loops such as the overfit-100 regeneration gate.
func (ds *TokenDataset) Get(i int) (tokens []int, supervised []int, task string) {
	info := ds.samples[i]
	n := info.Len
	if ds.cfg.MaxTrainSeq > 0 && n > ds.cfg.MaxTrainSeq {
		n = ds.cfg.MaxTrainSeq
	}
	tokens = make([]int, n)
	for j := 0; j < n; j++ {
		tokens[j] = int(ds.tokens[info.Offset+int64(j)])
	}
	for _, sp := range info.Supervised {
		for pos := sp.Start; pos < sp.End; pos++ {
			if pos < n-1 {
				supervised = append(supervised, pos)
			}
		}
	}
	return tokens, supervised, info.Task
}

// State returns the resumable sampling position.
func (ds *TokenDataset) State() DatasetState {
	return DatasetState{Seed: ds.cfg.Seed, Draws: ds.draws}
}

// Restore rewinds the dataset to a saved state by re-seeding and
// replaying the draw stream (sampling is cheap: replay is O(draws)).
// The dataset must have been constructed with the same shards and
// config as the one that produced the state.
func (ds *TokenDataset) Restore(st DatasetState) error {
	if st.Seed != ds.cfg.Seed {
		return fmt.Errorf("data: state seed %d != dataset seed %d", st.Seed, ds.cfg.Seed)
	}
	ds.rng = rand.New(rand.NewSource(ds.cfg.Seed))
	ds.draws = 0
	for _, p := range ds.pools {
		p.epoch = 0
		ds.reshuffle(p)
	}
	for i := int64(0); i < st.Draws; i++ {
		ds.Sample()
	}
	return nil
}
