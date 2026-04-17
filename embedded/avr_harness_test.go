// AVR test harness: drives a real int8 inference on a simulated
// ATmega2560 via simavr, and asserts the firmware output matches the
// Go reference kernel byte-for-byte.
//
// The test skips cleanly if any of avr-gcc / libsimavr / libelf / make is
// missing; CI on machines without the AVR toolchain should still build
// and run the pure-Go tests.
//
// Pipeline:
//  1. Construct a deterministic TinyModel (fixed RNG; small enough to eyeball).
//  2. Emit firmware/tiny_model.h and firmware/tiny_input.h from it.
//  3. Build avr_runner (libsimavr-based host runner) and firmware.elf.
//  4. Run ./avr_runner firmware.elf, parse "CLASS n LOGITS l0 l1 ..."
//  5. Compare against model.Infer(input).

package embedded

import (
	"bufio"
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func requireTools(t *testing.T) {
	t.Helper()
	for _, tool := range []string{"make", "avr-gcc"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("AVR harness requires %q on PATH: %v", tool, err)
		}
	}
	// libsimavr / libelf presence is checked implicitly when we build the
	// runner — that link step will fail clearly if they're missing.
}

// buildRunner compiles the libsimavr host runner once. Returns path to
// the executable. Later test runs in the same `go test` invocation reuse it.
func buildRunner(t *testing.T) string {
	t.Helper()
	runnerDir, err := filepath.Abs("avr_runner")
	if err != nil {
		t.Fatalf("abs runner dir: %v", err)
	}
	cmd := exec.Command("make", "-C", runnerDir, "all")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build avr_runner failed: %v\n%s", err, out)
	}
	return filepath.Join(runnerDir, "avr_runner")
}

// buildFirmware regenerates the model + input headers and then compiles
// firmware.elf against them.
func buildFirmware(t *testing.T, m *TinyModel, input []int8) string {
	t.Helper()
	fwDir, err := filepath.Abs("firmware")
	if err != nil {
		t.Fatalf("abs firmware dir: %v", err)
	}
	modelPath := filepath.Join(fwDir, "tiny_model.h")
	inputPath := filepath.Join(fwDir, "tiny_input.h")

	mf, err := os.Create(modelPath)
	if err != nil {
		t.Fatalf("create tiny_model.h: %v", err)
	}
	if err := m.EmitCHeader(mf, "TINY_MODEL_H"); err != nil {
		mf.Close()
		t.Fatalf("emit model header: %v", err)
	}
	mf.Close()

	inf, err := os.Create(inputPath)
	if err != nil {
		t.Fatalf("create tiny_input.h: %v", err)
	}
	if err := EmitInputHeader(inf, "TINY_INPUT_H", "GM1_INPUT", input); err != nil {
		inf.Close()
		t.Fatalf("emit input header: %v", err)
	}
	inf.Close()

	cmd := exec.Command("make", "-C", fwDir, "clean", "all")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build firmware failed: %v\n%s", err, out)
	}
	return filepath.Join(fwDir, "firmware.elf")
}

type firmwareResult struct {
	class  uint8
	logits []int32
}

// parseFirmwareOutput extracts the "CLASS n LOGITS ..." line from runner
// stdout. Any other lines (e.g. the libsimavr "Loaded ..." banner) are
// ignored.
func parseFirmwareOutput(out []byte) (*firmwareResult, error) {
	s := bufio.NewScanner(bytes.NewReader(out))
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if !strings.HasPrefix(line, "CLASS ") {
			continue
		}
		fields := strings.Fields(line)
		// expect: CLASS <n> LOGITS <l0> <l1> ...
		if len(fields) < 4 || fields[2] != "LOGITS" {
			return nil, fmt.Errorf("malformed CLASS line: %q", line)
		}
		cls64, err := strconv.ParseUint(fields[1], 10, 8)
		if err != nil {
			return nil, fmt.Errorf("bad class %q: %w", fields[1], err)
		}
		logits := make([]int32, 0, len(fields)-3)
		for _, f := range fields[3:] {
			v, err := strconv.ParseInt(f, 10, 32)
			if err != nil {
				return nil, fmt.Errorf("bad logit %q: %w", f, err)
			}
			logits = append(logits, int32(v))
		}
		return &firmwareResult{class: uint8(cls64), logits: logits}, nil
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("no CLASS line in runner output:\n%s", out)
}

// makeFixtureModel returns a deterministic 4 -> 6 -> 3 MLP with random int8
// weights and sensible requant params. Small enough that a human can sanity-
// check the output in a shell; large enough to exercise saturation in the
// requant step.
func makeFixtureModel(seed int64) *TinyModel {
	r := rand.New(rand.NewSource(seed))
	rand8 := func(n int) []int8 {
		out := make([]int8, n)
		for i := range out {
			out[i] = int8(r.Intn(127) - 63) // -63..63, keep headroom
		}
		return out
	}
	rand32 := func(n int) []int32 {
		out := make([]int32, n)
		for i := range out {
			out[i] = int32(r.Intn(2001) - 1000)
		}
		return out
	}
	return &TinyModel{
		InputDim: 4,
		Layers: []Layer{
			{
				Kind: KindLinearI8, InDim: 4, OutDim: 6,
				W: rand8(4 * 6), B: rand32(6),
				// M, S chosen so the accumulator shifts down roughly 10 bits:
				// M = 2^30 / 1024 ~= 2^20; pick a round value.
				M: 1 << 20, S: 18,
			},
			{Kind: KindReLU, InDim: 6, OutDim: 6},
			{
				Kind: KindLinearI32, InDim: 6, OutDim: 3,
				W: rand8(6 * 3), B: rand32(3),
			},
		},
	}
}

func TestAVRInferenceMatchesReference(t *testing.T) {
	requireTools(t)

	// Build toolchain artifacts once per test run.
	runnerExe := buildRunner(t)

	// Exercise a handful of deterministic input vectors so we catch any
	// saturation / sign-extension mistakes that a single input might miss.
	inputs := [][]int8{
		{10, 20, -30, 40},
		{127, -128, 0, 0},
		{-50, -50, -50, -50},
		{1, 2, 3, 4},
	}

	model := makeFixtureModel(1)

	for i, in := range inputs {
		t.Run(fmt.Sprintf("input_%d", i), func(t *testing.T) {
			elf := buildFirmware(t, model, in)

			cmd := exec.Command(runnerExe, elf)
			out, err := cmd.Output()
			if err != nil {
				if ee, ok := err.(*exec.ExitError); ok {
					t.Fatalf("runner failed: %v\nstderr:\n%s\nstdout:\n%s",
						err, ee.Stderr, out)
				}
				t.Fatalf("runner failed: %v\nstdout:\n%s", err, out)
			}

			got, err := parseFirmwareOutput(out)
			if err != nil {
				t.Fatalf("parse firmware output: %v", err)
			}

			wantClass, wantLogits := model.Infer(in)
			if got.class != wantClass {
				t.Errorf("class: firmware=%d reference=%d", got.class, wantClass)
			}
			if len(got.logits) != len(wantLogits) {
				t.Fatalf("logit count: firmware=%d reference=%d",
					len(got.logits), len(wantLogits))
			}
			for j, lf := range got.logits {
				if lf != wantLogits[j] {
					t.Errorf("logit[%d]: firmware=%d reference=%d",
						j, lf, wantLogits[j])
				}
			}
		})
	}
}
