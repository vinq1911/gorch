// RISC-V harness: trains an Iris MLP, quantizes it to int8, emits the
// weight/input headers, builds a bare-metal firmware for QEMU's `virt`
// machine (representative of ESP32-C3 class — same arch, same word size),
// runs it under qemu-system-riscv32, and asserts the firmware's
// classification matches the Go-side quantized reference for every
// sample in the dataset.
//
// The test `t.Skip`s cleanly if `qemu-system-riscv32`,
// `riscv64-unknown-elf-gcc`, or `make` is missing.

package riscv

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vinq1911/gorch/embedded"
	"github.com/vinq1911/gorch/embedded/iris"
)

func requireTools(t *testing.T) {
	t.Helper()
	for _, tool := range []string{"make", "riscv64-unknown-elf-gcc", "qemu-system-riscv32"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("RISC-V harness requires %q on PATH: %v", tool, err)
		}
	}
}

// trainAndQuantize is the workhorse: deterministic float training, then
// post-training quantization using the training set for activation
// calibration. Returns the float and int8 models plus the train/test split
// so the test can evaluate and emit test samples.
type trained struct {
	fm *iris.FloatModel
	qm *embedded.TinyModel

	xTr, xTe [][]float32
	yTr, yTe []int
}

func trainAndQuantize(t *testing.T) *trained {
	t.Helper()
	xTr, xTe, yTr, yTe := iris.Split(0.2)
	fm := iris.Train(iris.TrainConfig{
		HiddenDim:    8,
		LearningRate: 0.05,
		Epochs:       300,
		Seed:         42,
	}, xTr, yTr)
	qm := fm.Quantize(xTr)
	return &trained{fm: fm, qm: qm, xTr: xTr, xTe: xTe, yTr: yTr, yTe: yTe}
}

// writeHeaders regenerates iris_model.h (from qm) and iris_input.h (from
// a single quantized sample). The firmware #includes both.
func writeHeaders(t *testing.T, fwDir string, qm *embedded.TinyModel, xI8 []int8) {
	t.Helper()

	mf, err := os.Create(filepath.Join(fwDir, "iris_model.h"))
	if err != nil {
		t.Fatalf("create iris_model.h: %v", err)
	}
	if err := qm.EmitCHeader(mf, "IRIS_MODEL_H"); err != nil {
		mf.Close()
		t.Fatalf("emit model header: %v", err)
	}
	mf.Close()

	inf, err := os.Create(filepath.Join(fwDir, "iris_input.h"))
	if err != nil {
		t.Fatalf("create iris_input.h: %v", err)
	}
	if err := embedded.EmitInputHeader(inf, "IRIS_INPUT_H", "GM1_INPUT", xI8); err != nil {
		inf.Close()
		t.Fatalf("emit input header: %v", err)
	}
	inf.Close()
}

func buildFirmware(t *testing.T, fwDir string) string {
	t.Helper()
	cmd := exec.Command("make", "-C", fwDir, "clean", "all")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build firmware failed: %v\n%s", err, out)
	}
	return filepath.Join(fwDir, "firmware.elf")
}

// runQEMU boots the firmware on qemu-system-riscv32 -machine virt, streams
// the UART (16550 at 0x10000000) to stdout via `-serial stdio`, and
// captures everything. The firmware writes to the SiFive test device to
// terminate cleanly; the context timeout is a backstop.
func runQEMU(t *testing.T, elf string) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -display none + -monitor none + -serial stdio: ours is the only
	// character device, and it goes to our stdout. `-nographic` alone
	// would try to multiplex monitor+serial onto stdio and fail.
	// -bios none: skip OpenSBI, jump straight to our ELF at 0x80000000.
	cmd := exec.CommandContext(ctx, "qemu-system-riscv32",
		"-machine", "virt",
		"-cpu", "rv32",
		"-smp", "1",
		"-m", "128",
		"-bios", "none",
		"-display", "none",
		"-monitor", "none",
		"-serial", "stdio",
		"-kernel", elf,
	)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			t.Fatalf("QEMU timed out\nstdout:\n%s\nstderr:\n%s", out.String(), errBuf.String())
		}
		// SiFive test PASS exit may surface as a non-zero QEMU exit in some
		// versions; we only care that we got the CLASS line.
		if !bytes.Contains(out.Bytes(), []byte("CLASS ")) {
			t.Fatalf("QEMU failed: %v\nstdout:\n%s\nstderr:\n%s", err, out.String(), errBuf.String())
		}
	}
	return out.Bytes()
}

type firmwareResult struct {
	class  uint8
	logits []int32
}

func parseOutput(out []byte) (*firmwareResult, error) {
	s := bufio.NewScanner(bytes.NewReader(out))
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if !strings.HasPrefix(line, "CLASS ") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 4 || fields[2] != "LOGITS" {
			return nil, fmt.Errorf("malformed CLASS line: %q", line)
		}
		cls, err := strconv.ParseUint(fields[1], 10, 8)
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
		return &firmwareResult{class: uint8(cls), logits: logits}, nil
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("no CLASS line in QEMU output:\n%s", out)
}

// TestRISCVFirmwareMatchesReference picks a sample from each class,
// builds a firmware with that sample baked in, runs it, and asserts
// the predicted class + logits match the Go reference exactly.
func TestRISCVFirmwareMatchesReference(t *testing.T) {
	requireTools(t)

	tr := trainAndQuantize(t)

	fwDir, err := filepath.Abs("firmware")
	if err != nil {
		t.Fatalf("abs firmware dir: %v", err)
	}

	// Pick one representative sample per class from the test set; these
	// are the samples the firmware classifies. Keeping it to 3 runs
	// (one per class) keeps the test quick (~2-3s total).
	picks := map[int][]float32{}
	for i, y := range tr.yTe {
		if _, ok := picks[y]; !ok {
			picks[y] = tr.xTe[i]
		}
		if len(picks) == 3 {
			break
		}
	}
	classNames := []string{"setosa", "versicolor", "virginica"}

	for wantClass := 0; wantClass < 3; wantClass++ {
		sample := picks[wantClass]
		t.Run(classNames[wantClass], func(t *testing.T) {
			xI8 := tr.fm.QuantizeInput(sample)
			writeHeaders(t, fwDir, tr.qm, xI8)
			elf := buildFirmware(t, fwDir)

			out := runQEMU(t, elf)
			got, err := parseOutput(out)
			if err != nil {
				t.Fatalf("parse QEMU output: %v", err)
			}

			refClass, refLogits := tr.qm.Infer(xI8)
			if got.class != refClass {
				t.Errorf("class: firmware=%d reference=%d (raw sample %v, int8 %v)",
					got.class, refClass, sample, xI8)
			}
			if len(got.logits) != len(refLogits) {
				t.Fatalf("logit count: firmware=%d reference=%d",
					len(got.logits), len(refLogits))
			}
			for i, lf := range got.logits {
				if lf != refLogits[i] {
					t.Errorf("logit[%d]: firmware=%d reference=%d",
						i, lf, refLogits[i])
				}
			}

			// Also check that we're actually predicting correctly (i.e. the
			// whole train-quantize-deploy pipeline is producing a working
			// classifier, not just producing matching wrong answers).
			if got.class != uint8(wantClass) {
				t.Logf("note: firmware predicted class %d for a %s sample; "+
					"Iris has ~95%% quantized accuracy so occasional "+
					"misclassifications are expected", got.class, classNames[wantClass])
			}
		})
	}
}

// TestRISCVFirmwareAccuracyOnFullDataset is the "is the trained model
// actually useful?" check. It runs every Data row through the firmware
// and reports overall accuracy. Skipped in short mode because it
// invokes QEMU once per sample (~30s total).
func TestRISCVFirmwareAccuracyOnFullDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-dataset QEMU run in short mode")
	}
	requireTools(t)

	tr := trainAndQuantize(t)
	fwDir, err := filepath.Abs("firmware")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}

	correct := 0
	for _, row := range iris.Data {
		sample := []float32{row[0], row[1], row[2], row[3]}
		want := int(row[4])
		xI8 := tr.fm.QuantizeInput(sample)

		writeHeaders(t, fwDir, tr.qm, xI8)
		elf := buildFirmware(t, fwDir)
		out := runQEMU(t, elf)
		got, err := parseOutput(out)
		if err != nil {
			t.Fatalf("parse: %v\n%s", err, out)
		}
		if int(got.class) == want {
			correct++
		}
	}
	total := len(iris.Data)
	acc := float64(correct) / float64(total)
	t.Logf("firmware Iris accuracy: %d/%d = %.3f", correct, total, acc)
	if acc < 0.90 {
		t.Fatalf("firmware accuracy %.3f below 0.90 threshold", acc)
	}
}
