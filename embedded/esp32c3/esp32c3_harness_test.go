// ESP32-C3 harness: trains an Iris MLP, quantizes int8, builds a
// bare-metal firmware, and runs it on Espressif's QEMU fork with
// `-machine esp32c3`. This is real ESP32-C3 emulation — same CPU core
// (rv32imc), same memory map, same UART peripheral — not a generic
// RISC-V virt machine.
//
// The Espressif QEMU binary is expected at /opt/esp-qemu/bin/ or on
// PATH as `qemu-esp-riscv32`. The test skips cleanly if the binary,
// the RISC-V cross compiler, or make isn't found.

package esp32c3

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

// findEspQEMU returns the path to Espressif's QEMU for RISC-V, checking
// a few well-known locations.
func findEspQEMU() string {
	candidates := []string{
		"/opt/esp-qemu/bin/qemu-system-riscv32",
		os.ExpandEnv("$HOME/.espressif/tools/qemu-riscv32/bin/qemu-system-riscv32"),
	}
	// Also check PATH for a renamed binary.
	if p, err := exec.LookPath("qemu-esp-riscv32"); err == nil {
		return p
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

func requireTools(t *testing.T) string {
	t.Helper()
	for _, tool := range []string{"make", "riscv64-unknown-elf-gcc"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("ESP32-C3 harness requires %q on PATH: %v", tool, err)
		}
	}
	qemu := findEspQEMU()
	if qemu == "" {
		t.Skip("ESP32-C3 harness requires Espressif QEMU (not found at /opt/esp-qemu/bin/ or on PATH)")
	}
	// Verify it actually has the esp32c3 machine.
	out, err := exec.Command(qemu, "-machine", "help").Output()
	if err != nil || !bytes.Contains(out, []byte("esp32c3")) {
		t.Skipf("QEMU at %s does not support -machine esp32c3", qemu)
	}
	return qemu
}

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

func runQEMU(t *testing.T, qemu, elf string) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, qemu,
		"-machine", "esp32c3",
		"-nographic",
		"-no-reboot",
		"-kernel", elf,
	)
	// Pipe stdout so we can kill QEMU as soon as the CLASS line arrives
	// instead of waiting for the full timeout.
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf
	if err := cmd.Start(); err != nil {
		t.Fatalf("start QEMU: %v", err)
	}

	var collected bytes.Buffer
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		collected.WriteString(line)
		collected.WriteByte('\n')
		if strings.HasPrefix(strings.TrimSpace(line), "CLASS ") {
			break
		}
	}
	// Kill QEMU now that we have the output.
	cmd.Process.Kill()
	cmd.Wait()

	if !bytes.Contains(collected.Bytes(), []byte("CLASS ")) {
		t.Fatalf("QEMU produced no CLASS line\nstdout:\n%s\nstderr:\n%s",
			collected.String(), errBuf.String())
	}
	return collected.Bytes()
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

func TestESP32C3FirmwareMatchesReference(t *testing.T) {
	qemu := requireTools(t)
	tr := trainAndQuantize(t)

	fwDir, err := filepath.Abs("firmware")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}

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

			out := runQEMU(t, qemu, elf)
			got, err := parseOutput(out)
			if err != nil {
				t.Fatalf("parse QEMU output: %v", err)
			}

			refClass, refLogits := tr.qm.Infer(xI8)
			if got.class != refClass {
				t.Errorf("class: firmware=%d reference=%d", got.class, refClass)
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
			if got.class != uint8(wantClass) {
				t.Logf("note: firmware predicted class %d for a %s sample",
					got.class, classNames[wantClass])
			}
		})
	}
}
