//go:build darwin && e2e

package e2e

// Temporary diagnostic: CPU profile of the X1K1 Metal-resident block
// step (plan 0009). Run with TA_PROFILE=1. Not part of the bench gates.

import (
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"testing"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/optim"
)

func TestX1K1Profile(t *testing.T) {
	if os.Getenv("TA_PROFILE") == "" {
		t.Skip("set TA_PROFILE=1 to run")
	}
	if _, err := g.InitMetal(); err != nil {
		t.Skipf("metal not available: %v", err)
	}
	seq := 1024
	rng := rand.New(rand.NewSource(taSeed))
	block := newTABlock(rng, seq+1)
	block.toMetal(g.MetalDev())
	x := taSeedTensor(rng, 1.0, seq, taHidden).ToMetal(g.MetalDev())
	x.SetRequiresGrad(true)
	opt := optim.NewAdamW(block.parameters(), 1e-4, 0.01)

	step := func() {
		opt.ZeroGrad()
		x.ZeroGrad()
		g.Sum(block.forward(x, true)).Backward()
		opt.Step()
	}
	step() // warmup

	path := filepath.Join(os.TempDir(), fmt.Sprintf("x1k1_block%d.pprof", seq))
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := pprof.StartCPUProfile(f); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		step()
	}
	pprof.StopCPUProfile()

	out, err := exec.Command("go", "tool", "pprof", "-top", "-nodecount=20", path).Output()
	if err != nil {
		t.Fatalf("pprof: %v", err)
	}
	for _, ln := range strings.Split(string(out), "\n") {
		if ln != "" {
			t.Log(ln)
		}
	}
}
