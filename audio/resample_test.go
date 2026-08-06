//go:build darwin

package audio

import (
	"fmt"
	"math"
	"testing"
)

// TestResampleMatchesScipy compares Resample against scipy.signal.resample_poly
// outputs for all four target ratios at max relative error <= 1e-4.
func TestResampleMatchesScipy(t *testing.T) {
	fx := loadFixtures(t)
	for _, srIn := range []int{8000, 16000, 48000, 44100} {
		t.Run(fmt.Sprintf("%d_to_24000", srIn), func(t *testing.T) {
			in := fixtureTensor(t, fx, fmt.Sprintf("resample_%d_in", srIn)).Data()
			want := fixtureTensor(t, fx, fmt.Sprintf("resample_%d_out", srIn)).Data()
			got := Resample(in, srIn, 24000)
			if len(got) != len(want) {
				t.Fatalf("output length = %d, want %d", len(got), len(want))
			}
			var maxRel float64
			for i := range got {
				a, b := float64(got[i]), float64(want[i])
				rel := math.Abs(a-b) / (math.Abs(b) + 1e-5)
				if rel > maxRel {
					maxRel = rel
				}
			}
			t.Logf("%d -> 24000: n=%d, max rel err = %.3g", srIn, len(got), maxRel)
			if maxRel > 1e-4 {
				t.Errorf("max rel err vs scipy = %g, want <= 1e-4", maxRel)
			}
		})
	}
}

// TestResampleOutputLength checks the ceil(n*up/down) output-length formula
// for all four ratios across assorted input lengths.
func TestResampleOutputLength(t *testing.T) {
	ratios := [][2]int{{8000, 24000}, {16000, 24000}, {48000, 24000}, {44100, 24000}}
	lengths := []int{1, 2, 7, 100, 147, 960, 1001, 22063}
	for _, r := range ratios {
		srIn, srOut := r[0], r[1]
		d := gcd(srIn, srOut)
		up, down := srOut/d, srIn/d
		for _, n := range lengths {
			x := make([]float32, n)
			got := len(Resample(x, srIn, srOut))
			want := (n*up + down - 1) / down
			if got != want {
				t.Errorf("Resample(len %d, %d->%d): output length %d, want %d",
					n, srIn, srOut, got, want)
			}
		}
	}
}

// TestResampleIdentity: equal rates return an equal-length copy.
func TestResampleIdentity(t *testing.T) {
	x := []float32{1, -2, 3, -4}
	y := Resample(x, 24000, 24000)
	if len(y) != len(x) {
		t.Fatalf("length %d, want %d", len(y), len(x))
	}
	for i := range x {
		if y[i] != x[i] {
			t.Fatalf("y[%d] = %g, want %g", i, y[i], x[i])
		}
	}
	y[0] = 99
	if x[0] == 99 {
		t.Error("identity resample aliases the input; want a copy")
	}
}

// TestResampleSpectral: a 3 kHz tone at 8 kHz survives resampling to 24 kHz,
// and energy above the old Nyquist (4 kHz) is strongly attenuated.
//
// Note on the threshold: the plan asks for < -60 dB, but scipy's own
// resample_poly defaults (Kaiser beta=5.0, half_len=30 for up=3) leave the
// 5 kHz image of a 3 kHz tone at -57.3 dB — measured directly on scipy
// 1.15.3 output. Since Resample must match scipy exactly, the test asserts
// < -55 dB, which the reference and this implementation both satisfy while
// still catching any real filtering bug (an unfiltered image sits at -0 dB).
func TestResampleSpectral(t *testing.T) {
	const (
		srIn  = 8000
		srOut = 24000
		tone  = 3000.0
		nIn   = 8000 // 1 s
		N     = 4096 // DFT length: 3 kHz falls exactly on bin 512
	)
	x := make([]float32, nIn)
	for i := range x {
		x[i] = float32(math.Sin(2 * math.Pi * tone * float64(i) / srIn))
	}
	y := Resample(x, srIn, srOut)

	// Analyze a steady-state segment away from the filter edge transients.
	seg := y[8000 : 8000+N]
	power := make([]float64, N/2+1)
	var toneAmp float64
	for k := 0; k <= N/2; k++ {
		var re, im float64
		for n := 0; n < N; n++ {
			ph := 2 * math.Pi * float64(k) * float64(n) / N
			s, c := math.Sincos(ph)
			re += float64(seg[n]) * c
			im -= float64(seg[n]) * s
		}
		power[k] = re*re + im*im
		if k == tone*N/srOut { // bin 512
			toneAmp = 2 * math.Sqrt(re*re+im*im) / N
		}
	}

	if toneAmp < 0.97 || toneAmp > 1.03 {
		t.Errorf("3 kHz tone amplitude after resample = %.4f, want within [0.97, 1.03]", toneAmp)
	}

	var total, high float64
	for k := 0; k <= N/2; k++ {
		f := float64(k) * srOut / N
		total += power[k]
		if f > srIn/2 {
			high += power[k]
		}
	}
	ratioDB := 10 * math.Log10(high/total)
	t.Logf("tone amplitude = %.4f, energy above %d Hz = %.2f dB", toneAmp, srIn/2, ratioDB)
	if ratioDB > -55 {
		t.Errorf("energy above old Nyquist = %.2f dB, want < -55 dB", ratioDB)
	}
}
