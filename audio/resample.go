//go:build darwin

package audio

import "math"

// Resample converts x from srIn to srOut using a polyphase windowed-sinc
// filter that reproduces scipy.signal.resample_poly's defaults exactly
// (window=('kaiser', 5.0), padtype='constant' with cval=0):
//
//   - up/down = srOut/srIn reduced by their gcd
//   - halfLen = 10*max(up, down); FIR = firwin(2*halfLen+1, 1/max(up, down))
//     with a Kaiser window (beta=5.0), normalized to unit DC gain, then
//     scaled by up
//   - zero-phase alignment: the filter is pre-padded so scipy's trim of
//     halfLen taps lands on whole output samples
//   - output length ceil(len(x)*up/down)
//
// The polyphase evaluation computes only the needed output phases: the
// upsampled intermediate signal is never materialized, so large up factors
// (44100 -> 24000 is up=80/down=147) stay fast.
//
// Precision: the filter and accumulation are float64, matching scipy's
// float64 path (resample_poly on float64 input). For float32 input scipy
// instead casts the filter to float32 and accumulates in float32 with a
// compiler-dependent (vectorized) summation order, so bit-exactness with
// that path is unattainable; float64 here is strictly more accurate, and
// the two differ only by float32 rounding noise (~1e-7 absolute).
func Resample(x []float32, srIn, srOut int) []float32 {
	if srIn <= 0 || srOut <= 0 {
		panic("audio: Resample sample rates must be positive")
	}
	d := gcd(srIn, srOut)
	up := srOut / d
	down := srIn / d
	if up == down {
		out := make([]float32, len(x))
		copy(out, x)
		return out
	}
	nIn := len(x)
	if nIn == 0 {
		return nil
	}
	nOut := (nIn*up + down - 1) / down // ceil(nIn*up/down)

	maxRate := up
	if down > maxRate {
		maxRate = down
	}
	halfLen := 10 * maxRate
	h := firwinKaiser(2*halfLen+1, 1/float64(maxRate), 5.0)
	for i := range h {
		h[i] *= float64(up)
	}

	// Zero-pad the filter so the retained output samples sit at whole
	// downsampled positions (scipy's n_pre_pad / n_post_pad logic).
	nPrePad := down - halfLen%down
	nPostPad := 0
	nPreRemove := (halfLen + nPrePad) / down
	for upfirdnLen(len(h)+nPrePad+nPostPad, nIn, up, down) < nOut+nPreRemove {
		nPostPad++
	}

	// upfirdn: y[t] = sum_i x[i] * hPadded[t*down - i*up], where hPadded is
	// h with nPrePad leading and nPostPad trailing zeros. Only the non-zero
	// taps contribute, so index directly into h with the pre-pad offset and
	// keep i inside both the signal and the filter support.
	out := make([]float32, nOut)
	lenH := len(h)
	for t := 0; t < nOut; t++ {
		jj := (t+nPreRemove)*down - nPrePad // filter index for i = 0
		iMin := floorDiv(jj-lenH, up) + 1   // smallest i with jj-i*up < lenH
		if iMin < 0 {
			iMin = 0
		}
		iMax := floorDiv(jj, up) // largest i with jj-i*up >= 0
		if iMax > nIn-1 {
			iMax = nIn - 1
		}
		var acc float64
		for i := iMin; i <= iMax; i++ {
			acc += float64(x[i]) * h[jj-i*up]
		}
		out[t] = float32(acc)
	}
	return out
}

// firwinKaiser designs an odd-length linear-phase lowpass FIR, matching
// scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta)): a sinc
// truncated by a Kaiser window and normalized to unit DC gain. cutoff is
// in Nyquist units (1.0 = fs/2).
func firwinKaiser(numtaps int, cutoff, beta float64) []float64 {
	alpha := float64(numtaps-1) / 2
	i0beta := besselI0(beta)
	h := make([]float64, numtaps)
	var sum float64
	for i := range h {
		m := float64(i) - alpha
		r := m / alpha
		q := 1 - r*r
		if q < 0 {
			q = 0
		}
		w := besselI0(beta*math.Sqrt(q)) / i0beta
		h[i] = cutoff * sinc(cutoff*m) * w
		sum += h[i]
	}
	for i := range h {
		h[i] /= sum
	}
	return h
}

// sinc is the normalized sinc function sin(pi*x)/(pi*x).
func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// besselI0 is the zeroth-order modified Bessel function of the first kind,
// via the standard power series I0(x) = sum_k ((x/2)^k / k!)^2.
func besselI0(x float64) float64 {
	half := x / 2
	term := 1.0
	sum := 1.0
	for k := 1; k <= 200; k++ {
		t := half / float64(k)
		term *= t * t
		sum += term
		if term < sum*1e-18 {
			break
		}
	}
	return sum
}

// upfirdnLen mirrors scipy.signal._upfirdn._output_len: the number of
// output samples produced by upsample-filter-downsample for a length-lenH
// filter over nIn input samples.
func upfirdnLen(lenH, nIn, up, down int) int {
	return ((nIn-1)*up+lenH-1)/down + 1
}

// floorDiv returns floor(a/b) for b > 0 (Go's / truncates toward zero).
func floorDiv(a, b int) int {
	q := a / b
	if a%b != 0 && (a < 0) != (b < 0) {
		q--
	}
	return q
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
