//go:build darwin

package gorch

// MSELoss computes mean squared error: mean((pred - target)^2).
func MSELoss(pred, target *Tensor) *Tensor {
	diff := Sub(pred, target)
	sq := Mul(diff, diff)
	return Mean(sq)
}
