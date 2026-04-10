//go:build darwin

package gorch

// MSELoss computes mean squared error: mean((pred - target)^2).
func MSELoss(pred, target *Tensor) *Tensor {
	diff := Sub(pred, target)
	sq := Mul(diff, diff)
	return Mean(sq)
}

// CrossEntropyLoss computes cross-entropy loss for classification.
// logits: (batch, classes) raw scores (pre-softmax).
// targets: (batch, 1) integer class labels stored as float32.
// Returns a scalar loss = -mean(logsoftmax(logits)[i, target[i]]).
func CrossEntropyLoss(logits, targets *Tensor) *Tensor {
	if logits.Dim() != 2 {
		panic("gorch: CrossEntropyLoss requires 2-D logits (batch, classes)")
	}
	batch := logits.shape[0]
	classes := logits.shape[1]

	// Compute log-softmax
	ls := LogSoftmax(logits)

	// Pick the log-prob of the correct class for each sample and negate
	var total float32
	for i := 0; i < batch; i++ {
		cls := int(targets.data[i])
		total -= ls.data[i*classes+cls]
	}
	loss := NewTensor([]float32{total / float32(batch)}, 1)

	if logits.requiresGrad {
		loss.requiresGrad = true
		loss.gradFn = &GradFn{
			name:   "CrossEntropyLoss",
			inputs: []*Tensor{logits},
			backward: func(grad *Tensor) []*Tensor {
				// Gradient of CE w.r.t. logits = softmax(logits) - one_hot(targets)
				// scaled by grad / batch
				sm := Softmax(logits)
				dx := Zeros(batch, classes)
				scale := grad.data[0] / float32(batch)
				for i := 0; i < batch; i++ {
					cls := int(targets.data[i])
					for j := 0; j < classes; j++ {
						dx.data[i*classes+j] = sm.data[i*classes+j] * scale
					}
					dx.data[i*classes+cls] -= scale
				}
				return []*Tensor{dx}
			},
		}
	}
	return loss
}
