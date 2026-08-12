//go:build darwin

package qwen

// Plan 0009 X4 — trainer integration of the GPU+bf16 path.
//
// The accelerated configuration is: frozen base loaded with LoadNative
// (block Linear weights bf16, everything else f32), every block weight
// + norm gamma + LoRA factor + the embedding tables moved into Metal
// unified memory, and the hidden-state chain Metal-resident from the
// embedding lookup onward. All the actual kernels/dispatch paths were
// shipped in X1–X3; this file is only the wiring that points the REAL
// VoiceModel at them.

import (
	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/metal"
	"github.com/vinq1911/gorch/nn"
)

// ToMetal moves every weight the training step touches into Metal
// unified memory:
//
//   - block Linear layers (bf16 weights from LoadNative — the MPS
//     dtyped matmul operands — plus their f32 zero biases),
//   - norm gammas (block norms, QK-norms, final norm; f32 by the
//     RMSNorm kernel contract),
//   - LoRA A/B factors (trainable f32; resident so the adapter matmuls
//     and their grads dispatch MPS and the delta path never breaks the
//     chain's residency),
//   - the extended embedding's Base/Ext tables (f32; makes the hidden
//     chain resident from the first op and puts the tied-head GEMMs on
//     MPS).
//
// RoPE cos/sin tables upload themselves lazily on the first resident
// Apply (X2b). Safe to call once after NewVoiceModel; idempotent
// because Tensor.ToMetal is.
func (vm *VoiceModel) ToMetal(dev *metal.Device) {
	for _, blk := range vm.Base.Blocks {
		for _, l := range []*nn.Linear{blk.Attn.Wq, blk.Attn.Wk, blk.Attn.Wv, blk.Attn.Wo,
			blk.FFN.Wgate, blk.FFN.Wup, blk.FFN.Wdown} {
			l.ToMetal(dev)
		}
		blk.NormAttn.Weight.ToMetal(dev)
		blk.NormFFN.Weight.ToMetal(dev)
		if blk.Attn.QNorm != nil {
			blk.Attn.QNorm.Weight.ToMetal(dev)
		}
		if blk.Attn.KNorm != nil {
			blk.Attn.KNorm.Weight.ToMetal(dev)
		}
	}
	vm.Base.Norm.Weight.ToMetal(dev)
	for _, a := range vm.adapters {
		a.A.ToMetal(dev)
		a.B.ToMetal(dev)
	}
	vm.Embed.ToMetal(dev)
}

// AccelSupported reports whether the full accelerated path is
// available on this machine: Metal initialized and the MPS dtyped
// bf16 matmul probe green (ADR-012). Callers that get false should
// stay on the CPU f32 path (`--accel=off` semantics).
func AccelSupported() bool {
	return g.MetalDev() != nil && g.MetalBF16MatMulSupported()
}
