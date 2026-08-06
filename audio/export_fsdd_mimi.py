"""Export Mimi embeddings for the Free Spoken Digit Dataset (FSDD).

Mimi (kyutai/mimi) is a streaming neural audio codec: 24 kHz mono in,
512-dim continuous latent at 12.5 Hz out (pre-quantizer). We use the
frozen encoder as a feature extractor and mean+std-pool the latent over
time, giving one 1024-dim vector per clip. The gorch side
(e2e/mimi_fsdd_test.go) trains a classifier on these vectors.

Usage:
    python audio/export_fsdd_mimi.py /path/to/free-spoken-digit-dataset/recordings out.safetensors

FSDD filenames are {digit}_{speaker}_{index}.wav at 8 kHz; index 0-4 is
the conventional test split, 5-49 train.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from scipy.signal import resample_poly
from transformers import MimiModel

FSDD_SR = 8000
MIMI_SR = 24000


def wav_to_float(path: Path) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    assert sr == FSDD_SR, f"{path}: expected {FSDD_SR} Hz, got {sr}"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav


@torch.no_grad()
def mimi_latent(model: MimiModel, wav24k: np.ndarray) -> torch.Tensor:
    """Continuous pre-quantizer latent, shape (T, 512) at 12.5 Hz."""
    x = torch.from_numpy(wav24k)[None, None, :]
    h = model.encoder(x)
    h = model.encoder_transformer(h.transpose(1, 2))[0]
    h = model.downsample(h.transpose(1, 2))
    return h[0].T  # (T, 512)


def pool(latent: torch.Tensor) -> torch.Tensor:
    """Mean+std pooling over time: (T, 512) -> (1024,)."""
    mean = latent.mean(dim=0)
    std = latent.std(dim=0, unbiased=False) if latent.shape[0] > 1 else torch.zeros_like(mean)
    return torch.cat([mean, std])


def main() -> None:
    rec_dir = Path(sys.argv[1])
    out_path = sys.argv[2]

    model = MimiModel.from_pretrained("kyutai/mimi").eval()

    splits = {"train": ([], [], []), "test": ([], [], [])}
    files = sorted(rec_dir.glob("*.wav"))
    assert files, f"no wav files in {rec_dir}"
    speakers = sorted({f.stem.split("_")[1] for f in files})
    print(f"speakers: {speakers}")

    for i, f in enumerate(files):
        digit, speaker, idx = f.stem.split("_")
        wav = wav_to_float(f)
        wav24k = resample_poly(wav, MIMI_SR, FSDD_SR).astype(np.float32)
        vec = pool(mimi_latent(model, wav24k))
        split = "test" if int(idx) < 5 else "train"
        splits[split][0].append(vec)
        splits[split][1].append(float(digit))
        splits[split][2].append(float(speakers.index(speaker)))
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(files)}")

    tensors = {}
    for name, (xs, ys, spk) in splits.items():
        tensors[f"{name}_x"] = torch.stack(xs)
        tensors[f"{name}_y"] = torch.tensor(ys).unsqueeze(1)
        tensors[f"{name}_spk"] = torch.tensor(spk).unsqueeze(1)
        print(f"{name}: x={tuple(tensors[f'{name}_x'].shape)} y={tuple(tensors[f'{name}_y'].shape)}")

    save_file(tensors, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
