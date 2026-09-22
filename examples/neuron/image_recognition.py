#!/usr/bin/env python3
"""A small image classifier, trained and measured honestly.

"Basic and quick image analysis" — a few hundred labelled pictures and
something that tells them apart in under a millisecond. Not a ResNet:
if you need one of those, fine-tune a pretrained model instead.

Run it:

    python examples/neuron/image_recognition.py

The data here is synthetic (four shapes drawn into 32x32 arrays) so the
example runs with no download and no dataset directory. Swap
`make_dataset` for your own loader and nothing else changes.

The part worth copying is not the training loop — it is that the
held-out split is not optional, and that the run keeps the *best*
epoch rather than the last. On a few hundred examples a small network
memorises in about four epochs, and the last epoch is reliably worse
than the middle.
"""
from __future__ import annotations

import numpy as np
import torch

from hypernix.neuron import nets, supervised

SHAPES = ("square", "cross", "ring", "diagonal")


def draw(kind: int, size: int = 32, rng: np.random.Generator | None = None) -> np.ndarray:
    """One 1-channel image of the given shape, with noise."""
    rng = rng or np.random.default_rng()
    canvas = np.zeros((1, size, size), dtype=np.float32)
    pad = rng.integers(4, 10)
    if kind == 0:                                    # square
        canvas[0, pad:size - pad, pad:size - pad] = 1.0
        canvas[0, pad + 3:size - pad - 3, pad + 3:size - pad - 3] = 0.0
    elif kind == 1:                                  # cross
        middle = size // 2
        canvas[0, middle - 2:middle + 2, pad:size - pad] = 1.0
        canvas[0, pad:size - pad, middle - 2:middle + 2] = 1.0
    elif kind == 2:                                  # ring
        y, x = np.ogrid[:size, :size]
        centre = size / 2
        distance = np.sqrt((y - centre) ** 2 + (x - centre) ** 2)
        radius = size / 2 - pad
        canvas[0] = ((distance < radius) & (distance > radius - 3)).astype(np.float32)
    else:                                            # diagonal
        for offset in range(-2, 3):
            index = np.arange(pad, size - pad)
            canvas[0, index, np.clip(index + offset, 0, size - 1)] = 1.0

    canvas += rng.normal(0.0, 0.08, canvas.shape).astype(np.float32)
    return np.clip(canvas, 0.0, 1.0)


def make_dataset(count: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(SHAPES), count)
    images = np.stack([draw(int(label), rng=rng) for label in labels])
    return torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))


def main() -> int:
    print(f"1. Making 900 images of {len(SHAPES)} shapes: {', '.join(SHAPES)}")
    x, y = make_dataset(900, seed=0)
    train_x, train_y = x[:700], y[:700]
    test_x, test_y = x[700:], y[700:]
    print(f"   {len(train_x)} to train on, {len(test_x)} held out\n")

    print("2. A small CNN")
    model = nets.small_cnn((1, 32, 32), len(SHAPES))
    print(f"   {nets.count_parameters(model):,} parameters\n")

    print("3. Training")
    result = supervised.fit(
        model, train_x, train_y,
        validation=(test_x, test_y),
        epochs=60, batch_size=32, seed=0,
        on_epoch=lambda epoch, row: (
            print(f"   epoch {epoch:3d}  loss {row['train_loss']:.4f}  "
                  f"held-out {row.get('val_accuracy', 0):.3f}")
            if epoch % 10 == 0 else None
        ),
    )
    print(f"\n   {result.summary()}\n")

    print("4. Per-class accuracy on the held-out set")
    predicted = supervised.predict(model, test_x.numpy())
    actual = test_y.numpy()
    for index, name in enumerate(SHAPES):
        mask = actual == index
        if mask.any():
            share = float((predicted[mask] == index).mean())
            print(f"   {name:9} {share:.3f}  ({int(mask.sum())} examples)")

    print("\n5. Saving")
    spec = nets.NetSpec(kind="cnn", image=(1, 32, 32), outputs=len(SHAPES))
    path = nets.save(model, spec, "shapes.pt")
    print(f"   {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
