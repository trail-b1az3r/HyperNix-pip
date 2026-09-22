"""neuron.supervised — labels in, classifier out.

The ordinary case, and the one the "quick image analysis or
recognition" jobs land in: you have pictures and you have what they
are, and you want something small that tells them apart in a
millisecond.

What this does that a fifteen-line training loop does not
---------------------------------------------------------
**It holds data out and reports on it.** Not optional, not a flag. A
training accuracy is a measure of memorisation, and on the dataset sizes
this package is aimed at — a few hundred examples somebody labelled by
hand — memorisation is the default outcome, reached in about four
epochs.

**It keeps the best epoch, not the last.** Small datasets overfit, and
the last epoch is reliably worse than the middle. Returning the final
weights silently hands back a worse model than the run produced.

**It stops when held-out performance stops moving.** Which is also what
makes `epochs` a ceiling rather than a duration.

Why stopping watches the held-out *loss* and not the accuracy
--------------------------------------------------------------
It watched accuracy first, and that was wrong in a way worth recording.
On the dataset sizes this package is for, a 20% holdout is twenty-odd
examples, so accuracy can only move in jumps of 1/25 — it sits on a
plateau for ten epochs at a time while the model is still learning
perfectly well. Patience counted those flat epochs and stopped runs at
epoch 41 with the training loss still falling, producing a 48%-accurate
policy on a task where the same code reaches 100% given a few more
examples.

Cross-entropy on the same twenty-five examples moves every epoch. So
the *stopping decision* and the *best-epoch choice* both run on
held-out loss, and accuracy is reported alongside rather than steering
anything. A tiny validation set is still a bad validation set, which is
why :class:`FitResult` carries ``validation_size`` and says so.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

__all__ = ["FitResult", "fit", "accuracy", "predict"]


@dataclass
class FitResult:
    """What a run did. Printed, asserted on, and saved beside the model."""

    epochs_run: int = 0
    best_epoch: int = 0
    #: Held-out accuracy at the chosen epoch. Reported, not optimised.
    best_score: float = 0.0
    #: Held-out loss at the chosen epoch — what actually chose it.
    best_loss: float = float("inf")
    final_train_loss: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)
    stopped_early: bool = False
    validation_size: int = 0

    #: Below this many held-out examples, every number here is noisy
    #: enough to mislead. Not enforced — refusing to train on a small
    #: dataset would be worse — but reported.
    SMALL_VALIDATION = 50

    @property
    def validation_is_small(self) -> bool:
        return 0 < self.validation_size < self.SMALL_VALIDATION

    def summary(self) -> str:
        line = (
            f"{self.epochs_run} epochs, best held-out accuracy "
            f"{self.best_score:.4f} (loss {self.best_loss:.4f}) "
            f"at epoch {self.best_epoch}"
        )
        if self.stopped_early:
            line += " (stopped early — held-out loss stopped improving)"
        if self.validation_is_small:
            line += (
                f"\n  note: only {self.validation_size} held-out examples, so "
                f"that accuracy is coarse. Record more demonstrations before "
                f"trusting it."
            )
        return line


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction correct. The metric for a classifier, evaluated properly."""
    if len(y) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        predicted = model(x).argmax(dim=-1)
        return float((predicted == y).float().mean().item())


def predict(model: nn.Module, x) -> np.ndarray:
    """Class indices for a batch, as numpy. For use outside torch."""
    model.eval()
    array = np.asarray(x, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("predict needs at least one example")
    tensor = torch.from_numpy(array)
    with torch.no_grad():
        return model(tensor).argmax(dim=-1).cpu().numpy()


def fit(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    validation: tuple[torch.Tensor, torch.Tensor] | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 20,
    seed: int = 0,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> FitResult:
    """Train *model* on (x, y). Returns the run, leaves the best weights.

    *model* is modified in place and ends holding the **best** epoch's
    weights, not the last — see the module docstring.

    *patience* is how many epochs without a held-out improvement to
    tolerate before stopping. With no ``validation`` there is nothing to
    be patient about, so early stopping is off and the last epoch is
    what you get; that is the honest behaviour but it is not the one you
    want, which is why ``validation`` is the first keyword.
    """
    if len(x) != len(y):
        raise ValueError(
            f"{len(x)} examples but {len(y)} labels — these have to match"
        )
    if len(x) == 0:
        raise ValueError("nothing to train on")
    if batch_size < 1:
        raise ValueError("batch size has to be at least 1")

    torch.manual_seed(seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    result = FitResult(
        validation_size=len(validation[1]) if validation is not None else 0
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_accuracy = 0.0
    since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.0
        batches = 0

        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_function(model(x[index]), y[index])
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())
            batches += 1

        train_loss = total_loss / max(batches, 1)
        row = {"epoch": float(epoch), "train_loss": train_loss}

        if validation is not None:
            score = accuracy(model, validation[0], validation[1])
            held_out_loss = _loss_on(model, loss_function, validation)
            row["val_accuracy"] = score
            row["val_loss"] = held_out_loss
            # Loss, not accuracy: accuracy on a small holdout only moves
            # in steps of 1/n and plateaus for many epochs while the
            # model is still learning. See the module docstring.
            if held_out_loss < best_loss - 1e-6:
                best_loss = held_out_loss
                best_accuracy = score
                best_state = copy.deepcopy(model.state_dict())
                result.best_epoch = epoch
                since_improvement = 0
            else:
                since_improvement += 1
        else:
            # Without held-out data the last epoch is all there is.
            best_state = copy.deepcopy(model.state_dict())
            result.best_epoch = epoch
            best_loss = train_loss
            best_accuracy = 0.0

        result.history.append(row)
        result.epochs_run = epoch
        result.final_train_loss = train_loss
        if on_epoch is not None:
            on_epoch(epoch, row)

        if validation is not None and since_improvement >= patience:
            result.stopped_early = True
            break

    model.load_state_dict(best_state)
    model.eval()
    result.best_loss = float(best_loss)
    result.best_score = float(best_accuracy)
    return result


def _loss_on(
    model: nn.Module,
    loss_function: nn.Module,
    validation: tuple[torch.Tensor, torch.Tensor],
) -> float:
    model.eval()
    with torch.no_grad():
        return float(loss_function(model(validation[0]), validation[1]).item())
