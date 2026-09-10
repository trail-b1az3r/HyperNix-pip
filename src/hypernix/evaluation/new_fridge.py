"""new_fridge — graphing and analytics.

A thin wrapper around matplotlib that stays out of the import path
until actually called. matplotlib is not a hypernix dependency; the
first plotting call uses :func:`hypernix.deps.ensure` to pull it in
on demand (respecting ``HYPERNIX_AUTO_INSTALL=0``).

Three entry points cover 90% of the job:

* :func:`parse_training_log` — extract ``(step, loss)`` pairs from the
  stdout that :func:`hypernix.train.train` prints.
* :func:`plot_loss_curve` — save a PNG of the loss curve.
* :func:`plot_score_distribution` — histogram of judge scores.

Each plotting function accepts an ``out_path``; no GUI is ever opened.
"""
from __future__ import annotations

from hypernix.system.deprecation import deprecated_module

# Announced before the heavy imports below, so the notice reaches the
# operator immediately rather than after torch has finished loading.
deprecated_module(
    "hypernix.evaluation.new_fridge",
    instead="hypernix.models.neo_oven",
    since="0.71.5a2",
)

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hypernix.system import deps

_LOG_LINE_RE = re.compile(
    r"step\s+(\d+)\s*/\s*\d+\s+loss=([-+0-9.eE]+)"
)


def parse_training_log(text: str) -> list[tuple[int, float]]:
    """Return ``(step, loss)`` pairs parsed from ``hypernix.train`` stdout."""
    pairs: list[tuple[int, float]] = []
    for m in _LOG_LINE_RE.finditer(text):
        step = int(m.group(1))
        try:
            loss = float(m.group(2))
        except ValueError:
            continue
        pairs.append((step, loss))
    return pairs


def _load_matplotlib() -> Any:
    """Return a matplotlib.pyplot module, installing matplotlib if needed."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ModuleNotFoundError:
        deps.ensure(["matplotlib>=3.7"], reimport=["matplotlib", "matplotlib.pyplot"])
        import matplotlib.pyplot as plt  # type: ignore
        return plt


def plot_loss_curve(
    pairs: Sequence[tuple[int, float]],
    out_path: Path | str,
    *,
    title: str = "Training loss",
) -> Path:
    """Save a loss curve as a PNG. Returns the path written."""
    plt = _load_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    steps = [s for s, _ in pairs]
    losses = [loss for _, loss in pairs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plot_score_distribution(
    scores: Sequence[float],
    out_path: Path | str,
    *,
    title: str = "Judge score distribution",
    bins: int = 20,
) -> Path:
    """Histogram of judge scores (one pass through a dataset)."""
    plt = _load_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(list(scores), bins=bins, edgecolor="black")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plot_round_losses(
    rounds: Sequence[Sequence[tuple[int, float]]],
    out_path: Path | str,
    *,
    title: str = "Multi-round training loss",
    round_labels: Sequence[str] | None = None,
) -> Path:
    """Plot loss curves for multiple training rounds (e.g. from :mod:`tupperware`)."""
    plt = _load_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, pairs in enumerate(rounds):
        if not pairs:
            continue
        label = round_labels[i] if round_labels and i < len(round_labels) else f"round {i + 1}"
        steps = [s for s, _ in pairs]
        losses = [loss for _, loss in pairs]
        ax.plot(steps, losses, marker="o", markersize=3, label=label)

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
