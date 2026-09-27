"""hypernix.audio.wakeup — train a wake word, then listen for it.

What openWakeWord does, without using openWakeWord: take a phrase the
operator chooses, gather examples of it and of everything that is not
it, train a small classifier on log-mel frames, and run that classifier
over a sliding window of live audio.

Three ways to get the examples, and they combine
------------------------------------------------
1. **Your own voice, live.** :func:`record_examples` captures takes from
   a microphone. Best quality per sample and the slowest to gather.
2. **A folder you already have.** :func:`examples_from_folder` reads
   WAV, MP3, FLAC and fragmented MP3 (see :mod:`hypernix.audio.audiofile`)
   — a recorder's output, an archive, a stream cut into pieces.
3. **One to four TTS voices, overnight.** :func:`synthesize_examples`
   generates the phrase across voices, speeds and pitches while nobody is
   awake. No human input, and the least like a real microphone.

They are meant to be mixed. A model trained only on TTS learns what
synthesised speech sounds like, which is not the task; a handful of real
recordings alongside a few hundred synthetic ones is worth far more than
either alone, and :func:`build_dataset` takes all three at once.

The processing, and why it is not optional
------------------------------------------
Every example is augmented: gain, time shift, speed, background noise,
and a cheap reverb. This is not padding the dataset — a wake word is
heard across a room, over a fan, at whatever distance and volume the
speaker happens to be, and a model trained on clean centred clips learns
"clean and centred" as part of the phrase. The augmentation is where most
of the real-world accuracy comes from, which is why it is on by default
and the parameters are stated rather than hidden.

Negatives
---------
A wake-word model is mostly a rejector: it will hear a thousand hours of
not-the-phrase for every utterance of it. Negatives from ambient
recordings and from other speech matter more than more positives, and
:class:`WakeUpDataset` reports the ratio because a model trained at 1:1
will fire at the television.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audiofile import (
    TARGET_SAMPLE_RATE,
    Audio,
    AudioError,
    iter_audio_files,
    load_audio,
    load_fragments,
)
from .features import MelConfig, log_mel_frames, mel_filterbank

logger = logging.getLogger(__name__)

__all__ = [
    "WakeUpError",
    "WakeUpConfig",
    "Example",
    "WakeUpDataset",
    "AugmentConfig",
    "augment",
    "examples_from_folder",
    "synthesize_examples",
    "record_examples",
    "build_dataset",
    "train_wakeword",
    "WakeUpDetector",
    "load_detector",
]


class WakeUpError(Exception):
    """Training or detection could not proceed."""


@dataclass(frozen=True)
class WakeUpConfig:
    """The shape of one wake-word model."""

    #: What the model listens for. Only ever used for labelling and for
    #: driving TTS — the model never sees the text.
    wake_words: tuple[str, ...] = ()
    #: Seconds of audio the classifier sees at once. A wake word is
    #: 0.5-1.5 s; the window has to hold the longest one plus room for it
    #: to arrive late in the buffer.
    window_seconds: float = 1.5
    sample_rate: int = TARGET_SAMPLE_RATE
    mel: MelConfig = field(default_factory=MelConfig)
    hidden_size: int = 96
    #: Score above which a detection fires. Deliberately high: a false
    #: accept is a device waking in a silent room, which people notice
    #: much more than a missed word they can simply repeat.
    threshold: float = 0.75

    @property
    def window_frames(self) -> int:
        return max(1, int(self.window_seconds * 1000 / (self.mel.hop_length / self.mel.sample_rate * 1000)))

    @property
    def window_samples(self) -> int:
        return int(self.window_seconds * self.sample_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wake_words": list(self.wake_words),
            "window_seconds": self.window_seconds,
            "sample_rate": self.sample_rate,
            "hidden_size": self.hidden_size,
            "threshold": self.threshold,
            "mel": {
                "sample_rate": self.mel.sample_rate,
                "frame_length": self.mel.frame_length,
                "hop_length": self.mel.hop_length,
                "n_fft": self.mel.n_fft,
                "n_mels": self.mel.n_mels,
                "f_min": self.mel.f_min,
                "f_max": self.mel.f_max,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WakeUpConfig:
        mel = MelConfig(**data.get("mel", {}))
        return cls(
            wake_words=tuple(data.get("wake_words", ())),
            window_seconds=float(data.get("window_seconds", 1.5)),
            sample_rate=int(data.get("sample_rate", TARGET_SAMPLE_RATE)),
            mel=mel,
            hidden_size=int(data.get("hidden_size", 96)),
            threshold=float(data.get("threshold", 0.75)),
        )


@dataclass
class Example:
    """One labelled clip."""

    samples: list[float]
    positive: bool
    source: str = ""


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AugmentConfig:
    """How far each example is moved.

    The ranges are deliberately wide. A wake word is heard from across a
    room, over a fan, from someone who is walking away — a model trained
    on clean centred clips learns "clean and centred" as part of the
    phrase and then does not fire in a kitchen.
    """

    gain_db: tuple[float, float] = (-12.0, 6.0)
    #: Fraction of the window the clip may slide within.
    shift: float = 0.35
    speed: tuple[float, float] = (0.9, 1.1)
    #: Signal-to-noise ratios to mix background in at, in dB.
    snr_db: tuple[float, float] = (5.0, 30.0)
    reverb_probability: float = 0.3
    noise_probability: float = 0.8


def _apply_gain(samples: list[float], db: float) -> list[float]:
    factor = 10.0 ** (db / 20.0)
    # Clipped, because a gain that pushes past 1.0 and is then written to
    # 16-bit wraps around — which sounds like a loud click and teaches the
    # model that the phrase contains one.
    return [max(-1.0, min(1.0, value * factor)) for value in samples]


def _change_speed(samples: list[float], factor: float) -> list[float]:
    if factor == 1.0 or not samples:
        return list(samples)
    length = max(1, int(len(samples) / factor))
    out: list[float] = []
    for index in range(length):
        position = index * factor
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        out.append(samples[left] * (1 - fraction) + samples[right] * fraction)
    return out


def _reverb(samples: list[float], sample_rate: int, rng: random.Random) -> list[float]:
    """A few decaying taps. Not a room model — a hint of one.

    Convolving with a measured impulse response would be better and would
    mean shipping impulse responses. Three taps at plausible delays move
    the spectrum in the direction a room does, which is what the model
    needs to stop treating anechoic audio as part of the phrase.
    """
    out = list(samples)
    for _ in range(3):
        delay = rng.randint(int(0.01 * sample_rate), int(0.08 * sample_rate))
        decay = rng.uniform(0.15, 0.4)
        for index in range(delay, len(out)):
            out[index] += samples[index - delay] * decay
    peak = max((abs(v) for v in out), default=0.0)
    if peak > 1.0:
        out = [v / peak for v in out]
    return out


def _mix_noise(
    samples: list[float], noise: list[float], snr_db: float, rng: random.Random
) -> list[float]:
    if not noise:
        return samples
    # A noise clip shorter than the signal is tiled; a longer one is cut
    # from a random offset, so the same background does not always arrive
    # at the same moment in the window.
    if len(noise) < len(samples):
        repeats = (len(samples) // len(noise)) + 1
        noise = (noise * repeats)[: len(samples)]
    else:
        start = rng.randint(0, len(noise) - len(samples))
        noise = noise[start:start + len(samples)]

    signal_power = sum(v * v for v in samples) / max(1, len(samples))
    noise_power = sum(v * v for v in noise) / max(1, len(noise))
    if noise_power <= 0 or signal_power <= 0:
        return samples
    target = signal_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(target / noise_power)
    return [
        max(-1.0, min(1.0, s + n * scale))
        for s, n in zip(samples, noise, strict=True)
    ]


def augment(
    samples: list[float],
    *,
    window_samples: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
    config: AugmentConfig | None = None,
    noise: Sequence[list[float]] = (),
    rng: random.Random | None = None,
) -> list[float]:
    """One augmented copy of *samples*, fitted to the window."""
    config = config or AugmentConfig()
    rng = rng or random.Random()

    out = _change_speed(samples, rng.uniform(*config.speed))
    out = _apply_gain(out, rng.uniform(*config.gain_db))

    # Place it in the window with a random offset, so the model does not
    # learn that the phrase always starts at frame zero.
    window = [0.0] * window_samples
    if len(out) >= window_samples:
        start = rng.randint(0, len(out) - window_samples)
        window = out[start:start + window_samples]
    else:
        room = window_samples - len(out)
        offset = rng.randint(0, min(room, int(window_samples * config.shift) + room // 2))
        window[offset:offset + len(out)] = out

    if noise and rng.random() < config.noise_probability:
        window = _mix_noise(
            window, list(rng.choice(noise)), rng.uniform(*config.snr_db), rng
        )
    if rng.random() < config.reverb_probability:
        window = _reverb(window, sample_rate, rng)
    return window


# ---------------------------------------------------------------------------
# Gathering examples
# ---------------------------------------------------------------------------

def examples_from_folder(
    directory: str | Path,
    *,
    positive: bool,
    sample_rate: int = TARGET_SAMPLE_RATE,
    fragments: bool = False,
    on_skip: Callable[[Path, str], None] | None = None,
) -> list[Example]:
    """Every readable clip under *directory*, labelled *positive*.

    A file that cannot be decoded is skipped and reported, not raised: a
    folder of a few hundred recordings with one corrupt file should train
    on the rest, and finding out which one it was after an hour of
    gathering is worse than being told at the time.

    ``fragments=True`` treats the folder as one recording cut into
    pieces, joined in natural order.
    """
    paths = iter_audio_files(directory)
    if not paths:
        raise WakeUpError(f"No audio files under {directory}")

    if fragments:
        joined = load_fragments(list(paths), sample_rate=sample_rate)
        return [Example(joined.samples, positive, joined.source)]

    out: list[Example] = []
    for path in paths:
        try:
            audio = load_audio(path, sample_rate=sample_rate)
        except AudioError as exc:
            logger.warning("wakeup: skipping %s: %s", path, exc)
            if on_skip is not None:
                on_skip(path, str(exc))
            continue
        out.append(Example(audio.samples, positive, str(path)))
    if not out:
        raise WakeUpError(
            f"Nothing under {directory} could be decoded. Install ffmpeg, "
            "which reads every format this accepts."
        )
    return out


def synthesize_examples(
    wake_words: Sequence[str],
    *,
    voices: Sequence[str] = (),
    per_voice: int = 40,
    sample_rate: int = TARGET_SAMPLE_RATE,
    engine: Any = None,
    progress: Callable[[dict], None] | None = None,
) -> list[Example]:
    """Generate positives with 1-4 TTS voices.

    The overnight path: no human input, and the point is volume rather
    than fidelity. A model trained *only* on this learns what synthesised
    speech sounds like, which is not the task — the docstring for this
    module says to mix in real recordings, and it means it.

    ``voices`` beyond four is refused rather than silently truncated;
    past that the marginal voice adds less than the time it costs, and a
    caller passing twelve has usually misunderstood what this does.
    """
    if not wake_words:
        raise WakeUpError("No wake words given.")
    voices = tuple(voices) or ("default",)
    if len(voices) > 4:
        raise WakeUpError(
            f"{len(voices)} voices given; this uses at most 4. More voices "
            "cost hours and add less than a handful of real recordings would."
        )

    if engine is None:
        try:
            from ..models.workshop import TTSConfig, TTSEngine

            engine = TTSEngine(TTSConfig(sample_rate=sample_rate))
            engine.initialize()
        except Exception as exc:  # noqa: BLE001
            raise WakeUpError(
                f"No TTS engine available ({exc}). Pass engine=..., or gather "
                "examples with record_examples() or examples_from_folder()."
            ) from exc

    out: list[Example] = []
    for phrase in wake_words:
        for voice in voices:
            for index in range(per_voice):
                try:
                    audio = engine.synthesize(phrase, voice=voice)
                except TypeError:
                    audio = engine.synthesize(phrase)
                except Exception as exc:  # noqa: BLE001
                    raise WakeUpError(f"TTS failed on {phrase!r}: {exc}") from exc
                samples = _as_samples(audio)
                out.append(Example(samples, True, f"tts:{voice}:{phrase}:{index}"))
                if progress is not None:
                    progress({
                        "event": "synth", "phrase": phrase,
                        "voice": voice, "index": index, "total": per_voice,
                    })
    return out


def _as_samples(audio: Any) -> list[float]:
    """Whatever a TTS engine returned, as a float list."""
    if isinstance(audio, Audio):
        return audio.samples
    if hasattr(audio, "tolist"):
        flat = audio.tolist()
        while flat and isinstance(flat[0], list):
            flat = [sum(row) / len(row) for row in flat]
        return [float(v) for v in flat]
    if isinstance(audio, (list, tuple)):
        return [float(v) for v in audio]
    raise WakeUpError(f"TTS returned {type(audio).__name__}, which is not audio.")


def record_examples(
    count: int,
    *,
    seconds: float = 2.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
    prompt: Callable[[int, int], None] | None = None,
    recorder: Any = None,
) -> list[Example]:
    """Capture *count* takes from a microphone.

    ``recorder`` is injectable so this is testable without a microphone —
    and so a caller with their own capture stack does not have to adopt
    this one.
    """
    if recorder is None:
        try:
            import sounddevice  # type: ignore
        except ImportError as exc:
            raise WakeUpError(
                "Recording needs `pip install sounddevice`, or pass "
                "recorder=... . You can also skip recording entirely and use "
                "examples_from_folder() or synthesize_examples()."
            ) from exc

        def recorder(duration: float, rate: int):  # type: ignore[misc]
            data = sounddevice.rec(
                int(duration * rate), samplerate=rate, channels=1, dtype="float32"
            )
            sounddevice.wait()
            return [float(frame[0]) for frame in data]

    out: list[Example] = []
    for index in range(count):
        if prompt is not None:
            prompt(index + 1, count)
        out.append(Example(list(recorder(seconds, sample_rate)), True, f"mic:{index}"))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class WakeUpDataset:
    """Feature windows and labels, plus what went into them."""

    features: list[list[list[float]]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    config: WakeUpConfig = field(default_factory=WakeUpConfig)
    sources: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def positives(self) -> int:
        return sum(self.labels)

    @property
    def negatives(self) -> int:
        return len(self.labels) - self.positives

    @property
    def negative_ratio(self) -> float:
        return (self.negatives / self.positives) if self.positives else 0.0

    def warnings(self) -> list[str]:
        """What is wrong with this dataset, said before training on it."""
        notes: list[str] = []
        if not self.positives:
            notes.append("No positive examples — there is nothing to learn.")
        if not self.negatives:
            notes.append(
                "No negative examples. A wake-word model is mostly a rejector; "
                "trained on positives alone it will fire on everything."
            )
        elif self.negative_ratio < 3:
            notes.append(
                f"Only {self.negative_ratio:.1f} negatives per positive. A model "
                "trained near 1:1 fires at the television — aim for 5-20x, from "
                "ambient recordings and other speech."
            )
        if self.positives and self.positives < 50:
            notes.append(
                f"{self.positives} positives is thin. Under about 100 the model "
                "learns the takes rather than the phrase."
            )
        return notes


def build_dataset(
    positives: Iterable[Example],
    negatives: Iterable[Example],
    *,
    config: WakeUpConfig | None = None,
    augment_config: AugmentConfig | None = None,
    copies: int = 4,
    noise: Sequence[list[float]] = (),
    seed: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> WakeUpDataset:
    """Turn labelled clips into augmented feature windows.

    Each clip becomes ``copies`` augmented windows. The augmentation is
    where most of the real-world accuracy comes from, so it is applied to
    negatives too: negatives that are all clean while positives are all
    noisy teaches the model to detect noise.
    """
    config = config or WakeUpConfig()
    rng = random.Random(seed)
    filters = mel_filterbank(config.mel)
    dataset = WakeUpDataset(config=config)

    for group, positive in ((positives, True), (negatives, False)):
        for example in group:
            kind = example.source.split(":", 1)[0] if ":" in example.source else "file"
            dataset.sources[kind] = dataset.sources.get(kind, 0) + 1
            for _ in range(max(1, copies)):
                window = augment(
                    example.samples,
                    window_samples=config.window_samples,
                    sample_rate=config.sample_rate,
                    config=augment_config,
                    noise=noise,
                    rng=rng,
                )
                frames = log_mel_frames(window, config.mel, filters=filters)
                if not frames:
                    continue
                dataset.features.append(frames)
                dataset.labels.append(1 if positive else 0)
            if progress is not None:
                progress({"event": "example", "source": example.source, "positive": positive})

    return dataset


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _build_model(config: WakeUpConfig):
    """A small conv + GRU classifier over mel frames.

    Small on purpose. This runs continuously on whatever the operator has
    — a laptop, a Pi, a phone — and a model that needs a GPU to decide
    whether someone said two words is not a wake-word model, it is a
    reason to not use one.

    Conv over the mel axis first (a phoneme is a local pattern in
    frequency), then a GRU over time (the phrase is an ordering of them),
    then one output. Bidirectional would score better offline and is
    useless here: the detector has to answer before the utterance ends.
    """
    try:
        from torch import nn
    except ImportError as exc:  # pragma: no cover - torch is a hard dep
        raise WakeUpError("Training a wake word needs torch installed.") from exc

    class WakeWordNet(nn.Module):
        def __init__(self, n_mels: int, hidden: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, frames):          # (batch, time, mels)
            x = self.conv(frames.transpose(1, 2)).transpose(1, 2)
            output, _ = self.gru(x)
            # The last step, not a mean: the decision is "has the phrase
            # finished by now", and averaging lets a long silence dilute
            # a clear detection at the end of the window.
            return self.head(output[:, -1, :]).squeeze(-1)

    return WakeWordNet(config.mel.n_mels, config.hidden_size)


def train_wakeword(
    dataset: WakeUpDataset,
    *,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    seed: int | None = 0,
    progress: Callable[[dict], None] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Train on *dataset*. Returns the model, config and metrics.

    ``strict`` turns the dataset warnings into an error. Off by default
    because an operator experimenting with twenty clips should be allowed
    to; on for anything that will be deployed, because "it trained fine"
    is not the same as "it will work in a kitchen".
    """
    import torch
    from torch import nn

    notes = dataset.warnings()
    if strict and notes:
        raise WakeUpError("; ".join(notes))
    for note in notes:
        logger.warning("wakeup: %s", note)

    if not len(dataset):
        raise WakeUpError("The dataset is empty.")

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    # Frames vary in length by a frame or two depending on the window; pad
    # to the longest so a batch is a tensor. Padding with the dataset's
    # own minimum rather than zero, since zero is a *loud* value in log-mel
    # space and would read as a click at the end of every clip.
    longest = max(len(f) for f in dataset.features)
    quiet = min(min(row) for frames in dataset.features for row in frames)
    padded = [
        frames + [[quiet] * dataset.config.mel.n_mels] * (longest - len(frames))
        for frames in dataset.features
    ]

    features = torch.tensor(padded, dtype=torch.float32)
    labels = torch.tensor(dataset.labels, dtype=torch.float32)

    indices = list(range(len(labels)))
    random.Random(seed).shuffle(indices)
    split = max(1, int(len(indices) * (1 - validation_split)))
    train_index, val_index = indices[:split], indices[split:] or indices[:1]

    model = _build_model(dataset.config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Positives are the minority by design — the negative ratio is the
    # whole point — so the loss is weighted to match, or the model learns
    # that answering "no" is right 95% of the time.
    positive_count = max(1, int(labels[train_index].sum().item()))
    negative_count = max(1, len(train_index) - positive_count)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count)
    )

    history: list[dict[str, float]] = []
    best_accuracy = 0.0
    best_state: dict[str, Any] | None = None

    for epoch in range(epochs):
        model.train()
        random.Random(epoch).shuffle(train_index)
        total_loss = 0.0
        for start in range(0, len(train_index), batch_size):
            batch = train_index[start:start + batch_size]
            optimiser.zero_grad(set_to_none=True)
            logits = model(features[batch])
            loss = loss_fn(logits, labels[batch])
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item()) * len(batch)

        model.eval()
        with torch.no_grad():
            val_logits = model(features[val_index])
            predicted = (torch.sigmoid(val_logits) >= dataset.config.threshold).float()
            actual = labels[val_index]
            accuracy = float((predicted == actual).float().mean().item())
            true_positive = float(((predicted == 1) & (actual == 1)).sum().item())
            false_positive = float(((predicted == 1) & (actual == 0)).sum().item())
            false_negative = float(((predicted == 0) & (actual == 1)).sum().item())

        record = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, len(train_index)),
            "val_accuracy": accuracy,
            "false_accepts": false_positive,
            "false_rejects": false_negative,
        }
        history.append(record)
        if progress is not None:
            progress({"event": "epoch", **record})
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) else 0.0
    )
    return {
        "model": model,
        "config": dataset.config,
        "history": history,
        "val_accuracy": best_accuracy,
        "precision": precision,
        "recall": recall,
        "examples": len(dataset),
        "negative_ratio": dataset.negative_ratio,
        "warnings": notes,
        "sources": dict(dataset.sources),
    }


def save_wakeword(result: dict[str, Any], path: str | Path) -> Path:
    """Write the model and everything needed to run it.

    The config travels with the weights. A wake-word model is useless
    without the exact mel settings it was trained on — different hop or
    band count and the model sees a different spectrum — and keeping them
    in a separate file is how that pairing gets broken.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hypernix.wakeup": 1,
            "state_dict": result["model"].state_dict(),
            "config": result["config"].to_dict(),
            "metrics": {
                "val_accuracy": result.get("val_accuracy"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "examples": result.get("examples"),
                "negative_ratio": result.get("negative_ratio"),
            },
            "trained": time.time(),
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class WakeUpDetector:
    """Streaming detection over a sliding window.

    Fed audio in whatever chunks the capture layer produces. Keeps a ring
    of the last ``window_seconds`` and scores it as often as
    ``hop_seconds`` allows, so the cost is bounded by the hop and not by
    how small the caller's chunks happen to be.

    After a detection it holds off for ``refractory_seconds``. Without it
    one utterance fires on every overlapping window it appears in and the
    caller gets six wakes for one word.
    """

    def __init__(
        self,
        model: Any,
        config: WakeUpConfig,
        *,
        hop_seconds: float = 0.25,
        refractory_seconds: float = 1.5,
    ) -> None:
        self.model = model
        self.config = config
        self.hop_samples = max(1, int(hop_seconds * config.sample_rate))
        self.refractory_seconds = refractory_seconds
        self._buffer: list[float] = []
        self._since_score = 0
        self._last_fire = 0.0
        self._filters = mel_filterbank(config.mel)
        try:
            self.model.eval()
        except AttributeError:
            pass

    def reset(self) -> None:
        self._buffer.clear()
        self._since_score = 0

    def score(self, samples: Sequence[float]) -> float:
        """Probability that *samples* contains the wake word."""
        import torch

        window = list(samples)[-self.config.window_samples:]
        if len(window) < self.config.window_samples:
            window = [0.0] * (self.config.window_samples - len(window)) + window
        frames = log_mel_frames(window, self.config.mel, filters=self._filters)
        if not frames:
            return 0.0
        with torch.no_grad():
            tensor = torch.tensor([frames], dtype=torch.float32)
            return float(torch.sigmoid(self.model(tensor)).item())

    def push(self, chunk: Sequence[float], *, now: float | None = None) -> dict[str, Any] | None:
        """Feed audio. Returns a detection dict, or None.

        Returns rather than calls back, so the caller decides what a wake
        means — this module has no business ringing a bell.
        """
        now = time.monotonic() if now is None else now
        self._buffer.extend(chunk)
        self._since_score += len(chunk)
        if len(self._buffer) > self.config.window_samples * 2:
            self._buffer = self._buffer[-self.config.window_samples * 2:]

        if len(self._buffer) < self.config.window_samples:
            return None
        if self._since_score < self.hop_samples:
            return None
        self._since_score = 0

        confidence = self.score(self._buffer)
        if confidence < self.config.threshold:
            return None
        if now - self._last_fire < self.refractory_seconds:
            return None
        self._last_fire = now
        return {
            "detected": True,
            "confidence": confidence,
            "wake_words": list(self.config.wake_words),
            "at": now,
        }


def load_detector(path: str | Path, **kwargs: Any) -> WakeUpDetector:
    """Load a model saved by :func:`save_wakeword`."""
    from ..security.safeload import load_checkpoint

    path = Path(path)
    if not path.exists():
        raise WakeUpError(f"No wake-word model at {path}")
    payload = load_checkpoint(path)
    if not isinstance(payload, dict) or "hypernix.wakeup" not in payload:
        raise WakeUpError(
            f"{path} is not a HyperNix wake-word model (saved by a different tool?)."
        )
    config = WakeUpConfig.from_dict(payload["config"])
    model = _build_model(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return WakeUpDetector(model, config, **kwargs)
