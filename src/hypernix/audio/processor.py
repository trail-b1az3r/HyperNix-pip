"""hypernix.audio.processor — the DSP that everything else assumed.

:mod:`hypernix.audio.audiofile` reads audio and
:mod:`hypernix.audio.features` turns it into log-mel frames, and between
those two sat a gap everybody filled by hand: the recording is 44.1 kHz
and the model wants 16 kHz, it is stereo and the model wants mono, it is
twelve seconds of which nine are room tone, it clips, it hums at 50 Hz
because of the cable. Every caller wrote its own half of that, slightly
differently, and a model trained on one normalisation and served on
another quietly gets worse in a way nothing reports.

So this is the missing middle, and it is real signal processing rather
than a wrapper: the filters are biquads with coefficients computed from
the Audio EQ Cookbook, the resampler is windowed-sinc, the gate and the
compressor have actual attack and release envelopes. All of it is numpy
on float32 in [-1, 1], which is the one representation everything else
here already speaks.

What is deliberately not here
-----------------------------
No codec. :mod:`~hypernix.audio.audiofile` decodes and this processes;
merging them would mean a resampling bug and an MP3 bug landing in the
same file and being hard to tell apart.

No FFT-based spectral subtraction beyond :func:`reduce_noise`'s simple
version. Good denoising is a model, not a filter, and pretending a
three-line spectral gate is "noise reduction" is how people end up
shipping audio with musical artefacts they cannot hear because they
listened to it forty times.

Everything is out-of-place: a function returns a new array and never
writes through its argument. In-place DSP saves an allocation and costs
an afternoon the first time a caller reuses a buffer it thought was
untouched.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Audio",
    "to_mono",
    "resample",
    "normalise_peak",
    "normalise_rms",
    "apply_gain",
    "fade",
    "biquad",
    "high_pass",
    "low_pass",
    "band_pass",
    "notch",
    "pre_emphasis",
    "dc_offset_removed",
    "noise_gate",
    "compress",
    "limit",
    "trim_silence",
    "split_on_silence",
    "detect_speech",
    "reduce_noise",
    "rms",
    "peak",
    "dbfs",
    "clipping_fraction",
    "chunks",
    "Pipeline",
]

#: Smallest amplitude treated as non-zero. Below this, dB is meaningless
#: and the reciprocal is a division by almost-zero that produces inf.
EPSILON = 1e-10


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


@dataclass
class Audio:
    """Samples plus the one fact you cannot recover from them.

    A bare array of samples is not audio: without the sample rate it
    cannot be resampled, filtered at a named frequency, or played. Every
    function here takes and returns this, so a rate cannot be lost
    halfway through a chain and then guessed at the end — which is the
    bug that produces audio at the right pitch and the wrong speed.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32)
        if self.sample_rate <= 0:
            raise ValueError(f"sample rate must be positive, got {self.sample_rate}")
        if self.samples.ndim > 2:
            raise ValueError(
                f"audio is 1-D (mono) or 2-D (channels, samples); got "
                f"{self.samples.ndim} dimensions"
            )

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[-1])

    @property
    def duration(self) -> float:
        """Seconds."""
        return self.frames / self.sample_rate

    def __len__(self) -> int:
        return self.frames

    def with_samples(self, samples: np.ndarray) -> Audio:
        """A new Audio at the same rate. Keeps the rate attached."""
        return Audio(samples, self.sample_rate)

    def describe(self) -> str:
        return (
            f"{self.duration:.2f}s · {self.sample_rate} Hz · "
            f"{self.channels}ch · peak {dbfs(peak(self)):.1f} dBFS · "
            f"rms {dbfs(rms(self)):.1f} dBFS"
        )


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def peak(audio: Audio) -> float:
    """Largest absolute sample."""
    if audio.frames == 0:
        return 0.0
    return float(np.max(np.abs(audio.samples)))


def rms(audio: Audio) -> float:
    """Root mean square — loudness, roughly, and what normalising uses.

    Peak says whether it clips; RMS says how loud it sounds. Normalising
    a podcast by peak makes one cough set the level for the whole file.
    """
    if audio.frames == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.samples, dtype=np.float64))))


def dbfs(amplitude: float) -> float:
    """Amplitude to dB relative to full scale. Silence is -inf, honestly."""
    if amplitude <= EPSILON:
        return -float("inf")
    return 20.0 * math.log10(amplitude)


def clipping_fraction(audio: Audio, threshold: float = 0.999) -> float:
    """How much of this is pinned at full scale.

    Worth checking before anything else: a recording that clips cannot
    be fixed by gain, and every measurement downstream of it is of the
    clipping rather than of the sound.
    """
    if audio.frames == 0:
        return 0.0
    return float(np.mean(np.abs(audio.samples) >= threshold))


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def to_mono(audio: Audio) -> Audio:
    """Average the channels.

    Averaging rather than taking the left channel: a stereo recording
    with the speaker panned right becomes silence if you take the left,
    and that failure is invisible until someone plays the file.
    """
    if audio.samples.ndim == 1:
        return audio
    return audio.with_samples(audio.samples.mean(axis=0, dtype=np.float32))


def resample(audio: Audio, rate: int) -> Audio:
    """Change the sample rate with a windowed-sinc interpolator.

    Not `samples[::2]`. Dropping samples aliases everything above the
    new Nyquist frequency back down into the audible band as a
    metallic ring, and it is permanent. This low-passes first, then
    interpolates.

    Kaiser-windowed sinc with a modest half-width: good enough that the
    artefacts are below the noise floor of anything this project does,
    cheap enough to run on a whole folder.
    """
    if rate <= 0:
        raise ValueError(f"target sample rate must be positive, got {rate}")
    if rate == audio.sample_rate or audio.frames == 0:
        return Audio(audio.samples.copy(), rate)

    if audio.samples.ndim == 2:
        rows = [
            resample(Audio(channel, audio.sample_rate), rate).samples
            for channel in audio.samples
        ]
        return Audio(np.stack(rows), rate)

    ratio = rate / audio.sample_rate
    output_length = int(math.ceil(audio.frames * ratio))
    if output_length <= 0:
        return Audio(np.zeros(0, dtype=np.float32), rate)

    # Downsampling has to band-limit to the *new* Nyquist; upsampling
    # only has to interpolate, so its cutoff stays at the old one.
    cutoff = min(1.0, ratio) * 0.95
    half_width = 16
    positions = np.arange(output_length, dtype=np.float64) / ratio
    left = np.floor(positions).astype(np.int64)

    accumulated = np.zeros(output_length, dtype=np.float64)
    weights = np.zeros(output_length, dtype=np.float64)
    padded = audio.samples.astype(np.float64)

    for offset in range(-half_width, half_width + 1):
        index = left + offset
        distance = (positions - index) * cutoff
        # Normalised sinc, Kaiser-windowed. np.sinc is sin(pi x)/(pi x).
        window_argument = np.clip((positions - index) / half_width, -1.0, 1.0)
        window = np.i0(8.6 * np.sqrt(np.maximum(0.0, 1.0 - window_argument**2))) / np.i0(8.6)
        taps = np.sinc(distance) * window
        clamped = np.clip(index, 0, audio.frames - 1)
        inside = (index >= 0) & (index < audio.frames)
        accumulated += np.where(inside, padded[clamped] * taps, 0.0)
        weights += np.where(inside, taps, 0.0)

    # Normalising by the realised tap sum rather than by cutoff keeps the
    # gain flat at the edges, where part of the kernel hangs off the end.
    safe = np.where(np.abs(weights) < 1e-9, 1.0, weights)
    return Audio((accumulated / safe).astype(np.float32), rate)


# ---------------------------------------------------------------------------
# Level
# ---------------------------------------------------------------------------


def apply_gain(audio: Audio, db: float) -> Audio:
    """Multiply by a decibel amount. Does not clamp — see :func:`limit`."""
    return audio.with_samples(audio.samples * (10.0 ** (db / 20.0)))


def normalise_peak(audio: Audio, target_db: float = -1.0) -> Audio:
    """Scale so the loudest sample sits at *target_db*.

    Default -1 rather than 0: a file normalised to exactly full scale
    clips the moment anything downstream resamples or encodes it, because
    reconstruction overshoots between samples.
    """
    current = peak(audio)
    if current <= EPSILON:
        return audio.with_samples(audio.samples.copy())
    target = 10.0 ** (target_db / 20.0)
    return audio.with_samples(audio.samples * (target / current))


def normalise_rms(audio: Audio, target_db: float = -20.0) -> Audio:
    """Scale so the *average* level sits at *target_db*.

    What you want for speech. Peak normalising a spoken recording lets
    one door slam decide the level for the whole thing.

    Can push peaks past full scale by design — follow it with
    :func:`limit` if that matters, which is the usual chain.
    """
    current = rms(audio)
    if current <= EPSILON:
        return audio.with_samples(audio.samples.copy())
    target = 10.0 ** (target_db / 20.0)
    return audio.with_samples(audio.samples * (target / current))


def dc_offset_removed(audio: Audio) -> Audio:
    """Subtract the mean.

    A recording with DC offset uses part of its headroom holding a
    constant, clips asymmetrically, and makes every RMS measurement
    read high. Cheap to fix and usually forgotten.
    """
    if audio.frames == 0:
        return audio.with_samples(audio.samples.copy())
    axis = -1
    mean = audio.samples.mean(axis=axis, keepdims=True, dtype=np.float64)
    return audio.with_samples((audio.samples - mean).astype(np.float32))


def fade(audio: Audio, seconds_in: float = 0.01, seconds_out: float = 0.01) -> Audio:
    """Ramp the start and end.

    Ten milliseconds by default, which is inaudible as a fade and
    removes the click you get from a waveform that starts partway up.
    Anything that cuts audio — :func:`trim_silence`, :func:`split_on_silence`
    — wants this after it.
    """
    out = audio.samples.astype(np.float32).copy()
    length = audio.frames
    if length == 0:
        return audio.with_samples(out)

    ramp_in = min(int(seconds_in * audio.sample_rate), length)
    ramp_out = min(int(seconds_out * audio.sample_rate), length)
    if ramp_in > 0:
        shape = np.linspace(0.0, 1.0, ramp_in, dtype=np.float32)
        out[..., :ramp_in] *= shape
    if ramp_out > 0:
        shape = np.linspace(1.0, 0.0, ramp_out, dtype=np.float32)
        out[..., length - ramp_out :] *= shape
    return audio.with_samples(out)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def biquad(
    audio: Audio, b: Sequence[float], a: Sequence[float]
) -> Audio:
    """Run a second-order section. ``a[0]`` is assumed normalised to 1.

    The direct-form-I difference equation, in float64. Float32 state in a
    recursive filter accumulates error that shows up as a slow drift at
    low cutoffs, and the cost of the wider accumulator is nothing.
    """
    if len(b) != 3 or len(a) != 3:
        raise ValueError("a biquad needs three b coefficients and three a")
    if audio.samples.ndim == 2:
        rows = [
            biquad(Audio(channel, audio.sample_rate), b, a).samples
            for channel in audio.samples
        ]
        return audio.with_samples(np.stack(rows))

    x = audio.samples.astype(np.float64)
    y = np.zeros_like(x)
    b0, b1, b2 = (float(v) for v in b)
    a0, a1, a2 = (float(v) for v in a)
    if abs(a0) < 1e-12:
        raise ValueError("a[0] cannot be zero")
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

    x1 = x2 = y1 = y2 = 0.0
    for index in range(x.size):
        current = x[index]
        value = b0 * current + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[index] = value
        x2, x1 = x1, current
        y2, y1 = y1, value
    return audio.with_samples(y.astype(np.float32))


def _cookbook(kind: str, rate: int, frequency: float, q: float):
    """Audio EQ Cookbook coefficients. The standard formulas, not mine."""
    if frequency <= 0 or frequency >= rate / 2:
        raise ValueError(
            f"cutoff {frequency} Hz must be between 0 and Nyquist "
            f"({rate / 2} Hz) for a {rate} Hz signal"
        )
    if q <= 0:
        raise ValueError("Q must be positive")

    w0 = 2.0 * math.pi * frequency / rate
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)

    if kind == "low":
        b = [(1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2]
    elif kind == "high":
        b = [(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2]
    elif kind == "band":
        b = [alpha, 0.0, -alpha]
    elif kind == "notch":
        b = [1.0, -2.0 * cos_w0, 1.0]
    else:  # pragma: no cover - guarded by the public wrappers
        raise ValueError(f"unknown filter kind {kind!r}")

    a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    return b, a


def low_pass(audio: Audio, cutoff: float, q: float = 0.7071) -> Audio:
    """Keep what is below *cutoff*. Default Q is Butterworth."""
    return biquad(audio, *_cookbook("low", audio.sample_rate, cutoff, q))


def high_pass(audio: Audio, cutoff: float, q: float = 0.7071) -> Audio:
    """Keep what is above *cutoff*.

    The single most useful filter for speech: 80 Hz removes handling
    noise, air conditioning and desk rumble, none of which carry any of
    the voice, and all of which eat headroom.
    """
    return biquad(audio, *_cookbook("high", audio.sample_rate, cutoff, q))


def band_pass(audio: Audio, centre: float, q: float = 1.0) -> Audio:
    """Keep a band around *centre*."""
    return biquad(audio, *_cookbook("band", audio.sample_rate, centre, q))


def notch(audio: Audio, frequency: float, q: float = 30.0) -> Audio:
    """Remove one narrow frequency. Mains hum is 50 Hz or 60 Hz.

    High Q by default because a notch wide enough to be gentle is wide
    enough to take the voice with it.
    """
    return biquad(audio, *_cookbook("notch", audio.sample_rate, frequency, q))


def pre_emphasis(audio: Audio, coefficient: float = 0.97) -> Audio:
    """``y[n] = x[n] - k*x[n-1]`` — lift the high end before features.

    Speech has far more energy at low frequencies, so a mel spectrogram
    of un-emphasised audio spends most of its dynamic range on the parts
    that carry the least information. Standard front-end step, and the
    exact coefficient matters only in that training and serving must
    agree on it.
    """
    if audio.frames == 0:
        return audio.with_samples(audio.samples.copy())
    out = np.empty_like(audio.samples)
    out[..., 0] = audio.samples[..., 0]
    out[..., 1:] = audio.samples[..., 1:] - coefficient * audio.samples[..., :-1]
    return audio.with_samples(out)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def _envelope(
    magnitude: np.ndarray, attack_samples: float, release_samples: float
) -> np.ndarray:
    """A one-pole follower. Rises at attack, falls at release.

    Written as a loop rather than vectorised because it is genuinely
    recursive — each output depends on the previous one. A vectorised
    version of this is a different filter that happens to look similar.
    """
    attack = math.exp(-1.0 / max(attack_samples, 1.0))
    release = math.exp(-1.0 / max(release_samples, 1.0))
    out = np.zeros_like(magnitude)
    level = 0.0
    for index in range(magnitude.size):
        target = magnitude[index]
        coefficient = attack if target > level else release
        level = target + coefficient * (level - target)
        out[index] = level
    return out


def noise_gate(
    audio: Audio,
    threshold_db: float = -45.0,
    attack: float = 0.005,
    release: float = 0.08,
) -> Audio:
    """Silence anything quieter than *threshold_db*.

    With real attack and release, because a gate that switches
    instantaneously chatters on every sample near the threshold and
    turns room tone into a stutter that is far more distracting than the
    noise was.
    """
    if audio.samples.ndim == 2:
        rows = [
            noise_gate(
                Audio(channel, audio.sample_rate), threshold_db, attack, release
            ).samples
            for channel in audio.samples
        ]
        return audio.with_samples(np.stack(rows))
    if audio.frames == 0:
        return audio.with_samples(audio.samples.copy())

    threshold = 10.0 ** (threshold_db / 20.0)
    level = _envelope(
        np.abs(audio.samples.astype(np.float64)),
        attack * audio.sample_rate,
        release * audio.sample_rate,
    )
    open_gate = (level >= threshold).astype(np.float64)
    smoothed = _envelope(
        open_gate, attack * audio.sample_rate, release * audio.sample_rate
    )
    return audio.with_samples((audio.samples * smoothed).astype(np.float32))


def compress(
    audio: Audio,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack: float = 0.005,
    release: float = 0.1,
    makeup_db: float | None = None,
) -> Audio:
    """Pull loud parts down so the quiet parts can come up.

    *makeup_db* defaults to the gain the threshold and ratio imply, so
    the result comes out at roughly the level it went in at rather than
    quieter — which is what makes an A/B comparison of a compressor
    honest instead of "the louder one sounds better".
    """
    if ratio < 1.0:
        raise ValueError("a compression ratio below 1:1 is an expander")
    if audio.samples.ndim == 2:
        rows = [
            compress(
                Audio(channel, audio.sample_rate),
                threshold_db, ratio, attack, release, makeup_db,
            ).samples
            for channel in audio.samples
        ]
        return audio.with_samples(np.stack(rows))
    if audio.frames == 0:
        return audio.with_samples(audio.samples.copy())

    level = _envelope(
        np.abs(audio.samples.astype(np.float64)),
        attack * audio.sample_rate,
        release * audio.sample_rate,
    )
    level_db = 20.0 * np.log10(np.maximum(level, EPSILON))
    excess = np.maximum(0.0, level_db - threshold_db)
    reduction_db = excess * (1.0 - 1.0 / ratio)

    if makeup_db is None:
        makeup_db = abs(threshold_db) * (1.0 - 1.0 / ratio) * 0.5
    gain = 10.0 ** ((makeup_db - reduction_db) / 20.0)
    return audio.with_samples((audio.samples * gain).astype(np.float32))


def limit(audio: Audio, ceiling_db: float = -0.3) -> Audio:
    """Hard ceiling. Nothing comes out above *ceiling_db*.

    Scales the whole signal if it is over rather than clipping each
    sample: clipping generates harmonics across the spectrum, and this
    is the last thing in a chain where everything before it was trying
    not to do that.
    """
    ceiling = 10.0 ** (ceiling_db / 20.0)
    current = peak(audio)
    if current <= ceiling or current <= EPSILON:
        return audio.with_samples(audio.samples.copy())
    return audio.with_samples(audio.samples * (ceiling / current))


# ---------------------------------------------------------------------------
# Silence and speech
# ---------------------------------------------------------------------------


def _frame_energy(
    audio: Audio, frame: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    """(starts, rms per frame) for a mono signal."""
    mono = to_mono(audio).samples.astype(np.float64)
    if mono.size < frame:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    starts = np.arange(0, mono.size - frame + 1, hop, dtype=np.int64)
    view = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop]
    return starts, np.sqrt(np.mean(view**2, axis=1))


def detect_speech(
    audio: Audio,
    threshold_db: float = -40.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> list[tuple[float, float]]:
    """Where something is happening. ``[(start_s, end_s), ...]``.

    Energy-based, and honest about it: this finds *sound*, not speech,
    so a slammed door is a region. That is the right trade for trimming
    and chunking, which is what it is for. If you need actual
    voice-activity detection, that is a model.
    """
    frame = max(1, int(frame_ms * audio.sample_rate / 1000))
    hop = max(1, int(hop_ms * audio.sample_rate / 1000))
    starts, energy = _frame_energy(audio, frame, hop)
    if starts.size == 0:
        return []

    threshold = 10.0 ** (threshold_db / 20.0)
    loud = energy >= threshold
    regions: list[tuple[float, float]] = []
    begin: int | None = None
    for index, is_loud in enumerate(loud):
        if is_loud and begin is None:
            begin = index
        elif not is_loud and begin is not None:
            regions.append(
                (
                    starts[begin] / audio.sample_rate,
                    (starts[index] + frame) / audio.sample_rate,
                )
            )
            begin = None
    if begin is not None:
        regions.append(
            (starts[begin] / audio.sample_rate, audio.duration)
        )
    return regions


def trim_silence(
    audio: Audio, threshold_db: float = -40.0, pad: float = 0.05
) -> Audio:
    """Cut the quiet from both ends, keeping *pad* seconds of it.

    Keeping a little is not politeness — a clip that starts exactly on
    the first loud sample has its onset cut off, which is audible and
    also removes the attack a classifier may be keying on.
    """
    regions = detect_speech(audio, threshold_db)
    if not regions:
        return audio.with_samples(audio.samples[..., :0].copy())
    start = max(0.0, regions[0][0] - pad)
    end = min(audio.duration, regions[-1][1] + pad)
    first = int(start * audio.sample_rate)
    last = int(end * audio.sample_rate)
    return audio.with_samples(audio.samples[..., first:last].copy())


def split_on_silence(
    audio: Audio,
    threshold_db: float = -40.0,
    min_silence: float = 0.3,
    pad: float = 0.05,
) -> list[Audio]:
    """Cut into pieces wherever it goes quiet for *min_silence*.

    What you want for turning one long recording into training clips, or
    a meeting into utterances.
    """
    regions = detect_speech(audio, threshold_db)
    if not regions:
        return []

    merged: list[list[float]] = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] < min_silence:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    pieces: list[Audio] = []
    for start, end in merged:
        first = max(0, int((start - pad) * audio.sample_rate))
        last = min(audio.frames, int((end + pad) * audio.sample_rate))
        if last > first:
            pieces.append(audio.with_samples(audio.samples[..., first:last].copy()))
    return pieces


def reduce_noise(
    audio: Audio,
    noise_seconds: float = 0.5,
    reduction_db: float = 12.0,
    frame: int = 1024,
) -> Audio:
    """Spectral subtraction against a noise profile from the opening.

    Takes the first *noise_seconds* as "this is what the room sounds
    like", and attenuates each frequency bin by how much of it is
    explained by that profile. Overlap-add with a Hann window at 50%,
    which is what stops the frame boundaries being audible.

    *reduction_db* is capped deliberately. Subtracting the whole profile
    gives the classic underwater warble — isolated bins surviving in
    otherwise-silent frames — and 12 dB of honest reduction sounds far
    better than 40 dB of artefacts.
    """
    mono = to_mono(audio)
    if mono.frames < frame * 2:
        return audio.with_samples(audio.samples.copy())

    hop = frame // 2
    window = np.hanning(frame).astype(np.float64)
    noise_frames = max(1, int(noise_seconds * audio.sample_rate) // hop)

    # Padded by a whole frame at each end, and trimmed again at the
    # bottom. Without this the first and last frames are the only
    # samples covered by a single window rather than two, so the
    # overlap-add normalisation divides them by a weight near zero --
    # which does not attenuate the opening, it amplifies it about
    # tenfold. The denoiser made its own noise profile louder.
    original = mono.samples.astype(np.float64)
    signal = np.concatenate(
        [np.zeros(frame, dtype=np.float64), original, np.zeros(frame, dtype=np.float64)]
    )
    starts = range(0, signal.size - frame + 1, hop)

    spectra = [np.fft.rfft(signal[s : s + frame] * window) for s in starts]
    if not spectra:
        return audio.with_samples(audio.samples.copy())

    # Skip the frames that only see the zero padding -- averaging them
    # into the profile would understate the real noise floor.
    first_real = frame // hop
    sampled = spectra[first_real : first_real + max(1, noise_frames)] or spectra[:1]
    profile = np.mean([np.abs(s) for s in sampled], axis=0)
    floor = 10.0 ** (-abs(reduction_db) / 20.0)

    out = np.zeros(signal.size, dtype=np.float64)
    weights = np.zeros(signal.size, dtype=np.float64)
    for index, start in enumerate(starts):
        magnitude = np.abs(spectra[index])
        # Wiener-ish gain, floored so a bin is attenuated rather than
        # removed. Removing it is what produces the warble.
        gain = np.maximum(floor, (magnitude - profile) / np.maximum(magnitude, EPSILON))
        rebuilt = np.fft.irfft(spectra[index] * gain, n=frame)
        out[start : start + frame] += rebuilt * window
        weights[start : start + frame] += window**2

    # Hann at 50% hop sums its squares to a constant in the interior;
    # only the padded ends fall short, and those get trimmed away.
    safe = np.where(weights < 1e-9, 1.0, weights)
    recovered = (out / safe)[frame : frame + original.size]
    return Audio(recovered.astype(np.float32), audio.sample_rate)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def chunks(
    audio: Audio, seconds: float, overlap: float = 0.0
) -> Iterator[Audio]:
    """Fixed-length pieces, optionally overlapping.

    The last piece is returned short rather than zero-padded. Padding
    would be a silent lie about how much audio there was, and anything
    that needs a fixed length can pad it itself knowing that it did.
    """
    if seconds <= 0:
        raise ValueError("chunk length must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap is a fraction in [0, 1)")

    size = max(1, int(seconds * audio.sample_rate))
    step = max(1, int(size * (1.0 - overlap)))
    for start in range(0, audio.frames, step):
        piece = audio.samples[..., start : start + size]
        if piece.shape[-1] == 0:
            break
        yield audio.with_samples(piece.copy())
        if start + size >= audio.frames:
            break


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    """An ordered chain, so a preset is a value you can save and compare.

    The reason this exists rather than "just call the functions": the
    order matters enormously and is not obvious. Gate before compress
    and you gate the quiet parts then amplify what is left; compress
    before gate and the compressor lifts the noise floor above the gate
    threshold so the gate never closes. Having the chain be data means a
    project can write its order down once.
    """

    steps: list[tuple[str, dict]] = field(default_factory=list)

    #: Sensible default for getting speech into a model: remove rumble,
    #: even out the level, gate the room, land at a known loudness.
    @classmethod
    def for_speech(cls, sample_rate: int = 16000) -> Pipeline:
        return cls(
            steps=[
                ("dc_offset_removed", {}),
                ("high_pass", {"cutoff": 80.0}),
                ("resample", {"rate": sample_rate}),
                ("to_mono", {}),
                ("noise_gate", {"threshold_db": -45.0}),
                ("compress", {"threshold_db": -20.0, "ratio": 3.0}),
                ("normalise_rms", {"target_db": -20.0}),
                ("limit", {"ceiling_db": -1.0}),
            ]
        )

    def then(self, name: str, **options) -> Pipeline:
        self.steps.append((name, options))
        return self

    def run(self, audio: Audio) -> Audio:
        current = audio
        for name, options in self.steps:
            function = globals().get(name)
            if function is None or name.startswith("_"):
                raise ValueError(
                    f"no audio step named {name!r}; the ones there are: "
                    f"{', '.join(sorted(n for n in __all__ if n[0].islower()))}"
                )
            current = function(current, **options)
        return current

    def describe(self) -> str:
        return " -> ".join(
            name + (f"({', '.join(f'{k}={v}' for k, v in o.items())})" if o else "")
            for name, o in self.steps
        )
