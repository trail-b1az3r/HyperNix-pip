#!/usr/bin/env python3
"""Get a messy recording into the shape a model wants.

The gap between `audiofile` (which reads audio) and `features` (which
turns it into log-mel frames), and the part everybody used to write by
hand slightly differently — which is how a model trained on one
normalisation gets served another and quietly gets worse.

Run it:

    python examples/audio/clean_up_a_recording.py [path/to/audio.wav]

With no path it synthesises something with the usual problems — 44.1
kHz, stereo, a DC offset, 50 Hz mains hum, a noise floor, and long
silences at both ends — and shows what each stage does to the numbers.

The order in `Pipeline.for_speech` is the part to copy. It is not
arbitrary: gate before compress and you gate the quiet parts then
amplify what is left; compress before gate and the compressor lifts the
noise floor above the threshold so the gate never closes.
"""
from __future__ import annotations

import sys

import numpy as np

from hypernix.audio.processor import (
    Audio,
    Pipeline,
    clipping_fraction,
    dbfs,
    detect_speech,
    high_pass,
    normalise_rms,
    notch,
    peak,
    reduce_noise,
    rms,
    split_on_silence,
    to_mono,
    trim_silence,
)


def synthesise() -> Audio:
    """A recording with every problem this module exists for."""
    rate = 44_100
    rng = np.random.default_rng(0)
    seconds = 4.0
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)

    # "Speech": a couple of formant-ish tones, only in the middle.
    voice = 0.08 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 700 * t)
    envelope = ((t > 1.0) & (t < 3.0)).astype(np.float64)
    voice *= envelope

    hum = 0.05 * np.sin(2 * np.pi * 50 * t)          # mains
    floor = rng.normal(0.0, 0.004, t.shape)          # room tone
    offset = 0.03                                     # DC

    left = voice + hum + floor + offset
    right = voice * 0.9 + hum + rng.normal(0.0, 0.004, t.shape) + offset
    return Audio(np.stack([left, right]).astype(np.float32), rate)


def describe(label: str, audio: Audio) -> None:
    print(f"   {label:24} {audio.describe()}")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        from hypernix.audio.audiofile import load

        loaded = load(argv[1])
        audio = Audio(np.asarray(loaded.samples, dtype=np.float32),
                      loaded.sample_rate)
        print(f"1. Loaded {argv[1]}")
    else:
        audio = synthesise()
        print("1. Synthesised a recording with the usual problems")
        print("   (44.1 kHz stereo, DC offset, 50 Hz hum, room tone, "
              "silence at both ends)")

    print()
    describe("as recorded", audio)
    print(f"   {'DC offset':24} {float(np.mean(audio.samples)):+.4f}")
    print(f"   {'clipping':24} {clipping_fraction(audio):.1%}\n")

    print("2. One stage at a time")
    mono = to_mono(audio)
    describe("mono", mono)

    unhummed = notch(mono, 50.0)
    describe("50 Hz notched", unhummed)

    rumbleless = high_pass(unhummed, 80.0)
    describe("80 Hz high-passed", rumbleless)

    denoised = reduce_noise(rumbleless, noise_seconds=0.5)
    describe("noise reduced", denoised)

    trimmed = trim_silence(denoised, threshold_db=-45.0)
    describe("silence trimmed", trimmed)

    levelled = normalise_rms(trimmed, target_db=-20.0)
    describe("levelled", levelled)
    print()

    print("3. Or the whole thing as one pipeline")
    pipeline = Pipeline.for_speech(16_000)
    print(f"   {pipeline.describe()}\n")
    finished = pipeline.run(audio)
    describe("result", finished)
    print(f"   {'DC offset':24} {float(np.mean(finished.samples)):+.4f}")
    print(f"   {'clipping':24} {clipping_fraction(finished):.1%}\n")

    print("4. Where the sound actually is")
    for start, end in detect_speech(finished, threshold_db=-45.0)[:6]:
        print(f"   {start:6.2f}s -> {end:6.2f}s")

    pieces = split_on_silence(finished, threshold_db=-45.0, min_silence=0.3)
    print(f"\n5. Split into {len(pieces)} piece(s) on the silences")
    for index, piece in enumerate(pieces, start=1):
        print(f"   {index}: {piece.duration:.2f}s, "
              f"peak {dbfs(peak(piece)):.1f} dBFS, "
              f"rms {dbfs(rms(piece)):.1f} dBFS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
