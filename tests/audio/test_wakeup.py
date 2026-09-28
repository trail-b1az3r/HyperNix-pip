"""wake-up: train a wake word, then listen for it.

What openWakeWord does, without using it. The parts worth testing are
the ones that fail quietly: a fragmented stream reassembled in the wrong
order sounds almost right, a dataset with no negatives trains to 100%
and fires at the television, and a model saved without its mel settings
loads and sees a different spectrum than it was trained on.
"""
from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

import pytest

from hypernix.audio.audiofile import (
    AudioError,
    decoders_available,
    iter_audio_files,
    load_audio,
    load_fragments,
    resample,
)
from hypernix.audio.features import (
    MelConfig,
    frame_count,
    hz_to_mel,
    log_mel_frames,
    mel_filterbank,
    mel_to_hz,
)
from hypernix.audio.wakeup import (
    AugmentConfig,
    Example,
    WakeUpConfig,
    WakeUpDataset,
    WakeUpError,
    augment,
    build_dataset,
    examples_from_folder,
    synthesize_examples,
)

SR = 16000


def _tone(freq: float, seconds: float = 0.5, sample_rate: int = SR) -> list[float]:
    return [
        math.sin(2 * math.pi * freq * i / sample_rate)
        for i in range(int(seconds * sample_rate))
    ]


def _write_wav(path: Path, samples: list[float], *, rate: int = SR,
               channels: int = 1, width: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        frames = b""
        for value in samples:
            clipped = max(-1.0, min(1.0, value))
            for _ in range(channels):
                frames += struct.pack("<h", int(clipped * 32767))
        handle.writeframes(frames)
    return path


class TestReadingAudio:
    def test_a_wav_round_trips(self, tmp_path):
        original = _tone(440, 0.2)
        loaded = load_audio(_write_wav(tmp_path / "a.wav", original))
        assert loaded.sample_rate == SR
        assert len(loaded.samples) == len(original)
        assert loaded.samples[100] == pytest.approx(original[100], abs=1e-4)

    def test_stereo_is_averaged_not_halved(self, tmp_path):
        """Taking one channel silently halves the signal for anything
        recorded quieter on one side."""
        loaded = load_audio(_write_wav(tmp_path / "s.wav", _tone(440, 0.2), channels=2))
        assert len(loaded.samples) == int(0.2 * SR)

    def test_it_resamples_to_the_target_rate(self, tmp_path):
        # 0.2 s *at 8 kHz*, so the file's own rate and its contents agree.
        # Writing 16 kHz samples into an 8 kHz header would make it a
        # 0.4 s file, which is a different test than the one intended.
        quiet = _tone(440, 0.2, sample_rate=8000)
        loaded = load_audio(_write_wav(tmp_path / "b.wav", quiet, rate=8000))
        assert loaded.sample_rate == SR
        assert len(loaded.samples) == pytest.approx(int(0.2 * SR), rel=0.01)

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(AudioError, match="No such audio file"):
            load_audio(tmp_path / "absent.wav")

    def test_an_undecodable_format_names_the_remedy(self, tmp_path):
        """Not an ImportError from whichever decoder was tried last."""
        fake = tmp_path / "x.opus"
        fake.write_bytes(b"not really an opus file")
        if decoders_available()["ffmpeg"]:
            pytest.skip("ffmpeg is installed; it will produce its own error")
        with pytest.raises(AudioError, match="ffmpeg"):
            load_audio(fake)

    def test_resample_is_a_no_op_at_the_same_rate(self):
        values = _tone(440, 0.05)
        assert resample(values, SR, SR) == values

    def test_iter_finds_only_audio(self, tmp_path):
        _write_wav(tmp_path / "a.wav", _tone(440, 0.05))
        _write_wav(tmp_path / "nested" / "b.wav", _tone(880, 0.05))
        (tmp_path / "notes.txt").write_text("not audio")
        found = iter_audio_files(tmp_path)
        assert [p.name for p in found] == ["a.wav", "b.wav"]


class TestFragmentOrder:
    """A stream reassembled wrong sounds almost right, which is the worst
    kind of wrong for training data."""

    def test_part10_comes_after_part2(self, tmp_path):
        for index in (1, 2, 10):
            _write_wav(tmp_path / f"part{index}.wav", [index / 100.0] * 1600)
        joined = load_fragments(list(iter_audio_files(tmp_path)))
        # Each fragment is a constant; read the value at the start of each.
        thirds = [joined.samples[0], joined.samples[1600], joined.samples[3200]]
        assert thirds == pytest.approx([0.01, 0.02, 0.10], abs=1e-3)

    def test_no_fragments_is_an_error(self):
        with pytest.raises(AudioError, match="No fragments"):
            load_fragments([])


class TestMelFeatures:
    def test_the_mel_scale_round_trips(self):
        for hz in (100.0, 1000.0, 4000.0):
            assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-6)

    def test_a_tone_lands_in_the_expected_band(self):
        """A model trained on one mel definition and served on another
        sees a shifted spectrum and quietly gets worse."""
        config = MelConfig()
        frames = log_mel_frames(_tone(1000, 0.2), config)
        peak = max(range(config.n_mels), key=lambda i: frames[5][i])
        expected = int(
            (hz_to_mel(1000) - hz_to_mel(config.f_min))
            / (hz_to_mel(config.f_max) - hz_to_mel(config.f_min))
            * (config.n_mels + 1)
        )
        assert abs(peak - expected) <= 1

    def test_frame_count_matches_what_is_produced(self):
        config = MelConfig()
        samples = _tone(440, 0.31)
        assert len(log_mel_frames(samples, config)) == frame_count(len(samples), config)

    def test_audio_shorter_than_a_frame_yields_nothing(self):
        assert log_mel_frames([0.0] * 10, MelConfig()) == []

    def test_silence_does_not_produce_infinities(self):
        """A -inf propagates as NaN through the first layer."""
        frames = log_mel_frames([0.0] * (SR // 2), MelConfig())
        assert all(math.isfinite(v) for row in frames for v in row)

    def test_every_band_has_some_response(self):
        """A band collapsed to zeros contributes nothing and wastes a
        model input."""
        for row in mel_filterbank(MelConfig()):
            assert sum(row) > 0

    def test_the_numpy_and_pure_python_paths_agree(self, monkeypatch):
        config = MelConfig(n_fft=512)
        samples = _tone(700, 0.1)
        with_numpy = log_mel_frames(samples, config)

        real_import = __import__

        def _no_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy disabled for this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_numpy)
        without_numpy = log_mel_frames(samples, config)

        assert len(with_numpy) == len(without_numpy)
        for a, b in zip(with_numpy[0], without_numpy[0], strict=True):
            assert a == pytest.approx(b, abs=1e-6)


class TestAugmentation:
    def test_it_fits_the_window_exactly(self):
        window = 2 * SR
        for length in (SR // 2, SR, 3 * SR):
            out = augment(
                _tone(440, length / SR), window_samples=window, rng=random.Random(0)
            )
            assert len(out) == window

    def test_it_never_clips_past_full_scale(self):
        """A gain past 1.0 wraps when written to 16-bit, which sounds
        like a click and teaches the model the phrase contains one."""
        loud = [0.9] * SR
        out = augment(
            loud, window_samples=SR,
            config=AugmentConfig(gain_db=(20.0, 20.0), noise_probability=0.0,
                                 reverb_probability=0.0),
            rng=random.Random(0),
        )
        assert all(-1.0 <= v <= 1.0 for v in out)

    def test_two_copies_differ(self):
        """Otherwise 'copies' multiplies the dataset without adding
        anything."""
        base = _tone(440, 0.5)
        first = augment(base, window_samples=SR, rng=random.Random(1))
        second = augment(base, window_samples=SR, rng=random.Random(2))
        assert first != second

    def test_noise_is_mixed_at_the_requested_snr(self):
        signal = _tone(440, 1.0)
        noisy = augment(
            signal, window_samples=SR,
            config=AugmentConfig(gain_db=(0.0, 0.0), speed=(1.0, 1.0), shift=0.0,
                                 snr_db=(0.0, 0.0), noise_probability=1.0,
                                 reverb_probability=0.0),
            noise=[[random.Random(0).gauss(0, 0.3) for _ in range(SR)]],
            rng=random.Random(0),
        )
        assert noisy != signal


class TestTheDatasetSaysWhatIsWrongWithIt:
    def _dataset(self, positives: int, negatives: int) -> WakeUpDataset:
        return build_dataset(
            [Example(_tone(440, 0.4), True, f"a:{i}") for i in range(positives)],
            [Example(_tone(880, 0.4), False, f"b:{i}") for i in range(negatives)],
            config=WakeUpConfig(window_seconds=0.6),
            copies=1, seed=0,
        )

    def test_no_negatives_is_called_out(self):
        """A wake-word model is mostly a rejector."""
        notes = self._dataset(5, 0).warnings()
        assert any("rejector" in note for note in notes)

    def test_a_thin_negative_ratio_is_called_out(self):
        notes = self._dataset(10, 10).warnings()
        assert any("television" in note for note in notes)

    def test_a_healthy_dataset_only_notes_the_positive_count(self):
        notes = self._dataset(10, 100).warnings()
        assert not any("television" in note for note in notes)

    def test_it_counts_where_examples_came_from(self):
        """Mixing sources is the advice; reporting the mix is how anyone
        can tell whether they took it."""
        dataset = build_dataset(
            [Example(_tone(440, 0.4), True, "tts:voice1:hey:0"),
             Example(_tone(450, 0.4), True, "mic:0")],
            [Example(_tone(880, 0.4), False, "/a/b.wav")],
            config=WakeUpConfig(window_seconds=0.6), copies=1, seed=0,
        )
        assert dataset.sources["tts"] == 1
        assert dataset.sources["mic"] == 1

    def test_negatives_are_augmented_too(self):
        """Clean negatives against noisy positives teaches the model to
        detect noise."""
        source = Path("src/hypernix/audio/wakeup.py").read_text()
        build = source.split("def build_dataset(")[1].split("\ndef ")[0]
        assert "for group, positive in" in build


class TestGatheringExamples:
    def test_a_folder_becomes_examples(self, tmp_path):
        for index in range(3):
            _write_wav(tmp_path / f"take{index}.wav", _tone(440 + index, 0.3))
        examples = examples_from_folder(tmp_path, positive=True)
        assert len(examples) == 3
        assert all(e.positive for e in examples)

    def test_an_empty_folder_is_an_error(self, tmp_path):
        with pytest.raises(WakeUpError, match="No audio files"):
            examples_from_folder(tmp_path, positive=True)

    def test_one_bad_file_does_not_lose_the_folder(self, tmp_path):
        """Finding out after an hour of gathering is worse than being
        told at the time."""
        _write_wav(tmp_path / "good.wav", _tone(440, 0.3))
        (tmp_path / "bad.wav").write_bytes(b"RIFF corrupted nonsense")
        skipped = []
        examples = examples_from_folder(
            tmp_path, positive=True, on_skip=lambda p, r: skipped.append(p.name)
        )
        assert len(examples) == 1
        if skipped:
            assert skipped == ["bad.wav"]

    def test_fragments_load_as_one_recording(self, tmp_path):
        for index in (1, 2, 3):
            _write_wav(tmp_path / f"part{index}.wav", _tone(440, 0.2))
        examples = examples_from_folder(tmp_path, positive=True, fragments=True)
        assert len(examples) == 1
        assert len(examples[0].samples) == pytest.approx(int(0.6 * SR), rel=0.01)


class TestSynthesis:
    class _Engine:
        def synthesize(self, phrase, voice="default"):
            return _tone(300 + len(phrase) * 10 + len(voice), 0.4)

    def test_it_generates_across_voices(self):
        examples = synthesize_examples(
            ["hey nix"], voices=("a", "b"), per_voice=3, engine=self._Engine()
        )
        assert len(examples) == 6
        assert all(e.positive for e in examples)

    def test_more_than_four_voices_is_refused(self):
        """Past four the marginal voice adds less than the time it costs."""
        with pytest.raises(WakeUpError, match="at most 4"):
            synthesize_examples(
                ["hey"], voices=tuple("abcde"), engine=self._Engine()
            )

    def test_no_wake_words_is_refused(self):
        with pytest.raises(WakeUpError, match="No wake words"):
            synthesize_examples([], engine=self._Engine())

    def test_the_source_records_that_it_is_tts(self):
        examples = synthesize_examples(
            ["hey"], voices=("a",), per_voice=1, engine=self._Engine()
        )
        assert examples[0].source.startswith("tts:")


class TestConfigTravelsWithTheModel:
    def test_it_round_trips(self):
        config = WakeUpConfig(
            wake_words=("hey nix", "ok nix"), window_seconds=1.2, threshold=0.8
        )
        restored = WakeUpConfig.from_dict(config.to_dict())
        assert restored.wake_words == config.wake_words
        assert restored.window_seconds == config.window_seconds
        assert restored.threshold == config.threshold
        assert restored.mel.n_mels == config.mel.n_mels

    def test_the_window_is_a_whole_number_of_samples(self):
        assert WakeUpConfig(window_seconds=1.5).window_samples == int(1.5 * SR)


def _chirp(f0: float, f1: float, seconds: float = 0.6, *, noise: float = 0.02,
           rng: random.Random | None = None) -> list[float]:
    """A rising or falling sweep: a stand-in for a phrase.

    Real speech would need real recordings, which a unit test cannot
    carry. What this checks is that the pipeline can learn *a* separable
    acoustic pattern from noisy augmented examples — if it cannot do that
    it certainly cannot do a wake word.
    """
    rng = rng or random.Random(0)
    count = int(seconds * SR)
    return [
        math.sin(2 * math.pi * (f0 + (f1 - f0) * i / count) * i / SR) * 0.5
        + rng.gauss(0, noise)
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def trained():
    """One trained model, shared: training is the slow part.

    Sized to be the smallest run that still demonstrates learning, and
    pinned to one torch thread. Under `pytest -n 4` every worker that
    draws a test from this module builds its own copy of the fixture, and
    four torch processes each spawning one intra-op thread per core turn
    a 17-second fixture into minutes of contention — which showed up as
    this suite going from 80 seconds to over five.
    """
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)

    from hypernix.audio.wakeup import train_wakeword

    rng = random.Random(0)
    config = WakeUpConfig(wake_words=("hey nix",), window_seconds=0.8)
    positives = [Example(_chirp(300, 1800, rng=rng), True, f"synthetic:{i}")
                 for i in range(24)]
    negatives = [Example(_chirp(1800, 300, rng=rng), False, f"synthetic:{i}")
                 for i in range(72)]
    negatives += [
        Example([rng.gauss(0, 0.1) for _ in range(int(0.6 * SR))], False, f"noise:{i}")
        for i in range(48)
    ]
    dataset = build_dataset(positives, negatives, config=config, copies=2, seed=0)
    return train_wakeword(dataset, epochs=8, seed=0), config


class TestItActuallyLearns:
    def test_it_separates_the_pattern_from_everything_else(self, trained):
        result, _ = trained
        assert result["val_accuracy"] > 0.9, result["history"][-1]

    def test_it_reports_precision_and_recall_not_just_accuracy(self, trained):
        """Accuracy on a 5:1 dataset is 83% for a model that always says
        no — the two numbers that matter are the other two."""
        result, _ = trained
        assert result["precision"] > 0.8
        assert result["recall"] > 0.8

    def test_a_held_out_positive_scores_above_a_negative(self, trained):
        from hypernix.audio.wakeup import WakeUpDetector

        result, config = trained
        detector = WakeUpDetector(result["model"], config)
        rng = random.Random(99)      # not a seed used in training
        assert detector.score(_chirp(300, 1800, rng=rng)) > detector.score(
            _chirp(1800, 300, rng=rng)
        )

    def test_the_loss_is_weighted_for_the_class_imbalance(self):
        """Negatives outnumber positives by design; unweighted, the model
        learns that answering 'no' is right most of the time."""
        source = Path("src/hypernix/audio/wakeup.py").read_text()
        assert "pos_weight" in source

    def test_training_on_nothing_is_refused(self):
        pytest.importorskip("torch")
        from hypernix.audio.wakeup import train_wakeword

        with pytest.raises(WakeUpError, match="empty"):
            train_wakeword(WakeUpDataset())

    def test_strict_refuses_a_dataset_with_warnings(self):
        pytest.importorskip("torch")
        from hypernix.audio.wakeup import train_wakeword

        dataset = build_dataset(
            [Example(_tone(440, 0.4), True, "a")], [],
            config=WakeUpConfig(window_seconds=0.6), copies=1, seed=0,
        )
        with pytest.raises(WakeUpError, match="rejector"):
            train_wakeword(dataset, epochs=1, strict=True)


class TestStreamingDetection:
    def test_it_needs_a_full_window_before_scoring(self, trained):
        from hypernix.audio.wakeup import WakeUpDetector

        result, config = trained
        detector = WakeUpDetector(result["model"], config)
        assert detector.push([0.0] * 100) is None

    def test_it_fires_on_the_pattern(self, trained):
        from hypernix.audio.wakeup import WakeUpDetector

        result, config = trained
        detector = WakeUpDetector(result["model"], config, hop_seconds=0.1)
        fired = None
        chunk = 1600
        audio = [0.0] * SR + _chirp(300, 1800, rng=random.Random(5))
        for start in range(0, len(audio), chunk):
            fired = detector.push(audio[start:start + chunk], now=start / SR) or fired
        assert fired is not None and fired["detected"]

    def test_one_utterance_does_not_fire_six_times(self, trained):
        """Without a refractory period every overlapping window that
        contains the phrase fires."""
        from hypernix.audio.wakeup import WakeUpDetector

        result, config = trained
        detector = WakeUpDetector(
            result["model"], config, hop_seconds=0.05, refractory_seconds=2.0
        )
        audio = _chirp(300, 1800, seconds=1.5, rng=random.Random(6))
        fires = 0
        chunk = 800
        for start in range(0, len(audio), chunk):
            now = start / SR
            if detector.push(audio[start:start + chunk], now=now):
                fires += 1
        assert fires <= 1

    def test_reset_clears_the_buffer(self, trained):
        from hypernix.audio.wakeup import WakeUpDetector

        result, config = trained
        detector = WakeUpDetector(result["model"], config)
        detector.push([0.1] * config.window_samples)
        detector.reset()
        assert detector.push([0.0] * 10) is None

    def test_it_returns_rather_than_calls_back(self):
        """This module has no business ringing a bell."""
        source = Path("src/hypernix/audio/wakeup.py").read_text()
        push = source.split("    def push(")[1].split("\n\ndef ")[0]
        assert "return {" in push


class TestSavingAndLoading:
    def test_the_config_travels_with_the_weights(self, trained, tmp_path):
        """A model is useless without the exact mel settings it was
        trained on — a different hop and it sees a different spectrum."""
        from hypernix.audio.wakeup import load_detector, save_wakeword

        result, config = trained
        path = save_wakeword(result, tmp_path / "model.pt")
        detector = load_detector(path)
        assert detector.config.wake_words == config.wake_words
        assert detector.config.mel.n_mels == config.mel.n_mels
        assert detector.config.mel.hop_length == config.mel.hop_length

    def test_a_saved_model_scores_the_same(self, trained, tmp_path):
        from hypernix.audio.wakeup import WakeUpDetector, load_detector, save_wakeword

        result, config = trained
        clip = _chirp(300, 1800, rng=random.Random(11))
        before = WakeUpDetector(result["model"], config).score(clip)
        after = load_detector(save_wakeword(result, tmp_path / "m.pt")).score(clip)
        assert after == pytest.approx(before, abs=1e-5)

    def test_a_missing_model_says_so(self, tmp_path):
        pytest.importorskip("torch")
        from hypernix.audio.wakeup import load_detector

        with pytest.raises(WakeUpError, match="No wake-word model"):
            load_detector(tmp_path / "absent.pt")

    def test_someone_elses_checkpoint_is_refused(self, tmp_path):
        torch = pytest.importorskip("torch")
        from hypernix.audio.wakeup import load_detector

        path = tmp_path / "other.pt"
        torch.save({"something": "else"}, path)
        with pytest.raises(WakeUpError, match="not a HyperNix wake-word model"):
            load_detector(path)


class TestTheCLI:
    def test_it_is_registered_as_a_subcommand(self):
        source = Path("src/hypernix/interfaces/cli.py").read_text()
        assert '"wakeup",' in source
        assert 'cmd == "wakeup"' in source

    def test_the_subcommands_exist(self):
        from hypernix.audio.wakeup_cli import main

        with pytest.raises(SystemExit):
            main(["train", "--help"])

    def test_training_with_no_positives_says_what_to_do(self, capsys):
        from hypernix.audio.wakeup_cli import main

        code = main(["train", "hey nix"])
        assert code == 2
        assert "--tts" in capsys.readouterr().err
