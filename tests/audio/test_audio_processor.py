"""hypernix.audio.processor — the DSP between decoding and features.

Signal processing is unusually testable and unusually often untested.
Every claim a filter makes is measurable on a sine wave: a second-order
low-pass is exactly -3 dB at its cutoff and falls 40 dB per decade, a
notch removes one frequency and leaves its neighbours alone, and a
resampler either rejects the energy above the new Nyquist or folds it
back into the audible band as a permanent metallic ring.

So none of these assert "it ran". They measure the thing the docstring
promises, against tones with known frequencies, and the numbers are
tight enough that a wrong coefficient fails rather than merely shifting.
"""
from __future__ import annotations

import numpy as np
import pytest

from hypernix.audio.processor import (
    Audio,
    Pipeline,
    apply_gain,
    band_pass,
    biquad,
    chunks,
    clipping_fraction,
    compress,
    dbfs,
    dc_offset_removed,
    detect_speech,
    fade,
    high_pass,
    limit,
    low_pass,
    noise_gate,
    normalise_peak,
    normalise_rms,
    notch,
    peak,
    pre_emphasis,
    reduce_noise,
    resample,
    rms,
    split_on_silence,
    to_mono,
    trim_silence,
)

RATE = 16_000


def tone(frequency: float, *, rate: int = RATE, seconds: float = 1.0,
         amplitude: float = 0.5) -> Audio:
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False,
                    dtype=np.float64)
    return Audio((amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32),
                 rate)


def silence(seconds: float, *, rate: int = RATE) -> Audio:
    return Audio(np.zeros(int(rate * seconds), dtype=np.float32), rate)


def settled(audio: Audio) -> Audio:
    """The second half, past the filter's start-up transient."""
    return Audio(audio.samples[audio.frames // 2 :], audio.sample_rate)


def gain_db(before: Audio, after: Audio) -> float:
    return dbfs(rms(settled(after))) - dbfs(rms(before))


def dominant_hz(audio: Audio) -> float:
    x = audio.samples.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.fft.rfftfreq(x.size, 1.0 / audio.sample_rate)[int(np.argmax(spectrum))])


# ---------------------------------------------------------------------------


class TestAudioCarriesItsRate:
    def test_a_rate_cannot_be_lost_mid_chain(self):
        """The bug this container exists to stop: audio that comes out
        at the right pitch and the wrong speed."""
        a = tone(440)
        assert high_pass(a, 100).sample_rate == RATE
        assert resample(a, 8000).sample_rate == 8000

    def test_it_refuses_a_nonsense_rate(self):
        with pytest.raises(ValueError, match="positive"):
            Audio(np.zeros(4, dtype=np.float32), 0)

    def test_it_refuses_more_than_two_dimensions(self):
        with pytest.raises(ValueError, match="dimensions"):
            Audio(np.zeros((2, 2, 2), dtype=np.float32), RATE)

    def test_duration_and_channels_read_correctly(self):
        stereo = Audio(np.zeros((2, RATE), dtype=np.float32), RATE)
        assert stereo.channels == 2
        assert stereo.duration == pytest.approx(1.0)


class TestNothingIsModifiedInPlace:
    """In-place DSP costs an afternoon the first time a caller reuses a
    buffer it thought was untouched."""

    @pytest.mark.parametrize(
        "operation",
        [
            lambda a: apply_gain(a, -6.0),
            lambda a: high_pass(a, 100.0),
            lambda a: normalise_peak(a),
            lambda a: fade(a),
            lambda a: dc_offset_removed(a),
            lambda a: limit(a),
            lambda a: to_mono(a),
        ],
    )
    def test_the_input_survives(self, operation):
        audio = tone(440)
        before = audio.samples.copy()
        operation(audio)
        assert np.array_equal(audio.samples, before)


class TestFilters:
    """Every number here is the textbook value for a second-order
    section. A wrong coefficient moves them."""

    def test_a_low_pass_is_minus_three_db_at_its_cutoff(self):
        """The definition of cutoff, and the tightest single check that
        the biquad coefficients are right."""
        assert gain_db(tone(1000), low_pass(tone(1000), 1000)) == pytest.approx(
            -3.0, abs=0.3
        )

    def test_a_high_pass_is_minus_three_db_at_its_cutoff(self):
        assert gain_db(tone(1000), high_pass(tone(1000), 1000)) == pytest.approx(
            -3.0, abs=0.3
        )

    def test_second_order_falls_forty_db_per_decade(self):
        """100 Hz is one decade below a 1 kHz high-pass."""
        assert gain_db(tone(100), high_pass(tone(100), 1000)) == pytest.approx(
            -40.0, abs=3.0
        )

    def test_a_passband_is_left_alone(self):
        assert gain_db(tone(4000), high_pass(tone(4000), 1000)) == pytest.approx(
            0.0, abs=0.5
        )
        assert gain_db(tone(100), low_pass(tone(100), 1000)) == pytest.approx(
            0.0, abs=0.5
        )

    def test_a_notch_removes_one_frequency_and_spares_its_neighbours(self):
        """Mains hum at 50 Hz, with the voice at 200 Hz untouched. A
        notch wide enough to be gentle takes the voice with it."""
        assert gain_db(tone(50), notch(tone(50), 50)) < -20.0
        assert gain_db(tone(200), notch(tone(200), 50)) == pytest.approx(0.0, abs=0.5)

    def test_a_band_pass_keeps_its_centre(self):
        assert gain_db(tone(1000), band_pass(tone(1000), 1000)) == pytest.approx(
            0.0, abs=1.0
        )
        assert gain_db(tone(100), band_pass(tone(100), 1000)) < -15.0

    def test_a_cutoff_above_nyquist_is_refused(self):
        """Silently wrapping produces a filter that does the opposite of
        what was asked."""
        with pytest.raises(ValueError, match="Nyquist"):
            high_pass(tone(440), 9000)

    def test_a_cutoff_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="Nyquist"):
            low_pass(tone(440), 0)

    def test_a_biquad_needs_three_and_three(self):
        with pytest.raises(ValueError, match="three"):
            biquad(tone(440), [1.0, 0.0], [1.0, 0.0, 0.0])

    def test_pre_emphasis_lifts_the_top(self):
        assert gain_db(tone(4000), pre_emphasis(tone(4000))) > gain_db(
            tone(200), pre_emphasis(tone(200))
        )


class TestResampling:
    def test_it_does_not_alias_where_naive_decimation_does(self):
        """The whole reason this is not `samples[::2]`. A 6 kHz tone
        resampled to 8 kHz has nowhere legal to go -- naive decimation
        folds it to 2 kHz at full strength, permanently."""
        source = tone(6000, seconds=0.5)
        naive = Audio(source.samples[::2], 8000)
        proper = resample(source, 8000)

        assert dominant_hz(naive) == pytest.approx(2000.0, abs=50.0)
        # Same fold-down frequency, but 40+ dB quieter: rejected, not moved.
        assert dbfs(rms(proper)) < dbfs(rms(naive)) - 40.0

    def test_a_tone_inside_the_new_band_is_untouched(self):
        source = tone(500, seconds=0.5)
        out = resample(source, 8000)
        assert dominant_hz(out) == pytest.approx(500.0, abs=10.0)
        assert dbfs(rms(out)) - dbfs(rms(source)) == pytest.approx(0.0, abs=0.5)

    def test_a_round_trip_comes_back(self):
        source = tone(1000, seconds=0.5)
        back = resample(resample(source, 48_000), RATE)
        count = min(source.frames, back.frames)
        # Edges excluded: the kernel hangs off the end there by design.
        error = np.abs(
            source.samples[:count].astype(np.float64)
            - back.samples[:count].astype(np.float64)
        )[800:-800]
        assert error.max() < 0.01

    def test_upsampling_lengthens_proportionally(self):
        assert resample(tone(440, seconds=0.25), 32_000).frames == pytest.approx(
            8000, abs=2
        )

    def test_the_same_rate_is_a_copy_not_a_share(self):
        source = tone(440)
        out = resample(source, RATE)
        out.samples[0] = 0.9
        assert source.samples[0] != 0.9

    def test_a_negative_rate_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            resample(tone(440), -1)

    def test_stereo_resamples_both_channels(self):
        stereo = Audio(np.stack([tone(440).samples, tone(880).samples]), RATE)
        out = resample(stereo, 8000)
        assert out.channels == 2 and out.frames == 8000


class TestLevels:
    def test_peak_normalising_lands_on_the_target(self):
        out = normalise_peak(tone(440, amplitude=0.2), target_db=-1.0)
        assert dbfs(peak(out)) == pytest.approx(-1.0, abs=0.05)

    def test_the_default_leaves_headroom(self):
        """Exactly full scale clips as soon as anything resamples or
        encodes it, because reconstruction overshoots between samples."""
        assert dbfs(peak(normalise_peak(tone(440)))) < 0.0

    def test_rms_normalising_lands_on_the_target(self):
        out = normalise_rms(tone(440, amplitude=0.05), target_db=-20.0)
        assert dbfs(rms(out)) == pytest.approx(-20.0, abs=0.05)

    def test_rms_normalising_ignores_one_loud_bang(self):
        """Peak normalising a spoken recording lets one door slam set
        the level for the whole thing."""
        speech = tone(300, seconds=1.0, amplitude=0.1)
        banged = speech.samples.copy()
        banged[8000] = 0.99
        with_bang = Audio(banged, RATE)

        by_rms = normalise_rms(with_bang, target_db=-20.0)
        by_peak = normalise_peak(with_bang, target_db=-1.0)
        assert dbfs(rms(by_rms)) > dbfs(rms(by_peak))

    def test_silence_normalises_to_silence_rather_than_exploding(self):
        assert peak(normalise_rms(silence(0.1))) == 0.0
        assert peak(normalise_peak(silence(0.1))) == 0.0

    def test_gain_is_decibels(self):
        assert peak(apply_gain(tone(440, amplitude=0.5), -6.02)) == pytest.approx(
            0.25, abs=0.005
        )

    def test_dc_offset_is_removed(self):
        offset = Audio(tone(440).samples + 0.3, RATE)
        assert float(np.mean(dc_offset_removed(offset).samples)) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_limiting_scales_rather_than_clips(self):
        """Clipping generates harmonics across the spectrum, which is
        what everything before this was trying to avoid."""
        loud = apply_gain(tone(440), 12.0)
        out = limit(loud, ceiling_db=-0.3)
        assert dbfs(peak(out)) == pytest.approx(-0.3, abs=0.05)
        assert clipping_fraction(out) == 0.0
        assert dominant_hz(out) == pytest.approx(440.0, abs=10.0)

    def test_limiting_something_already_quiet_changes_nothing(self):
        quiet = tone(440, amplitude=0.1)
        assert np.allclose(limit(quiet).samples, quiet.samples)

    def test_dbfs_of_silence_is_minus_infinity(self):
        assert dbfs(0.0) == -float("inf")


class TestDynamics:
    def test_compression_narrows_the_range(self):
        quiet = tone(440, seconds=0.3, amplitude=0.02)
        loud = tone(440, seconds=0.3, amplitude=0.8)
        joined = Audio(np.concatenate([quiet.samples, loud.samples]), RATE)

        before = dbfs(peak(loud)) - dbfs(peak(quiet))
        out = compress(joined, threshold_db=-20.0, ratio=4.0)
        half = out.frames // 2
        after = dbfs(peak(Audio(out.samples[half:], RATE))) - dbfs(
            peak(Audio(out.samples[:half], RATE))
        )
        assert after < before

    def test_a_ratio_below_one_is_refused(self):
        with pytest.raises(ValueError, match="expander"):
            compress(tone(440), ratio=0.5)

    def test_the_gate_closes_on_quiet_and_opens_on_loud(self):
        quiet = tone(440, seconds=0.4, amplitude=0.001)
        loud = tone(440, seconds=0.4, amplitude=0.5)
        joined = Audio(np.concatenate([quiet.samples, loud.samples]), RATE)
        out = noise_gate(joined, threshold_db=-40.0)

        half = out.frames // 2
        assert rms(Audio(out.samples[: half - 800], RATE)) < 1e-4
        assert rms(Audio(out.samples[half + 2000 :], RATE)) > 0.1

    def test_the_gate_does_not_chatter(self):
        """A gate that switches instantaneously stutters on every sample
        near the threshold, which is worse than the noise was."""
        rng = np.random.default_rng(0)
        borderline = Audio(
            (rng.normal(0, 0.01, RATE // 2)).astype(np.float32), RATE
        )
        out = noise_gate(borderline, threshold_db=-40.0)
        # Count sign-flips of the gate's effect; a chattering gate makes
        # the ratio jump thousands of times.
        ratio = np.abs(out.samples) > np.abs(borderline.samples) * 0.5
        flips = int(np.sum(np.diff(ratio.astype(np.int8)) != 0))
        assert flips < 50


class TestSilenceAndSplitting:
    @staticmethod
    def _speech_then_quiet() -> Audio:
        return Audio(
            np.concatenate(
                [
                    silence(0.5).samples,
                    tone(300, seconds=0.5).samples,
                    silence(0.5).samples,
                ]
            ),
            RATE,
        )

    def test_it_finds_where_the_sound_is(self):
        regions = detect_speech(self._speech_then_quiet())
        assert len(regions) == 1
        assert regions[0][0] == pytest.approx(0.5, abs=0.05)
        assert regions[0][1] == pytest.approx(1.0, abs=0.05)

    def test_trimming_keeps_a_little_padding(self):
        """A clip starting exactly on the first loud sample has its
        onset cut off, which is audible."""
        out = trim_silence(self._speech_then_quiet(), pad=0.05)
        assert out.duration == pytest.approx(0.6, abs=0.06)

    def test_trimming_silence_gives_back_nothing(self):
        assert trim_silence(silence(0.5)).frames == 0

    def test_splitting_finds_each_piece(self):
        joined = Audio(
            np.concatenate(
                [
                    tone(300, seconds=0.3).samples,
                    silence(0.5).samples,
                    tone(300, seconds=0.3).samples,
                    silence(0.5).samples,
                    tone(300, seconds=0.3).samples,
                ]
            ),
            RATE,
        )
        assert len(split_on_silence(joined, min_silence=0.3)) == 3

    def test_a_short_gap_does_not_split(self):
        joined = Audio(
            np.concatenate(
                [
                    tone(300, seconds=0.3).samples,
                    silence(0.05).samples,
                    tone(300, seconds=0.3).samples,
                ]
            ),
            RATE,
        )
        assert len(split_on_silence(joined, min_silence=0.3)) == 1


class TestNoiseReduction:
    def test_it_lowers_the_floor_and_keeps_the_tone(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 0.02, RATE).astype(np.float32)
        signal = tone(440, seconds=1.0, amplitude=0.3).samples
        # The first half-second is noise alone -- the profile it learns.
        noisy = Audio(np.concatenate([noise[: RATE // 2], signal + noise]), RATE)

        out = reduce_noise(noisy, noise_seconds=0.4)
        quiet_before = rms(Audio(noisy.samples[: RATE // 2], RATE))
        quiet_after = rms(Audio(out.samples[: RATE // 2], RATE))

        assert quiet_after < quiet_before
        assert dominant_hz(Audio(out.samples[RATE:], RATE)) == pytest.approx(
            440.0, abs=15.0
        )

    def test_something_too_short_to_profile_is_returned_unchanged(self):
        tiny = tone(440, seconds=0.01)
        assert np.allclose(reduce_noise(tiny).samples, tiny.samples)


class TestChunking:
    def test_the_last_piece_is_short_not_padded(self):
        """Padding would be a silent lie about how much audio there was."""
        pieces = list(chunks(tone(440, seconds=2.5), 1.0))
        assert len(pieces) == 3
        assert pieces[-1].duration == pytest.approx(0.5, abs=0.01)

    def test_overlap_produces_more_pieces(self):
        plain = list(chunks(tone(440, seconds=2.0), 1.0))
        lapped = list(chunks(tone(440, seconds=2.0), 1.0, overlap=0.5))
        assert len(lapped) > len(plain)

    def test_a_silly_overlap_is_refused(self):
        with pytest.raises(ValueError, match="fraction"):
            list(chunks(tone(440), 1.0, overlap=1.0))


class TestPipeline:
    def test_the_speech_preset_runs_end_to_end(self):
        messy = Audio(
            (tone(300, rate=44_100, seconds=1.0, amplitude=0.05).samples
             + tone(50, rate=44_100, seconds=1.0, amplitude=0.2).samples
             + 0.1),
            44_100,
        )
        out = Pipeline.for_speech(16_000).run(messy)

        assert out.sample_rate == 16_000
        assert out.channels == 1
        assert dbfs(peak(out)) <= -1.0 + 0.05           # limited
        assert float(np.mean(out.samples)) == pytest.approx(0.0, abs=0.01)

    def test_the_speech_preset_removes_mains_hum(self):
        """Found by running the example rather than by reading it.

        An 80 Hz high-pass is 0.7 of an octave above 50 Hz, so it takes
        about 10 dB off the hum and leaves the rest — enough to sit
        *above* the -45 dB gate, which then never closes. The silences
        stop being silent and everything that looks for them finds one
        continuous region.
        """
        rate = 44_100
        t = np.linspace(0.0, 4.0, rate * 4, endpoint=False)
        speech = 0.08 * np.sin(2 * np.pi * 220 * t) * ((t > 1.0) & (t < 3.0))
        hum = 0.05 * np.sin(2 * np.pi * 50 * t)
        noisy = Audio((speech + hum).astype(np.float32), rate)

        with_notch = Pipeline.for_speech(16_000, mains=50.0).run(noisy)
        without = Pipeline.for_speech(16_000, mains=0).run(noisy)

        # The lead-in is silence in the source. With the hum removed it
        # gates to nothing; with the hum left in it does not.
        def lead_in(audio: Audio) -> float:
            return rms(Audio(audio.samples[: int(0.8 * audio.sample_rate)],
                             audio.sample_rate))

        # The consequence that actually matters, and the unambiguous
        # one: with the hum notched, the silences are silent and the
        # speech is found where it is. Without, the whole recording
        # reads as one continuous region.
        found = detect_speech(with_notch, threshold_db=-45.0)
        missed = detect_speech(without, threshold_db=-45.0)

        assert found, "no speech found at all"
        assert found[-1][1] < 3.5, found
        assert missed == [(0.0, without.duration)], missed

        # Supporting: about 10 dB, measured rather than guessed. The
        # first version of this asserted a factor of four picked out of
        # the air and the real figure is three.
        assert lead_in(with_notch) < lead_in(without) / 2

    def test_the_preset_can_skip_the_notch(self):
        """60 Hz elsewhere, and 0 for a recording with no mains in it."""
        assert "notch" not in Pipeline.for_speech(16_000, mains=0).describe()
        assert "60" in Pipeline.for_speech(16_000, mains=60.0).describe()

    def test_order_is_data_so_it_can_be_written_down(self):
        """Gate-then-compress and compress-then-gate are different
        processors, and which one you have is not obvious from the
        output."""
        chain = Pipeline().then("high_pass", cutoff=80.0).then("limit")
        assert "high_pass" in chain.describe()
        assert chain.run(tone(440)).sample_rate == RATE

    def test_an_unknown_step_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="no audio step"):
            Pipeline().then("reverb").run(tone(440))

    def test_a_private_name_is_not_reachable_as_a_step(self):
        with pytest.raises(ValueError, match="no audio step"):
            Pipeline().then("_envelope").run(tone(440))


class TestMono:
    def test_it_averages_rather_than_taking_the_left(self):
        """A stereo recording panned right becomes silence if you take
        the left, and nothing reports it."""
        stereo = Audio(
            np.stack(
                [np.zeros(RATE, dtype=np.float32), tone(440).samples]
            ),
            RATE,
        )
        assert rms(to_mono(stereo)) > 0.1

    def test_mono_in_mono_out(self):
        mono = tone(440)
        assert to_mono(mono).samples is mono.samples
