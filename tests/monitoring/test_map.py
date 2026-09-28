"""Tests for ``hnx map`` (hypernix.map) — the steampunk schematic TUI.

Covers:
  - Config subsystem (poly/acc/use-gpu/tps/file-mode persistence + validation)
  - acc string parsing (1 / 1k / 1m / 1b / 1t / decimals / garbage)
  - Data model: tensor-name -> layer bucketing, safetensors file/folder
    scanning (exact counts), and the config.json analytical fallback
  - Rendering primitives: poly-tiered glyphs, dial math, throttle sweep,
    pipe/steam animation, engine icons, the Canvas grid, snake layout
  - Mouse SGR parsing + bottom-right hover detection + no-tty fallback
  - HyperMap end-to-end rendering (consistent row widths, all poly tiers,
    0/1/many layers, legend overlay alignment)
  - CLI argument parsing for every ``hnx map config ...`` subcommand
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import hypernix.map as hmap


@pytest.fixture(autouse=True)
def _isolated_map_config(tmp_path, monkeypatch):
    """Every test gets its own map_config.json so nothing touches the
    real ``~/.hypernix`` directory or leaks state between tests."""
    monkeypatch.setattr(hmap, "_MAP_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(hmap, "_MAP_CONFIG_FILE", tmp_path / "map_config.json")
    yield


# ---------------------------------------------------------------------------
# Config subsystem
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_defaults_when_no_file_exists(self):
        cfg = hmap.get_map_config()
        assert cfg["poly"] == 32
        # "auto" for both, so a first run with no configuration at all
        # finds a model and scales its dials to it. The old defaults
        # (acc "1m", file_mode 2) drew an empty schematic reading
        # "Σ params ?" and gave no hint that a path was needed.
        assert cfg["acc"] == hmap.ACC_AUTO
        assert cfg["use_gpu"] is True
        assert cfg["tps"] == 8
        assert cfg["file_mode"] == 0
        assert cfg["file_path"] is None

    def test_round_trip_persists_to_disk(self):
        hmap.set_poly(64)
        cfg = hmap.get_map_config()
        assert cfg["poly"] == 64
        assert hmap._MAP_CONFIG_FILE.exists()

    def test_corrupt_config_file_falls_back_to_defaults(self, capsys):
        hmap._MAP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        hmap._MAP_CONFIG_FILE.write_text("{not valid json", encoding="utf-8")
        cfg = hmap._load_map_config()
        assert cfg["poly"] == 32
        assert "Warning" in capsys.readouterr().err


class TestSetPoly:
    @pytest.mark.parametrize("value", [16, 32, 64, 128, "16", "128"])
    def test_valid_values(self, value):
        result = hmap.set_poly(value)
        assert result == int(value)
        assert hmap.get_map_config()["poly"] == int(value)

    @pytest.mark.parametrize("value", [8, 17, 256, 0, -1])
    def test_rejects_values_outside_the_four_levels(self, value):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_poly(value)

    def test_rejects_non_integer(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_poly("fast")


class TestParseAcc:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", 1.0),
            ("1k", 1e3),
            ("1K", 1e3),
            ("1m", 1e6),
            ("1b", 1e9),
            ("1t", 1e12),
            ("2.5m", 2.5e6),
            ("0.5k", 500.0),
            ("10", 10.0),
        ],
    )
    def test_valid_strings(self, value, expected):
        assert hmap.parse_acc(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["", "abc", "1x", "-1", "1kk", "m1"])
    def test_invalid_strings_raise(self, value):
        with pytest.raises(hmap.MapConfigError):
            hmap.parse_acc(value)

    def test_zero_is_rejected(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.parse_acc("0")

    def test_set_acc_persists_valid_value(self):
        hmap.set_acc("10m")
        assert hmap.get_map_config()["acc"] == "10m"

    def test_set_acc_rejects_invalid_without_persisting(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_acc("bogus")
        assert hmap.get_map_config()["acc"] == hmap.ACC_AUTO  # unchanged


class TestSetUseGpu:
    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on", True])
    def test_truthy(self, value):
        assert hmap.set_use_gpu(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", False])
    def test_falsy(self, value):
        assert hmap.set_use_gpu(value) is False

    def test_invalid_raises(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_use_gpu("maybe")


class TestSetTps:
    @pytest.mark.parametrize("value", [1, 8, 30, "15"])
    def test_valid_range(self, value):
        assert hmap.set_tps(value) == int(value)

    @pytest.mark.parametrize("value", [0, 31, -5, 100])
    def test_out_of_range_raises(self, value):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_tps(value)

    def test_non_integer_raises(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_tps("fast")


class TestSetFileMode:
    def test_mode_1_requires_path(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode(1, None)

    def test_mode_1_with_f_flag_succeeds(self):
        result = hmap.set_file_mode(1, "/tmp/model.safetensors", flag="-f")
        assert result == {"file_mode": 1, "file_path": "/tmp/model.safetensors"}
        assert hmap.get_map_config()["file_mode"] == 1

    def test_mode_1_rejects_capital_f_flag(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode(1, "/tmp/model.safetensors", flag="-F")

    def test_mode_3_requires_path(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode(3, None)

    def test_mode_3_with_capital_f_flag_succeeds(self):
        result = hmap.set_file_mode(3, "/tmp/mydir", flag="-F")
        assert result == {"file_mode": 3, "file_path": "/tmp/mydir"}

    def test_mode_3_rejects_lowercase_f_flag(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode(3, "/tmp/mydir", flag="-f")

    def test_mode_2_rejects_a_path(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode(2, "/tmp/mydir")

    def test_mode_2_with_no_path_succeeds(self):
        result = hmap.set_file_mode(2, None)
        assert result == {"file_mode": 2, "file_path": None}

    def test_invalid_mode_number_raises(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode(4)

    def test_non_integer_mode_raises(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode("model")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TestBucketForTensor:
    @pytest.mark.parametrize(
        "name,expected_label",
        [
            ("model.layers.0.self_attn.q_proj.weight", "L0"),
            ("model.layers.12.mlp.gate_proj.weight", "L12"),
            ("transformer.h.3.attn.c_attn.weight", "L3"),
            ("model.blocks.5.mlp.weight", "L5"),
            ("model.embed_tokens.weight", "Embed"),
            ("lm_head.weight", "LMHead"),
            ("model.norm.weight", "Norm"),
            ("something_unrecognised", "Other"),
        ],
    )
    def test_bucket_labels(self, name, expected_label):
        _, label = hmap._bucket_for_tensor(name)
        assert label == expected_label

    def test_layer_indices_sort_numerically_not_lexically(self):
        keys = [hmap._bucket_for_tensor(f"model.layers.{i}.weight")[0] for i in (2, 10, 1)]
        assert sorted(keys) == [
            hmap._bucket_for_tensor("model.layers.1.weight")[0],
            hmap._bucket_for_tensor("model.layers.2.weight")[0],
            hmap._bucket_for_tensor("model.layers.10.weight")[0],
        ]


class TestSnapshotFromTensorIter:
    def test_sums_param_counts_and_tensor_counts_per_bucket(self):
        shapes = {
            "model.layers.0.self_attn.q_proj.weight": ([4, 4], "F32"),
            "model.layers.0.mlp.gate_proj.weight": ([8, 4], "F32"),
            "model.layers.1.self_attn.q_proj.weight": ([4, 4], "F32"),
            "model.embed_tokens.weight": ([10, 4], "F32"),
        }
        snap = hmap._snapshot_from_tensor_iter("test", shapes)
        by_label = {n.label: n for n in snap.layers}
        assert by_label["L0"].param_count == 16 + 32
        assert by_label["L0"].tensor_count == 2
        assert by_label["L1"].param_count == 16
        assert by_label["Embed"].param_count == 40
        assert snap.total_params == 16 + 32 + 16 + 40
        assert snap.ok

    def test_layers_are_returned_sorted(self):
        shapes = {
            "model.layers.5.weight": ([2], "F32"),
            "model.layers.1.weight": ([2], "F32"),
            "model.embed_tokens.weight": ([2], "F32"),
            "lm_head.weight": ([2], "F32"),
        }
        snap = hmap._snapshot_from_tensor_iter("test", shapes)
        labels = [n.label for n in snap.layers]
        # Natural front-to-back model flow: embeddings, then layers in
        # numeric order, then the head -- not a plain lexical sort of
        # "layer:" vs "head" style keys, which would misorder these.
        assert labels == ["Embed", "L1", "L5", "LMHead"]


@pytest.fixture
def synthetic_safetensors_file(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    state = {
        "model.embed_tokens.weight": torch.zeros(100, 16),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 16),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(32, 16),
        "model.layers.1.self_attn.q_proj.weight": torch.zeros(16, 16),
        "model.norm.weight": torch.zeros(16),
        "lm_head.weight": torch.zeros(100, 16),
    }
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    return path


class TestSnapshotFromSafetensorsFile:
    def test_reads_exact_shapes_without_loading_tensor_data(self, synthetic_safetensors_file):
        snap = hmap._snapshot_from_safetensors_file(synthetic_safetensors_file)
        assert snap.ok
        assert snap.total_params == 100 * 16 + 16 * 16 + 32 * 16 + 16 * 16 + 16 + 100 * 16
        assert snap.dtype == "F32"
        labels = {n.label for n in snap.layers}
        assert labels == {"Embed", "L0", "L1", "Norm", "LMHead"}

    def test_missing_file_reports_error_not_exception(self, tmp_path):
        snap = hmap._snapshot_from_safetensors_file(tmp_path / "nope.safetensors")
        assert not snap.ok
        assert "not found" in snap.error


class TestSnapshotFromFolder:
    def test_single_safetensors_file_in_folder(self, tmp_path, synthetic_safetensors_file):
        snap = hmap._snapshot_from_folder(tmp_path)
        assert snap.ok
        assert snap.total_params > 0

    def test_sharded_safetensors_files(self, tmp_path):
        torch = pytest.importorskip("torch")
        from safetensors.torch import save_file

        save_file({"model.layers.0.weight": torch.zeros(4, 4)}, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file({"model.layers.1.weight": torch.zeros(4, 4)}, str(tmp_path / "model-00002-of-00002.safetensors"))
        snap = hmap._snapshot_from_folder(tmp_path)
        assert snap.ok
        assert snap.total_params == 32
        assert len(snap.layers) == 2

    def test_config_json_analytical_fallback_when_no_weights(self, tmp_path):
        cfg = {
            "vocab_size": 1000, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 3, "tie_word_embeddings": False,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        snap = hmap._snapshot_from_folder(tmp_path)
        assert snap.ok
        assert snap.dtype == "estimated"
        # embed + 3 layers + norm + head (untied embeddings => head is kept)
        assert len(snap.layers) == 6
        assert [n.label for n in snap.layers] == ["Embed", "L0", "L1", "L2", "Norm", "LMHead"]

    def test_config_json_with_tied_embeddings_omits_lm_head(self, tmp_path):
        cfg = {
            "vocab_size": 1000, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "tie_word_embeddings": True,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        snap = hmap._snapshot_from_folder(tmp_path)
        assert "LMHead" not in {n.label for n in snap.layers}

    def test_missing_config_fields_reports_error(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 64}), encoding="utf-8")
        snap = hmap._snapshot_from_folder(tmp_path)
        assert not snap.ok

    def test_no_config_and_no_weights_reports_error(self, tmp_path):
        snap = hmap._snapshot_from_folder(tmp_path)
        assert not snap.ok

    def test_nonexistent_folder_reports_error(self, tmp_path):
        snap = hmap._snapshot_from_folder(tmp_path / "does-not-exist")
        assert not snap.ok


class TestBuildSnapshot:
    def test_mode_1_without_path_errors_gracefully(self):
        snap = hmap.build_snapshot({"file_mode": 1, "file_path": None})
        assert not snap.ok

    def test_mode_3_without_path_errors_gracefully(self):
        snap = hmap.build_snapshot({"file_mode": 3, "file_path": None})
        assert not snap.ok

    def test_mode_2_has_no_architecture_but_no_error(self):
        snap = hmap.build_snapshot({"file_mode": 2, "file_path": None})
        assert snap.ok
        assert snap.layers == []

    def test_mode_1_dispatches_to_safetensors_scanner(self, synthetic_safetensors_file):
        snap = hmap.build_snapshot({"file_mode": 1, "file_path": str(synthetic_safetensors_file)})
        assert snap.ok
        assert snap.total_params > 0


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------

class TestGlyphsForPoly:
    @pytest.mark.parametrize("poly", [16, 32, 64, 128])
    def test_every_poly_level_has_glyphs(self, poly):
        g = hmap.glyphs_for_poly(poly)
        assert isinstance(g, hmap.PolyGlyphs)
        assert g.track_width > 0
        assert len(g.steam_frames) >= 2
        assert len(g.engine) >= 1

    def test_higher_poly_gives_more_steam_frames_and_wider_track(self):
        levels = [hmap.glyphs_for_poly(p) for p in hmap.POLY_LEVELS]
        widths = [g.track_width for g in levels]
        frames = [len(g.steam_frames) for g in levels]
        assert widths == sorted(widths)
        assert frames == sorted(frames)

    def test_unknown_poly_falls_back_to_32(self):
        assert hmap.glyphs_for_poly(999) is hmap.glyphs_for_poly(32)


class TestDialFrac:
    def test_half_full(self):
        assert hmap.dial_frac(500, 1000) == pytest.approx(0.5)

    def test_clamped_to_one_when_exceeding_acc(self):
        assert hmap.dial_frac(5000, 1000) == 1.0

    def test_zero_when_empty(self):
        assert hmap.dial_frac(0, 1000) == 0.0

    def test_zero_acc_does_not_divide_by_zero(self):
        assert hmap.dial_frac(500, 0) == 0.0


class TestRenderDial:
    def test_returns_two_lines(self):
        g = hmap.glyphs_for_poly(32)
        lines = hmap.render_dial("L0", 0.5, g)
        assert len(lines) == 2

    def test_needle_at_zero_sits_at_left_rim(self):
        g = hmap.glyphs_for_poly(32)
        _, track = hmap.render_dial("X", 0.0, g)
        inner = track[1:-1]  # strip rim_l/rim_r
        assert inner[0] == "\u25cf"
        assert set(inner[1:]) == {g.pipe_h}

    def test_needle_at_one_sits_at_right_rim(self):
        g = hmap.glyphs_for_poly(32)
        _, track = hmap.render_dial("X", 1.0, g)
        inner = track[1:-1]
        assert inner[-1] == "\u25cf"

    def test_track_lines_are_equal_length(self):
        g = hmap.glyphs_for_poly(64)
        label, track = hmap.render_dial("VeryLongLabelName", 0.3, g)
        assert len(label) == len(track)


class TestThrottleFrac:
    def test_idle_is_always_zero(self):
        for tick in range(10):
            assert hmap.throttle_frac(tick, False, 32) == 0.0

    def test_active_sweeps_up_and_back_down(self):
        values = [hmap.throttle_frac(t, True, 8) for t in range(9)]
        assert values[0] == 0.0
        assert values[4] == 1.0  # peak at half the period
        assert values[8] == 0.0  # back to zero after a full period
        assert values == [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]

    def test_bounded_between_zero_and_one(self):
        for tick in range(50):
            v = hmap.throttle_frac(tick, True, 32)
            assert 0.0 <= v <= 1.0


class TestRenderPipeSegment:
    def test_inactive_is_plain_pipe_chars(self):
        g = hmap.glyphs_for_poly(32)
        seg = hmap.render_pipe_segment(g, 10, 3, active=False)
        assert seg == g.pipe_h * 10

    def test_active_embeds_a_steam_frame(self):
        g = hmap.glyphs_for_poly(32)
        seg = hmap.render_pipe_segment(g, 10, 3, active=True)
        assert len(seg) == 10
        assert seg != g.pipe_h * 10
        assert any(ch in g.steam_frames for ch in seg)

    def test_zero_length_is_empty_string(self):
        g = hmap.glyphs_for_poly(32)
        assert hmap.render_pipe_segment(g, 0, 0, active=True) == ""


class TestRenderEngine:
    def test_row_count_matches_engine_art_plus_label_and_puff(self):
        g = hmap.glyphs_for_poly(32)
        rows = hmap.render_engine(g, "PROMPT", 1, active=True)
        assert len(rows) == len(g.engine) + 2

    def test_idle_shows_no_puff(self):
        g = hmap.glyphs_for_poly(32)
        rows = hmap.render_engine(g, "DATA", 1, active=False)
        assert rows[0].strip() == ""


class TestCanvas:
    def test_stamp_and_render(self):
        c = hmap.Canvas(10, 3)
        c.stamp(2, 1, "hi")
        rendered = c.render()
        lines = rendered.splitlines()
        assert lines[1][2:4] == "hi"

    def test_stamp_out_of_bounds_does_not_raise(self):
        c = hmap.Canvas(5, 5)
        c.stamp(-3, -3, "oops")
        c.stamp(100, 100, "oops")
        c.stamp(3, 100, "oops")
        assert len(c.render().splitlines()) == 5

    def test_stamp_lines_places_each_row(self):
        c = hmap.Canvas(10, 4)
        c.stamp_lines(0, 1, ["ab", "cd"])
        lines = c.render().splitlines()
        assert lines[1].startswith("ab")
        assert lines[2].startswith("cd")

    def test_every_row_is_exactly_width_chars(self):
        c = hmap.Canvas(15, 4)
        c.stamp(0, 0, "hello")
        for line in c.render().splitlines():
            assert len(line) == 15


class TestLayoutPositions:
    def test_single_row_left_to_right(self):
        assert hmap._layout_positions(3, 5) == [(0, 0), (1, 0), (2, 0)]

    def test_wraps_boustrophedon_snake(self):
        # 4 items, 2 per row -> row0 L-to-R, row1 R-to-L
        assert hmap._layout_positions(4, 2) == [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_empty_input(self):
        assert hmap._layout_positions(0, 5) == []

    def test_every_position_is_unique(self):
        positions = hmap._layout_positions(23, 4)
        assert len(set(positions)) == len(positions)


class TestFmtCount:
    @pytest.mark.parametrize(
        "n,expected_suffix",
        [(500, ""), (1_500, "K"), (2_500_000, "M"), (3_000_000_000, "B"), (4_000_000_000_000, "T")],
    )
    def test_thresholds_pick_the_right_suffix(self, n, expected_suffix):
        result = hmap._fmt_count(n)
        if expected_suffix:
            assert result.endswith(expected_suffix)
        else:
            assert result == str(n)


# ---------------------------------------------------------------------------
# Mouse hover
# ---------------------------------------------------------------------------

class TestParseSgrMouse:
    def test_parses_a_simple_report(self):
        assert hmap._parse_sgr_mouse("\x1b[<0;45;12M") == (45, 12)

    def test_takes_the_last_report_when_several_are_buffered(self):
        assert hmap._parse_sgr_mouse("\x1b[<0;1;1M\x1b[<0;99;29M") == (99, 29)

    def test_release_event_lowercase_m_also_parses(self):
        assert hmap._parse_sgr_mouse("\x1b[<0;10;20m") == (10, 20)

    def test_no_sequence_returns_none(self):
        assert hmap._parse_sgr_mouse("just some text") is None

    def test_empty_string_returns_none(self):
        assert hmap._parse_sgr_mouse("") is None

    def test_partial_sequence_returns_none(self):
        assert hmap._parse_sgr_mouse("\x1b[<0;45;") is None


class TestIsBottomRight:
    def test_inside_the_corner_margin(self):
        assert hmap._is_bottom_right((95, 28), 100, 30, margin=12) is True

    def test_outside_the_corner_margin(self):
        assert hmap._is_bottom_right((5, 28), 100, 30, margin=12) is False
        assert hmap._is_bottom_right((95, 5), 100, 30, margin=12) is False

    def test_top_left_corner_is_never_bottom_right(self):
        assert hmap._is_bottom_right((0, 0), 100, 30) is False


class TestMouseReader:
    def test_unavailable_on_a_non_tty(self):
        reader = hmap._MouseReader()
        # The test runner's stdin is never a live interactive TTY.
        assert reader.available is False

    def test_start_and_stop_are_safe_no_ops_when_unavailable(self):
        reader = hmap._MouseReader()
        reader.start()
        reader.stop()  # must not raise even though start() did nothing


# ---------------------------------------------------------------------------
# HyperMap end-to-end rendering
# ---------------------------------------------------------------------------

class TestHyperMapRendering:
    def _app(self, **cfg_overrides):
        cfg = dict(hmap._MAP_DEFAULTS)
        cfg.update(cfg_overrides)
        return hmap.HyperMap(cfg, width=90, height=30)

    def test_falls_back_to_a_train_placeholder_when_no_architecture(self):
        app = self._app(file_mode=2)
        nodes = app._nodes()
        assert len(nodes) == 1
        assert nodes[0].label == "TRAIN"

    def test_invalid_persisted_acc_falls_back_gracefully(self):
        app = self._app(acc="not-a-number")
        assert app.acc_value == hmap.parse_acc("1m")

    def test_invalid_persisted_poly_falls_back_to_32(self):
        app = self._app(poly=999)
        assert app.poly == 32

    @pytest.mark.parametrize("poly", [16, 32, 64, 128])
    def test_every_row_is_the_same_width_at_every_poly_level(self, poly):
        app = self._app(poly=poly, file_mode=2)
        app.snapshot = hmap.ModelSnapshot(
            source="synthetic",
            layers=[hmap.LayerNode(key=f"l{i}", label=f"L{i}", param_count=i * 1000) for i in range(9)],
            total_params=sum(i * 1000 for i in range(9)),
        )
        frame = app.render()
        widths = {len(line) for line in frame.splitlines()}
        assert widths == {90}

    @pytest.mark.parametrize("n_layers", [0, 1, 6, 25])
    def test_renders_without_crashing_for_various_layer_counts(self, n_layers):
        app = self._app(file_mode=2)
        app.snapshot = hmap.ModelSnapshot(
            source="synthetic",
            layers=[hmap.LayerNode(key=f"l{i}", label=f"L{i}", param_count=i + 1) for i in range(n_layers)],
            total_params=sum(range(1, n_layers + 1)),
        )
        frame = app.render()
        assert frame  # non-empty, no exception

    def test_narrow_terminal_does_not_crash(self):
        app = hmap.HyperMap(dict(hmap._MAP_DEFAULTS), width=25, height=15)
        app.render()

    def test_legend_overlay_every_row_still_matches_canvas_width(self):
        app = self._app(file_mode=2)
        app.legend_visible = True
        frame = app.render()
        widths = {len(line) for line in frame.splitlines()}
        assert widths == {90}

    def test_legend_box_lines_are_all_the_same_length(self):
        canvas = hmap.Canvas(90, 30)
        hmap._stamp_legend(canvas, hmap.glyphs_for_poly(32))
        rendered_lines = canvas.render().splitlines()
        # every line of the canvas is still uniform width (composited, not glued)
        assert {len(line) for line in rendered_lines} == {90}

    def test_is_active_reflects_log_training_data(self):
        app = self._app(file_mode=2)
        assert app.is_active is False
        app.log.has_training_data = True
        assert app.is_active is True

    def test_gpu_reading_is_none_when_use_gpu_false(self):
        app = self._app(use_gpu=False)
        assert app.gpu_reading() is None

    def test_gpu_reading_calls_nvidia_probe_when_use_gpu_true(self, monkeypatch):
        app = self._app(use_gpu=True)
        called = {}

        def _fake_probe():
            called["yes"] = True
            return {"util_percent": 42.0, "temp_c": 60.0}

        import hypernix.tv as tv
        monkeypatch.setattr(tv, "_query_nvidia_smi_full", _fake_probe)
        reading = app.gpu_reading()
        assert called.get("yes") is True
        assert reading["util_percent"] == 42.0

    def test_tick_advances_and_steam_position_changes(self):
        app = self._app(file_mode=2)
        app.log.has_training_data = True
        frame1 = app.render()
        app.tick = 1
        frame2 = app.render()
        # Different ticks should (almost always) animate the pipe/steam
        # differently somewhere in the frame.
        assert frame1 != frame2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCliMain:
    def test_help_returns_zero(self, capsys):
        assert hmap.cli_main(["--help"]) == 0
        assert "hnx map" in capsys.readouterr().out

    def test_unknown_argument_returns_nonzero(self, capsys):
        assert hmap.cli_main(["bogus"]) != 0
        assert "Unknown argument" in capsys.readouterr().err

    def test_config_poly_round_trip(self):
        assert hmap.cli_main(["config", "poly", "64"]) == 0
        assert hmap.get_map_config()["poly"] == 64

    def test_config_poly_invalid_returns_nonzero(self, capsys):
        assert hmap.cli_main(["config", "poly", "7"]) != 0
        assert "poly" in capsys.readouterr().err

    def test_config_acc_round_trip(self):
        assert hmap.cli_main(["config", "acc", "5m"]) == 0
        assert hmap.get_map_config()["acc"] == "5m"

    def test_config_main_use_gpu_round_trip(self):
        assert hmap.cli_main(["config", "main", "use-gpu", "false"]) == 0
        assert hmap.get_map_config()["use_gpu"] is False

    def test_config_main_tps_round_trip(self):
        assert hmap.cli_main(["config", "main", "tps", "20"]) == 0
        assert hmap.get_map_config()["tps"] == 20

    def test_config_main_tps_out_of_range_returns_nonzero(self, capsys):
        assert hmap.cli_main(["config", "main", "tps", "999"]) != 0

    def test_config_main_file_model_1_requires_f_flag(self, capsys):
        assert hmap.cli_main(["config", "main", "file", "model", "1"]) != 0
        assert "-f" in capsys.readouterr().err

    def test_config_main_file_model_1_with_f_flag_succeeds(self, tmp_path):
        target = tmp_path / "m.safetensors"
        assert hmap.cli_main(["config", "main", "file", "model", "1", "-f", str(target)]) == 0
        cfg = hmap.get_map_config()
        assert cfg["file_mode"] == 1
        assert cfg["file_path"] == str(target)

    def test_config_main_file_model_3_with_capital_f_flag_succeeds(self, tmp_path):
        assert hmap.cli_main(["config", "main", "file", "model", "3", "-F", str(tmp_path)]) == 0
        assert hmap.get_map_config()["file_mode"] == 3

    def test_config_main_file_model_2_rejects_path(self, capsys):
        assert hmap.cli_main(["config", "main", "file", "model", "2", "-f", "/tmp/x"]) != 0

    def test_config_main_file_missing_model_keyword(self, capsys):
        assert hmap.cli_main(["config", "main", "file", "1"]) != 0

    def test_config_unknown_key_returns_nonzero(self, capsys):
        assert hmap.cli_main(["config", "bogus"]) != 0
        assert "unknown config key" in capsys.readouterr().err

    def test_config_main_unknown_subkey_returns_nonzero(self, capsys):
        assert hmap.cli_main(["config", "main", "bogus"]) != 0

    def test_config_with_no_args_prints_usage(self, capsys):
        assert hmap.cli_main(["config"]) == 0
        assert "hnx map" in capsys.readouterr().out


# ===========================================================================
# Auto-discovery, auto acc, and the failure banner
# ===========================================================================


class TestDiscoverModel:
    """`hnx map` with no configuration has to find something to draw.

    The map's rendering and safetensors parsing were always fine; what
    made it look broken was that the default mode (train.log) carries no
    architecture at all, so the very first run drew one placeholder dial
    and "Σ params ?" with nothing on screen saying why.
    """

    def test_finds_a_weight_file_in_the_working_directory(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert hmap.discover_model(tmp_path) == tmp_path / "model.safetensors"

    def test_finds_a_weight_file_in_checkpoints(self, tmp_path):
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        (ckpt / "step-100.safetensors").write_bytes(b"")
        assert hmap.discover_model(tmp_path) == ckpt / "step-100.safetensors"

    def test_returns_the_directory_when_a_model_is_sharded(self, tmp_path):
        """Several shards are one model, not several — handing back the
        first shard would draw a fraction of the network."""
        for i in range(3):
            (tmp_path / f"model-0000{i}-of-00003.safetensors").write_bytes(b"")
        assert hmap.discover_model(tmp_path) == tmp_path

    def test_prefers_the_working_directory_over_a_subfolder(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        (ckpt / "other.safetensors").write_bytes(b"")
        assert hmap.discover_model(tmp_path) == tmp_path / "model.safetensors"

    def test_prefers_safetensors_over_pickle_formats(self, tmp_path):
        """Shapes can be read from safetensors without torch.load, so
        there is no reason to execute a pickle to count parameters."""
        (tmp_path / "pytorch_model.bin").write_bytes(b"")
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert hmap.discover_model(tmp_path) == tmp_path / "model.safetensors"

    def test_returns_none_when_there_is_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(tmp_path / "no-such-dir"))
        assert hmap.discover_model(tmp_path) is None

    def test_an_unreadable_directory_does_not_raise(self, tmp_path, monkeypatch):
        """Discovery is best-effort: a permissions error somewhere in the
        search path must not stop the TUI from starting."""
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(tmp_path / "nope"))
        monkeypatch.setattr(
            hmap.Path, "glob",
            lambda self, pattern: (_ for _ in ()).throw(OSError("permission denied")),
        )
        assert hmap.discover_model(tmp_path) is None


class TestAutoAcc:
    def test_scales_to_the_model(self):
        """A fixed acc pegs every dial on anything but a toy model, which
        makes the schematic a row of identical full needles."""
        acc = hmap.auto_acc(total_params=40_000_000, layer_count=6)
        assert acc > 1e6
        # The mean node reads somewhere in the readable middle, not pinned.
        assert 0.1 < (40_000_000 / 6) / acc < 1.0

    def test_falls_back_for_an_empty_model(self):
        assert hmap.auto_acc(0, 0) == 1e6
        assert hmap.auto_acc(100, 0) == 1e6

    def test_parse_acc_rejects_auto_rather_than_silently_defaulting(self):
        """parse_acc has no view of the model, so it cannot resolve
        "auto" — it says so instead of quietly returning 1M."""
        with pytest.raises(hmap.MapConfigError):
            hmap.parse_acc(hmap.ACC_AUTO)

    def test_set_acc_accepts_auto(self):
        assert hmap.set_acc("auto") == hmap.ACC_AUTO
        assert hmap.get_map_config()["acc"] == hmap.ACC_AUTO


class TestFileModeZero:
    def test_mode_zero_takes_no_path(self):
        with pytest.raises(hmap.MapConfigError):
            hmap.set_file_mode(0, "/some/path", flag="-f")

    def test_mode_zero_is_settable(self):
        assert hmap.set_file_mode(0)["file_mode"] == 0

    def test_build_snapshot_reports_when_nothing_is_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(tmp_path / "nope"))
        snap = hmap.build_snapshot({"file_mode": 0})
        assert snap.error is not None
        assert "no model found" in snap.error
        # ...and it says what to do about it.
        assert "-f" in snap.error


class TestErrorBanner:
    def test_a_failed_read_is_shown_on_the_canvas(self, tmp_path, monkeypatch):
        """An empty schematic and "Σ params ?" looks identical to a model
        with no layers. The reason belongs on screen."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(tmp_path / "nope"))
        app = hmap.HyperMap(
            {"file_mode": 0, "poly": 32, "acc": "auto", "use_gpu": False, "tps": 8},
            width=100, height=24,
        )
        frame = app.render()
        assert "no model found" in frame

    def test_the_banner_does_not_overwrite_the_schematic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(tmp_path / "nope"))
        app = hmap.HyperMap(
            {"file_mode": 0, "poly": 32, "acc": "auto", "use_gpu": False, "tps": 8},
            width=100, height=24,
        )
        lines = app.render().splitlines()
        # "! " is the banner's own marker. Matching on "no model found"
        # alone also hits the title bar, which shows it as the source.
        banner_rows = [i for i, ln in enumerate(lines) if "! no model found" in ln]
        engine_rows = [i for i, ln in enumerate(lines) if "PROMPT" in ln or "DATA" in ln]
        assert banner_rows and engine_rows
        assert min(banner_rows) > max(engine_rows), "banner drew through the engines"


class TestModuleEntryPoint:
    def test_module_has_a_main_guard(self):
        """`python -m hypernix.monitoring.map` used to import the module,
        run nothing and exit 0 — indistinguishable from a TUI that
        started and instantly quit."""
        source = Path(hmap.__file__).read_text(encoding="utf-8")
        assert '__name__ == "__main__"' in source
