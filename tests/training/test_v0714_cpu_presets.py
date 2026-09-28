"""Tests for the CPU-oriented presets added to hypernix.brewer (small
architecture presets sized for CPU-only training) and hypernix.freezer
(AMD Ryzen hardware presets, closing the gap where the README claimed
Ryzen support that didn't actually exist in CPU_PRESETS).
"""
from __future__ import annotations

from hypernix.brewer import _PRESET_MAP, Brewer, BrewerModel
from hypernix.freezer import CPU_PRESETS, cpu_preset

# ---------------------------------------------------------------------------
# brewer.py CPU-size presets
# ---------------------------------------------------------------------------

def test_cpu_presets_registered_in_preset_map():
    for key in ("cpu-nano", "cpu-tiny", "cpu-small"):
        assert key in _PRESET_MAP


def test_cpu_presets_use_plain_mha_no_sliding_window():
    for key in ("cpu-nano", "cpu-tiny", "cpu-small"):
        cfg = _PRESET_MAP[key]()
        assert cfg.attention_type == "mha"
        assert cfg.use_sliding_window is False


def test_cpu_presets_are_smaller_than_smallest_gpu_preset():
    smallest_gpu = _PRESET_MAP["33m"]()
    smallest_gpu_params = BrewerModel(smallest_gpu).num_params()
    for key in ("cpu-nano", "cpu-tiny", "cpu-small"):
        cfg = _PRESET_MAP[key]()
        n = BrewerModel(cfg).num_params()
        assert n < smallest_gpu_params, f"{key} ({n:,}) should be smaller than 33m ({smallest_gpu_params:,})"


def test_cpu_presets_size_ordering():
    nano = BrewerModel(_PRESET_MAP["cpu-nano"]()).num_params()
    tiny = BrewerModel(_PRESET_MAP["cpu-tiny"]()).num_params()
    small = BrewerModel(_PRESET_MAP["cpu-small"]()).num_params()
    assert nano < tiny < small


def test_cpu_nano_exact_param_count():
    # Locks in the documented, measured count so a later unrelated config
    # tweak can't silently drift it without a test failing.
    cfg = _PRESET_MAP["cpu-nano"]()
    assert BrewerModel(cfg).num_params() == 2_073_728


def test_cpu_tiny_exact_param_count():
    cfg = _PRESET_MAP["cpu-tiny"]()
    assert BrewerModel(cfg).num_params() == 9_211_136


def test_cpu_small_exact_param_count():
    cfg = _PRESET_MAP["cpu-small"]()
    assert BrewerModel(cfg).num_params() == 26_450_304


def test_cpu_preset_forward_pass_produces_correct_shape():
    import torch
    cfg = _PRESET_MAP["cpu-nano"]()
    model = Brewer(cfg, name="fwd-test").build()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(x)
    assert tuple(out.shape) == (1, 16, cfg.vocab_size)


def test_cpu_presets_have_valid_head_dimensions():
    for key in ("cpu-nano", "cpu-tiny", "cpu-small"):
        cfg = _PRESET_MAP[key]()
        assert cfg.d_model % cfg.n_heads == 0, f"{key}: d_model not divisible by n_heads"


# ---------------------------------------------------------------------------
# freezer.py Ryzen CPU hardware presets
# ---------------------------------------------------------------------------

RYZEN_KEYS = [
    "ryzen-5-5600x", "ryzen-7-5800x", "ryzen-9-5900x", "ryzen-9-5950x",
    "ryzen-5-7600x", "ryzen-7-7700x", "ryzen-9-7900x", "ryzen-9-7950x",
    "ryzen-5-9600x", "ryzen-7-9700x", "ryzen-9-9900x", "ryzen-9-9950x",
]


def test_all_ryzen_presets_registered():
    for key in RYZEN_KEYS:
        assert key in CPU_PRESETS, f"{key} missing from CPU_PRESETS"


def test_ryzen_preset_count():
    ryzen_keys = [k for k in CPU_PRESETS if k.startswith("ryzen")]
    assert len(ryzen_keys) == 12


def test_zen3_ryzen_presets_are_avx2_only_no_avx512():
    for key in ("ryzen-5-5600x", "ryzen-7-5800x", "ryzen-9-5900x", "ryzen-9-5950x"):
        preset = CPU_PRESETS[key]
        assert "AVX-512" not in preset.avx_levels
        assert "AVX2" in preset.avx_levels


def test_zen4_and_zen5_ryzen_presets_support_avx512():
    zen4_zen5 = (
        "ryzen-5-7600x", "ryzen-7-7700x", "ryzen-9-7900x", "ryzen-9-7950x",
        "ryzen-5-9600x", "ryzen-7-9700x", "ryzen-9-9900x", "ryzen-9-9950x",
    )
    for key in zen4_zen5:
        preset = CPU_PRESETS[key]
        assert "AVX-512" in preset.avx_levels


def test_ryzen_recommended_threads_matches_physical_cores():
    for key in RYZEN_KEYS:
        preset = CPU_PRESETS[key]
        assert preset.recommended_threads == preset.cores


def test_ryzen_thread_count_is_double_cores_smt():
    for key in RYZEN_KEYS:
        preset = CPU_PRESETS[key]
        assert preset.threads == preset.cores * 2


def test_cpu_preset_lookup_is_case_and_separator_insensitive():
    assert cpu_preset("RYZEN_9_9950X") is not None
    assert cpu_preset("ryzen 9 9950x") is not None
    assert cpu_preset("ryzen-9-9950x").name == "AMD Ryzen 9 9950X"


def test_ryzen_generational_aliases_resolve():
    assert cpu_preset("ryzen-9000").name == "AMD Ryzen 9 9900X"
    assert cpu_preset("ryzen-7-7000").name == "AMD Ryzen 7 7700X"
    assert cpu_preset("ryzen-5000").name == "AMD Ryzen 9 5900X"


def test_ryzen_9950x_is_highest_scoring_cpu_preset():
    # Zen 5's full-width AVX-512 (no frequency throttle penalty, unlike
    # early Intel implementations) plus IPC gains puts it at the top of
    # the whole table, not just the Ryzen entries -- documents that this
    # was an intentional call, not an oversight, if it's ever challenged.
    top = max(CPU_PRESETS.values(), key=lambda p: p.gflops_per_thread)
    assert top.name == "AMD Ryzen 9 9950X"


def test_no_duplicate_cpu_preset_display_names():
    names = [p.name for p in CPU_PRESETS.values()]
    assert len(names) == len(set(names))
