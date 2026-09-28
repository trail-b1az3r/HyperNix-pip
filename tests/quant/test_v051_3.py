"""Tests for v0.51.3 — quantize catalog rewrite.

Covers the new ``QuantSpec`` dataclass + ``CATALOG`` registry,
the helper functions (``recommended``, ``by_category``,
``for_size``, ``estimate_size``, ``resolve_spec``,
``list_types``), the expanded ``QUANT_TYPES`` alias dict, and
the v0.51.3 top-level re-exports on ``hypernix``.

Backward-compatibility: the original 6 types (F32 / F16 / Q8_0 /
Q6_K / Q4_K_M / Q5_K_M) and their pre-existing aliases must
still resolve unchanged.
"""
from __future__ import annotations

import pytest

import hypernix
from hypernix import quantize
from hypernix.quantize import (
    CATALOG,
    QUANT_TYPES,
    QuantSpec,
    by_category,
    estimate_size,
    for_size,
    list_types,
    recommended,
    resolve_spec,
)

# ---------------------------------------------------------------------------
# CATALOG / QuantSpec
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_catalog_has_at_least_30_specs(self) -> None:
        assert len(CATALOG) >= 30

    def test_every_quant_type_alias_resolves_into_catalog(self) -> None:
        for alias, target in QUANT_TYPES.items():
            assert target in CATALOG, f"{alias} -> {target} missing from CATALOG"

    def test_every_spec_has_positive_bpw(self) -> None:
        for spec in CATALOG.values():
            assert spec.bits_per_weight > 0
            assert spec.size_factor == pytest.approx(spec.bits_per_weight / 16.0)

    def test_every_spec_has_known_category(self) -> None:
        for spec in CATALOG.values():
            assert spec.category in {"float", "legacy", "k", "iq"}

    def test_every_spec_has_nonempty_notes(self) -> None:
        for spec in CATALOG.values():
            assert spec.notes.strip()

    def test_quant_spec_is_frozen_dataclass(self) -> None:
        from dataclasses import FrozenInstanceError
        spec = CATALOG["Q4_K_M"]
        with pytest.raises(FrozenInstanceError):
            spec.name = "MUTATED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_recommended_short_list(self) -> None:
        names = {s.name for s in recommended()}
        # The curated short-list must include the headline types.
        assert {"F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"}.issubset(names)
        # And must NOT include extreme IQ-quants by default.
        assert "IQ1_S" not in names

    def test_by_category_float(self) -> None:
        names = [s.name for s in by_category("float")]
        assert "F16" in names and "F32" in names and "BF16" in names
        # Sorted ascending by bpw.
        bpws = [s.bits_per_weight for s in by_category("float")]
        assert bpws == sorted(bpws)

    def test_by_category_iq_includes_iq4_xs(self) -> None:
        names = {s.name for s in by_category("iq")}
        assert "IQ4_XS" in names and "IQ2_M" in names

    def test_by_category_unknown_returns_empty(self) -> None:
        assert by_category("does-not-exist") == []

    def test_for_size_picks_largest_fitting_quant(self) -> None:
        # 2 GB fp16 model, 1 GB target — should land on a k-quant ≤ 8 bpw.
        spec = for_size(1_000_000_000, 2_000_000_000)
        assert spec.category in {"legacy", "k", "iq"}
        # At 1 GB target / 2 GB fp16 → ratio = 0.5 → bpw ≤ 8.
        assert spec.bits_per_weight <= 8.0

    def test_for_size_falls_back_to_smallest_when_target_is_tiny(self) -> None:
        spec = for_size(1, 1_000_000_000)  # absurdly small target
        # Smallest bpw in the catalog wins.
        assert spec.bits_per_weight == min(s.bits_per_weight for s in CATALOG.values())

    def test_for_size_rejects_zero_fp16(self) -> None:
        with pytest.raises(ValueError):
            for_size(1_000, 0)

    def test_estimate_size_q4_k_m(self) -> None:
        # 1 GB fp16 ≈ 302 MB at Q4_K_M (4.83 bpw).
        out = estimate_size("q4km", 1_000_000_000)
        assert 250_000_000 <= out <= 350_000_000

    def test_estimate_size_f16_is_passthrough(self) -> None:
        assert estimate_size("F16", 100_000) == 100_000

    def test_list_types_is_sorted_and_complete(self) -> None:
        names = list_types()
        assert names == sorted(names)
        assert set(names) == set(CATALOG.keys())


# ---------------------------------------------------------------------------
# resolve_spec / aliases
# ---------------------------------------------------------------------------

class TestResolveSpec:
    def test_canonical_uppercase(self) -> None:
        assert resolve_spec("Q4_K_M").name == "Q4_K_M"
        assert resolve_spec("IQ4_XS").name == "IQ4_XS"

    def test_short_alias_q4km(self) -> None:
        assert resolve_spec("q4km").name == "Q4_K_M"

    def test_dash_alias(self) -> None:
        assert resolve_spec("q4-k-m").name == "Q4_K_M"

    def test_case_insensitive(self) -> None:
        assert resolve_spec("Q4_k_m").name == "Q4_K_M"
        assert resolve_spec("iq4_nl").name == "IQ4_NL"

    def test_unknown_alias_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            resolve_spec("definitely-not-a-quant")


# ---------------------------------------------------------------------------
# Backward compatibility — pre-0.51.3 callers must keep working
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    @pytest.mark.parametrize(
        "alias",
        ["fp32", "f32", "fp16", "f16", "q8", "q8_0",
         "q6", "q6_k", "q4km", "q4_k_m", "q5km", "q5_k_m"],
    )
    def test_pre_0_51_3_aliases_still_resolve(self, alias: str) -> None:
        # Every alias that existed before the rewrite must still map.
        target = QUANT_TYPES[alias]
        assert target in CATALOG

    def test_old_api_rejects_unknown_with_value_error(self, monkeypatch) -> None:
        # Stub out the binary finder so quantize_gguf only exercises the
        # validation path (no llama-quantize on this machine).
        monkeypatch.setattr(
            quantize, "_find_llama_quantize",
            lambda *a, **kw: "/bin/true",
        )
        with pytest.raises(ValueError):
            quantize.quantize_gguf("in.gguf", "out.gguf", "totally-unknown")


# ---------------------------------------------------------------------------
# Top-level hypernix exports
# ---------------------------------------------------------------------------

class TestTopLevelExports:
    def test_hypernix_reexports_quant_helpers(self) -> None:
        for name in (
            "QUANT_CATALOG", "QUANT_TYPES", "QuantSpec",
            "quant_recommended", "quant_by_category", "quant_for_size",
            "quant_estimate_size", "quant_list_types", "quant_resolve_spec",
            "quantize_gguf",
        ):
            assert hasattr(hypernix, name), f"missing top-level {name!r}"

    def test_quant_catalog_is_the_same_object(self) -> None:
        assert hypernix.QUANT_CATALOG is CATALOG

    def test_quant_recommended_returns_specs(self) -> None:
        out = hypernix.quant_recommended()
        assert all(isinstance(s, QuantSpec) for s in out)
        assert len(out) >= 5
