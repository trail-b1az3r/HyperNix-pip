"""Tests for HyperNixQuantizer and quantize v0.70.3b2 facade."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import hypernix
from hypernix.quantize import HyperNixQuantizer, QuantJob, recommend_profile


def test_recommend_profile_chat() -> None:
    specs = recommend_profile("chat")
    names = {s.name for s in specs}
    assert "Q4_K_M" in names
    assert "Q6_K" in names


def test_recommend_profile_unknown_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        recommend_profile("not-a-profile")


def test_plan_batch_paths(tmp_path: Path) -> None:
    q = HyperNixQuantizer()
    jobs = q.plan_batch(tmp_path / "src.gguf", tmp_path / "out", ["q4_k_m", "q6_k"])
    assert len(jobs) == 2
    assert all(isinstance(j, QuantJob) for j in jobs)
    assert jobs[0].spec.name in {"Q4_K_M", "Q6_K"}


def test_format_catalog_nonempty() -> None:
    text = HyperNixQuantizer().format_catalog(category="k")
    assert "Q4_K_M" in text
    assert "bpw" in text


def test_batch_quantize_top_level_export() -> None:
    assert hasattr(hypernix, "HyperNixQuantizer")
    assert hasattr(hypernix, "quant_recommend_profile")
    assert hasattr(hypernix, "quant_batch")


def test_run_batch_calls_quantize(tmp_path: Path) -> None:
    src = tmp_path / "in.gguf"
    src.write_bytes(b"gguf")
    q = HyperNixQuantizer(auto_fetch=False)
    with mock.patch("hypernix.quantize.quantize_gguf", side_effect=lambda *a, **k: Path(a[1])) as m:
        out = q.run_batch(src, tmp_path / "q", ["q4_k_m"])
    assert len(out) == 1
    assert m.called
