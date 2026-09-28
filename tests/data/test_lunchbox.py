"""Tests for hypernix.lunchbox — consistent-schema dataset packager."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_lunchbox_collects_heterogeneous_records() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(id="r1", prompt="Q1", score=0.5)
    box.add(id="r2", prompt="Q2", score=0.9, latency_s=0.1)

    # Columns is the union.
    assert set(box.columns()) == {"id", "prompt", "score", "latency_s"}
    assert len(box) == 2


def test_normalize_fills_missing_with_none() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(a=1, b=2)
    box.add(a=3)           # missing b

    rows = box.normalize()
    assert rows[0] == {"a": 1, "b": 2}
    assert rows[1] == {"a": 3, "b": None}


def test_required_columns_reject_unknown_keys() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox(required_columns=("id", "prompt"))
    box.add(id="r1", prompt="Q")
    with pytest.raises(ValueError, match="unknown columns"):
        box.add(id="r2", unexpected="oops")


def test_required_columns_fixes_output_order() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox(required_columns=("b", "a", "c"))
    box.add(a=1, b=2, c=3)
    # columns() preserves required order.
    assert box.columns() == ["b", "a", "c"]


def test_validate_catches_mixed_types() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(score=1.0)
    box.add(score="oops")
    with pytest.raises(ValueError, match="mixed types"):
        box.validate()


def test_validate_tolerates_int_float_mix() -> None:
    """int ↔ float co-occur in the same column a lot in practice; Arrow
    promotes to float, so Lunchbox accepts the pair."""
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(score=1)
    box.add(score=2.5)
    box.validate()


def test_validate_ignores_none() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(score=1.0)
    box.add(score=None)
    box.validate()


def test_for_eval_has_recommended_schema() -> None:
    from hypernix.lunchbox import EVAL_SCHEMA, Lunchbox

    box = Lunchbox.for_eval()
    assert box.required_columns == EVAL_SCHEMA
    # "latency_s" is one of the recommended columns — it should not be
    # rejected as unknown.
    box.add(id="r", prompt="Q", keyword_score=1.0, latency_s=0.2)


def test_pack_jsonl_has_every_column_per_row(tmp_path: Path) -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(id="r1", prompt="Q1")
    box.add(id="r2", prompt="Q2", latency_s=0.3)

    out = box.pack_jsonl(tmp_path / "data.jsonl")
    lines = [json.loads(line) for line in out.read_text().splitlines()]

    # Every row carries every column — the whole point of the packer.
    for row in lines:
        assert set(row.keys()) == {"id", "prompt", "latency_s"}
    # Missing cells emitted as null.
    assert lines[0]["latency_s"] is None


def test_pack_invokes_datasets_from_list(tmp_path: Path) -> None:
    """pack() must route through datasets.Dataset.from_list so the
    huggingface metadata key inside the Parquet header matches the
    column set — that's the fix for the CastError seen in the Hub
    dataset viewer."""
    import hypernix.lunchbox as lb

    fake_ds = MagicMock()
    fake_ds_cls = MagicMock()
    fake_ds_cls.from_list.return_value = fake_ds
    fake_module = MagicMock(Dataset=fake_ds_cls)

    box = lb.Lunchbox()
    box.add(id="r1", prompt="Q1")
    box.add(id="r2", prompt="Q2", latency_s=0.3)

    with patch.dict("sys.modules", {"datasets": fake_module}):
        out = box.pack(tmp_path / "data.parquet")

    # Normalised rows (filled with None) were passed to from_list.
    (args, kwargs) = fake_ds_cls.from_list.call_args
    rows = args[0]
    assert rows == [
        {"id": "r1", "prompt": "Q1", "latency_s": None},
        {"id": "r2", "prompt": "Q2", "latency_s": 0.3},
    ]
    fake_ds.to_parquet.assert_called_once_with(str(out))


def test_push_to_hub_uses_datasets_push(tmp_path: Path) -> None:
    import hypernix.lunchbox as lb

    fake_ds = MagicMock()
    fake_ds_cls = MagicMock()
    fake_ds_cls.from_list.return_value = fake_ds
    fake_module = MagicMock(Dataset=fake_ds_cls)

    box = lb.Lunchbox.for_eval()
    box.add(id="r1", prompt="Q1", keyword_score=1.0)

    with patch.dict("sys.modules", {"datasets": fake_module}):
        url = box.push_to_hub(
            "ray0rf1re/eval", token="hf_xxx",
            private=False, split="train",
        )

    fake_ds.push_to_hub.assert_called_once()
    call_args = fake_ds.push_to_hub.call_args
    assert call_args.args[0] == "ray0rf1re/eval"
    assert call_args.kwargs["token"] == "hf_xxx"
    assert call_args.kwargs["split"] == "train"
    assert url == "https://huggingface.co/datasets/ray0rf1re/eval"


def test_lunchbox_function_preloads() -> None:
    from hypernix.lunchbox import lunchbox

    box = lunchbox([
        {"id": "r1", "prompt": "Q"},
        {"id": "r2", "prompt": "Q"},
    ])
    assert len(box) == 2


def test_from_records_classmethod() -> None:
    from hypernix.lunchbox import EVAL_SCHEMA, Lunchbox

    records = [
        {"id": "r1", "prompt": "Q", "keyword_score": 1.0},
        {"id": "r2", "prompt": "Q", "keyword_score": 0.5, "latency_s": 0.1},
    ]
    box = Lunchbox.from_records(records, required_columns=EVAL_SCHEMA)
    assert len(box) == 2
    assert box.required_columns == EVAL_SCHEMA


def test_lunchbox_exposed_on_package() -> None:
    import hypernix

    assert hypernix.lunchbox is not None
    assert hasattr(hypernix.lunchbox, "Lunchbox")
    assert hasattr(hypernix.lunchbox, "EVAL_SCHEMA")


def test_regression_report_cast_error_is_prevented() -> None:
    """The reported ray0rf1re/eval error had a Parquet schema of 11
    columns while the embedded ``huggingface`` metadata only described
    4.  Lunchbox's normalize + datasets.from_list route guarantees
    the two match by construction.  This test encodes that guarantee
    in the only way we can without uploading: the normalised rows
    must include every declared column in the schema, and every row
    must have exactly the same key set."""
    from hypernix.lunchbox import EVAL_SCHEMA, Lunchbox

    box = Lunchbox.for_eval()
    # Simulate the reported scenario: early rows only carry the
    # first four columns; later rows pick up the newer columns.
    box.add(id="r1", category="math", difficulty="easy", prompt="2+2?")
    box.add(
        id="r2", category="trivia", difficulty="hard", tier="t3",
        prompt="Who invented Python?", reference="van Rossum",
        model_response="Guido", keyword_score=1.0, latency_s=0.4,
        variant="a", pipeline_meta="{}",
    )

    rows = box.normalize()
    # Every row gets every declared column.
    for row in rows:
        assert set(row.keys()) == set(EVAL_SCHEMA)
    # Missing cells are None (Arrow null on write).
    assert rows[0]["latency_s"] is None
    assert rows[0]["model_response"] is None
