"""The Stats page's downloads by hypernix version (0.72.8rc1).

scripts/update_json_stats.py adds ``series.by_version`` (a daily line per
release) and ``versions`` (every exact version's downloads) from ClickPy,
merged with what the stats file already holds so a failed fetch loses
nothing.
"""
from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("requests")


@pytest.fixture(scope="module")
def stats():
    spec = importlib.util.spec_from_file_location(
        "update_json_stats", Path(__file__).resolve().parents[2] / "scripts" / "update_json_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _day(n: int) -> str:
    return (datetime.now(UTC).date() - timedelta(days=n)).isoformat()


class TestReleases:
    @pytest.mark.parametrize("version,release", [
        ("0.72.7", "0.72.7"), ("0.72.7.post1", "0.72.7"), ("0.72.8rc1", "0.72.8"),
        ("0.72.8a4", "0.72.8"), ("0.72.8b1", "0.72.8"), ("1.0.dev3", "1.0"),
    ])
    def test_a_version_counts_toward_its_release(self, stats, version, release):
        assert stats.release_of(version) == release

    def test_versions_sort_as_numbers(self, stats):
        assert sorted(["0.9.0", "0.10.0", "0.72.7.post1"], key=stats.version_key) == ["0.9.0", "0.10.0", "0.72.7.post1"]


class TestByVersion:
    def test_lines_per_release_and_a_row_per_version(self, stats):
        rows = [(_day(40), "0.72.6", 5), (_day(2), "0.72.7", 3), (_day(1), "0.72.7.post1", 4), (_day(1), "0.72.7", 1)]
        lines, versions = stats.by_version(rows, None, None, current="0.72.7.post1")
        assert lines["0.72.7"] == {_day(2): 3, _day(1): 5}
        assert [v["version"] for v in versions] == ["0.72.7.post1", "0.72.7", "0.72.6"]
        post = versions[0]
        assert post["current"] and post["release"] == "0.72.7" and post["last_30_days"] == 4
        old = versions[-1]
        assert old["downloads"] == 5 and old["last_30_days"] == 0 and not old["current"]

    def test_a_failed_fetch_keeps_what_the_file_had(self, stats):
        existing = {"dates": ["2026-01-01", "2026-01-02"], "by_version": {"0.70.0": [2, 3]}}
        kept = [{"version": "0.70.0", "downloads": 5}]
        lines, versions = stats.by_version([], existing, kept)
        assert lines == {"0.70.0": {"2026-01-01": 2, "2026-01-02": 3}} and versions == kept

    def test_the_series_carries_them_newest_first(self, stats, monkeypatch):
        monkeypatch.setattr(stats, "get_category_series", lambda url: {})
        lines = {"0.9.0": {"2026-01-01": 1}, "0.10.0": {"2026-01-02": 2}}
        series = stats.build_series(None, lines)
        assert series["dates"] == ["2026-01-01", "2026-01-02"]
        assert list(series["by_version"]) == ["0.10.0", "0.9.0"]
        assert series["by_version"]["0.10.0"] == [0, 2]

    def test_clickpy_is_asked_with_the_project_as_a_parameter(self, stats, monkeypatch):
        """The package name is a query parameter, never formatted into the SQL."""
        asked = {}

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [["2026-10-01", "0.72.7", "9"], ["bad"]]}

        def post(url, data, params, timeout, headers):
            asked.update(url=url, sql=data.decode(), params=params)
            return Response()

        monkeypatch.setattr(stats.requests, "post", post)
        assert stats.get_version_rows() == [("2026-10-01", "0.72.7", 9)]
        assert "{project:String}" in asked["sql"] and stats.PACKAGE not in asked["sql"]
        assert asked["params"] == {"param_project": stats.PACKAGE}
        assert asked["url"].startswith("https://")

    def test_clickpy_down_is_no_rows(self, stats, monkeypatch):
        def post(*a, **k):
            raise stats.requests.ConnectionError("down")

        monkeypatch.setattr(stats.requests, "post", post)
        assert stats.get_version_rows() == []
