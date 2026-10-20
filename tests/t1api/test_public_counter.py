"""The public-server counter (0.72.8a3): who sends a heartbeat, what it carries, and the counting side."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hypernix.t1api import counter
from hypernix.t1api.counter import Counter, CounterPinger, heartbeat, is_public, should_count


def _config(**kw):
    base = dict(public=False, public_counter=True, public_counter_url="", server_url="", hyperlink_public_url="")
    return SimpleNamespace(**{**base, **kw})


class TestWhoIsCounted:
    @pytest.mark.parametrize("url, public", [
        ("https://t1.example.org", True), ("http://93.184.215.14:8000", True), ("http://203.0.113.5:8000", False),
        ("http://127.0.0.1:8000", False), ("http://192.168.1.4:8000", False), ("http://100.64.0.9", False),
        ("https://box.tail1234.ts.net", False), ("http://box.local:8000", False), ("http://box:8000", False),
        ("", False),
    ])
    def test_public_means_an_internet_address(self, url, public):
        assert is_public(_config(server_url=url)) is public

    def test_t1_public_says_so_outright(self):
        assert is_public(_config(public=True))

    def test_turned_off(self):
        ok, why = should_count(_config(public=True, public_counter=False))
        assert not ok and "T1_PUBLIC_COUNTER=0" in why

    def test_a_private_server_sends_nothing(self):
        sent = []
        pinger = CounterPinger(_config(), None, post=lambda url, body: sent.append(body) or True,
                               endpoint=lambda: "https://counter.example/counter/heartbeat")
        assert pinger.start() is False and pinger.beat() is False and sent == []


class TestTheHeartbeat:
    def test_it_carries_three_things_and_no_address(self, tmp_path):
        body = heartbeat(tmp_path)
        assert set(body) == {"id", "t1_version", "version"}
        assert len(body["id"]) == 32 and int(body["id"], 16) >= 0
        assert heartbeat(tmp_path)["id"] == body["id"]                 # stable, so one server is one
        secret = (tmp_path / "counter-id").read_text()
        assert secret not in json.dumps(body)                          # the kept value is never sent

    def test_no_published_counter_means_nothing_is_sent(self, tmp_path):
        sent = []
        pinger = CounterPinger(_config(public=True), tmp_path, post=lambda url, body: sent.append(url) or True,
                               endpoint=lambda: "")
        assert pinger.beat() is False and sent == []

    def test_a_configured_counter_wins(self, tmp_path):
        sent = []
        pinger = CounterPinger(_config(public=True, public_counter_url="https://mine.example/counter/heartbeat"),
                               tmp_path, post=lambda url, body: sent.append((url, body)) or True,
                               endpoint=lambda: pytest.fail("asked for the published one"))
        assert pinger.beat() and sent[0][0] == "https://mine.example/counter/heartbeat"


class TestTheCounter:
    def test_counting(self, tmp_path):
        now = [1_800_000_000.0]
        c = Counter(tmp_path / "c.json", clock=lambda: now[0])
        c.record({"id": "a" * 32, "t1_version": "1.1.26.10.2.0", "version": "0.72.8a3"})
        c.record({"id": "b" * 32, "t1_version": "1.1.26.10.1.0", "version": "0.72.8a1"})
        c.record({"id": "a" * 32, "t1_version": "1.1.26.10.2.0", "version": "0.72.8a3"})
        assert c.summary()["up"] == 2
        assert c.summary()["t1_versions"] == {"1.1.26.10.1.0": 1, "1.1.26.10.2.0": 1}
        now[0] += 3 * 3600
        assert Counter(tmp_path / "c.json", clock=lambda: now[0]).summary()["up"] == 0

    @pytest.mark.parametrize("body, match", [
        ({"id": "x", "t1_version": "1.1.26.10.1.0"}, "32 lower-case hex"),
        ({"id": "a" * 32, "t1_version": "1.0.26.9.2.0"}, "only T1"),
        ({"id": "a" * 32, "t1_version": "nope"}, "t1_version"),
        ([], "JSON object"),
    ])
    def test_a_bad_heartbeat_is_refused(self, body, match):
        with pytest.raises(ValueError, match=match):
            Counter().record(body)


def test_the_routes(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from conftest import clear_t1_config
    from fastapi.testclient import TestClient

    from hypernix.t1api.app import create_app

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_WEB_ENABLED", "0")
    monkeypatch.setenv("T1_COUNTER_HOST", "1")
    client = TestClient(create_app(), client=("203.0.113.7", 5000))
    good = {"id": "c" * 32, "t1_version": "1.1.26.10.2.0", "version": "0.72.8a3"}
    assert client.post("/counter/heartbeat", json=good).json() == {"ok": True}       # no key needed
    assert client.post("/counter/heartbeat", content=b"x" * 600).status_code == 413
    assert client.post("/counter/heartbeat", json={"id": "bad"}).status_code == 400
    summary = client.get("/counter").json()
    assert summary["up"] == 1 and "203.0.113.7" not in json.dumps(summary)
    assert "203.0.113.7" not in (tmp_path / "counter.json").read_text()


def test_the_badge_script(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "update_t1_counter", Path(__file__).resolve().parents[2] / "scripts" / "update_t1_counter.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "DATA", tmp_path)
    (tmp_path / "t1-counter-endpoint.json").write_text('{"summary": ""}')
    monkeypatch.delenv("T1_COUNTER_SUMMARY_URL", raising=False)
    assert script.main() == 0
    assert json.loads((tmp_path / "t1-counter-badge.json").read_text())["message"] == "counter not hosted yet"
    monkeypatch.setenv("T1_COUNTER_SUMMARY_URL", "https://counter.example/counter")
    monkeypatch.setattr(script, "_fetch", lambda url: {"up": 3, "t1_versions": {"1.1.26.10.2.0": 3}})
    script.main()
    assert json.loads((tmp_path / "t1-counter-badge.json").read_text())["message"] == "3 up"
    monkeypatch.setattr(script, "_fetch", lambda url: (_ for _ in ()).throw(OSError("down")))
    script.main()
    record = json.loads((tmp_path / "t1-counter.json").read_text())
    assert record["up"] == 3 and record["state"] == "stale"
    assert counter.COUNTED_SINCE.short == "1.1.26.10.1.0"
