"""/web/v2 (T1 v1.1.26.10.1.0): the grammar, reading sites, keeping
searches, the five settings, /web/v1/ai/config and the v1 deprecation."""
from __future__ import annotations

import json

import pytest

from hypernix.t1api import webconfig_cli
from hypernix.t1api import websearch as ws
from hypernix.t1api import websearch_v2 as v2

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.routers import web as web_router  # noqa: E402


class TestTheGrammar:
    def test_the_documented_form(self):
        r = v2.parse_v2_query("=red+shoes&q?=2&mx?=5&sq?=1&&kpd?=true")
        assert (r.query, r.depth, r.max_items, r.read_sites, r.keep) == ("red shoes", 2, 5, 1, True)

    def test_the_conventional_form(self):
        r = v2.parse_v2_query("q=cats&depth=3&max=7&read=2&keep=false")
        assert (r.query, r.depth, r.max_items, r.read_sites, r.keep) == ("cats", 3, 7, 2, False)

    def test_brackets_around_the_flag(self):
        assert v2.parse_v2_query("=x&&kpd?=[true]").keep is True

    def test_the_settings_cap_what_may_be_asked(self):
        settings = v2.WebAdminSettings(max_items=4, max_read_sites=2)
        r = v2.parse_v2_query("=x&mx?=999&sq?=50&q?=99", settings)
        assert (r.max_items, r.read_sites, r.depth) == (4, 2, ws.MAX_DEPTH)


class TestTheSettings:
    def test_round_trip_and_defaults(self, tmp_path):
        assert v2.load_settings(tmp_path) == v2.WebAdminSettings()
        s = v2.WebAdminSettings()
        s.set("store_enabled", "true")
        s.set("store_format", "parquet")
        s.set("max_items", "30")
        v2.save_settings(s, tmp_path)
        assert v2.load_settings(tmp_path).to_dict() == {
            "store_enabled": True, "store_format": "parquet", "max_items": 30,
            "max_read_sites": 5, "ai_config": "partial"}

    @pytest.mark.parametrize("name, value", [
        ("store_format", "csv"), ("max_items", "0"), ("max_items", "x"), ("ai_config", "everyone"),
        ("store_enabled", "maybe"), ("nope", "1")])
    def test_bad_values_are_refused(self, name, value):
        with pytest.raises(ValueError):
            v2.WebAdminSettings().set(name, value)

    def test_a_damaged_file_is_the_defaults(self, tmp_path):
        (tmp_path / "web.json").write_text("{not json")
        assert v2.load_settings(tmp_path) == v2.WebAdminSettings()

    def test_keeping_is_off_until_turned_on_then_jsonl(self, tmp_path):
        off = v2.keep({"query": "q"}, settings=v2.WebAdminSettings(), config_dir=tmp_path)
        assert off["kept"] is False and "off" in off["why"]
        on = v2.keep({"query": "q"}, settings=v2.WebAdminSettings(store_enabled=True), config_dir=tmp_path)
        line = json.loads(open(on["path"], encoding="utf-8").read().splitlines()[0])
        assert on["format"] == "jsonl" and line["query"] == "q" and line["id"] == on["id"]

    def test_parquet_without_pyarrow_keeps_jsonl_and_says_so(self, tmp_path, monkeypatch):
        import builtins

        real = builtins.__import__

        def no_pyarrow(name, *args, **kwargs):
            if name.startswith("pyarrow"):
                raise ImportError(name)
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pyarrow)
        kept = v2.keep({"query": "q"}, settings=v2.WebAdminSettings(store_enabled=True, store_format="parquet"),
                       config_dir=tmp_path)
        assert kept["format"] == "jsonl" and "pyarrow" in kept["note"]


class TestTheCli:
    def test_show_set_and_reset(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        assert webconfig_cli.main(["web", "--set", "store_enabled=true", "--set", "max_read_sites=3"]) == 0
        assert v2.load_settings(tmp_path).max_read_sites == 3
        assert "max_read_sites" in capsys.readouterr().out  # --set shows the table
        assert webconfig_cli.main(["web", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["store_enabled"] is True
        assert webconfig_cli.main(["web", "--set", "max_items=500"]) == 2
        assert webconfig_cli.main(["web", "--reset"]) == 0
        assert v2.load_settings(tmp_path) == v2.WebAdminSettings()


@pytest.fixture
def server(tmp_path, monkeypatch):
    from conftest import clear_t1_config

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    km = Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)
    app = create_app(config=T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough", db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"), hyperlink_files_dir=str(tmp_path / "files"),
    ), keymaster=km, gatekeeper=gk)
    hits = [ws.SearchHit(f"Site {i}", f"https://site{i}.test/", f"snippet {i}") for i in range(1, 9)]
    monkeypatch.setattr(web_router.ws, "search",
                        lambda query, depth=1, settings=None: ws.SearchOutcome(query, hits=list(hits), engine="duckduckgo",
                                                                               depth=depth))
    fetched: list[str] = []

    def fetch(url):
        fetched.append(url)
        return f"The page at {url} talks about the query at length. It has several sentences. More here."

    monkeypatch.setattr(web_router, "_fetch_for_summary", fetch)
    client = TestClient(app, client=("127.0.0.1", 5000))
    writer = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
    reader = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
    admin = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.READ, KeyScope.WRITE, KeyScope.ADMIN})
    return client, writer, reader, admin, fetched, tmp_path


def _h(key):
    return {"Authorization": f"Bearer {key.key}"}


class TestV2Search:
    def test_max_items_and_reading_only_the_first_site(self, server):
        client, writer, _r, _a, fetched, _ = server
        body = client.get("/web/v2/I?=widgets&q?=1&mx?=3&sq?=1&&kpd?=false", headers=_h(writer)).json()
        assert body["count"] == 3 and len(body["hits"]) == 3
        assert fetched == ["https://site1.test/"]
        assert body["sites"][0]["rank"] == 1 and "summary" in body["sites"][0]
        assert body["kept"] is None

    def test_keeping_needs_storage_on(self, server):
        client, writer, _r, _a, _f, tmp = server
        off = client.get("/web/v2/I?=widgets&&kpd?=true", headers=_h(writer)).json()
        assert off["kept"]["kept"] is False
        v2.save_settings(v2.WebAdminSettings(store_enabled=True), tmp)
        on = client.get("/web/v2/I?=widgets&sq?=2&&kpd?=true", headers=_h(writer)).json()
        record = json.loads(open(on["kept"]["path"], encoding="utf-8").read().splitlines()[-1])
        assert record["query"] == "widgets" and len(record["sites"]) == 2
        assert "_text" in record["sites"][0]  # the site text is what is kept

    def test_no_query_is_a_400(self, server):
        client, writer, *_ = server
        assert client.get("/web/v2/I?=&q?=2", headers=_h(writer)).status_code == 400

    def test_v1_search_still_answers_and_says_it_is_going(self, server):
        client, writer, *_ = server
        response = client.get("/web/v1/search?q=widgets", headers=_h(writer))
        assert response.status_code == 200
        assert response.headers["Deprecation"] == "true" and "0.73.0" in response.json()["deprecated"]
        assert "Deprecation" not in client.get("/web/v1/I?=widgets", headers=_h(writer)).headers


class TestAiConfig:
    def test_reading_needs_partial_admin(self, server):
        client, writer, reader, *_ = server
        assert client.get("/web/v1/ai/config", headers=_h(reader)).status_code == 403
        body = client.get("/web/v1/ai/config", headers=_h(writer)).json()
        assert body["ai_may_change"] == list(v2.AI_SAFE)

    def test_partial_admin_changes_only_the_less_risky_ones(self, server):
        client, writer, _r, _a, _f, tmp = server
        ok = client.post("/web/v1/ai/config", headers=_h(writer), json={"max_items": 12})
        assert ok.status_code == 200 and v2.load_settings(tmp).max_items == 12
        refused = client.post("/web/v1/ai/config", headers=_h(writer), json={"store_enabled": True})
        assert refused.status_code == 403 and v2.load_settings(tmp).store_enabled is False

    def test_an_admin_may_turn_storage_on(self, server):
        client, _w, _r, admin, _f, tmp = server
        assert client.post("/web/v1/ai/config", headers=_h(admin), json={"store_enabled": True}).status_code == 200
        assert v2.load_settings(tmp).store_enabled is True

    def test_off_means_off(self, server):
        client, writer, _r, _a, _f, tmp = server
        v2.save_settings(v2.WebAdminSettings(ai_config="off"), tmp)
        assert client.post("/web/v1/ai/config", headers=_h(writer), json={"max_items": 5}).status_code == 403

    def test_bad_values_and_empty_bodies(self, server):
        client, writer, *_ = server
        assert client.post("/web/v1/ai/config", headers=_h(writer), json={"max_items": 0}).status_code == 400
        assert client.post("/web/v1/ai/config", headers=_h(writer), json={}).status_code == 400
