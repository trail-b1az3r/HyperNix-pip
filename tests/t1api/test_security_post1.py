"""The high-rated scan findings fixed in 0.72.6.post1.

CodeQL's Python security suite, run over the tree, reported 27 findings
rated high or critical. Most were the scanner not recognising a guard
that was already there, or a CLI printing a key it had just made for the
person who asked. These are the ones that were real, or where the
guard was worth making obvious, each held to what it now does.
"""
from __future__ import annotations

import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import clear_t1_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_t1_config(monkeypatch)


@pytest.fixture
def local_site():
    """A server on loopback that records every path it is asked for."""
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            body = b"User-agent: *\nDisallow: /private\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", hits
    server.shutdown()


class TestRobotsTxtOnACallersBehalf:
    """py/partial-ssrf, gather.py: robots.txt was fetched before the page,
    with a plain urlopen that follows redirects, so a public site's
    robots.txt could send the server to the LAN or 169.254.169.254."""

    def test_public_only_never_touches_a_private_address(self, local_site):
        from hypernix.data import gather

        base, hits = local_site
        robots = gather.RobotsCache(public_only=True)
        robots.allows(f"{base}/private/page")
        assert hits == [], "robots.txt was fetched from a loopback address"

    def test_the_local_tools_still_read_a_lan_sites_robots(self, local_site):
        from hypernix.data import gather

        base, hits = local_site
        robots = gather.RobotsCache()
        assert robots.allows(f"{base}/private/page") is False
        assert hits == ["/robots.txt"]

    def test_the_server_summary_route_asks_for_the_public_only_cache(self):
        source = (ROOT / "src/hypernix/t1api/routers/web.py").read_text(encoding="utf-8")
        assert "_shared_robots(public_only=True)" in source
        assert "_shared_robots()." not in source

    def test_the_two_caches_are_different_objects(self):
        from hypernix.interfaces import websearch as iw

        assert iw._shared_robots(public_only=True).public_only is True
        assert iw._shared_robots().public_only is False


class TestScriptAndStyleAreStripped:
    """py/bad-tag-filter: `</script foo>` and upper case ended the element in a
    browser but not in the pattern, so script text reached the page text."""

    @pytest.mark.parametrize("markup", [
        "<p>keep</p><script>evil()</script><p>this</p>",
        "<p>keep</p><SCRIPT>evil()</SCRIPT><p>this</p>",
        "<p>keep</p><script>evil()</script\t\n bar><p>this</p>",
        "<p>keep</p><style>.evil{}</style foo><p>this</p>",
    ])
    def test_gather(self, markup):
        from hypernix.data import gather

        text = gather.extract_text(markup)
        assert "evil" not in text
        assert "keep" in text and "this" in text


class TestUsageSqlUsesOnlyLiteralColumns:
    """py/sql-injection: `group_by` from the query string was checked against
    an allowlist and then interpolated. It now picks a literal."""

    def test_an_unknown_dimension_is_refused(self, tmp_path):
        from hypernix.t1api.storage import UsageStore

        store = UsageStore(tmp_path / "u.db")
        with pytest.raises(ValueError):
            store.aggregate("model_id; DROP TABLE usage_events")
        with pytest.raises(ValueError):
            store.aggregate("model_id", **{"1=1 OR key_id": "x"})

    def test_every_dimension_maps_to_itself(self):
        from hypernix.t1api import storage

        assert storage._COLUMNS == {name: name for name in storage.GROUPABLE}


class TestModulePathsStayInside:
    """py/path-injection, modules.py: the guard was right, but a shape the
    scanner could not see. It is now realpath plus a separator-aware prefix."""

    def test_traversal_is_refused(self, tmp_path):
        from hypernix.t1api.errors import T1APIError
        from hypernix.t1api.security import sanitize_module_path

        base = tmp_path / "modules"
        base.mkdir()
        for bad in ("../outside", "a/../../outside", "/etc/passwd"):
            with pytest.raises(T1APIError):
                sanitize_module_path(bad, base)

    def test_a_sibling_with_the_same_prefix_is_refused(self, tmp_path):
        """`/srv/modules-evil` starts with `/srv/modules`."""
        from hypernix.t1api.errors import T1APIError
        from hypernix.t1api.security import sanitize_module_path

        base = tmp_path / "modules"
        base.mkdir()
        (tmp_path / "modules-evil").mkdir()
        with pytest.raises(T1APIError):
            sanitize_module_path("../modules-evil/x", base)

    def test_a_symlink_out_is_refused(self, tmp_path):
        from hypernix.t1api.errors import T1APIError
        from hypernix.t1api.security import sanitize_module_path

        base = tmp_path / "modules"
        base.mkdir()
        (base / "link").symlink_to(tmp_path)
        with pytest.raises(T1APIError):
            sanitize_module_path("link/secret", base)

    def test_a_path_inside_is_returned_resolved(self, tmp_path):
        from hypernix.t1api.security import sanitize_module_path

        base = tmp_path / "modules"
        base.mkdir()
        assert sanitize_module_path("m1/file.bin", base) == (base / "m1" / "file.bin").resolve()


class TestKeyIdsAreNotPaths:
    """py/path-injection, keymaster.py: a key id from the rotate route
    became a file name. Only ids that already exist reached it, but the
    file name now refuses anything that is not an id."""

    @pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "", "x" * 200, "id.json"])
    def test_a_path_is_not_a_key_id(self, bad):
        from hypernix.security.keymaster import _safe_key_id

        with pytest.raises(ValueError):
            _safe_key_id(bad)

    def test_a_minted_id_is(self, tmp_path):
        from hypernix.security.keymaster import Keymaster, KeyScope, KeyType

        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        meta = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        assert km._key_path(meta.key_id).parent == km._store


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits; Windows guards a profile's files by ACL, and chmod there only sets read-only")
class TestWaitersConfigIsNeverReadable:
    """The key sat in a file written under the default umask and chmodded
    afterwards. It is now created 0600."""

    def test_created_owner_only(self, tmp_path):
        from hypernix.waiter.local_config import WaiterConfigStore, WaiterLocalConfig

        old = os.umask(0o022)
        try:
            path = tmp_path / "waiter.jsonl"
            WaiterConfigStore(path).save(WaiterLocalConfig(server="http://x", key="T1_secret"))
        finally:
            os.umask(old)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_an_existing_loose_file_is_tightened(self, tmp_path):
        from hypernix.waiter.local_config import WaiterConfigStore, WaiterLocalConfig

        path = tmp_path / "waiter.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o644)
        WaiterConfigStore(path).save(WaiterLocalConfig(server="http://x", key="T1_secret"))
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


class TestLmStudiosAddressIsHttp:
    """The admin-only override is the one LM Studio URL that arrives over
    HTTP, and urlopen also speaks file:// and ftp://."""

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "gopher://h"])
    def test_other_schemes_are_refused(self, url):
        from hypernix.bridge.lmstudio import LMStudioError, _normalise_url

        with pytest.raises(LMStudioError):
            _normalise_url(url)

    @pytest.mark.parametrize("url,expected", [
        ("desktop:1234", "http://desktop:1234"),
        ("https://box:1234/v1", "https://box:1234"),
    ])
    def test_http_still_works(self, url, expected):
        from hypernix.bridge.lmstudio import _normalise_url

        assert _normalise_url(url) == expected


def test_the_examples_script_holds_no_secret():
    """HNX-T1-001: the example server's secrets were committed literals."""
    source = (ROOT / "scripts/t1api_examples.py").read_text(encoding="utf-8")
    assert "not-a-real-one" not in source
    assert "secrets.token_hex(32)" in source
