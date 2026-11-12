"""0.72.9: ``waiter serv -Pro KEY -K ADMIN_KEY -L LEVEL``.

An administrator grants a key an access level (1-9). The level a T2 key
states is its own last character, and until now the server took that at
its word, so a holder could rewrite "-1" as "-9". The grant is now the
key's recorded ``access_level`` (gkey already wrote it), a presented level
is held to it, and promotion is how the record is raised.
"""
from __future__ import annotations

import contextlib
import io
import re

import pytest

from hypernix.security.gkey_cli import main as gkey
from hypernix.waiter.servargs import ServArgsError, expand

try:  # a real import, not find_spec
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
needs_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="needs the [t1api] extra (fastapi)")

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class TestTheCommandLine:
    def test_pro_takes_the_key_and_l_the_level(self):
        argv = expand(["-Pro", "T2_key", "-K", "T2_admin", "-L", "5"]).argv
        assert argv == ["--promote", "T2_key", "-K", "T2_admin", "--level", "5"]

    def test_without_pro_l_is_still_local_only(self):
        assert expand(["-A", "-L"]).argv == ["-A", "-L"]
        assert expand(["-AL"]).argv == ["--auto", "--local-only"]

    def test_pro_is_not_read_as_a_port_cluster(self):
        assert expand(["-Pro", "k", "-L", "2"]).argv[0] == "--promote"

    def test_a_clustered_l_with_pro_is_refused_not_guessed(self):
        with pytest.raises(ServArgsError, match="-L is the level"):
            expand(["-Pro", "k", "-AL", "5"])

    def test_the_parser_accepts_it(self):
        from hypernix.waiter.cli import _build_serv_parser

        args = _build_serv_parser().parse_args(expand(["-Pro", "k", "-K", "a", "-L", "7"]).argv)
        assert (args.promote, args.key, args.level, args.local_only) == ("k", "a", 7, False)

    def test_a_level_outside_1_to_9_is_refused(self):
        from hypernix.waiter.cli import _build_serv_parser

        with pytest.raises(SystemExit):
            _build_serv_parser().parse_args(expand(["-Pro", "k", "-L", "12"]).argv)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster")
    monkeypatch.setattr("hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper")
    monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
    monkeypatch.setenv("T1_BACKUP_DIR", str(tmp_path / "backups"))
    return tmp_path / "keymaster"


def _gkey(*argv: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert gkey(list(argv)) == 0
    return ANSI.sub("", buf.getvalue())


def _field(text: str, label: str) -> str:
    pattern = re.compile(rf"\b{re.escape(label)}:\s*(.+)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1).strip().rstrip("│").strip()
    raise AssertionError(f"no {label!r} in:\n{text}")


@needs_server
class TestTheServer:
    @pytest.fixture
    def wired(self, store):
        from fastapi.testclient import TestClient

        from hypernix.t1api.app import create_app

        admin = _gkey("create", "--type", "admin", "--scopes", "admin,read,write")
        user = _gkey("create", "-v", "v2", "--scopes", "read,write", "--level", "2")
        with TestClient(create_app(), client=("127.0.0.1", 5000)) as client:
            yield client, {"Authorization": "Bearer " + _field(admin, "Key")}, _field(user, "Key")

    @staticmethod
    def _level(client, key: str) -> int:
        reply = client.post("/auth/t1/validate", json={"key": key})
        assert reply.status_code == 200, reply.text
        return reply.json()["access_level"]

    def test_rewriting_the_suffix_buys_nothing(self, wired):
        client, _, user = wired
        assert user.endswith("-2")
        assert self._level(client, user[:-1] + "9") == 2

    def test_promotion_raises_the_grant_and_answers_the_new_spelling(self, wired):
        client, admin, user = wired
        reply = client.post("/auth/t1/admin/access-level",
                            json={"key": user, "access_level": 7}, headers=admin)
        assert reply.status_code == 200, reply.text
        body = reply.json()
        assert (body["access_level"], body["previous_level"]) == (7, 2)
        assert body["key"] == user[:-1] + "7"
        assert self._level(client, body["key"]) == 7
        # The old spelling still works, at the level it states.
        assert self._level(client, user) == 2

    def test_only_an_admin_may(self, wired):
        client, _, user = wired
        reply = client.post("/auth/t1/admin/access-level", json={"key": user, "access_level": 9},
                            headers={"Authorization": "Bearer " + user})
        assert reply.status_code in (401, 403)

    def test_an_unknown_key_is_a_404(self, wired):
        client, admin, user = wired
        from hypernix.security.keymaster import T1KeyGenerator

        stranger = "T1_" + T1KeyGenerator.generate()[3:]
        reply = client.post("/auth/t1/admin/access-level",
                            json={"key": stranger, "access_level": 3}, headers=admin)
        assert reply.status_code in (400, 404)

    def test_it_can_be_undone(self, wired):
        client, admin, user = wired
        promoted = client.post("/auth/t1/admin/access-level",
                               json={"key": user, "access_level": 8}, headers=admin).json()["key"]
        assert self._level(client, promoted) == 8
        assert client.post("/t1/auth/undo", headers=admin).status_code == 200
        assert self._level(client, promoted) == 2


@needs_server
def test_waiter_serv_pro_end_to_end(store, monkeypatch, capsys):
    """The command as typed, against a live app: the key comes back promoted."""
    from fastapi.testclient import TestClient

    from hypernix.t1api.app import create_app
    from hypernix.waiter import cli

    admin = _field(_gkey("create", "--type", "admin", "--scopes", "admin,read,write"), "Key")
    user = _field(_gkey("create", "-v", "v2", "--scopes", "read", "--level", "1"), "Key")
    with TestClient(create_app(), client=("127.0.0.1", 5000)) as client:
        def via_testclient(self, method, path, **kwargs):
            from types import SimpleNamespace

            headers = {"Authorization": "Bearer " + admin}
            reply = client.request(method, path, json=kwargs.get("body"), headers=headers)
            assert reply.status_code == 200, reply.text
            return SimpleNamespace(body=reply.json())

        monkeypatch.setattr(type(cli.T1Client("http://x").transport), "request", via_testclient)
        code = cli._cmd_serv(["-Pro", user, "-K", admin, "-I", "http://127.0.0.1:8000", "-L", "6"])
    out = capsys.readouterr().out
    assert code == 0
    assert user[:-1] + "6" in out
