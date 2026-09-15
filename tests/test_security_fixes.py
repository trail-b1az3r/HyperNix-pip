"""Three security fixes, and the tests that would have caught them.

Each of these is a regression test in the strict sense: it fails against
the code as it was, and the failure it describes is the vulnerability
rather than a proxy for it.

The permissions one is the most serious and the least interesting:
``~/.hypernix/config.json`` holds the user's Anthropic, OpenAI, Moonshot
and DashScope API keys and their HuggingFace token, and it was written at
the default umask — 0644 on every mainstream distribution, in a 0755
directory. Every other secret store in this package already wrote 0600.

The key-lookup one is the classic shape: a secret compared with ``==``
inside a loop. What made it measurable was not the string compare (from
Python, that difference is well under the interpreter's own noise) but
the loop — the number of iterations depended on where the presented key
sat in the store, so the time to answer told the caller both whether a
key was valid and roughly where it was. With 200 keys the spread was 0.06
to 3.84 microseconds.
"""
from __future__ import annotations

import importlib
import os
import stat
import time

import pytest


# ---------------------------------------------------------------------------
# 1. The config file held API keys world-readable
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A fresh config module rooted at a temporary home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    import hypernix.system.config as module

    monkeypatch.setattr(module, "_CONFIG_DIR", tmp_path / ".hypernix")
    monkeypatch.setattr(
        module, "_CONFIG_FILE", tmp_path / ".hypernix" / "config.json"
    )
    return module


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
class TestTheConfigFilePermissions:
    def test_a_saved_api_key_is_not_world_readable(self, config):
        """The fix. Anything else on the machine could read every key the
        user had set."""
        config.set_provider_key("anthropic", "sk-ant-secret-value")
        assert _mode(config._CONFIG_FILE) == 0o600
        assert not _mode(config._CONFIG_FILE) & 0o077

    def test_the_directory_is_not_world_traversable_either(self, config):
        config.set_provider_key("openai", "sk-secret")
        assert not _mode(config._CONFIG_DIR) & 0o077

    def test_an_umask_cannot_loosen_it(self, config):
        """The mode is passed to os.open rather than left to the umask,
        so a permissive umask does not widen the file."""
        previous = os.umask(0o000)
        try:
            config.set_provider_key("openai", "sk-secret")
        finally:
            os.umask(previous)
        assert _mode(config._CONFIG_FILE) == 0o600

    def test_a_legacy_file_is_tightened_on_read(self, config):
        """Somebody who set their keys six months ago and has not touched
        one since would otherwise keep the 0644 file forever, and the
        upgrade would have fixed nothing for them."""
        config.set_provider_key("anthropic", "sk-ant-secret")
        os.chmod(config._CONFIG_FILE, 0o644)
        os.chmod(config._CONFIG_DIR, 0o755)

        config.get_provider_key("anthropic")

        assert _mode(config._CONFIG_FILE) == 0o600
        assert _mode(config._CONFIG_DIR) == 0o700

    def test_the_key_still_round_trips(self, config):
        """A permissions fix that broke reading would be worse than the
        bug."""
        config.set_provider_key("anthropic", "sk-ant-round-trip")
        assert config.get_provider_key("anthropic") == "sk-ant-round-trip"

    def test_an_interrupted_write_keeps_the_old_config(self, config, monkeypatch):
        """The file holds the only copy of keys the user typed once, and
        half a JSON object is not recoverable."""
        config.set_provider_key("anthropic", "sk-ant-original")

        real_replace = os.replace

        def _fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", _fail)
        with pytest.raises(OSError):
            config.set_provider_key("anthropic", "sk-ant-replacement")
        monkeypatch.setattr(os, "replace", real_replace)

        assert config.get_provider_key("anthropic") == "sk-ant-original"

    def test_no_temporary_file_is_left_behind(self, config, monkeypatch):
        config.set_provider_key("openai", "sk-a")

        def _fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", _fail)
        with pytest.raises(OSError):
            config.set_provider_key("openai", "sk-b")
        assert [p.name for p in config._CONFIG_DIR.iterdir()] == ["config.json"]


# ---------------------------------------------------------------------------
# 2. Key authentication leaked which key was presented, and whether it was one
# ---------------------------------------------------------------------------


@pytest.fixture
def keymaster(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from hypernix.security.keymaster import Keymaster

    return Keymaster(store_dir=tmp_path / "keys")


def _make(km, count=1):
    from hypernix.security.keymaster import KeyScope, KeyType

    return [
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        for _ in range(count)
    ]


class TestTheKeyLookupIsConstantTime:
    def test_a_valid_key_is_found(self, keymaster):
        meta = _make(keymaster)[0]
        assert keymaster.get_by_key(meta.key).key_id == meta.key_id

    def test_an_invalid_key_is_not(self, keymaster):
        _make(keymaster)
        assert keymaster.get_by_key("T1_" + "x" * 24 + "ab!@#$%/1") is None

    def test_an_empty_key_is_not(self, keymaster):
        _make(keymaster)
        assert keymaster.get_by_key("") is None

    def test_the_answer_does_not_depend_on_position(self, keymaster):
        """The measurable half of the old bug. The loop ran until it
        found a match, so answering took time proportional to where the
        key sat -- which told the caller both whether the key was valid
        and roughly where. With 200 keys the old spread was 0.06 us for
        the first to 3.84 us for a miss."""
        metas = _make(keymaster, 120)

        def _timed(candidate, rounds=400, repeats=5):
            best = float("inf")
            for _ in range(repeats):
                started = time.perf_counter()
                for _ in range(rounds):
                    keymaster.get_by_key(candidate)
                best = min(best, time.perf_counter() - started)
            return best / rounds

        first = _timed(metas[0].key)
        last = _timed(metas[-1].key)
        miss = _timed("T1_" + "z" * 24 + "ab!@#$%/1")
        slowest, fastest = max(first, last, miss), min(first, last, miss)

        # Generous, because a CI runner's scheduler is noisier than the
        # signal being ruled out. The old code was 60x across this range;
        # anything under 3x is not an oracle.
        assert slowest / fastest < 3.0, (
            f"first={first * 1e6:.2f}us last={last * 1e6:.2f}us "
            f"miss={miss * 1e6:.2f}us"
        )

    def test_it_uses_a_digest_index_rather_than_a_scan(self, keymaster):
        """Named explicitly so that someone simplifying this back into a
        loop has to delete a test that says why not."""
        metas = _make(keymaster, 3)
        assert len(keymaster._by_digest) == 3
        from hypernix.security.keymaster import _key_digest

        assert keymaster._by_digest[_key_digest(metas[0].key)] is metas[0]

    def test_the_index_survives_a_revoke(self, keymaster):
        metas = _make(keymaster, 3)
        keymaster.revoke(metas[1].key_id)
        assert keymaster.get_by_key(metas[1].key) is None
        assert keymaster.get_by_key(metas[0].key) is not None
        assert keymaster.get_by_key(metas[2].key) is not None

    def test_the_index_survives_a_rotate(self, keymaster):
        meta = _make(keymaster)[0]
        replacement = keymaster.rotate(meta.key_id)
        assert keymaster.get_by_key(meta.key) is None
        assert keymaster.get_by_key(replacement.key).key_id == replacement.key_id

    def test_the_index_is_rebuilt_from_disk(self, keymaster, tmp_path):
        """A reload that did not reindex would make every stored key stop
        authenticating -- the kind of bug that gets "fixed" by going back
        to the string compare."""
        from hypernix.security.keymaster import Keymaster

        meta = _make(keymaster)[0]
        reloaded = Keymaster(store_dir=tmp_path / "keys")
        assert reloaded.get_by_key(meta.key).key_id == meta.key_id

    def test_the_gatekeeper_path_still_works(self, keymaster):
        """get_by_key's only caller is the T1 API's primary auth path,
        reached with whatever an unauthenticated caller sent."""
        from hypernix.security.gatekeeper import Gatekeeper

        meta = _make(keymaster)[0]
        gate = Gatekeeper(keymaster=keymaster)
        assert gate.authenticate(meta.key).key_id == meta.key_id
        with pytest.raises(PermissionError):
            gate.authenticate("T1_" + "q" * 24 + "ab!@#$%/1")

    def test_a_revoked_key_is_refused_by_name(self, keymaster):
        from hypernix.security.gatekeeper import Gatekeeper

        meta = _make(keymaster)[0]
        gate = Gatekeeper(keymaster=keymaster)
        keymaster.revoke(meta.key_id)
        with pytest.raises(PermissionError):
            gate.authenticate(meta.key)


# ---------------------------------------------------------------------------
# 3. Sandboxes and bind defaults
# ---------------------------------------------------------------------------


class TestTheNoodleSandbox:
    """Not a fix — a boundary that this release made reachable from a
    chat TUI, so it is worth pinning down what it holds against."""

    @pytest.fixture
    def context(self, tmp_path):
        from hypernix.interfaces.noodle.tools import ToolContext

        return ToolContext(root=tmp_path / "workspace")

    @pytest.mark.parametrize("escape", [
        "/etc/passwd",
        "../../etc/passwd",
        "../outside.txt",
        "sub/../../../etc/passwd",
    ])
    def test_it_refuses_an_escape(self, context, escape):
        from hypernix.interfaces.noodle.tools import ToolError

        with pytest.raises(ToolError) as caught:
            context.resolve(escape)
        assert caught.value.code == "outside_workspace"

    def test_it_refuses_a_symlink_out(self, context):
        """resolve() follows symlinks before the containment check, so a
        link planted by an earlier tool call cannot be used to step
        outside."""
        from hypernix.interfaces.noodle.tools import ToolError

        (context.root / "escape").symlink_to("/etc")
        with pytest.raises(ToolError):
            context.resolve("escape/passwd")

    def test_a_tilde_stays_inside(self, context):
        """expanduser() runs after the join, where a `~` is a literal
        directory name. Expanding it first would produce an absolute path
        to the real home, and `Path(root) / "/abs"` is `/abs`."""
        assert context.root in context.resolve("~/.ssh/id_rsa").parents

    @pytest.mark.parametrize("ok", ["file.txt", "./file.txt", "sub/file.txt"])
    def test_ordinary_paths_still_work(self, context, ok):
        assert context.root in context.resolve(ok).parents or \
            context.resolve(ok).parent == context.root


class TestBindDefaults:
    """Anything that listens should default to loopback."""

    def test_the_livestream_server_defaults_to_loopback(self):
        """It publishes training logs, model output and hardware details
        with no authentication."""
        import inspect

        from hypernix.monitoring.livestream import LiveStreamServer

        assert inspect.signature(
            LiveStreamServer.__init__
        ).parameters["host"].default == "127.0.0.1"

    def test_the_t1api_example_does_not_bind_everywhere(self):
        """The first example in a docstring is the one that gets copied
        into production.

        Read from the file rather than imported: t1api.app pulls in
        FastAPI, which is an optional extra, and a security assertion
        about a docstring should not be skipped on a machine that has not
        installed the server dependencies.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        source = (root / "src/hypernix/t1api/app.py").read_text()
        docstring = source.split('"""')[1]
        # The runnable lines, not the prose around them — the docstring
        # goes on to explain what 0.0.0.0 would do, and mentioning it is
        # the opposite of recommending it.
        runnable = [
            line for line in docstring.splitlines()
            if "uvicorn.run(" in line and not line.lstrip().startswith("#")
        ]
        assert runnable, "the mounting example lost its uvicorn.run line"
        assert all('host="127.0.0.1"' in line for line in runnable), runnable

    def test_the_remote_desktop_defaults_to_localhost(self):
        from hypernix.monitoring import remote_desktop

        assert remote_desktop._listen_args("x11vnc", "localhost") == ["-localhost"]
