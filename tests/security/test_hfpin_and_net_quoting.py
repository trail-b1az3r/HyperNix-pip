"""0.72.8 security pass: Hub downloads can be pinned, and `hnx net` quotes
what it sends to a peer's shell."""
from __future__ import annotations

import json

import pytest

from hypernix.security import hfpin
from hypernix.system import net


class TestPinning:
    def test_an_explicit_revision_wins(self, monkeypatch):
        monkeypatch.setenv("HYPERNIX_HF_REVISION_OWNER_REPO", "abc1234")
        assert hfpin.hub_revision("owner/repo", "v2") == "v2"

    def test_the_environment_pins_one_repo(self, monkeypatch):
        monkeypatch.setenv("HYPERNIX_HF_REVISION_LIQUIDAI_LFM2_5_2_6B", "0123abcd")
        assert hfpin.hub_revision("LiquidAI/LFM2.5-2.6B") == "0123abcd"

    def test_the_pins_file(self, tmp_path, monkeypatch):
        pins = tmp_path / "pins.json"
        pins.write_text(json.dumps({"owner/repo": "deadbeef"}))
        monkeypatch.setenv("HYPERNIX_HF_PINS", str(pins))
        assert hfpin.hub_revision("owner/repo") == "deadbeef"
        assert hfpin.hub_revision("owner/other") is None

    def test_a_local_path_is_never_pinned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERNIX_HF_PINS", str(tmp_path / "missing.json"))
        assert hfpin.hub_revision(tmp_path) is None
        assert hfpin.hub_revision("./model") is None

    def test_no_pin_is_the_default_branch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERNIX_HF_PINS", str(tmp_path / "missing.json"))
        assert hfpin.hub_revision("owner/repo") is None


class TestRemoteCommands:
    def test_a_file_name_cannot_run_a_second_command(self):
        argv = net._remote("peer", "test", "-f", "x; rm -rf ~")
        assert argv[1:3] == ["--", "peer"]
        assert argv[3] == "test -f 'x; rm -rf ~'"

    def test_a_peer_cannot_be_an_ssh_option(self):
        assert net._remote("-oProxyCommand=evil", "true")[1] == "--"

    @pytest.mark.parametrize("cmd", ["tail_acheck", "tail_stop"])
    def test_the_peer_commands_quote_the_file(self, monkeypatch, cmd):
        seen = []

        class Done:
            returncode = 1

        monkeypatch.setattr(net, "get_tailscale_peers", lambda: ["100.64.0.2"])
        monkeypatch.setattr(net.subprocess, "run", lambda argv, **kw: seen.append(argv) or Done())
        args = type("A", (), {"file": "a b;c", "r": False})()
        getattr(net, f"cmd_{cmd}")(args)
        assert "'a b;c'" in seen[0][-1]
