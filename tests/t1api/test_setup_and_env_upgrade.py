"""``hnx-t1 setup`` (installer b1) and ``hnx-t1 upgrade -S env|venv|all`` (0.72.8a4)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hypernix.t1api import envupgrade, setup_cli

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bin" / "hypernix-t1"


def _keys(text: str) -> dict[str, str]:
    return setup_cli.read_env_text(text) if hasattr(setup_cli, "read_env_text") else {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in text.splitlines() if line and not line.startswith("#") and "=" in line}


def _setup(tmp_path, *argv, read=None):
    config = tmp_path / "t1"
    args = ["--config-dir", str(config), "--no-start", *argv]
    if read is None:
        return setup_cli.cli_main(["--yes", *args]), config
    return setup_cli.cli_main(args, read=read), config


class TestSetup:
    def test_every_setting_it_writes_is_one_hypernix_reads(self, tmp_path):
        answers = setup_cli.Answers(public_url="https://t1.example.org", public=True, models="lmstudio",
                                    default_model="none", claude_model="qwen3.5", trusted=True)
        text = setup_cli.build_env(answers, config_dir=tmp_path, old={}, version="x")
        written = set(_keys(text))
        assert written <= envupgrade.read_by_hypernix(), written - envupgrade.read_by_hypernix()
        assert {"T1_TOKEN_SECRET", "T1_ANTHROPIC_ENABLED", "T1_MCP_HF", "T1_CODE_SANDBOX"} <= written

    def test_yes_writes_the_defaults_privately(self, tmp_path):
        code, config = _setup(tmp_path)
        env = config / ".env"
        assert code == 0 and stat.S_IMODE(env.stat().st_mode) == 0o600
        values = _keys(env.read_text())
        assert values["T1_HOST"] == "127.0.0.1" and values["T1_TRUSTED_NETWORK"] == "0"
        assert values["T1_CODE_SANDBOX"] == "0" and len(values["T1_TOKEN_SECRET"]) == 64
        assert "T1_DEFAULT_MODEL" not in values          # unset: HyperNix's default model
        assert stat.S_IMODE((config / "keymaster").stat().st_mode) == 0o700

    def test_dry_run_writes_nothing_and_stdout_is_the_file(self, tmp_path, capsys):
        code, config = _setup(tmp_path, "--dry-run")
        out = capsys.readouterr().out
        assert code == 0 and not config.exists()
        assert out.startswith("# Written by hnx-t1 setup (installer b1")
        # The file, but not its secret (CodeQL: clear-text logging).
        values = _keys(out)
        assert values["T1_TOKEN_SECRET"] == "<redacted>"
        # Switches are not secrets because their names say KEY.
        assert all(v != "<redacted>" for k, v in values.items() if k != "T1_TOKEN_SECRET" and
                   v.lower() in ("0", "1", "true", "false"))

    def test_a_second_setup_keeps_the_secret_and_what_it_does_not_ask(self, tmp_path):
        _, config = _setup(tmp_path)
        env = config / ".env"
        secret = _keys(env.read_text())["T1_TOKEN_SECRET"]
        env.write_text(env.read_text() + "T1_HF_TOKEN=hf_kept\n")
        answers = tmp_path / "a.json"
        answers.write_text(json.dumps({"reach": "network", "port": 9001, "code_sandbox": True,
                                       "default_model": "none"}))
        code, _ = _setup(tmp_path, "--answers", str(answers))
        values = _keys(env.read_text())
        assert code == 0 and values["T1_TOKEN_SECRET"] == secret
        assert values["T1_HOST"] == "0.0.0.0" and values["T1_PORT"] == "9001"
        assert values["T1_CODE_SANDBOX"] == "1" and values["T1_DEFAULT_MODEL"] == ""
        assert values["T1_HF_TOKEN"] == "hf_kept"
        assert list(config.glob(".env.bak-*"))

    def test_a_wrong_answer_is_refused_with_its_code(self, tmp_path, capsys):
        answers = tmp_path / "a.json"
        answers.write_text(json.dumps({"port": "eighty"}))
        code, config = _setup(tmp_path, "--answers", str(answers))
        assert code == 2 and not (config / ".env").exists()
        assert "S1-02091.a2" in capsys.readouterr().err
        answers.write_text(json.dumps({"colour": "blue"}))
        assert _setup(tmp_path, "--answers", str(answers))[0] == 2

    def test_asked_interactively_it_asks_again_until_an_answer_fits(self, tmp_path):
        replies = iter(["box", "3", "70000", "8100", "", "n", "4", "1", "", "n", "n", "n", "n",
                        "n", "n", "n", "y"])
        code, config = _setup(tmp_path, read=lambda prompt: next(replies))
        values = _keys((config / ".env").read_text())
        assert code == 0 and values["T1_SERVER_NAME"] == "box" and values["T1_PORT"] == "8100"
        assert values["T1_RATE_LIMIT_ENABLED"] == "0" and values["T1_HYPERLINK_ENABLED"] == "0"
        assert values["T1_LMSTUDIO_ENABLED"] == "0"


class TestEnvUpgrade:
    def _env(self, tmp_path, text: str, mode: int = 0o600) -> Path:
        env = tmp_path / ".env"
        env.write_text(text)
        env.chmod(mode)
        return env

    def test_new_settings_are_added_commented_at_their_default(self, tmp_path):
        env = self._env(tmp_path, "T1_TOKEN_SECRET=abc\nT1_HOST=127.0.0.1\n")
        report, _ = envupgrade.upgrade_env(env, version="x")
        text = env.read_text()
        assert "T1_CODE_SANDBOX" in report.added and "# T1_CODE_SANDBOX=0" in text
        assert "\nT1_CODE_SANDBOX=" not in text      # nothing is turned on
        assert "T1_TOKEN_SECRET" not in report.added and text.count("T1_TOKEN_SECRET") == 1
        assert report.backup and Path(report.backup).read_text() == "T1_TOKEN_SECRET=abc\nT1_HOST=127.0.0.1\n"
        again, _ = envupgrade.upgrade_env(env, version="x")
        assert not again.changed

    def test_a_missing_secret_is_added_and_a_readable_file_is_closed(self, tmp_path):
        env = self._env(tmp_path, "T1_HOST=127.0.0.1\n", mode=0o644)
        report, _ = envupgrade.upgrade_env(env, version="x")
        assert report.secret_added and report.mode_fixed
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
        assert len(_keys(env.read_text())["T1_TOKEN_SECRET"]) == 64

    def test_a_typo_is_reported_with_what_it_meant_and_left_in(self, tmp_path):
        env = self._env(tmp_path, "T1_TOKEN_SECRET=abc\nT1_TRUSTED_NETWROK=1\n")
        report, _ = envupgrade.upgrade_env(env, version="x")
        assert report.unknown == {"T1_TRUSTED_NETWROK": "T1_TRUSTED_NETWORK"}
        assert "T1_TRUSTED_NETWROK=1" in env.read_text()

    def test_dry_run_changes_nothing(self, tmp_path):
        env = self._env(tmp_path, "T1_HOST=127.0.0.1\n", mode=0o644)
        report, new = envupgrade.upgrade_env(env, version="x", dry_run=True)
        assert report.changed and "T1_TOKEN_SECRET" in new
        assert env.read_text() == "T1_HOST=127.0.0.1\n" and not list(tmp_path.glob(".env.bak-*"))

    def test_two_backups_in_one_second_are_two_files(self, tmp_path):
        env = self._env(tmp_path, "T1_HOST=1\n")
        first = envupgrade.backup_name(env)
        first.write_text("x")
        assert envupgrade.backup_name(env) != first

    def test_every_setting_has_a_name_and_most_say_what_they_do(self):
        settings = envupgrade.known_settings()
        assert len(settings) > 80
        assert sum(1 for s in settings.values() if s.help) > len(settings) / 2


@pytest.mark.skipif(os.name == "nt", reason="the manager is bash")
class TestTheManager:
    def _run(self, tmp_path, *argv):
        env = {**os.environ, "HOME": str(tmp_path / "home"), "T1_CONFIG_DIR": str(tmp_path / "t1"),
               "NO_COLOR": "1", "PYTHONPATH": str(ROOT / "src"),
               "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        done = subprocess.run(["bash", str(SCRIPT), *argv], capture_output=True, text=True, env=env, timeout=120)
        return done.returncode, done.stdout + done.stderr

    def test_setup_runs_the_installer(self, tmp_path):
        code, out = self._run(tmp_path, "setup", "--yes", "--no-start")
        assert code == 0, out
        assert "installer b1" in out and (tmp_path / "t1" / ".env").is_file()

    def test_upgrade_env_upgrades_the_configuration(self, tmp_path):
        (tmp_path / "t1").mkdir()
        (tmp_path / "t1" / ".env").write_text("T1_HOST=127.0.0.1\n")
        code, out = self._run(tmp_path, "upgrade", "-S", "env")
        assert code == 0, out
        assert "token secret" in out and "T1_TOKEN_SECRET=" in (tmp_path / "t1" / ".env").read_text()

    def test_upgrade_refuses_a_target_it_does_not_have(self, tmp_path):
        code, out = self._run(tmp_path, "upgrade", "-S", "everything")
        assert code != 0 and "env, venv or all" in out

    def test_upgrade_env_without_a_config_says_to_run_setup(self, tmp_path):
        code, out = self._run(tmp_path, "upgrade", "-Senv")
        assert code != 0 and "hnx-t1 setup" in out
