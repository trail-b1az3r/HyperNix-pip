"""``hypernix-t1 upgrade``, and the warning that says one is needed.

The failure these exist for: someone runs ``pip install -U hypernix`` in
their shell, ``hypernix-t1 restart``, and nothing changes -- because the
server runs from the private venv install-t1.sh made, which that pip
never touched. ``hypernix-t1 version`` then reads the old version, and
features the upgrade was for (the HyperLink site) are silently missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from shell_support import BASH, NO_BASH_REASON

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "bin" / "hypernix-t1"

pytestmark = pytest.mark.skipif(BASH is None or os.name == "nt", reason=NO_BASH_REASON)


def stub_python(path: Path, version_file: Path, pip_log: Path | None = None) -> None:
    """An interpreter that reports the HyperNix version in *version_file*.

    `pip install -U` is recorded in *pip_log* and "upgrades" to
    0.72.6.post3; everything else runs on the real interpreter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pip = (
        f'printf "%s\\n" "$*" >> "{pip_log}"; echo 0.72.6.post3 > "{version_file}"; exit 0'
        if pip_log else "exit 1"
    )
    path.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'  *importlib.metadata*) cat "{version_file}" ;;\n'
        f'  "-m pip install"*) {pip} ;;\n'
        f'  *) exec "{sys.executable}" "$@" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def run(script: Path, *argv: str, home: Path, config: Path):
    result = subprocess.run(
        [BASH, str(script), *argv],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
        env={
            **os.environ,
            "HOME": str(home),
            "T1_CONFIG_DIR": str(config),
            "NO_COLOR": "1",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    result.output = result.stdout + result.stderr  # type: ignore[attr-defined]
    return result


@pytest.fixture
def server(tmp_path):
    """A config dir whose private venv has HyperNix 0.72.6."""
    config = tmp_path / "t1api"
    version = tmp_path / "venv_version"
    version.write_text("0.72.6\n", encoding="utf-8")
    pip_log = tmp_path / "pip.log"
    stub_python(config / "venv" / "bin" / "python", version, pip_log)
    return {"home": tmp_path, "config": config, "version": version, "pip_log": pip_log}


def test_upgrade_runs_pip_in_the_servers_own_python(server):
    result = run(SCRIPT, "upgrade", home=server["home"], config=server["config"])
    assert result.returncode == 0, result.output
    assert server["pip_log"].read_text(encoding="utf-8").split("\n")[0] == "-m pip install -U hypernix[t1api]"
    assert "0.72.6 → 0.72.6.post3" in result.output
    assert "hypernix-t1 start" in result.output


def test_upgrade_main_installs_from_github(server):
    result = run(SCRIPT, "upgrade", "--main", home=server["home"], config=server["config"])
    assert result.returncode == 0, result.output
    assert "git+https://github.com/trail-b1az3r/HyperNix-pip@main" in server["pip_log"].read_text(encoding="utf-8")


def test_upgrade_takes_a_requirement(server):
    result = run(SCRIPT, "upgrade", "hypernix[t1api]==0.72.7",
                 home=server["home"], config=server["config"])
    assert result.returncode == 0, result.output
    assert "hypernix[t1api]==0.72.7" in server["pip_log"].read_text(encoding="utf-8")


def test_upgrade_refuses_an_unknown_option(server):
    result = run(SCRIPT, "upgrade", "--bogus", home=server["home"], config=server["config"])
    assert result.returncode != 0
    assert "Unknown option: --bogus" in result.output
    assert not server["pip_log"].exists()


def test_upgrade_says_when_nothing_changed(server):
    server["version"].write_text("0.72.6.post3\n", encoding="utf-8")
    result = run(SCRIPT, "upgrade", home=server["home"], config=server["config"])
    assert result.returncode == 0, result.output
    assert "already the newest" in result.output


def _installed_copy(tmp_path: Path, version: str) -> Path:
    """hypernix-t1 as pip installs it: next to a python with *version*."""
    envbin = tmp_path / "shellenv" / "bin"
    envbin.mkdir(parents=True)
    script = envbin / "hypernix-t1"
    shutil.copy(SCRIPT, script)
    own_version = tmp_path / "own_version"
    own_version.write_text(version + "\n", encoding="utf-8")
    stub_python(envbin / "python3", own_version)
    return script


def test_status_warns_when_the_server_runs_an_older_hypernix(server, tmp_path):
    script = _installed_copy(tmp_path, "0.72.6.post3")
    result = run(script, "status", home=server["home"], config=server["config"])
    assert "HyperNix 0.72.6" in result.output
    assert "older than the 0.72.6.post3 this command came with" in result.output
    assert "hypernix-t1 upgrade" in result.output
    assert "hypernix  0.72.6 (" in result.output


def test_no_warning_when_the_versions_match(server, tmp_path):
    script = _installed_copy(tmp_path, "0.72.6")
    result = run(script, "status", home=server["home"], config=server["config"])
    assert "older than" not in result.output


def test_no_warning_when_the_server_is_newer(server, tmp_path):
    server["version"].write_text("0.72.7\n", encoding="utf-8")
    script = _installed_copy(tmp_path, "0.72.6.post3")
    result = run(script, "status", home=server["home"], config=server["config"])
    assert "older than" not in result.output


def test_no_warning_from_a_checkout(server):
    # bin/ in a checkout has no python next to it: nothing to compare.
    result = run(SCRIPT, "status", home=server["home"], config=server["config"])
    assert "older than" not in result.output


def test_status_says_when_the_installed_hypernix_has_no_site():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "has no HyperLink site" in source
    assert "except ImportError:" in source


def _venv_script(server, body: str) -> Path:
    script = server["config"] / "venv" / "bin" / "hypernix-t1"
    script.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    script.chmod(0o755)
    return script


def test_a_different_copy_in_the_servers_venv_is_the_one_that_runs(server):
    """`pip install -U` into the venv upgrades venv/bin/hypernix-t1, not the
    copy on PATH; two builds of one version cannot be told apart by it.
    Seen on a real server: status described the old model sync."""
    _venv_script(server, 'echo "server copy: $* delegated=$HNX_T1_DELEGATED"\n')
    result = run(SCRIPT, "status", "--x", home=server["home"], config=server["config"])
    assert result.returncode == 0, result.output
    assert "server copy: status --x delegated=1" in result.output


def test_an_identical_copy_is_not_handed_off_to(server):
    copy = server["config"] / "venv" / "bin" / "hypernix-t1"
    shutil.copy(SCRIPT, copy)
    result = run(SCRIPT, "status", home=server["home"], config=server["config"])
    assert "HyperNix T1 API" in result.output


def test_the_hand_off_happens_once(server, monkeypatch):
    _venv_script(server, 'echo "server copy"\n')
    monkeypatch.setenv("HNX_T1_DELEGATED", "1")
    result = run(SCRIPT, "status", home=server["home"], config=server["config"])
    assert "server copy" not in result.output
    assert "HyperNix T1 API" in result.output


def test_a_newer_copy_keeps_running_and_says_the_server_is_behind(server, tmp_path):
    _venv_script(server, 'echo "server copy"\n')
    script = _installed_copy(tmp_path, "0.72.7")
    result = run(script, "status", home=server["home"], config=server["config"])
    assert "server copy" not in result.output
    assert "older than the 0.72.7 this command came with" in result.output


def test_without_a_private_venv_nothing_is_handed_off_to(tmp_path):
    config = tmp_path / "t1api"
    config.mkdir()
    result = run(SCRIPT, "status", home=tmp_path, config=config)
    assert "HyperNix T1 API" in result.output


def _status_with_path_copy(server, tmp_path, path_copy_text: str | None, *, link: bool = False):
    venv_copy = server["config"] / "venv" / "bin" / "hypernix-t1"
    shutil.copy(SCRIPT, venv_copy)
    venv_copy.chmod(0o755)
    on_path = tmp_path / "pathbin"
    on_path.mkdir()
    first = on_path / "hypernix-t1"
    if link:
        first.symlink_to(venv_copy)
    else:
        first.write_text(path_copy_text or "", encoding="utf-8")
        first.chmod(0o755)
    result = subprocess.run(
        [BASH, str(venv_copy), "status"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
        env={**os.environ, "HOME": str(server["home"]), "T1_CONFIG_DIR": str(server["config"]),
             "NO_COLOR": "1", "PYTHONPATH": str(REPO_ROOT / "src"),
             "PATH": f"{on_path}{os.pathsep}{os.environ.get('PATH', '')}"},
    )
    return first, venv_copy, result.stdout + result.stderr


def test_status_says_when_an_old_copy_on_path_runs_instead(server, tmp_path):
    """Seen on a real server: two copies on PATH from before the hand-off,
    one in a uv-managed Python, both running instead of the venv's."""
    first, venv_copy, out = _status_with_path_copy(
        server, tmp_path, "#!/usr/bin/env bash\necho old copy\n")
    assert f"{first} is an older hypernix-t1 than the server's" in out
    assert f"ln -sf {venv_copy} {first}" in out


def test_no_word_about_a_copy_that_hands_over(server, tmp_path):
    _first, _venv, out = _status_with_path_copy(
        server, tmp_path, SCRIPT.read_text(encoding="utf-8"))
    assert "older hypernix-t1 than the server's" not in out


def test_no_word_about_a_link_to_the_servers_copy(server, tmp_path):
    _first, _venv, out = _status_with_path_copy(server, tmp_path, None, link=True)
    assert "older hypernix-t1 than the server's" not in out
