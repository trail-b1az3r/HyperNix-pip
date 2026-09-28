"""Tests for hypernix.system.pathfix.

Everything here runs against a temporary HOME and an explicit profile path
— no test may ever touch the real ``~/.bashrc``. The autoconfigure tests in
particular assert the *refusals*, since a wrong refusal is a papercut and a
wrong write edits somebody's shell startup file.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hypernix.system import pathfix


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("HOME", str(home))
    # Windows resolves home from USERPROFILE; without this a test could
    # write into the real user's profile directory.
    monkeypatch.setenv("USERPROFILE", str(home))
    # Every opt-out and CI marker cleared, so a test that expects a write
    # gets one and a test that expects a refusal names its own reason.
    for var in (*pathfix.OPT_OUT_ENV_VARS, *pathfix.CI_ENV_VARS,
                "VIRTUAL_ENV", "CONDA_PREFIX", "ZDOTDIR", "HYPERNIX_PATH_SHELL"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def not_isolated(monkeypatch):
    """Pretend we're on a system Python, not in a venv."""
    monkeypatch.setattr(pathfix, "in_isolated_env", lambda: False)


# ---------------------------------------------------------------------------
# PATH inspection
# ---------------------------------------------------------------------------


def test_path_entries_drops_empty_segments():
    value = os.pathsep.join(["/a", "", "/b"])
    # Compared as Path, not str: on Windows `str(Path("/a"))` comes back
    # with a backslash, and what this asserts is that the empty segment is
    # dropped — not how the host spells a separator.
    assert pathfix.path_entries(value) == [Path("/a"), Path("/b")]


def test_is_on_path_matches_resolved_location(tmp_path):
    real = tmp_path / "bin"
    real.mkdir()
    # A trailing separator and a "." component must not make this look like
    # a different directory.
    noisy = f"{real}{os.sep}.{os.sep}"
    assert pathfix.is_on_path(real, path_value=noisy)


def test_is_on_path_follows_symlinks(tmp_path):
    real = tmp_path / "real-bin"
    real.mkdir()
    link = tmp_path / "link-bin"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/filesystem")
    assert pathfix.is_on_path(link, path_value=str(real))


def test_is_on_path_false_for_unrelated_dir(tmp_path):
    assert not pathfix.is_on_path(tmp_path / "nope", path_value=str(tmp_path))


def test_candidate_scripts_dirs_are_unique_and_exist():
    dirs = pathfix.candidate_scripts_dirs()
    assert all(d.is_dir() for d in dirs)
    normalized = [os.path.normcase(str(d.resolve())) for d in dirs]
    assert len(normalized) == len(set(normalized))


def test_scripts_dir_returns_a_path():
    assert isinstance(pathfix.scripts_dir(), Path)


# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shell_path,expected", [
    ("/bin/bash", "bash"),
    ("/usr/local/bin/bash", "bash"),
    ("/bin/zsh", "zsh"),
    ("/usr/bin/fish", "fish"),
    ("/bin/ksh", "sh"),
])
def test_detect_shell_from_shell_env(monkeypatch, shell_path, expected):
    """Passing os_name rather than patching ``os.name``: pathlib picks
    PosixPath vs WindowsPath from that attribute, so faking it on Windows
    breaks every later ``Path(...)`` — including pytest's own failure
    formatting, which turns one bad assert into a session-wide
    INTERNALERROR."""
    monkeypatch.delenv("HYPERNIX_PATH_SHELL", raising=False)
    monkeypatch.setenv("SHELL", shell_path)
    assert pathfix.detect_shell(os_name="posix") == expected


def test_detect_shell_is_powershell_on_windows(monkeypatch):
    monkeypatch.delenv("HYPERNIX_PATH_SHELL", raising=False)
    monkeypatch.setenv("SHELL", "/bin/bash")
    assert pathfix.detect_shell(os_name="nt") == "powershell"


def test_detect_shell_env_override_wins(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("HYPERNIX_PATH_SHELL", "fish")
    assert pathfix.detect_shell() == "fish"


def test_profile_for_zsh_respects_zdotdir(fake_home, monkeypatch, tmp_path):
    zdot = tmp_path / "zdot"
    monkeypatch.setenv("ZDOTDIR", str(zdot))
    assert pathfix.profile_for_shell("zsh") == zdot / ".zshrc"


def test_profile_for_fish_is_its_own_conf_d_file(fake_home):
    assert pathfix.profile_for_shell("fish") == fake_home / ".config/fish/conf.d/hypernix.fish"


def test_profile_for_unknown_shell_is_none():
    assert pathfix.profile_for_shell("nushell") is None


# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------


def test_posix_snippet_guards_against_duplicate_entries():
    snippet = pathfix.snippet_for_shell(Path("/opt/bin"), "bash")
    assert 'case ":$PATH:" in' in snippet
    assert '*":/opt/bin:"*' in snippet
    assert 'export PATH="/opt/bin:$PATH"' in snippet


def test_fish_snippet_uses_fish_add_path():
    assert pathfix.snippet_for_shell(Path("/opt/bin"), "fish") == 'fish_add_path "/opt/bin"'


def test_powershell_snippet_uses_env_path():
    snippet = pathfix.snippet_for_shell(Path(r"C:\\Scripts"), "powershell")
    assert snippet.startswith("$env:PATH =")


def test_session_hint_is_a_single_export():
    hint = pathfix.session_hint(Path("/opt/bin"), "bash")
    assert hint == 'export PATH="/opt/bin:$PATH"'
    assert "\n" not in hint


# ---------------------------------------------------------------------------
# ensure_on_path / remove_from_path
# ---------------------------------------------------------------------------


def test_ensure_on_path_is_a_noop_when_already_on_path(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", str(scripts))
    result = pathfix.ensure_on_path(directory=scripts, shell="bash",
                                    profile=tmp_path / "rc")
    assert result.status == "already-on-path"
    assert not result.changed
    assert not (tmp_path / "rc").exists()


def test_ensure_on_path_writes_a_marked_block(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"
    profile.write_text("# my shell\nalias ll='ls -l'\n")

    result = pathfix.ensure_on_path(directory=scripts, shell="bash", profile=profile)

    assert result.changed and result.status == "written"
    text = profile.read_text()
    assert pathfix.BLOCK_START in text and pathfix.BLOCK_END in text
    # as_posix: the block is a bash snippet, so the path in it uses
    # forward slashes even when a Windows host wrote the file.
    assert scripts.as_posix() in text
    # The person's own content survives untouched.
    assert "alias ll='ls -l'" in text


def test_ensure_on_path_check_mode_writes_nothing(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"
    result = pathfix.ensure_on_path(apply=False, directory=scripts,
                                    shell="bash", profile=profile)
    assert result.status == "would-write"
    assert not result.changed
    assert not profile.exists()


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish", "powershell", "sh"])
def test_ensure_on_path_is_idempotent(tmp_path, monkeypatch, shell):
    """Every shell, because recognising our own block is per-shell.

    The line in the file is spelled the way that shell needs it. A
    recogniser that looks for the platform's own spelling of the path
    instead misses the block it just wrote, and rewrites it on every
    run -- which on Windows was every ``hypernix`` invocation.
    """
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"

    pathfix.ensure_on_path(directory=scripts, shell=shell, profile=profile)
    first = profile.read_text()
    second_result = pathfix.ensure_on_path(directory=scripts, shell=shell, profile=profile)

    assert second_result.status == "already-configured"
    assert not second_result.changed
    assert profile.read_text() == first
    assert first.count(pathfix.BLOCK_START) == 1


def test_ensure_on_path_replaces_a_block_pointing_elsewhere(tmp_path, monkeypatch):
    """A second Python install must not leave two competing blocks."""
    old_scripts = tmp_path / "old-bin"
    new_scripts = tmp_path / "new-bin"
    old_scripts.mkdir()
    new_scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"

    pathfix.ensure_on_path(directory=old_scripts, shell="bash", profile=profile)
    pathfix.ensure_on_path(directory=new_scripts, shell="bash", profile=profile)

    text = profile.read_text()
    assert text.count(pathfix.BLOCK_START) == 1
    assert new_scripts.as_posix() in text
    assert old_scripts.as_posix() not in text


def test_ensure_on_path_creates_missing_parent_dirs(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "deep" / "conf.d" / "hypernix.fish"

    result = pathfix.ensure_on_path(directory=scripts, shell="fish", profile=profile)

    assert result.changed
    assert "fish_add_path" in profile.read_text()


def test_ensure_on_path_reports_unwritable_profile(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    # A directory where the file should be: writing it always fails.
    profile = tmp_path / "rc"
    profile.mkdir()

    result = pathfix.ensure_on_path(directory=scripts, shell="bash", profile=profile)

    assert result.status in ("unwritable", "unreadable")
    assert not result.changed
    # The message still tells the person what to do by hand -- and the line
    # it hands them is the bash one, so the path in it is spelled the way
    # bash needs rather than the way this platform writes paths.
    assert pathfix.snippet_for_shell(scripts, "bash") in result.message


def test_ensure_on_path_force_writes_even_when_on_path(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", str(scripts))
    profile = tmp_path / "rc"

    result = pathfix.ensure_on_path(directory=scripts, shell="bash",
                                    profile=profile, force=True)

    assert result.changed
    assert scripts.as_posix() in profile.read_text()


def test_ensure_on_path_unknown_shell_explains_instead_of_writing(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    result = pathfix.ensure_on_path(directory=scripts, shell="nushell")
    assert result.status == "no-profile"
    assert not result.changed
    # The message has to carry the line the person is being asked to add,
    # spelled the way that shell needs it -- which on Windows is not
    # ``str(scripts)``, because a POSIX-ish shell wants forward slashes.
    assert pathfix.snippet_for_shell(scripts, "nushell") in result.message


def test_remove_from_path_strips_only_the_block(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"
    profile.write_text("# top\nalias ll='ls -l'\n")

    pathfix.ensure_on_path(directory=scripts, shell="bash", profile=profile)
    result = pathfix.remove_from_path(shell="bash", profile=profile)

    assert result.changed and result.status == "removed"
    text = profile.read_text()
    assert pathfix.BLOCK_START not in text
    assert pathfix.BLOCK_END not in text
    assert "alias ll='ls -l'" in text
    assert "# top" in text


def test_remove_from_path_when_nothing_to_remove(tmp_path):
    profile = tmp_path / "rc"
    profile.write_text("# nothing here\n")
    result = pathfix.remove_from_path(shell="bash", profile=profile)
    assert result.status == "not-configured"
    assert not result.changed
    assert profile.read_text() == "# nothing here\n"


def test_remove_from_path_handles_a_truncated_block(tmp_path):
    """A half-written block must not make the file un-fixable."""
    profile = tmp_path / "rc"
    profile.write_text(f"# top\n{pathfix.BLOCK_START}\nexport PATH=broken\n")

    result = pathfix.remove_from_path(shell="bash", profile=profile)

    assert result.changed
    assert pathfix.BLOCK_START not in profile.read_text()
    assert "# top" in profile.read_text()


def test_round_trip_leaves_the_file_semantically_unchanged(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"
    original = "# top\nexport EDITOR=vim\n"
    profile.write_text(original)

    pathfix.ensure_on_path(directory=scripts, shell="bash", profile=profile)
    pathfix.remove_from_path(shell="bash", profile=profile)

    assert profile.read_text().strip() == original.strip()


# ---------------------------------------------------------------------------
# maybe_autoconfigure — the automatic path, and everything that stops it
# ---------------------------------------------------------------------------


def test_autoconfigure_noop_when_already_on_path(fake_home, monkeypatch, tmp_path):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setenv("PATH", str(scripts))

    said = []
    result = pathfix.maybe_autoconfigure(echo=said.append)

    assert result.status == "already-on-path"
    assert said == []


@pytest.mark.parametrize("var", pathfix.OPT_OUT_ENV_VARS)
def test_autoconfigure_respects_opt_out(fake_home, not_isolated, monkeypatch, tmp_path, var):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv(var, "1")

    said = []
    result = pathfix.maybe_autoconfigure(echo=said.append)

    assert result.status == "skipped"
    assert var in result.message
    assert said == []
    assert not _expected_profile().exists()


@pytest.mark.parametrize("var", ["CI", "GITHUB_ACTIONS"])
def test_autoconfigure_never_edits_a_profile_in_ci(fake_home, not_isolated, monkeypatch, tmp_path, var):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv(var, "true")

    result = pathfix.maybe_autoconfigure(echo=lambda _m: None)

    assert result.status == "skipped"
    assert not _expected_profile().exists()


def test_autoconfigure_refuses_inside_a_virtualenv(fake_home, monkeypatch, tmp_path):
    """The case where a "helpful" edit does real damage: a venv's scripts
    directory baked into ~/.bashrc leaks that env into every shell."""
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))

    result = pathfix.maybe_autoconfigure(echo=lambda _m: None)

    assert result.status == "skipped"
    assert "virtualenv" in result.message
    assert not _expected_profile().exists()


def _expected_profile() -> Path:
    """The startup file this platform's shell actually reads.

    Asserting on a hardcoded ``~/.bashrc`` was a Linux assumption baked
    into a cross-platform test: on macOS the module correctly picks
    ``~/.bash_profile`` (a Terminal.app shell is a login shell and never
    reads ``.bashrc``), and on Windows a PowerShell profile. Asking the
    module keeps the test checking *that a write happened where this
    platform reads*, which is the actual property.
    """
    return pathfix.profile_for_shell(pathfix.detect_shell())


def test_autoconfigure_writes_once_and_announces_it(fake_home, not_isolated, monkeypatch, tmp_path):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("SHELL", "/bin/bash")

    said = []
    result = pathfix.maybe_autoconfigure(echo=said.append)

    assert result.changed and result.status == "written"
    # Which spelling of the directory lands in the file depends on the shell
    # this platform detects -- forward slashes for a POSIX shell, native
    # separators for PowerShell -- so ask the module rather than guessing.
    written = _expected_profile().read_text()
    assert pathfix.snippet_for_shell(scripts, pathfix.detect_shell()) in written
    # It must say what it did, and how to opt out.
    joined = "\n".join(said)
    assert str(scripts) in joined
    assert pathfix.OPT_OUT_ENV_VARS[0] in joined


def test_autoconfigure_does_not_retry_after_the_first_attempt(fake_home, not_isolated, monkeypatch, tmp_path):
    """Someone who deletes the block should not get it silently rewritten."""
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("SHELL", "/bin/bash")

    assert pathfix.maybe_autoconfigure(echo=lambda _m: None).changed
    profile = _expected_profile()
    pathfix.remove_from_path(profile=profile)

    said = []
    second = pathfix.maybe_autoconfigure(echo=said.append)

    assert second.status == "skipped"
    assert "already attempted" in second.message
    assert said == []
    assert pathfix.BLOCK_START not in profile.read_text()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_check_reports_without_writing(fake_home, capsys, tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"

    rc = pathfix.cli_main(["--check", "--shell", "bash", "--profile", str(profile)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "on PATH           : no" in out
    assert not profile.exists()


def test_cli_print_emits_only_the_snippet(fake_home, capsys, tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)

    assert pathfix.cli_main(["--print", "--shell", "fish"]) == 0
    assert (
        capsys.readouterr().out.strip()
        == f'fish_add_path "{scripts.as_posix()}"'
    )


def test_cli_apply_then_undo(fake_home, capsys, tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setattr(pathfix, "in_isolated_env", lambda: False)
    monkeypatch.setenv("PATH", "/usr/bin")
    profile = tmp_path / "rc"

    assert pathfix.cli_main(["--apply", "--shell", "bash", "--profile", str(profile)]) == 0
    assert pathfix.BLOCK_START in profile.read_text()

    assert pathfix.cli_main(["--undo", "--shell", "bash", "--profile", str(profile)]) == 0
    assert pathfix.BLOCK_START not in profile.read_text()


def test_cli_apply_refuses_inside_a_virtualenv(fake_home, capsys, tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(pathfix, "scripts_dir", lambda: scripts)
    monkeypatch.setattr(pathfix, "in_isolated_env", lambda: True)
    monkeypatch.setenv("PATH", "/usr/bin")

    rc = pathfix.cli_main(["--apply", "--shell", "bash"])

    assert rc == 1
    assert "virtualenv" in capsys.readouterr().err
    assert not _expected_profile().exists()


def test_cli_help_exits_zero(capsys):
    assert pathfix.cli_main(["--help"]) == 0
    assert "hypernix path" in capsys.readouterr().out


def test_cli_rejects_unknown_flags(capsys):
    assert pathfix.cli_main(["--nonsense"]) == 2
