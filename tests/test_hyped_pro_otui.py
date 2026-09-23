"""hyped-pro as an OpenTUI app, and hyped-plus as the readline one.

The TypeScript side has its own tests (``bun test`` in
src/hypernix/interfaces/hyped_pro_app); the last test here runs them when
bun is installed. Everything else checks the Python launcher and the
packaging, which decide whether the app ever starts.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import hypernix
from hypernix.interfaces import hyped_pro_otui as launcher

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "hypernix" / "interfaces" / "hyped_pro_app"


_FAKE_BUN = """
import json, os, sys
version, install_ok = {version!r}, {install_ok!r}
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print(version)
    sys.exit(0)
if args[:1] == ["install"]:
    if not install_ok:
        sys.exit(3)
    marker = os.path.join("node_modules", "@opentui", "core")
    os.makedirs(marker, exist_ok=True)
    with open(os.path.join(marker, "package.json"), "w") as f:
        f.write("{{}}")
    with open(".install-args", "w") as f:
        f.write(" ".join(args))
    sys.exit(0)
sys.exit(9)
"""


def _fake_bun(path: Path, version: str = "1.3.11", *, install_ok: bool = True) -> Path:
    """A stand-in for bun that answers --version and fakes `bun install`.

    Python, so it runs everywhere. On POSIX it is the script itself with a
    shebang; on Windows a script cannot be executed directly, so it is a
    .cmd beside it — which is also what makes `shutil.which("bun")` find
    it there, through PATHEXT.
    """
    body = _FAKE_BUN.format(version=version, install_ok=install_ok)
    if sys.platform == "win32":
        script = path.with_name(path.name + "-impl.py")
        script.write_text(body, encoding="utf-8")
        shim = path.with_name(path.name + ".cmd")
        shim.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return shim
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _app_copy(tmp_path: Path) -> Path:
    source = tmp_path / "hyped_pro_app"
    source.mkdir()
    for name in ("package.json", "bun.lock", "tsconfig.json"):
        shutil.copy2(APP / name, source / name)
    shutil.copytree(APP / "src", source / "src")
    shutil.copytree(APP / "test", source / "test")
    return source


# -- the look ---------------------------------------------------------------


def _css_palette() -> dict[str, str]:
    css = (REPO / "docs" / "src" / "index.css").read_text(encoding="utf-8")
    root = css[css.index(":root") : css.index("}", css.index(":root"))]
    return {name: value.lower() for name, value in re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", root)}


def _ts_palette() -> dict[str, str]:
    theme = (APP / "src" / "theme.ts").read_text(encoding="utf-8")
    body = theme[theme.index("export const palette") : theme.index("} as const")]
    found = dict(re.findall(r'(\w+):\s*"(#[0-9a-fA-F]{6})"', body))
    # camelCase in TypeScript, kebab-case in CSS.
    return {re.sub(r"(?<=[a-z])([A-Z0-9])", r"-\1", k).lower(): v.lower() for k, v in found.items()}


def test_palette_is_the_sites_palette() -> None:
    css = _css_palette()
    ts = _ts_palette()
    # Every colour the site defines is in the app, with the same value.
    missing = {name: value for name, value in css.items() if ts.get(name) != value}
    assert missing == {}
    assert set(ts) == set(css)


def test_palette_parsers_are_not_vacuous() -> None:
    assert len(_css_palette()) >= 10
    assert _ts_palette()["bg"] == "#0d0d0d"
    assert _ts_palette()["accent"] == "#c8192e"


def test_app_is_opentui_not_readline() -> None:
    package = (APP / "package.json").read_text(encoding="utf-8")
    assert '"@opentui/core"' in package
    sources = "\n".join(p.read_text(encoding="utf-8") for p in (APP / "src").glob("*.ts"))
    assert "createCliRenderer" in sources
    # Imports, not prose: a comment may mention the readline TUI.
    imports = re.findall(r'from\s+"([^"]+)"', sources)
    assert "@opentui/core" in imports
    assert not [name for name in imports if "readline" in name]


# -- which command runs what --------------------------------------------------


def _scripts() -> dict[str, str]:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]


def test_hyped_pro_runs_the_opentui_app() -> None:
    assert _scripts()["hyped-pro"] == "hypernix.interfaces.hyped_pro_otui:cli_main"


def test_hyped_plus_runs_the_previous_tui_and_nothing_else_does() -> None:
    scripts = _scripts()
    assert scripts["hyped-plus"] == "hypernix.interfaces.hyped_pro:cli_main"
    others = [name for name, target in scripts.items() if target == "hypernix.interfaces.hyped_pro:cli_main"]
    assert others == ["hyped-plus"]


def test_launcher_is_a_categorised_module() -> None:
    assert "hyped_pro_otui" in hypernix.MODULE_CATEGORIES["interfaces"]
    assert hypernix.CATEGORY_OF["hyped_pro_otui"] == "interfaces"


def test_every_app_source_is_packaged() -> None:
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["hypernix"]
    pkg = REPO / "src" / "hypernix"
    packaged = {p for pattern in patterns for p in pkg.glob(pattern)}
    needed = [APP / "package.json", APP / "bun.lock", APP / "tsconfig.json", *(APP / "src").glob("*.ts")]
    assert [str(p.relative_to(pkg)) for p in needed if p not in packaged] == []
    assert not any("node_modules" in str(p) for p in packaged)


def test_sdist_never_ships_node_modules() -> None:
    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    assert "prune src/hypernix/interfaces/hyped_pro_app/node_modules" in manifest
    assert "recursive-include src/hypernix/interfaces/hyped_pro_app" in manifest


def test_lockfile_pins_the_same_opentui() -> None:
    wanted = re.search(r'"@opentui/core":\s*"([^"]+)"', (APP / "package.json").read_text(encoding="utf-8")).group(1)
    assert f"@opentui/core@{wanted}" in (APP / "bun.lock").read_text(encoding="utf-8")


# -- finding bun --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1.3.11\n", (1, 3, 11)), ("bun 1.4.0-canary.2", (1, 4, 0)), ("", None), ("nope", None)],
)
def test_parse_version(text: str, expected: tuple[int, int, int] | None) -> None:
    assert launcher.parse_version(text) == expected


def test_candidates_explicit_first_then_path_then_installer(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_bun(bin_dir / "bun")
    env = {"HYPED_PRO_BUN": "/opt/bun", "PATH": str(bin_dir), "HOME": str(tmp_path / "home")}
    candidates = launcher.bun_candidates(env)
    assert candidates[0] == "/opt/bun"
    assert Path(candidates[1]).parent == bin_dir
    assert Path(candidates[1]).stem.lower() == "bun"
    assert candidates[2:] == [str(tmp_path / "home" / ".bun" / "bin" / launcher.BUN_EXECUTABLE)]


def test_candidates_are_deduplicated(tmp_path: Path) -> None:
    bun = _fake_bun(tmp_path / "bun")
    on_path = shutil.which("bun", path=str(tmp_path))
    assert on_path is not None
    env = {"HYPED_PRO_BUN": on_path, "PATH": str(tmp_path), "HOME": str(tmp_path)}
    assert launcher.bun_candidates(env).count(on_path) == 1
    assert Path(on_path).parent == bun.parent


def test_bun_install_env_is_honoured(tmp_path: Path) -> None:
    env = {"PATH": "", "HOME": str(tmp_path), "BUN_INSTALL": "/custom/bun"}
    assert launcher.bun_candidates(env)[-1] == str(Path("/custom/bun") / "bin" / launcher.BUN_EXECUTABLE)


def test_find_bun_takes_the_first_new_enough(tmp_path: Path) -> None:
    old = _fake_bun(tmp_path / "old-bun", "1.2.9")
    new = _fake_bun(tmp_path / "bun", "1.3.0")
    env = {"HYPED_PRO_BUN": str(old), "PATH": str(tmp_path), "HOME": str(tmp_path)}
    assert Path(launcher.find_bun(env)).resolve() == new.resolve()


def test_find_bun_too_old_says_so_and_names_hyped_plus(tmp_path: Path) -> None:
    old = _fake_bun(tmp_path / "bun", "1.2.21")
    env = {"PATH": str(tmp_path), "HOME": str(tmp_path / "nohome")}
    with pytest.raises(launcher.LaunchError) as caught:
        launcher.find_bun(env)
    assert caught.value.exit_code == 127
    # Case-folded: on Windows, which() spells the extension as PATHEXT does.
    assert "1.3.0" in str(caught.value) and str(old).lower() in str(caught.value).lower()
    assert "hyped-plus" in str(caught.value)


def test_find_bun_missing_explains_how_to_get_it(tmp_path: Path) -> None:
    env = {"PATH": str(tmp_path), "HOME": str(tmp_path)}
    with pytest.raises(launcher.LaunchError) as caught:
        launcher.find_bun(env)
    assert caught.value.exit_code == 127
    assert "bun.sh/install" in str(caught.value)
    assert "hyped-plus" in str(caught.value)


def test_find_bun_skips_a_broken_binary(tmp_path: Path) -> None:
    broken = _fake_bun(tmp_path / "broken", "not a version")
    good = _fake_bun(tmp_path / "bun")
    env = {"HYPED_PRO_BUN": str(broken), "PATH": str(tmp_path), "HOME": str(tmp_path)}
    assert Path(launcher.find_bun(env)).resolve() == good.resolve()


# -- getting the app ready ------------------------------------------------------


def test_a_ready_app_runs_where_it_is(tmp_path: Path) -> None:
    source = _app_copy(tmp_path)
    marker = source / launcher.DEPENDENCY_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    bun = tmp_path / "no-such-bun"  # never called
    assert launcher.prepare_app(source, bun=str(bun)) == source


def test_a_writable_app_is_installed_in_place(tmp_path: Path) -> None:
    source = _app_copy(tmp_path)
    bun = _fake_bun(tmp_path / "bun")
    assert launcher.prepare_app(source, bun=str(bun), env={"HOME": str(tmp_path)}) == source
    assert (source / launcher.DEPENDENCY_MARKER).is_file()
    # Only what running needs: no type checker.
    assert (source / ".install-args").read_text(encoding="utf-8").split() == ["install", "--production"]


def test_a_read_only_app_is_copied_home_by_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _app_copy(tmp_path)
    (source / "node_modules" / "junk").mkdir(parents=True)
    bun = _fake_bun(tmp_path / "bun")
    real_access = os.access
    monkeypatch.setattr(
        launcher.os, "access", lambda p, m: False if Path(p) == source and m == os.W_OK else real_access(p, m)
    )
    home = tmp_path / "hx"
    target = launcher.prepare_app(source, bun=str(bun), env={"HYPERNIX_HOME": str(home)})
    assert target == home / "hyped-pro" / launcher.app_version(source)
    assert (target / launcher.DEPENDENCY_MARKER).is_file()
    assert sorted(p.name for p in target.iterdir()) == sorted(
        ["package.json", "bun.lock", "tsconfig.json", "src", "node_modules", ".install-args"]
    )
    assert not (target / "node_modules" / "junk").exists()
    assert not (source / launcher.DEPENDENCY_MARKER).exists()

    # Second run: already installed, bun install is not run again, but the
    # sources are refreshed.
    (target / ".install-args").unlink()
    (source / "src" / "theme.ts").write_text("// changed\n")
    assert launcher.prepare_app(source, bun=str(bun), env={"HYPERNIX_HOME": str(home)}) == target
    assert not (target / ".install-args").exists()
    assert (target / "src" / "theme.ts").read_text(encoding="utf-8") == "// changed\n"


def test_a_failed_install_is_an_error_not_a_crash_later(tmp_path: Path) -> None:
    source = _app_copy(tmp_path)
    bun = _fake_bun(tmp_path / "bun", install_ok=False)
    with pytest.raises(launcher.LaunchError) as caught:
        launcher.prepare_app(source, bun=str(bun))
    assert "bun install" in str(caught.value)
    assert "hyped-plus" in str(caught.value)


def test_a_missing_app_says_reinstall(tmp_path: Path) -> None:
    with pytest.raises(launcher.LaunchError, match="reinstall"):
        launcher.prepare_app(tmp_path / "nothing", bun="bun")


def test_app_version_matches_the_package() -> None:
    assert launcher.app_version() == launcher.semver_of(hypernix.__version__)


@pytest.mark.parametrize("version, expected", [
    ("0.72.5.post17", "0.72.5-post17"),
    ("0.72.6.rc1", "0.72.6-rc1"),
    ("0.72.6rc1", "0.72.6-rc1"),
    ("0.72.6-rc1", "0.72.6-rc1"),
    ("0.72.6", "0.72.6"),
    ("0.72.6a2", "0.72.6-a2"),
    ("0.72.6postr1", "0.72.6-post1"),
    ("0.70.6-2", "0.70.6-post2"),
    ("v0.72.6", "0.72.6"),
    ("not a version", "not a version"),
])
def test_semver_of_spells_every_release_input_as_semver(version: str, expected: str) -> None:
    assert launcher.semver_of(version) == expected


def test_sync_app_version_writes_both_places(tmp_path: Path) -> None:
    app = tmp_path / "app"
    (app / "src").mkdir(parents=True)
    shutil.copy(APP / "package.json", app / "package.json")
    shutil.copy(APP / "src" / "app.ts", app / "src" / "app.ts")
    assert launcher.sync_app_version("0.72.6.rc1", app) == "0.72.6-rc1"
    assert launcher.app_version(app) == "0.72.6-rc1"
    assert 'export const VERSION = "0.72.6-rc1"' in (app / "src" / "app.ts").read_text(encoding="utf-8")
    # Nothing else in package.json moved.
    before = json.loads((APP / "package.json").read_text(encoding="utf-8"))
    after = json.loads((app / "package.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "version"} == {
        k: v for k, v in before.items() if k != "version"}


def test_app_and_launcher_agree_on_version() -> None:
    app_ts = (APP / "src" / "app.ts").read_text(encoding="utf-8")
    assert f'export const VERSION = "{launcher.app_version()}"' in app_ts


# -- running --------------------------------------------------------------------


def test_command_runs_the_entry_point_by_absolute_path(tmp_path: Path) -> None:
    command = launcher.command_for("/b/bun", tmp_path, ["--model", "kimi-k3"])
    assert command == ["/b/bun", "run", str(tmp_path / "src" / "index.ts"), "--model", "kimi-k3"]


def test_cli_main_without_bun_exits_127_with_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HYPED_PRO_BUN", raising=False)
    monkeypatch.delenv("BUN_INSTALL", raising=False)
    assert launcher.cli_main([]) == 127
    assert "hyped-plus" in capsys.readouterr().err


def test_cli_main_hands_over_with_the_bridge_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bun = "/opt/bun/bin/bun"
    monkeypatch.setattr(launcher, "find_bun", lambda **_: bun)
    monkeypatch.setenv("HYPED_PRO_PYTHON", "/opt/python3.12")
    monkeypatch.setattr(launcher, "prepare_app", lambda **_: tmp_path)
    seen: dict = {}

    def call(command, env):
        seen["command"], seen["env"] = command, env
        return 5

    monkeypatch.setattr(launcher.subprocess, "call", call)
    assert launcher.cli_main(["--debug", "--model", "x"]) == 5
    assert seen["command"] == [bun, "run", str(tmp_path / "src" / "index.ts"), "--model", "x"]
    assert seen["env"]["HYPED_PRO_PYTHON"] == "/opt/python3.12"


def test_hyped_plus_launcher_is_unchanged_in_what_it_runs() -> None:
    source = (REPO / "src" / "hypernix" / "interfaces" / "hyped_pro.py").read_text(encoding="utf-8")
    assert '"hyped_pro.js"' in source
    assert "hyped_pro_app" not in source


# -- the TypeScript tests -----------------------------------------------------------


def _bun() -> str | None:
    try:
        return launcher.find_bun()
    except launcher.LaunchError:
        return None


@pytest.mark.skipif(_bun() is None, reason="bun 1.3+ not installed")
@pytest.mark.skipif(not (APP / launcher.DEPENDENCY_MARKER).is_file(), reason="run `bun install` in hyped_pro_app")
def test_bun_suite_passes() -> None:
    result = subprocess.run(
        [_bun(), "test"], cwd=APP, capture_output=True, text=True, timeout=300, check=False
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert re.search(r"\b0 fail\b", output), output


def test_python_is_what_runs_the_bridge() -> None:
    # The app spawns the bridge as a module of the interpreter it is given.
    bridge = (APP / "src" / "bridge.ts").read_text(encoding="utf-8")
    assert '"hypernix.interfaces.hyped_pro_bridge", "serve"' in bridge
    assert "HYPED_PRO_PYTHON" in bridge


def test_ctrl_c_goes_through_quit() -> None:
    # OpenTUI's exitOnCtrlC destroys the renderer and nothing else; the
    # bridge's pipes then keep Bun running behind a blank terminal. Found
    # by driving the app in a pty: Ctrl+C left the process alive.
    index = (APP / "src" / "index.ts").read_text(encoding="utf-8")
    app = (APP / "src" / "app.ts").read_text(encoding="utf-8")
    assert "exitOnCtrlC: false" in index
    assert 'process.on("SIGINT", () => app.quit())' in index
    assert re.search(r'key\.ctrl && key\.name === "c"\) \{\s*this\.quit\(\)', app)
    quit_body = app[app.index("  quit(): void {") :]
    assert "this.bridge.close()" in quit_body and "process.exit(0)" in quit_body


def test_the_release_workflow_bumps_and_commits_the_app_version() -> None:
    """A release that bumped the package and not the app shipped
    0.72.6.rc1 with an app still calling itself 0.72.5-post17, and
    failed its own test run over it. The bump must write both files
    and the commit must stage both, or they are written and discarded."""
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows"
                / "public-release.yml").read_text(encoding="utf-8")
    bump = workflow[workflow.index("Bump versions in source"):]
    bump = bump[:bump.index("- name: Lint")]
    assert "sync_app_version(v)" in bump
    assert "src/hypernix/interfaces/hyped_pro_otui.py" in bump
    commit = workflow[workflow.index("- name: Commit version bump"):]
    staged = commit[commit.index("git add"):commit.index("git diff --cached")]
    for path in ("src/hypernix/interfaces/hyped_pro_app/package.json",
                 "src/hypernix/interfaces/hyped_pro_app/src/app.ts"):
        assert path in staged, f"the release bumps {path} and never commits it"
