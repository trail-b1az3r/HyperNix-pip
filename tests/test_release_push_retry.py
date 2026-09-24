"""The release's "Commit version bump" step, run for real against git.

0.72.6 built and tested for ten minutes and then failed here. main moved
during the build (the stats bot committed), so the first push was
rejected, and the retry's `git rebase` refused to start: the install and
the build rewrite the tracked `src/hypernix.egg-info`, and a rebase will
not run over unstaged changes. The script reported that as "the version
bump conflicts with main -- another release may be in flight", which it
was not.

These run the step's own shell, taken from the workflow file, against a
bare "origin" and a runner checkout, so the retry path is exercised the
way a release exercises it, rather than only the first push that nearly
always succeeds.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "public-release.yml"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None,
                                reason="needs git and bash")

#: Every file the step stages, so `git add` finds them all.
STAGED = [
    "pyproject.toml",
    "setup.cfg",
    "src/hypernix/__init__.py",
    "install-t1.sh",
    "src/hypernix/interfaces/hyped_pro_app/package.json",
    "src/hypernix/interfaces/hyped_pro_app/src/app.ts",
    "src/hypernix/monitoring/tvtop_max_app/package.json",
    "src/hypernix/monitoring/tvtop_max_app/src/app.ts",
    "wiki/Release-Timeline.md",
]


def step_script(name: str = "Commit version bump") -> str:
    """The `run:` block of one step, as the runner's bash receives it."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"- name: {name}")
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    body: list[str] = []
    for line in lines[run + 1:]:
        if line.strip().startswith("- name: ") or (line.strip() and not line.startswith(" " * 10)):
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)).strip() + "\n"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """A bare origin, the runner's checkout of it, and somebody else's."""
    origin = tmp_path / "origin.git"
    git("init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)

    seed = tmp_path / "seed"
    git("clone", "-q", str(origin), str(seed), cwd=tmp_path)
    for name in (*STAGED, "src/hypernix.egg-info/PKG-INFO", "docs/stats.json"):
        path = seed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name} at 0.72.6.rc3\n", encoding="utf-8")
    for who in (seed,):
        git("config", "user.email", "t@example.invalid", cwd=who)
        git("config", "user.name", "t", cwd=who)
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "tree", cwd=seed)
    git("push", "-q", "origin", "main", cwd=seed)

    runner = tmp_path / "runner"
    git("clone", "-q", str(origin), str(runner), cwd=tmp_path)
    other = tmp_path / "other"
    git("clone", "-q", str(origin), str(other), cwd=tmp_path)
    for who in (other,):
        git("config", "user.email", "bot@example.invalid", cwd=who)
        git("config", "user.name", "bot", cwd=who)
    return origin, runner, other


def run_step(runner: Path) -> subprocess.CompletedProcess:
    script = runner.parent / "step.sh"
    script.write_text(step_script(), encoding="utf-8")
    env = {**os.environ, "BRANCH": "main", "VERSION": "0.72.6"}
    return subprocess.run(["bash", "-e", str(script)], cwd=runner, env=env,
                          capture_output=True, text=True, timeout=120)


def release_leftovers(runner: Path) -> None:
    """What the bump and the build leave behind by the time this step runs."""
    (runner / "wiki/Release-Timeline.md").write_text("0.72.6 released\n", encoding="utf-8")
    # Rewritten by `pip install -e` and `python -m build`, tracked, never staged.
    (runner / "src/hypernix.egg-info/PKG-INFO").write_text("Version: 0.72.6\n", encoding="utf-8")


def someone_else_pushes(other: Path, name: str, text: str) -> str:
    (other / name).write_text(text, encoding="utf-8")
    git("commit", "-q", "-am", f"meanwhile: {name}", cwd=other)
    git("push", "-q", "origin", "main", cwd=other)
    return git("rev-parse", "HEAD", cwd=other)


def test_the_failure_main_moved_mid_release_with_a_dirty_tree(repos):
    """The 0.72.6 run: the stats bot committed while the release built."""
    origin, runner, other = repos
    release_leftovers(runner)
    bot = someone_else_pushes(other, "docs/stats.json", "stats after the merge\n")

    done = run_step(runner)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "push 1 rejected" in done.stdout + done.stderr

    history = git("log", "--format=%s", "main", cwd=origin).splitlines()
    assert history[0] == "hypernix v0.72.6 [skip ci]"
    assert git("rev-parse", "main~1", cwd=origin) == bot, "the bump sits on top of what arrived"
    # The build's leftovers were put back, and never pushed.
    assert (runner / "src/hypernix.egg-info/PKG-INFO").read_text() == "Version: 0.72.6\n"
    assert "rc3" in git("show", "main:src/hypernix.egg-info/PKG-INFO", cwd=origin)


def test_a_first_push_that_lands_needs_no_retry(repos):
    origin, runner, _other = repos
    release_leftovers(runner)
    done = run_step(runner)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "rejected" not in done.stdout + done.stderr
    assert git("log", "-1", "--format=%s", "main", cwd=origin) == "hypernix v0.72.6 [skip ci]"


def test_a_real_conflict_still_stops_and_names_the_file(repos):
    """Somebody else bumped the same file: not something to settle by script."""
    origin, runner, other = repos
    release_leftovers(runner)
    theirs = someone_else_pushes(other, "wiki/Release-Timeline.md", "0.72.7 released\n")

    done = run_step(runner)
    assert done.returncode == 1
    out = done.stdout + done.stderr
    assert "conflicts with main in: wiki/Release-Timeline.md" in out
    assert git("rev-parse", "main", cwd=origin) == theirs, "nothing was pushed over it"


def test_nothing_to_commit_is_not_an_error(repos):
    """A tree prepared with the version and no timeline change."""
    _origin, runner, _other = repos
    done = run_step(runner)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "no version change to commit" in done.stdout
