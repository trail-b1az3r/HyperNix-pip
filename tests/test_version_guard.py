"""`.github/scripts/version_guard.py` — may this release be cut?

The guard exists because `public-release` writes whatever version it is
handed: a dispatch naming an older number silently downgrades main, and
that has happened (v0.72.3.post2 against a tree already at .post4).

It also used to refuse a version *equal* to the tree's, which was wrong
and expensive. Preparing a release means writing the version into the
three source files and adding the changelog heading under it, and both
downstream steps already expect to find their work done -- "Commit
version bump" notices there is nothing to commit, "Tag and push" skips a
tag that exists. Only the guard disagreed, so releases were cut under
numbers invented at dispatch time to get past it: 0.72.4 `post1` and
`post3` have no changelog heading and say nothing about what shipped.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD = REPO_ROOT / ".github" / "scripts" / "version_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("version_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


guard = _load()


@pytest.fixture
def tree(tmp_path):
    """A checkout with a version, a changelog, and one commit."""
    def build(version: str, *, changelog: str | None = None, git: bool = True) -> Path:
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "hypernix"\nversion = "{version}"\n', encoding="utf-8"
        )
        wiki = tmp_path / "wiki"
        wiki.mkdir(exist_ok=True)
        body = changelog if changelog is not None else f"## Changelog\n\n## {version} — notes\n"
        (wiki / "Changelog.md").write_text(body, encoding="utf-8")
        if git:
            def run(*a):
                subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

            run("init", "-q", "-b", "main")
            run("config", "user.email", "t@example.invalid")
            run("config", "user.name", "t")
            run("add", "-A")
            run("commit", "-q", "-m", "tree")
        return tmp_path
    return build


def head_of(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def call(version: str, repo: Path, **kw):
    """Run the guard, returning (exit_code, printed)."""
    import contextlib
    import io
    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        try:
            guard.decide(version, repo=repo, **kw)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buffer.getvalue()


class TestTheSameVersionTheTreeIsPreparedWith:
    """The fix. This was a hard error, and it should never have been."""

    def test_it_is_allowed(self, tree):
        repo = tree("0.72.4.post5")
        code, out = call("0.72.4.post5", repo)
        assert code == 0, out
        assert "already prepared with" in out

    def test_it_does_not_demand_a_postn_suffix(self, tree):
        """The old refusal's advice is what produced the undocumented releases."""
        repo = tree("0.72.4.post5")
        _, out = call("0.72.4.post5", repo)
        assert "postN" not in out

    def test_a_missing_changelog_heading_warns_rather_than_blocks(self, tree):
        """A release with no notes is worth saying out loud, not worth stopping.

        Blocking here would only push people back to inventing a number
        at dispatch time, which is the behaviour that lost the notes in
        the first place.
        """
        repo = tree("0.72.4.post5", changelog="## Changelog\n\n## 0.72.4.post4 — old\n")
        code, out = call("0.72.4.post5", repo)
        assert code == 0, out
        assert "::warning::" in out
        assert "no heading for it" in out

    def test_the_heading_must_be_for_this_version_exactly(self, tree):
        """`## 0.72.4.post5` is not satisfied by `## 0.72.4.post50`."""
        repo = tree("0.72.4.post5", changelog="## Changelog\n\n## 0.72.4.post50 — other\n")
        _, out = call("0.72.4.post5", repo)
        assert "no heading for it" in out


class TestAVersionThatGoesBackwards:
    """Unchanged: this is what the guard was written for."""

    def test_it_is_refused(self, tree):
        repo = tree("0.72.4.post4")
        code, out = call("0.72.3.post2", repo)
        assert code == 1
        assert "OLDER" in out
        assert "allow_downgrade" in out

    def test_allow_downgrade_lets_it_through_and_says_so(self, tree):
        repo = tree("0.72.4.post4")
        code, out = call("0.72.3.post2", repo, allow_downgrade=True)
        assert code == 0, out
        assert "::warning::" in out

    def test_moving_forward_is_reported(self, tree):
        repo = tree("0.72.4.post4")
        code, out = call("0.72.5", repo)
        assert code == 0, out
        assert "0.72.4.post4 -> 0.72.5" in out


class TestANumberThatAlreadyNamesOtherCode:
    """The hazard the old message reached for, asked of a tag instead.

    "Re-releasing the same number publishes different code under one
    version" is true only when the number is already attached to
    different code. A tag can answer that; a version string cannot.
    """

    def test_a_tag_on_another_commit_is_refused(self, tree):
        repo = tree("0.72.4.post5")
        subprocess.run(["git", "tag", "v0.72.4.post5"], cwd=repo, check=True,
                       capture_output=True)
        released = head_of(repo)
        (repo / "extra.txt").write_text("more code\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "more"], cwd=repo, check=True,
                       capture_output=True)
        code, out = call("0.72.4.post5", repo)
        assert code == 1, out
        assert "two different trees under one release" in out
        assert released[:12] in out

    def test_a_tag_on_the_commit_being_released_is_allowed(self, tree):
        """Re-running a release that failed after the tag push.

        Same code under the same number is not the hazard -- the tag
        step already skips an existing tag rather than moving it.
        """
        repo = tree("0.72.4.post5")
        subprocess.run(["git", "tag", "v0.72.4.post5"], cwd=repo, check=True,
                       capture_output=True)
        code, out = call("0.72.4.post5", repo)
        assert code == 0, out
        assert "re-running the same release" in out

    def test_an_unknown_head_refuses_rather_than_guesses(self, tree, monkeypatch):
        """Cannot tell same code from different code, so do not pretend.

        And say *that*, rather than reporting a mismatch against a blank
        sha the operator cannot look up.
        """
        repo = tree("0.72.4.post5")
        subprocess.run(["git", "tag", "v0.72.4.post5"], cwd=repo, check=True,
                       capture_output=True)

        # What a checkout that cannot resolve HEAD looks like from here.
        # `--head ""` is "not supplied" and falls back to git, so the
        # only way to reach this branch is git declining to answer.
        real = guard._git

        def mute_rev_parse(*args, **kwargs):
            if args[:1] == ("rev-parse",):
                return None
            return real(*args, **kwargs)

        monkeypatch.setattr(guard, "_git", mute_rev_parse)
        code, out = call("0.72.4.post5", repo)
        assert code == 1, out
        assert "cannot say which commit" in out

    def test_no_git_at_all_does_not_block_a_prepared_release(self, tree):
        """A shallow or export-only checkout has no tags to consult.

        With nothing to contradict it, the prepared-tree case stands.
        """
        repo = tree("0.72.4.post5", git=False)
        code, out = call("0.72.4.post5", repo)
        assert code == 0, out


class TestTheDispatchInputShapes:
    """Everything the workflow's `version` input accepts."""

    @pytest.mark.parametrize(
        ("typed", "pep440"),
        [
            ("0.72.5", "0.72.5"),
            ("v0.72.5", "0.72.5"),
            ("0.72.5-rc1", "0.72.5-rc1"),
            ("0.70.6-2", "0.70.6.post2"),
            ("0.70.6postr1", "0.70.6.post1"),
            ("0.72.4.post5", "0.72.4.post5"),
        ],
    )
    def test_normalise(self, typed, pep440):
        assert guard.normalise(typed) == pep440

    def test_a_rebuild_suffix_matches_the_tree_written_as_postn(self, tree):
        """`0.70.6-2` and `0.70.6.post2` are the same request.

        The workflow writes `.post2` into pyproject and tags the raw
        form, so the guard has to see through both spellings or a
        re-run of a rebuild looks like a downgrade.
        """
        repo = tree("0.70.6.post2")
        code, out = call("0.70.6-2", repo)
        assert code == 0, out

    def test_a_version_that_is_not_pep_440_is_refused(self, tree):
        repo = tree("0.72.4.post5")
        code, out = call("not-a-version", repo)
        assert code == 1
        assert "::error::" in out

    def test_a_pyproject_with_no_version_line_is_refused(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        code, out = call("0.1.0", tmp_path)
        assert code == 1
        assert "no version line" in out


class TestTheWorkflowActuallyCallsIt:
    def test_the_step_runs_the_script(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "public-release.yml").read_text(
            encoding="utf-8"
        )
        assert ".github/scripts/version_guard.py" in workflow

    def test_the_old_inline_refusal_is_gone(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "public-release.yml").read_text(
            encoding="utf-8"
        )
        assert "is already what the tree says" not in workflow

    def test_allow_downgrade_is_still_wired_through(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "public-release.yml").read_text(
            encoding="utf-8"
        )
        assert "--allow-downgrade" in workflow
        assert "inputs.allow_downgrade" in workflow

    def test_it_runs_as_a_script_from_the_command_line(self, tree):
        repo = tree("0.72.4.post5")
        done = subprocess.run(
            [sys.executable, str(GUARD), "0.72.4.post5", "--repo", str(repo)],
            capture_output=True, text=True, timeout=60,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert "already prepared with" in done.stdout
