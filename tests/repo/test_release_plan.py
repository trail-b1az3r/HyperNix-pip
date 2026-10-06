"""Nightly builds in public-release.yml, and the script that decides them.

The one property that matters more than the rest: a nightly can never
turn into a release. No version bump in the tree, no commit, no v* tag,
nothing on PyPI -- and a version no package index would accept even if
one of those guards were lost.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "release_plan.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "public-release.yml"
HEAD = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"


def run(*args: str, pyproject: Path) -> tuple[int, dict[str, str], str]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--head", HEAD, "--date", "20260930",
         "--pyproject", str(pyproject), *args],
        capture_output=True, text=True, check=False,
    )
    values = dict(line.split("=", 1) for line in done.stdout.splitlines() if "=" in line)
    return done.returncode, values, done.stderr


@pytest.fixture
def pyproject(tmp_path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "hypernix"\nversion = "0.72.6.post3"\n', encoding="utf-8")
    return path


class TestThePlan:
    def test_a_dispatched_version_is_a_release(self, pyproject):
        code, out, _ = run("--event", "workflow_dispatch", "--version", "0.72.7",
                           pyproject=pyproject)
        assert code == 0
        assert out == {"run": "true", "nightly": "false", "version": "0.72.7",
                       "tag": "v0.72.7", "reason": "release"}

    def test_a_dispatch_with_nothing_is_a_mistake(self, pyproject):
        code, _, err = run("--event", "workflow_dispatch", pyproject=pyproject)
        assert code == 1
        assert "nightly" in err

    def test_the_schedule_is_off_until_the_setting_is_on(self, pyproject):
        code, out, _ = run("--event", "schedule", pyproject=pyproject)
        assert code == 0
        assert out["run"] == "false"
        assert "PUBLIC_RELEASE_NIGHTLY=true" in out["reason"]

    def test_the_schedule_with_the_setting_on_is_a_nightly(self, pyproject):
        code, out, _ = run("--event", "schedule", "--enabled", "true", pyproject=pyproject)
        assert code == 0
        assert out["run"] == "true" and out["nightly"] == "true"
        assert out["version"] == "0.72.6.post3+nightly.20260930.1a2b3c4"
        assert out["tag"] == "nightly"

    def test_a_night_with_nothing_new_is_skipped(self, pyproject):
        _, out, _ = run("--event", "schedule", "--enabled", "true",
                        "--last-nightly", HEAD, pyproject=pyproject)
        assert out["run"] == "false"
        assert "nothing new" in out["reason"]

    def test_a_nightly_by_hand_runs_even_with_nothing_new_and_the_setting_off(self, pyproject):
        _, out, _ = run("--event", "workflow_dispatch", "--nightly", "true",
                        "--last-nightly", HEAD, pyproject=pyproject)
        assert out["run"] == "true" and out["nightly"] == "true"

    def test_a_version_typed_with_nightly_is_ignored_and_says_so(self, pyproject):
        _, out, _ = run("--event", "workflow_dispatch", "--nightly", "true",
                        "--version", "9.9.9", pyproject=pyproject)
        assert out["version"].startswith("0.72.6.post3+nightly.")
        assert "9.9.9" in out["reason"]

    def test_the_nightly_version_is_one_no_index_accepts(self, pyproject):
        """PEP 440 local label: valid, installable from a URL, refused by
        PyPI and TestPyPI -- the last guard against a nightly publishing."""
        packaging = pytest.importorskip("packaging.version")
        _, out, _ = run("--event", "workflow_dispatch", "--nightly", "true",
                        pyproject=pyproject)
        parsed = packaging.Version(out["version"])
        assert parsed.local == "nightly.20260930.1a2b3c4"
        assert parsed.public == "0.72.6.post3"


class TestTheWorkflow:
    @pytest.fixture(scope="class")
    def flow(self):
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def text(self):
        return WORKFLOW.read_text(encoding="utf-8")

    def test_it_is_scheduled_and_dispatchable(self, flow):
        on = flow[True]            # YAML reads the bare key `on` as True
        assert on["schedule"][0]["cron"]
        inputs = on["workflow_dispatch"]["inputs"]
        assert inputs["nightly"]["type"] == "boolean"
        assert inputs["version"]["required"] is False

    def test_the_plan_reads_the_setting(self, flow):
        env = next(s for s in flow["jobs"]["plan"]["steps"] if s.get("id") == "plan")["env"]
        assert env["ENABLED"] == "${{ vars.PUBLIC_RELEASE_NIGHTLY }}"

    def test_cut_waits_on_the_plan(self, flow):
        cut = flow["jobs"]["cut"]
        assert cut["needs"] == "plan"
        assert cut["if"] == "needs.plan.outputs.run == 'true'"

    @pytest.mark.parametrize("step", [
        "Check the version this release was dispatched with",
        "Bump versions in source",
        "Update Release Timeline",
        "Commit version bump",
        "Tag and push",
    ])
    def test_a_nightly_skips_everything_that_makes_a_release(self, flow, step):
        found = next(s for s in flow["jobs"]["cut"]["steps"] if s.get("name") == step)
        assert "needs.plan.outputs.nightly != 'true'" in found["if"]

    @pytest.mark.parametrize("job", ["pypi-publish", "testpypi-publish"])
    def test_a_nightly_never_publishes(self, flow, job):
        assert "needs.plan.outputs.nightly != 'true'" in flow["jobs"][job]["if"]

    def test_the_stamp_is_after_the_tests_and_before_the_build(self, text):
        stamp = text.index("- name: Stamp the nightly version")
        assert text.index("- name: Tests") < stamp < text.index("- name: Build sdist + wheel")

    def test_only_the_nightly_tag_is_ever_forced(self, text):
        forced = [line.strip() for line in text.splitlines() if "push -f" in line or "tag -f" in line]
        assert forced == ["git tag -f nightly \"$GITHUB_SHA\"",
                          "git push -f origin refs/tags/nightly"]

    def test_the_ipa_is_opt_in_for_a_nightly(self, flow):
        assert "vars.PUBLIC_RELEASE_NIGHTLY_IPA == 'true'" in flow["jobs"]["hyperlink-ipa"]["if"]

    def test_the_previous_nightly_release_is_replaced(self, flow):
        steps = flow["jobs"]["github-release"]["steps"]
        names = [s.get("name") for s in steps]
        assert names.index("Remove the previous nightly release") < names.index("Create Release")
        remove = steps[names.index("Remove the previous nightly release")]
        assert "--cleanup-tag" not in remove["run"]    # the tag cut just moved stays

    def test_no_step_reads_the_raw_version_input_after_the_plan(self, text):
        after = text.split("\n  cut:\n", 1)[1]
        assert "${{ inputs.version }}" not in after


sys.path.insert(0, str(SCRIPT.parent))
import release_plan  # noqa: E402

STAT = ("chore: update JSON stats (scheduled)", ["docs/public/v1/json"])
DOCS = ("chore: refresh generated docs data and release summaries",
        ["docs/public/v1/t1-api.json", "docs/public/v1/api-deep.json"])
README = ("Auto-update README header (hourly)", ["README.md"])
CODE = ("hyprslug: a fix", ["src/hypernix/quant/hyprslug.py"])


class TestStatUpdatesDoNotMakeANightly:
    """The stat bots commit all day; a nightly of nothing but their
    commits is the same code as last night's under a new name."""

    def _plan(self, since):
        return release_plan.plan(
            event="schedule", version="", nightly=False, enabled=True,
            head=HEAD, last_nightly="0" * 40, date="20260930", base="0.72.6", since=since,
        )

    def test_only_stat_commits_is_skipped(self):
        out = self._plan([STAT, DOCS, README])
        assert out["run"] == "false"
        assert "only stat updates" in out["reason"] and "3 commits" in out["reason"]

    def test_one_real_commit_among_them_builds(self):
        assert self._plan([STAT, CODE, DOCS])["run"] == "true"

    def test_history_it_cannot_compare_builds(self):
        assert self._plan(None)["run"] == "true"

    def test_by_hand_it_builds_anyway(self):
        out = release_plan.plan(
            event="workflow_dispatch", version="", nightly=True, enabled=False,
            head=HEAD, last_nightly="0" * 40, date="20260930", base="0.72.6", since=[STAT],
        )
        assert out["run"] == "true"

    @pytest.mark.parametrize("subject,files", [STAT, DOCS, README])
    def test_each_bot_commit_is_recognised(self, subject, files):
        assert release_plan.is_stat_update(subject, files)

    def test_a_bot_subject_on_code_is_not_a_stat_update(self):
        """Both the subject and the files have to match: a stat commit
        that strays outside its files is a change like any other."""
        assert not release_plan.is_stat_update(README[0], ["README.md", "src/hypernix/cli.py"])
        assert not release_plan.is_stat_update(STAT[0], ["docs/public/v1/json-extra/x"])

    def test_a_person_editing_the_stat_files_is_a_change(self):
        assert not release_plan.is_stat_update("docs: fix the README intro", ["README.md"])

    def test_an_empty_commit_is_not_a_stat_update(self):
        assert not release_plan.is_stat_update(STAT[0], [])


class TestAgainstARealRepository:
    @pytest.fixture
    def repo(self, tmp_path):
        def git(*args):
            return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True,
                                  text=True, check=True).stdout.strip()

        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (tmp_path / "pyproject.toml").write_text('version = "0.72.6"\n', encoding="utf-8")
        (tmp_path / "README.md").write_text("x\n", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "start")

        def commit(subject, path, text):
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            git("add", "-A")
            git("commit", "-qm", subject)
            return git("rev-parse", "HEAD")

        return tmp_path, git, commit

    def _run(self, root, last, head):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--event", "schedule", "--enabled", "true",
             "--head", head, "--last-nightly", last, "--date", "20260930",
             "--pyproject", "pyproject.toml"],
            cwd=root, capture_output=True, text=True, check=True,
        )
        return dict(line.split("=", 1) for line in done.stdout.splitlines())

    def test_stat_commits_since_the_tag_skip_the_night(self, repo):
        root, git, commit = repo
        last = git("rev-parse", "HEAD")
        commit("chore: update JSON stats (scheduled)", "docs/public/v1/json", "1")
        head = commit("Auto-update README header (hourly)", "README.md", "y\n")
        assert self._run(root, last, head)["run"] == "false"

    def test_a_real_commit_since_the_tag_builds(self, repo):
        root, git, commit = repo
        last = git("rev-parse", "HEAD")
        commit("chore: update JSON stats (scheduled)", "docs/public/v1/json", "1")
        head = commit("fix: a real change", "src/x.py", "print(1)\n")
        assert self._run(root, last, head)["run"] == "true"

    def test_a_tag_off_the_branch_builds(self, repo):
        root, git, commit = repo
        git("checkout", "-qb", "side")
        side = commit("elsewhere", "side.txt", "s")
        git("checkout", "-q", "main")
        head = commit("chore: update JSON stats (scheduled)", "docs/public/v1/json", "1")
        assert self._run(root, side, head)["run"] == "true"
