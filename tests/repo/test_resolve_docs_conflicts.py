"""scripts/ci/resolve_docs_conflicts.py, against real git repositories.

Each test builds a base branch and a PR branch that both changed the same
files after diverging, with commit dates set explicitly so "the newest"
is decided by the test, not by how fast it ran.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "resolve_docs_conflicts.py"
DOCS = "docs/public/v1"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.clock = 1_800_000_000
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args: str, when: int | None = None) -> str:
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        if when is not None:
            env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"@{when} +0000"
        done = subprocess.run(["git", *args], cwd=self.path, env=env, capture_output=True,
                              text=True, encoding="utf-8", check=True)
        return done.stdout

    def commit(self, files: dict[str, str | None], *, at: int, message: str = "change") -> None:
        for name, content in files.items():
            target = self.path / name
            if content is None:
                self.git("rm", "-q", "--", name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            self.git("add", "--", name)
        self.git("commit", "-q", "-m", message, when=at)

    def read(self, name: str) -> str:
        return (self.path / name).read_text(encoding="utf-8")

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        return subprocess.run([sys.executable, str(SCRIPT), "--base", "main", *args], cwd=self.path,
                              env=env, capture_output=True, text=True, encoding="utf-8", check=False)


@pytest.fixture
def repo(tmp_path) -> Repo:
    repo = Repo(tmp_path)
    repo.commit({f"{DOCS}/api.json": '{"v": 0}\n', f"{DOCS}/stats.json": '{"n": 0}\n',
                 "src/app.py": "x = 0\n"}, at=1_000)
    repo.git("checkout", "-q", "-b", "pr")
    return repo


def _diverge(repo: Repo, *, pr: dict, pr_at: int, base: dict, base_at: int) -> None:
    repo.commit(pr, at=pr_at, message="pr change")
    repo.git("checkout", "-q", "main")
    repo.commit(base, at=base_at, message="base change")
    repo.git("checkout", "-q", "pr")


def test_the_base_branchs_newer_copy_wins(repo):
    _diverge(repo, pr={f"{DOCS}/api.json": '{"v": "pr"}\n'}, pr_at=2_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n'}, base_at=3_000)
    done = repo.run()
    assert done.returncode == 0, done.stderr
    assert repo.read(f"{DOCS}/api.json") == '{"v": "base"}\n'
    parents = repo.git("log", "-1", "--format=%P").split()
    assert len(parents) == 2, "a merge commit, so main is now an ancestor"
    assert repo.git("status", "--porcelain") == ""


def test_the_prs_newer_copy_wins(repo):
    _diverge(repo, pr={f"{DOCS}/api.json": '{"v": "pr"}\n'}, pr_at=5_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n'}, base_at=3_000)
    assert repo.run().returncode == 0
    assert repo.read(f"{DOCS}/api.json") == '{"v": "pr"}\n'


def test_each_file_is_decided_on_its_own(repo):
    repo.commit({f"{DOCS}/api.json": '{"v": "pr-early"}\n'}, at=2_000)
    repo.commit({f"{DOCS}/stats.json": '{"n": "pr-late"}\n'}, at=6_000)
    repo.git("checkout", "-q", "main")
    repo.commit({f"{DOCS}/api.json": '{"v": "base"}\n', f"{DOCS}/stats.json": '{"n": "base"}\n'}, at=4_000)
    repo.git("checkout", "-q", "pr")

    done = repo.run()
    assert done.returncode == 0, done.stderr
    assert repo.read(f"{DOCS}/api.json") == '{"v": "base"}\n'
    assert repo.read(f"{DOCS}/stats.json") == '{"n": "pr-late"}\n'
    message = repo.git("log", "-1", "--format=%B")
    assert f"{DOCS}/api.json: from main" in message and f"{DOCS}/stats.json: from this branch" in message


def test_a_tie_goes_to_the_base_branch(repo):
    _diverge(repo, pr={f"{DOCS}/api.json": '{"v": "pr"}\n'}, pr_at=3_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n'}, base_at=3_000)
    assert repo.run().returncode == 0
    assert repo.read(f"{DOCS}/api.json") == '{"v": "base"}\n'


def test_a_conflict_anywhere_else_changes_nothing(repo):
    _diverge(repo, pr={f"{DOCS}/api.json": '{"v": "pr"}\n', "src/app.py": "x = 1\n"}, pr_at=2_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n', "src/app.py": "x = 2\n"}, base_at=3_000)
    before = repo.head()

    done = repo.run()

    assert done.returncode == 2
    assert "src/app.py" in done.stdout and "Not merged automatically" in done.stdout
    assert repo.head() == before
    assert repo.git("status", "--porcelain") == ""
    assert not (repo.path / ".git" / "MERGE_HEAD").exists()
    assert repo.read("src/app.py") == "x = 1\n"


def test_a_clean_merge_is_left_alone(repo):
    _diverge(repo, pr={"src/app.py": "x = 1\n"}, pr_at=2_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n'}, base_at=3_000)
    before = repo.head()
    done = repo.run()
    assert done.returncode == 0 and "outcome: clean" in done.stdout
    assert repo.head() == before
    assert not (repo.path / ".git" / "MERGE_HEAD").exists()


def test_a_newer_deletion_wins_too(repo):
    _diverge(repo, pr={f"{DOCS}/stats.json": '{"n": "pr"}\n'}, pr_at=2_000,
             base={f"{DOCS}/stats.json": None}, base_at=3_000)
    done = repo.run()
    assert done.returncode == 0, done.stderr
    assert not (repo.path / DOCS / "stats.json").exists()
    assert "deleted, from main" in repo.git("log", "-1", "--format=%B")


def test_an_older_deletion_loses(repo):
    _diverge(repo, pr={f"{DOCS}/stats.json": '{"n": "pr"}\n'}, pr_at=4_000,
             base={f"{DOCS}/stats.json": None}, base_at=3_000)
    assert repo.run().returncode == 0
    assert repo.read(f"{DOCS}/stats.json") == '{"n": "pr"}\n'


def test_it_will_not_merge_into_uncommitted_work(repo):
    _diverge(repo, pr={f"{DOCS}/api.json": '{"v": "pr"}\n'}, pr_at=2_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n'}, base_at=3_000)
    (repo.path / "src/app.py").write_text("x = 'unsaved'\n", encoding="utf-8")
    done = repo.run()
    assert done.returncode == 1 and "uncommitted" in done.stderr
    assert repo.read("src/app.py") == "x = 'unsaved'\n"


def test_no_commit_leaves_the_resolution_staged(repo):
    _diverge(repo, pr={f"{DOCS}/api.json": '{"v": "pr"}\n'}, pr_at=2_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n'}, base_at=3_000)
    before = repo.head()
    assert repo.run("--no-commit").returncode == 0
    assert repo.head() == before
    assert (repo.path / ".git" / "MERGE_HEAD").exists()
    assert repo.git("diff", "--name-only", "--diff-filter=U") == ""


def test_the_summary_is_a_table_of_choices(repo, tmp_path):
    _diverge(repo, pr={f"{DOCS}/api.json": '{"v": "pr"}\n'}, pr_at=2_000,
             base={f"{DOCS}/api.json": '{"v": "base"}\n'}, base_at=3_000)
    out = tmp_path / "summary.md"
    assert repo.run("--summary", str(out)).returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "| File | Taken from |" in text and f"`{DOCS}/api.json` | `main`" in text


def test_other_paths_can_be_named(repo):
    _diverge(repo, pr={"src/app.py": "x = 1\n"}, pr_at=2_000,
             base={"src/app.py": "x = 2\n"}, base_at=3_000)
    assert repo.run().returncode == 2
    assert repo.run("--paths", "src").returncode == 0
    assert repo.read("src/app.py") == "x = 2\n"


# -- the workflow that runs it ---------------------------------------------


class TestTheWorkflow:
    WORKFLOWS = ROOT / ".github" / "workflows"

    @pytest.fixture(scope="class")
    def workflow(self):
        yaml = pytest.importorskip("yaml")
        return yaml.safe_load((self.WORKFLOWS / "resolve-docs-conflicts.yml").read_text(encoding="utf-8"))

    @staticmethod
    def _on(workflow):
        # PyYAML reads the bare key `on` as True.
        return workflow.get("on", workflow.get(True))

    def test_it_runs_when_a_pull_request_is_opened(self, workflow):
        assert "opened" in self._on(workflow)["pull_request"]["types"]

    def test_it_follows_the_jobs_that_rewrite_the_data(self, workflow):
        """workflow_run matches on the workflows' names; a rename that left
        this behind would make it silently stop running."""
        import yaml

        writers = {
            yaml.safe_load(path.read_text(encoding="utf-8"))["name"]
            for path in self.WORKFLOWS.glob("*.yml")
            if DOCS in path.read_text(encoding="utf-8") and path.name != "resolve-docs-conflicts.yml"
        }
        assert set(self._on(workflow)["workflow_run"]["workflows"]) == writers

    def test_it_settles_only_docs_public_v1(self, workflow):
        script = "\n".join(step.get("run", "") for step in workflow["jobs"]["resolve"]["steps"])
        assert "--paths docs/public/v1" in script

    def test_it_never_force_pushes(self, workflow):
        script = "\n".join(step.get("run", "") for step in workflow["jobs"]["resolve"]["steps"])
        pushes = [line for line in script.splitlines() if "git push" in line]
        assert pushes, "nothing is pushed"
        for line in pushes:
            assert "--force" not in line and " -f" not in line and "+HEAD" not in line, line

    def test_forks_are_skipped(self, workflow):
        lookup = workflow["jobs"]["resolve"]["steps"][0]["run"]
        assert "head.repo.full_name" in lookup and "fork" in lookup
