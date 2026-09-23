"""hyped-pro's git support and file editing, and the consent gate.

Against real repositories in tmp_path: the git module runs git itself,
and a parser tested only on invented porcelain would pass while real git
said something else.
"""
from __future__ import annotations

import io
import json
import queue
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from hypernix.interfaces import hyped_pro_bridge as bridge
from hypernix.interfaces import hyped_pro_git as git
from hypernix.interfaces import hyped_pro_tools as tools
from hypernix.interfaces.hyped_pro_tools import ToolError

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    sh(root, "config", "user.email", "t@example.com")
    sh(root, "config", "user.name", "Test")
    sh(root, "config", "commit.gpgsign", "false")
    (root / "a.py").write_text("one\n", encoding="utf-8")
    (root / "b.txt").write_text("bee\n", encoding="utf-8")
    sh(root, "add", "-A")
    sh(root, "commit", "-qm", "first")
    monkeypatch.setenv("HYPED_PRO_WORKSPACE", str(root))
    monkeypatch.delenv("HYPERNIX_TOOL_POLICY", raising=False)
    return root


# -- status ------------------------------------------------------------------


class TestStatus:
    def test_clean(self, repo):
        s = git.status()
        assert s.branch == "main" and s.clean and not s.detached
        assert Path(s.root).resolve() == repo.resolve()

    def test_every_kind_of_change(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")          # modified, unstaged
        (repo / "new.py").write_text("x\n", encoding="utf-8")          # untracked
        (repo / "staged.py").write_text("s\n", encoding="utf-8")
        sh(repo, "add", "staged.py")                                    # added, staged
        sh(repo, "mv", "b.txt", "bee.txt")                              # renamed, staged
        files = {f.path: f for f in git.status().files}
        assert files["a.py"].label == "modified" and files["a.py"].to_dict()["unstaged"]
        assert files["new.py"].label == "untracked"
        assert files["staged.py"].label == "added" and files["staged.py"].to_dict()["staged"]
        assert files["bee.txt"].label == "renamed" and files["bee.txt"].original == "b.txt"

    def test_a_path_with_spaces(self, repo):
        (repo / "has space.txt").write_text("x", encoding="utf-8")
        assert "has space.txt" in [f.path for f in git.status().files]

    def test_ahead_and_behind(self):
        porcelain = "\0".join([
            "# branch.oid abc", "# branch.head feature", "# branch.upstream origin/feature",
            "# branch.ab +2 -3", "",
        ])
        s = git.parse_status(porcelain)
        assert (s.branch, s.upstream, s.ahead, s.behind) == ("feature", "origin/feature", 2, 3)

    def test_detached_head(self, repo):
        sh(repo, "checkout", "-q", "--detach")
        s = git.status()
        assert s.detached and s.branch is None

    def test_not_a_repository(self, tmp_path, monkeypatch):
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.setenv("HYPED_PRO_WORKSPACE", str(plain))
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
        assert not git.is_repository()
        with pytest.raises(ToolError) as caught:
            git.status()
        assert caught.value.code == "TOOL-GIT-003"


# -- reading -------------------------------------------------------------------


class TestReading:
    def test_diff_unstaged_and_staged(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        assert "+two" in git.diff()
        assert git.diff(staged=True) == ""
        sh(repo, "add", "a.py")
        assert git.diff() == "" and "+two" in git.diff(staged=True)

    def test_diff_of_one_path(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        (repo / "b.txt").write_text("changed\n", encoding="utf-8")
        only = git.diff("b.txt")
        assert "b.txt" in only and "a.py" not in only

    def test_a_path_that_looks_like_an_option_is_diffed_as_a_path(self, repo):
        """Paths go after `--`. Without it git would read `-p` as an
        option and diff everything."""
        (repo / "-p").write_text("dash\n", encoding="utf-8")
        sh(repo, "add", "--", "-p")
        sh(repo, "commit", "-qm", "dash file")
        (repo / "-p").write_text("changed\n", encoding="utf-8")
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        out = git.diff("-p")
        assert "+changed" in out and "a.py" not in out

    def test_log(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        sh(repo, "commit", "-qam", "second: with a colon")
        commits = git.log(5)
        assert [c["subject"] for c in commits] == ["second: with a colon", "first"]
        assert commits[0]["author"] == "Test" and len(commits[0]["short"]) >= 7

    def test_branches(self, repo):
        sh(repo, "branch", "other")
        found = git.branches()
        assert found["current"] == "main"
        assert {b["name"] for b in found["branches"]} == {"main", "other"}

    def test_show(self, repo):
        assert "first" in git.show("HEAD")

    def test_big_output_is_cut_with_a_note(self, repo, monkeypatch):
        monkeypatch.setattr(git, "MAX_OUTPUT", 50)
        (repo / "a.py").write_text("x\n" * 500, encoding="utf-8")
        out = git.diff()
        assert "cut at 50" in out and len(out) < 200


# -- changing ------------------------------------------------------------------


class TestChanging:
    def test_add_and_commit(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        git.add(["a.py"])
        message = git.commit("Change a\n\nWith a body.")
        assert "Change a" in message
        assert sh(repo, "log", "-1", "--format=%B").strip() == "Change a\n\nWith a body."

    def test_a_message_that_looks_like_an_option_is_a_message(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        git.add(".")
        git.commit("--amend --no-verify")
        assert sh(repo, "log", "-1", "--format=%s").strip() == "--amend --no-verify"
        assert len(git.log(10)) == 2  # a new commit, not an amended one

    def test_commit_all_tracked(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        git.commit("all of it", all_tracked=True)
        assert git.status().clean

    def test_nothing_staged(self, repo):
        with pytest.raises(ToolError) as caught:
            git.commit("empty")
        assert caught.value.code == "TOOL-GIT-008"

    def test_an_empty_message(self, repo):
        with pytest.raises(ToolError):
            git.commit("   ")

    def test_hooks_still_run(self, repo):
        hook = repo / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\necho refused by hook >&2\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        git.add(".")
        with pytest.raises(ToolError, match="refused by hook"):
            git.commit("blocked")

    def test_switch_and_create(self, repo):
        git.switch("feature", create=True)
        assert git.status().branch == "feature"
        git.switch("main")
        assert git.status().branch == "main"

    def test_restore_discards(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        git.restore(["a.py"])
        assert (repo / "a.py").read_text(encoding="utf-8") == "one\n"

    def test_unstage(self, repo):
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        git.add("a.py")
        git.unstage("a.py")
        assert git.diff(staged=True) == "" and "+two" in git.diff()

    @pytest.mark.parametrize("name", ["--upload-pack=touch /tmp/x", "-b", "a..b", "a//b", "x@{1}", "end.", ""])
    def test_names_that_could_be_options_are_refused(self, repo, name):
        with pytest.raises(ToolError) as caught:
            git.switch(name)
        assert caught.value.code == "TOOL-GIT-005"

    def test_paths_outside_the_workspace_are_refused(self, repo):
        with pytest.raises(ToolError) as caught:
            git.add(["../../etc/passwd"])
        assert caught.value.code == "TOOL-PATH-001"

    def test_a_path_that_looks_like_an_option_stays_a_path(self, repo):
        (repo / "-rf").write_text("x", encoding="utf-8")
        git.add(["-rf"])
        assert "-rf" in git.diff(staged=True)

    def test_push_without_a_remote_fails_cleanly(self, repo):
        with pytest.raises(ToolError) as caught:
            git.push()
        assert caught.value.code == "TOOL-GIT-004"


# -- the file tools ------------------------------------------------------------


class TestFileTools:
    def test_write_creates_and_overwrites_keeping_mode(self, repo):
        tools.write_file("new/deep.txt", "hello")
        assert (repo / "new" / "deep.txt").read_text(encoding="utf-8") == "hello"
        script = repo / "run.sh"
        script.write_text("old", encoding="utf-8")
        script.chmod(0o755)
        tools.write_file("run.sh", "new")
        assert script.read_text(encoding="utf-8") == "new"
        assert script.stat().st_mode & 0o111

    def test_a_stale_write_is_refused(self, repo):
        seen = tools.content_hash((repo / "a.py").read_text(encoding="utf-8"))
        (repo / "a.py").write_text("someone else\n", encoding="utf-8")
        with pytest.raises(ToolError) as caught:
            tools.write_file("a.py", "mine\n", expected_hash=seen)
        assert caught.value.code == "TOOL-WRITE-002"
        assert (repo / "a.py").read_text(encoding="utf-8") == "someone else\n"

    def test_a_new_file_expects_nothing_there(self, repo):
        empty = tools.content_hash("")
        tools.write_file("fresh.txt", "x", expected_hash=empty)
        with pytest.raises(ToolError):
            tools.write_file("fresh.txt", "y", expected_hash=empty)

    @pytest.mark.parametrize("call", [
        lambda: tools.write_file(".git/hooks/post-commit", "#!/bin/sh\nrm -rf ~\n"),
        lambda: tools.create_file(".git/hooks/pre-push", "x"),
        lambda: tools.edit_file(".git/config", "[core]", "[core]\n\thooksPath = /tmp"),
        lambda: tools.move_file("a.py", ".git/hooks/pre-commit"),
        lambda: tools.delete_file(".git/HEAD"),
    ])
    def test_nothing_writes_under_dot_git(self, repo, call, monkeypatch):
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "allow")
        with pytest.raises(ToolError) as caught:
            call()
        assert caught.value.code == "TOOL-PATH-002"

    def test_move_refuses_to_overwrite(self, repo):
        with pytest.raises(ToolError) as caught:
            tools.move_file("a.py", "b.txt")
        assert caught.value.code == "TOOL-MOVE-002"
        tools.move_file("a.py", "moved/a.py")
        assert (repo / "moved" / "a.py").exists() and not (repo / "a.py").exists()

    def test_delete_one_file_not_a_directory(self, repo):
        (repo / "d").mkdir()
        with pytest.raises(ToolError):
            tools.delete_file("d")
        tools.delete_file("b.txt")
        assert not (repo / "b.txt").exists()


# -- consent -------------------------------------------------------------------


class TestConsent:
    def test_the_gated_set(self):
        assert tools.GATED_TOOLS == {"delete_file", "git_commit", "git_switch", "git_restore"}
        # Every gated name is a real tool.
        assert tools.GATED_TOOLS <= {t.name for t in tools.TOOLS}

    def test_nobody_to_ask_means_no(self, repo, monkeypatch):
        monkeypatch.setattr(tools.sys, "stdin", io.StringIO(""))
        with pytest.raises(ToolError) as caught:
            tools.execute_tool("delete_file", {"path": "b.txt"})
        assert caught.value.code == "TOOL-CONSENT-001"
        assert (repo / "b.txt").exists()

    def test_policy_allow_and_deny(self, repo, monkeypatch):
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "deny")
        with pytest.raises(ToolError):
            tools.execute_tool("delete_file", {"path": "b.txt"})
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "allow")
        tools.execute_tool("delete_file", {"path": "b.txt"})
        assert not (repo / "b.txt").exists()

    def test_deny_beats_a_hook_that_would_say_yes(self, repo, monkeypatch):
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "deny")
        with tools.hooks(consent=lambda n, a: (True, "")):
            with pytest.raises(ToolError):
                tools.execute_tool("delete_file", {"path": "b.txt"})

    def test_the_hook_decides_and_sees_the_real_arguments(self, repo):
        asked = []

        def consent(name, arguments):
            asked.append((name, arguments))
            return False, "no thanks"

        (repo / "a.py").write_text("two\n", encoding="utf-8")
        with tools.hooks(consent=consent):
            with pytest.raises(ToolError, match="no thanks"):
                tools.execute_tool("git_commit", {"message": "msg", "all_tracked": True})
        assert asked == [("git_commit", {"message": "msg", "all_tracked": True})]
        assert len(git.log(10)) == 1

    def test_ungated_tools_do_not_ask(self, repo):
        def consent(name, arguments):
            raise AssertionError(f"asked about {name}")

        with tools.hooks(consent=consent):
            tools.execute_tool("git_status", {})
            tools.execute_tool("write_file", {"path": "x.txt", "content": "x"})

    def test_hooks_are_per_thread_and_restored(self, repo):
        seen = []
        with tools.hooks(consent=lambda n, a: (seen.append(n) or (True, ""))):
            t = threading.Thread(target=lambda: seen.append(getattr(tools._hooks, "consent", None)))
            t.start()
            t.join()
        assert seen == [None]
        assert getattr(tools._hooks, "consent", None) is None

    def test_progress_reports_success_and_failure(self, repo):
        events = []
        with tools.hooks(progress=lambda n, a, r, e: events.append((n, r is not None, e))):
            tools.execute_tool("read_file", {"path": "a.py"})
            with pytest.raises(ToolError):
                tools.execute_tool("read_file", {"path": "missing.py"})
        assert events[0] == ("read_file", True, None)
        assert events[1][0] == "read_file" and events[1][1] is False and "does not exist" in events[1][2]

    def test_describe_shows_what_matters(self):
        assert tools.describe_call("git_commit", {"message": "Fix it"}) == "Fix it"
        assert "discard" in tools.describe_call("git_restore", {"paths": ["a.py"]})
        assert "create" in tools.describe_call("git_switch", {"branch": "x", "create": True})


# -- the bridge ----------------------------------------------------------------


@pytest.fixture
def wire(monkeypatch):
    """Capture what the bridge writes, and reset its client state."""
    lines: queue.Queue = queue.Queue()
    monkeypatch.setattr(bridge, "_write", lambda payload: lines.put(payload))
    monkeypatch.setattr(bridge, "_client_features", set())
    return lines


class TestBridge:
    def test_hello_keeps_only_known_features(self, wire):
        got = bridge.dispatch({"id": 1, "cmd": "hello", "features": ["consent", "events", "telepathy"]})
        assert got["data"]["features"] == ["consent", "events"]

    def test_a_consent_round_trip(self, repo, wire):
        bridge.dispatch({"id": 1, "cmd": "hello", "features": ["consent", "events"]})
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        tools.execute_tool("git_add", {"paths": ["a.py"]})
        result = {}

        def model_turn():
            with bridge._tool_hooks(42, None):
                try:
                    result["ok"] = tools.execute_tool("git_commit", {"message": "From the model"})
                except ToolError as exc:
                    result["error"] = exc.message

        worker = threading.Thread(target=model_turn)
        worker.start()
        asked = wire.get(timeout=10)
        assert asked["event"] == "consent" and asked["id"] == 42 and asked["tool"] == "git_commit"
        assert asked["detail"] == "From the model"
        reply = bridge.dispatch({"id": 2, "cmd": "consent_reply", "consent_id": asked["consent_id"], "allow": True})
        assert reply["data"]["delivered"] is True
        worker.join(10)
        assert "Committed" in result["ok"]
        progress = wire.get(timeout=5)
        assert progress["event"] == "tool" and progress["ok"] is True

    def test_a_refusal_reaches_the_model(self, repo, wire):
        bridge.dispatch({"id": 1, "cmd": "hello", "features": ["consent"]})
        result = {}

        def model_turn():
            with bridge._tool_hooks(7, None):
                try:
                    tools.execute_tool("delete_file", {"path": "a.py"})
                except ToolError as exc:
                    result["error"] = exc.message

        worker = threading.Thread(target=model_turn)
        worker.start()
        asked = wire.get(timeout=10)
        bridge.dispatch({"id": 2, "cmd": "consent_reply", "consent_id": asked["consent_id"], "allow": False})
        worker.join(10)
        assert "refused by the person" in result["error"]
        assert (repo / "a.py").exists()

    def test_cancelling_the_chat_answers_the_question_no(self, repo, wire):
        bridge.dispatch({"id": 1, "cmd": "hello", "features": ["consent"]})
        cancel = threading.Event()
        result = {}

        def model_turn():
            with bridge._tool_hooks(9, cancel):
                try:
                    tools.execute_tool("delete_file", {"path": "a.py"})
                except ToolError as exc:
                    result["error"] = exc.message

        worker = threading.Thread(target=model_turn)
        worker.start()
        wire.get(timeout=10)
        cancel.set()
        worker.join(10)
        assert "cancelled" in result["error"] and (repo / "a.py").exists()
        assert bridge._consents == {}

    def test_a_client_that_never_said_hello_gets_no_events_and_no_yes(self, repo, wire, monkeypatch):
        monkeypatch.setattr(tools.sys, "stdin", io.StringIO(""))
        with bridge._tool_hooks(3, None):
            with pytest.raises(ToolError):
                tools.execute_tool("delete_file", {"path": "a.py"})
            tools.execute_tool("read_file", {"path": "a.py"})
        assert wire.empty()

    def test_a_late_reply_is_harmless(self, wire):
        got = bridge.dispatch({"id": 1, "cmd": "consent_reply", "consent_id": 999, "allow": True})
        assert got["data"]["delivered"] is False

    def test_git_ops(self, repo, wire):
        status = bridge.dispatch({"id": 1, "cmd": "git", "op": "status"})["data"]
        assert status["repository"] and status["branch"] == "main"
        (repo / "a.py").write_text("two\n", encoding="utf-8")
        assert "+two" in bridge.dispatch({"id": 2, "cmd": "git", "op": "diff"})["data"]["diff"]
        bridge.dispatch({"id": 3, "cmd": "git", "op": "add", "paths": ["a.py"]})
        done = bridge.dispatch({"id": 4, "cmd": "git", "op": "commit", "message": "via bridge"})
        assert done["ok"] and "via bridge" in done["data"]["message"]
        bad = bridge.dispatch({"id": 5, "cmd": "git", "op": "switch", "branch": "--evil"})
        assert bad["ok"] is False and bad["code"] == "TOOL-GIT-005"
        assert bridge.dispatch({"id": 6, "cmd": "git", "op": "frobnicate"})["code"] == "TOOL-GIT-009"

    def test_git_outside_a_repository(self, tmp_path, wire, monkeypatch):
        monkeypatch.setenv("HYPED_PRO_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        assert bridge.dispatch({"id": 1, "cmd": "git", "op": "status"})["data"] == {"repository": False}

    def test_files_list_read_write(self, repo, wire):
        (repo / "src").mkdir()
        listing = bridge.dispatch({"id": 1, "cmd": "files_list", "path": "."})["data"]
        names = [e["name"] for e in listing["entries"]]
        assert names[0] == "src" and ".git" not in names and "a.py" in names
        read = bridge.dispatch({"id": 2, "cmd": "file_read", "path": "a.py"})["data"]
        assert read["content"] == "one\n" and read["exists"]
        wrote = bridge.dispatch({"id": 3, "cmd": "file_write", "path": "a.py", "content": "edited\n",
                                 "expected_hash": read["hash"]})
        assert wrote["ok"] and (repo / "a.py").read_text(encoding="utf-8") == "edited\n"
        stale = bridge.dispatch({"id": 4, "cmd": "file_write", "path": "a.py", "content": "again",
                                 "expected_hash": read["hash"]})
        assert stale["ok"] is False and stale["code"] == "TOOL-WRITE-002"

    def test_a_new_file_opens_empty(self, repo, wire):
        read = bridge.dispatch({"id": 1, "cmd": "file_read", "path": "later.md"})["data"]
        assert read == {"path": "later.md", "content": "", "hash": tools.content_hash(""), "exists": False}

    def test_binary_and_outside_files_are_refused(self, repo, wire):
        (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
        assert bridge.dispatch({"id": 1, "cmd": "file_read", "path": "blob.bin"})["code"] == "TOOL-READ-004"
        assert bridge.dispatch({"id": 2, "cmd": "file_read", "path": "../../etc/passwd"})["code"] == "TOOL-PATH-001"
        assert bridge.dispatch({"id": 3, "cmd": "file_write", "path": ".git/hooks/x", "content": "x"})["code"] == "TOOL-PATH-002"

    def test_git_runs_off_the_main_loop(self):
        # push and pull wait on a network; inline they would block cancel.
        assert "git" in bridge.BACKGROUND_COMMANDS


def test_the_bridge_serves_events_on_its_own_lines(repo, monkeypatch):
    """A real `serve` subprocess: hello, then a gated tool through the
    same hooks a chat uses, answered over stdin."""
    import os
    import sys

    script = (
        "import sys, threading\n"
        "from hypernix.interfaces import hyped_pro_bridge as b, hyped_pro_tools as t\n"
        "def turn():\n"
        "    with b._tool_hooks('chat-1', None):\n"
        "        try:\n"
        "            t.execute_tool('delete_file', {'path': 'b.txt'})\n"
        "        except t.ToolError as e:\n"
        "            print('refused', e.message, file=sys.stderr)\n"
        "b.dispatch({'id': 0, 'cmd': 'hello', 'features': ['consent']})\n"
        "threading.Thread(target=turn, daemon=True).start()\n"
        "b.serve()\n"
    )
    env = dict(os.environ, HYPED_PRO_WORKSPACE=str(repo), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()

    def next_line() -> dict:
        # A timeout rather than a bare readline: a bridge that never sends
        # the event must fail this test, not hang the suite.
        return json.loads(lines.get(timeout=20))

    try:
        event = next_line()
        assert event["event"] == "consent" and event["tool"] == "delete_file"
        proc.stdin.write(json.dumps({"id": 1, "cmd": "consent_reply", "consent_id": event["consent_id"], "allow": True}) + "\n")
        proc.stdin.flush()
        reply = next_line()
        assert reply == {"id": 1, "ok": True, "data": {"delivered": True}}
        for _ in range(50):
            if not (repo / "b.txt").exists():
                break
            threading.Event().wait(0.1)
        assert not (repo / "b.txt").exists()
    finally:
        proc.stdin.close()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
