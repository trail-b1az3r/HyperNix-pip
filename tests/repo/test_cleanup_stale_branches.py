from __future__ import annotations

import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

from cleanup_stale_branches import classify_branch, is_owned_branch  # noqa: E402, I001


NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


def make_branch(name: str, *, age_days: int = 45, protected: bool = False) -> dict:
    committed = (NOW - timedelta(days=age_days)).isoformat().replace("+00:00", "Z")
    return {
        "name": name,
        "protected": protected,
        "commit": {
            "commit": {
                "committer": {"date": committed},
                "author": {"date": committed},
            }
        },
    }


def make_commit(login: str, *, name: str = "", email: str = "") -> dict:
    return {
        "author": {"login": login},
        "committer": {"login": login},
        "commit": {
            "author": {"name": name, "email": email},
            "committer": {"name": name, "email": email},
        },
    }


def classify(branch: dict, *, open_refs: set[str] | None = None,
             owned_refs: set[str] | None = None, commits=None):
    return classify_branch(
        branch,
        default_branch="main",
        open_pr_refs=open_refs or set(),
        owned_pr_refs=owned_refs or set(),
        now=NOW,
        compare_commits=lambda _base, _head: commits if commits is not None else [],
        stale_days=30,
    )


def test_deletes_an_old_branch_with_no_allowed_authors():
    action, reason = classify(make_branch("feature/old"), commits=[make_commit("someone-else")])
    assert action == "delete"
    assert "30 days" in reason


def test_keeps_branch_with_a_trail_b1az3r_commit():
    action, _ = classify(make_branch("feature/mine"), commits=[make_commit("trail-b1az3r")])
    assert action == "keep"


def test_keeps_branch_with_a_claude_commit():
    action, _ = classify(
        make_branch("feature/claude-work"), commits=[make_commit("claude", name="Claude")]
    )
    assert action == "keep"


def test_keeps_claude_named_branch_even_if_commit_identity_is_unavailable():
    action, _ = classify(make_branch("claude/created-by-claude"), commits=[])
    assert action == "keep"


def test_keeps_default_branch():
    action, reason = classify(make_branch("main"), commits=[make_commit("someone-else")])
    assert action == "keep"
    assert reason == "default branch"


def test_keeps_protected_branch():
    action, reason = classify(make_branch("release/stable", protected=True),
                              commits=[make_commit("someone-else")])
    assert action == "keep"
    assert reason == "protected branch"


def test_keeps_branch_used_by_an_open_pull_request():
    action, reason = classify(make_branch("feature/open-pr"), open_refs={"feature/open-pr"},
                              commits=[make_commit("someone-else")])
    assert action == "keep"
    assert reason == "used by an open pull request"


def test_keeps_branch_with_a_recent_commit():
    action, reason = classify(make_branch("feature/recent", age_days=12),
                              commits=[make_commit("someone-else")])
    assert action == "keep"
    assert "only 12 day(s) old" in reason


def test_keeps_branch_when_history_and_tip_identity_are_unavailable():
    action, reason = classify(make_branch("feature/no-history"), commits=[])
    assert action == "keep"
    assert "no branch-only commits" in reason


def test_deletes_merged_branch_when_its_tip_is_not_by_an_allowed_author():
    branch = make_branch("feature/merged-other")
    branch["commit"]["author"] = {"login": "someone-else"}
    branch["commit"]["committer"] = {"login": "someone-else"}
    action, reason = classify(branch, commits=[])
    assert action == "delete"
    assert "already contained" in reason


def test_keeps_branch_when_its_tip_is_by_trail_b1az3r():
    branch = make_branch("feature/created-by-rayla")
    branch["commit"]["author"] = {"login": "trail-b1az3r"}
    action, _ = classify(branch, commits=[make_commit("someone-else")])
    assert action == "keep"


def test_keeps_branch_with_a_pull_request_authored_by_trail_b1az3r():
    action, reason = classify(
        make_branch("feature/owned-pr"), owned_refs={"feature/owned-pr"}, commits=[]
    )
    assert action == "keep"
    assert "pull request authored" in reason


def test_keeps_branch_when_commit_identity_is_uncertain():
    def fail_compare(_base, _head):
        raise RuntimeError("API unavailable")

    action, reason = classify_branch(
        make_branch("feature/api-error"), default_branch="main", open_pr_refs=set(),
        now=NOW, compare_commits=fail_compare, stale_days=30,
    )
    assert action == "keep"
    assert "could not verify branch ownership" in reason


def test_recognizes_claude_anthropic_email():
    assert is_owned_branch(
        "feature/claude", [make_commit("", name="Claude", email="noreply@anthropic.com")]
    )


def test_recognizes_the_owner_noreply_email():
    assert is_owned_branch(
        "feature/rayla",
        [make_commit("", name="Rayla", email="163888293+trail-b1az3r@users.noreply.github.com")],
    )
