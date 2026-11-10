#!/usr/bin/env python3
"""Delete abandoned branches that were not authored by trail-b1az3r or Claude.

Git refs do not expose who created a branch. Ownership is inferred
conservatively from commits unique to the branch compared with the default
branch: a branch is retained if any such commit is authored/committed by
trail-b1az3r or Claude. Claude branches using the claude/ prefix are also
retained. If history or identity cannot be established, the branch is retained.

A branch is eligible only when its tip commit is older than STALE_DAYS, it is
not protected/default, and it is not the head or base of an open pull request.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

OWNER_LOGIN = "trail-b1az3r"
CLAUDE_LOGIN = "claude"
API_VERSION = "2022-11-28"


class GitHubAPI:
    def __init__(self, repository: str, token: str) -> None:
        if "/" not in repository:
            raise ValueError("GITHUB_REPOSITORY must be owner/name")
        self.repository = repository
        self.base = f"https://api.github.com/repos/{repository}"
        self.token = token

    def request(self, method: str, path: str):
        request = Request(
            f"{self.base}{path}",
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "hypernix-stale-branch-cleanup",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
                return json.loads(body.decode("utf-8")) if body else None
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"GitHub API {method} {path}: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"GitHub API {method} {path}: {exc}") from exc

    def list_all(self, path: str) -> list[dict]:
        found: list[dict] = []
        page = 1
        while True:
            separator = "&" if "?" in path else "?"
            result = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            if not isinstance(result, list):
                raise RuntimeError(f"Expected a list from GitHub API {path}")
            found.extend(result)
            if len(result) < 100:
                return found
            page += 1

    def compare_commits(self, base: str, head: str) -> list[dict]:
        base_ref = quote(base, safe="/")
        head_ref = quote(head, safe="/")
        path = f"/compare/{base_ref}...{head_ref}"
        commits: list[dict] = []
        page = 1
        expected: int | None = None
        while True:
            result = self.request("GET", f"{path}?per_page=100&page={page}")
            if not isinstance(result, dict):
                raise RuntimeError(f"Invalid comparison response for {head}")
            if expected is None:
                expected = int(result.get("total_commits", 0))
            batch = result.get("commits")
            if not isinstance(batch, list):
                raise RuntimeError(f"Comparison response for {head} has no commit list")
            commits.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        if expected is not None and len(commits) < expected:
            raise RuntimeError(
                f"Incomplete history for {head}: received {len(commits)} of {expected} commits"
            )
        return commits

    def delete_branch(self, name: str) -> None:
        self.request("DELETE", f"/git/refs/heads/{quote(name, safe='/')}")


def parse_github_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def branch_tip_time(branch: dict) -> datetime | None:
    details = (branch.get("commit") or {}).get("commit") or {}
    committer = details.get("committer") or {}
    author = details.get("author") or {}
    return parse_github_time(committer.get("date") or author.get("date"))


def commit_is_owned(commit: dict) -> bool:
    """True for a commit attributed to the repository owner or Claude."""
    details = commit.get("commit") or {}
    for role in ("author", "committer"):
        linked_user = commit.get(role) or {}
        identity = details.get(role) or {}
        login = str(linked_user.get("login") or "").casefold()
        name = str(identity.get("name") or "").casefold()
        email = str(identity.get("email") or "").casefold()
        if login in {OWNER_LOGIN, CLAUDE_LOGIN}:
            return True
        if OWNER_LOGIN in email or email == "noreply@anthropic.com":
            return True
        if name in {"claude", OWNER_LOGIN}:
            return True
    return False


def is_owned_branch(name: str, commits: list[dict]) -> bool:
    if name.casefold().startswith("claude/"):
        return True
    return any(commit_is_owned(commit) for commit in commits)


def commit_identity_available(commit: dict) -> bool:
    details = commit.get("commit") or {}
    return any(
        bool((commit.get(role) or {}).get("login")
             or (details.get(role) or {}).get("name")
             or (details.get(role) or {}).get("email"))
        for role in ("author", "committer")
    )


def classify_branch(
    branch: dict,
    *,
    default_branch: str,
    open_pr_refs: set[str],
    now: datetime,
    compare_commits,
    owned_pr_refs: set[str] | None = None,
    stale_days: int = 30,
) -> tuple[str, str]:
    """Return (action, reason) without mutating the repository."""
    name = str(branch.get("name") or "")
    if not name:
        return "keep", "branch name unavailable"
    if name == default_branch:
        return "keep", "default branch"
    if branch.get("protected"):
        return "keep", "protected branch"
    if name in open_pr_refs:
        return "keep", "used by an open pull request"
    if name in (owned_pr_refs or set()):
        return "keep", "has a pull request authored by trail-b1az3r or Claude"
    tip_commit = branch.get("commit") or {}
    if is_owned_branch(name, [tip_commit]):
        return "keep", "branch tip is attributed to trail-b1az3r or Claude"

    last_commit = branch_tip_time(branch)
    if last_commit is None:
        return "keep", "could not determine the tip commit date"
    age = now.astimezone(timezone.utc) - last_commit
    if age <= timedelta(days=stale_days):
        return "keep", f"last commit is only {age.days} day(s) old"

    try:
        commits = compare_commits(default_branch, name)
    except Exception as exc:  # noqa: BLE001 - unknown history must fail closed
        return "keep", f"could not verify branch ownership: {exc}"
    if not commits:
        if commit_identity_available(tip_commit):
            return "delete", "branch is already contained in the default branch and its tip is not by an allowed author"
        return "keep", "no branch-only commits or reliable tip identity to establish ownership"
    if is_owned_branch(name, commits):
        return "keep", "contains a commit authored by trail-b1az3r or Claude"
    return "delete", f"no commits for at least {stale_days} days; no trail-b1az3r/Claude commits"


def open_pull_request_refs(pulls: list[dict], repository: str) -> set[str]:
    refs: set[str] = set()
    for pull in pulls:
        head = pull.get("head") or {}
        base = pull.get("base") or {}
        head_repo = head.get("repo") or {}
        base_repo = base.get("repo") or {}
        if head_repo.get("full_name") == repository and head.get("ref"):
            refs.add(str(head["ref"]))
        if base_repo.get("full_name") == repository and base.get("ref"):
            refs.add(str(base["ref"]))
    return refs


def owned_pull_request_refs(pulls: list[dict], repository: str) -> set[str]:
    refs: set[str] = set()
    for pull in pulls:
        head = pull.get("head") or {}
        head_repo = head.get("repo") or {}
        author = pull.get("user") or {}
        login = str(author.get("login") or "").casefold()
        if head_repo.get("full_name") == repository and head.get("ref") and login in {OWNER_LOGIN, CLAUDE_LOGIN}:
            refs.add(str(head["ref"]))
    return refs


def write_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("## Stale branch cleanup\n\n")
            summary.write("\n".join(lines))
            summary.write("\n")


def main() -> int:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    default_branch = os.environ.get("DEFAULT_BRANCH", "")
    stale_days = int(os.environ.get("STALE_DAYS", "30"))
    dry_run = os.environ.get("DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}

    if not repository or not token or not default_branch:
        print("Missing GITHUB_REPOSITORY, GITHUB_TOKEN, or DEFAULT_BRANCH", file=sys.stderr)
        return 2
    if stale_days < 1:
        print("STALE_DAYS must be at least 1", file=sys.stderr)
        return 2

    api = GitHubAPI(repository, token)
    branches = api.list_all("/branches")
    pulls = api.list_all("/pulls?state=all")
    open_pulls = [pull for pull in pulls if (pull.get("state") == "open")]
    protected_refs = open_pull_request_refs(open_pulls, repository)
    owned_refs = owned_pull_request_refs(pulls, repository)
    now = datetime.now(timezone.utc)

    report = [
        f"- Default branch: {default_branch}",
        f"- Inactivity threshold: **{stale_days} days**",
        f"- Mode: **{'DRY RUN — no refs changed' if dry_run else 'LIVE — eligible branches will be deleted'}**",
        f"- Branches inspected: **{len(branches)}**",
        "",
    ]
    deleted = 0
    skipped = 0

    for branch in branches:
        name = str(branch.get("name") or "")
        action, reason = classify_branch(
            branch,
            default_branch=default_branch,
            open_pr_refs=protected_refs,
            now=now,
            compare_commits=api.compare_commits,
            owned_pr_refs=owned_refs,
            stale_days=stale_days,
        )
        if action == "keep":
            skipped += 1
            print(f"KEEP   {name}: {reason}")
            report.append(f"- Kept {name} — {reason}")
            continue
        if dry_run:
            print(f"DRY RUN {name}: {reason}")
            report.append(f"- Would delete {name} — {reason}")
            continue
        try:
            api.delete_branch(name)
        except Exception as exc:  # noqa: BLE001 - report failures without hiding others
            skipped += 1
            print(f"ERROR  {name}: {exc}", file=sys.stderr)
            report.append(f"- **Failed to delete {name}:** {exc}")
            continue
        deleted += 1
        print(f"DELETED {name}: {reason}")
        report.append(f"- Deleted {name} — {reason}")

    report.extend(["", f"- Deleted: **{deleted}**", f"- Kept/skipped: **{skipped}**"])
    if dry_run:
        report.append("- Dry run completed. No branches were deleted.")
    write_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
