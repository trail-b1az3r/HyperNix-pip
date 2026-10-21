"""t1api.codegit — git inside a /code sandbox (0.72.8a3).

HyperLink's code section (and the model working in it) can version what
it writes: ``init``, ``clone``, ``status``, ``diff``, ``log``, ``add``,
``commit``, ``branch``, ``checkout``, ``restore`` and ``show``, each a
fixed argv, run in the sandbox's own directory.

Git runs code it finds in a repository unless told not to, so it is told:

* hooks are off (``core.hooksPath=/dev/null``), as is ``core.fsmonitor``;
* only ``https`` is spoken — no ``file://``, ``ssh``, or ``ext::``
  helpers — and cloning needs the sandbox's network permission (``s2``);
* the person's own git configuration is not read (``HOME`` is the
  sandbox, ``GIT_CONFIG_NOSYSTEM``), and nothing waits for a password;
* paths are checked to stay inside the sandbox and refs to be refs, and
  each is passed after ``--`` where git allows it.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any

from .codebox import CodePerms, Sandbox, SandboxError

__all__ = ["OPERATIONS", "git"]

OPERATIONS = ("init", "clone", "status", "diff", "log", "add", "commit", "branch", "checkout",
              "restore", "show")
MAX_OUTPUT = 100_000
_REF = re.compile(r"^(?!-)(?!.*\.\.)[A-Za-z0-9._/-]{1,200}$")
_SAFE = ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "protocol.allow=never",
         "-c", "protocol.https.allow=always", "-c", "safe.directory=*", "-c", "advice.detachedHead=false",
         "-c", "init.defaultBranch=main", "-c", "user.name=HyperLink", "-c", "user.email=hyperlink@localhost",
         "-c", "commit.gpgSign=false", "-c", "core.pager=cat", "-c", "color.ui=never"]


def _env(box: Sandbox) -> dict[str, str]:
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE",
                                                          "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy",
                                                          "GIT_SSL_CAINFO", "REQUESTS_CA_BUNDLE")}
    return {**keep, "HOME": str(box.root / ".hnx"), "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "", "SSH_ASKPASS": "", "GIT_PAGER": "cat", "GIT_OPTIONAL_LOCKS": "0"}


def _path(box: Sandbox, relative: str) -> str:
    text = str(relative or "").strip()
    if not text or text == ".":
        return "."
    if text.startswith(("/", "-")) or ".." in text.replace("\\", "/").split("/"):
        raise SandboxError(f"{text!r} is not a path inside the sandbox", code="bad_path")
    if text.split("/")[0] == ".hnx":
        raise SandboxError(".hnx is the sandbox's own folder", code="bad_path")
    return text


def _ref(name: str) -> str:
    if not _REF.match(str(name or "")):
        raise SandboxError(f"{name!r} is not a branch or commit name", code="bad_ref")
    return name


def _run(box: Sandbox, args: list[str], *, timeout: float, cwd: str | None = None) -> dict[str, Any]:
    exe = shutil.which("git")
    if not exe:
        raise SandboxError("git is not installed on this server", code="no_git", status=501)
    (box.root / ".hnx").mkdir(exist_ok=True)
    try:
        done = subprocess.run([exe, *_SAFE, *args], cwd=cwd or box.root, env=_env(box), capture_output=True,
                              text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"git {args[0]} did not finish within {timeout:.0f}s", code="timeout", status=408) from exc
    out, err = done.stdout or "", done.stderr or ""
    return {"ok": done.returncode == 0, "code": done.returncode, "output": out[:MAX_OUTPUT],
            "error": err[-4000:], "truncated": len(out) > MAX_OUTPUT}


def _is_repo(box: Sandbox) -> bool:
    return (box.root / ".git").is_dir()


def git(box: Sandbox, perms: CodePerms, op: str, **args: Any) -> dict[str, Any]:
    """Run one git operation in *box*. Raises :class:`SandboxError` for a refusal."""
    if op not in OPERATIONS:
        raise SandboxError(f"git operation is one of {', '.join(OPERATIONS)}, not {op!r}")
    timeout = float(max(5, perms.timeout))
    if op == "init":
        result = _run(box, ["init", "-q"], timeout=timeout)
        return {"op": op, **result}
    if op == "clone":
        from ..security.safeurl import check_url

        url = str(args.get("url") or "")
        if not url.startswith("https://"):
            raise SandboxError("only https:// repositories are cloned", code="bad_url")
        check_url(url)
        if not perms.network:
            raise SandboxError("cloning needs the network, which sandboxes do not have (s2). An admin can "
                               "turn it on with /code/create/sandbox/perms/s2?:=on", code="network_off", status=403)
        target = _path(box, args.get("path") or "repo")
        depth = ["--depth", str(int(args["depth"]))] if args.get("depth") else ["--depth", "50"]
        result = _run(box, ["clone", "--no-recurse-submodules", "-q", *depth, "--", url, target],
                      timeout=max(timeout, 300.0))
        return {"op": op, "path": target, **result}
    repo = _path(box, args.get("repo") or ".")
    cwd = str(box.root / repo)
    if not os.path.isdir(cwd) or not (box.root / repo / ".git").exists():
        raise SandboxError(f"{repo!r} is not a git repository here; init or clone one first", code="not_a_repo",
                           status=409)
    if op == "status":
        return {"op": op, **_run(box, ["status", "--porcelain=v1", "--branch"], timeout=timeout, cwd=cwd)}
    if op == "diff":
        argv = ["diff", "--no-ext-diff", "--no-color"]
        if args.get("staged"):
            argv.append("--cached")
        if args.get("ref"):
            argv.append(_ref(args["ref"]))
        paths = [_path(box, p) for p in args.get("paths") or []]
        return {"op": op, **_run(box, [*argv, "--", *paths], timeout=timeout, cwd=cwd)}
    if op == "log":
        count = max(1, min(int(args.get("count") or 20), 500))
        return {"op": op, **_run(box, ["log", f"-n{count}", "--date=short",
                                       "--pretty=format:%h%x09%ad%x09%an%x09%s"], timeout=timeout, cwd=cwd)}
    if op == "add":
        paths = [_path(box, p) for p in args.get("paths") or ["."]]
        return {"op": op, **_run(box, ["add", "--", *paths], timeout=timeout, cwd=cwd)}
    if op == "commit":
        message = str(args.get("message") or "").strip()
        if not message:
            raise SandboxError("a commit needs a message")
        argv = ["commit", "-q", "-m", message[:5000]]
        if args.get("all"):
            argv.insert(1, "-a")
        return {"op": op, **_run(box, argv, timeout=timeout, cwd=cwd)}
    if op == "branch":
        if args.get("name"):
            return {"op": op, **_run(box, ["branch", "--", _ref(args["name"])], timeout=timeout, cwd=cwd)}
        return {"op": op, **_run(box, ["branch", "--list", "--no-color"], timeout=timeout, cwd=cwd)}
    if op == "checkout":
        name = _ref(args.get("ref") or "")
        argv = ["checkout", "-q", "-b", name] if args.get("create") else ["checkout", "-q", name, "--"]
        return {"op": op, **_run(box, argv, timeout=timeout, cwd=cwd)}
    if op == "restore":
        paths = [_path(box, p) for p in args.get("paths") or []]
        if not paths:
            raise SandboxError("restore needs the paths to put back")
        return {"op": op, **_run(box, ["restore", "--", *paths], timeout=timeout, cwd=cwd)}
    # show
    return {"op": op, **_run(box, ["show", "--stat", "--no-color", _ref(args.get("ref") or "HEAD"), "--"],
                              timeout=timeout, cwd=cwd)}
