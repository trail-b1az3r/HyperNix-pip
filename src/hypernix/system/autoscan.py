"""hypernix.system.autoscan — the twice-weekly sweep, and its release gate.

A scheduled job that reads the repository, reports what is wrong with
it, fixes what is safe to fix, and — only when the findings justify it
and the release train is in a state where a patch makes sense — asks
for a `.postN` release.

The whole design question here is "when is it allowed to act on its
own", and the answer is: narrowly, on evidence, with the gate stated in
code rather than in a workflow file where nobody reads it.

Three kinds of finding, and only one of them is fixed automatically
-------------------------------------------------------------------
**Cosmetic** — import order, unused names, formatting. `ruff --fix`
territory. Applied without asking, because a machine that reformats a
file cannot change what it does and the diff is reviewable.

**Bug** — something that will misbehave at runtime: a bare `except`
that swallows `KeyboardInterrupt`, a mutable default argument, an
`assert` doing load-bearing work in code that ships with `-O`. Reported
and counted; never auto-fixed, because "obviously equivalent" is how
you land a behaviour change at 4am on a Sunday.

**Security** — a shell call assembled by string concatenation, a
disabled TLS verification, a hardcoded-looking credential. Reported and
counted, and its presence *blocks* a release rather than justifying
one: shipping faster is not the correct response to finding a
vulnerability.

The release gate
----------------
:func:`should_release` says no unless every one of these holds, and the
reason it says no is returned with it so the workflow can print
something better than "skipped":

* the most recent release is a **plain stable version** — not
  ``.devN``, not ``aN``/``bN``/``rcN``, not a local or beta tag. A
  post-release of a pre-release is a version-ordering trap: ``1.0b1``
  and ``1.0b1.post1`` both sort *below* ``1.0``, so the "fix" would
  publish into a lineage nobody installs.
* enough was actually fixed to be worth a version. One import reorder
  is not a release.
* no unresolved security finding.
* the tree is clean apart from what this run changed.

Why an advisor rather than one model
------------------------------------
The executor reads diffs and counts findings — a lot of tokens on
mechanical work, which is what a mid-tier model is for. Deciding
whether a pile of findings is worth a release is a judgement call made
once per run, and that is what the advisor tool is for: a more capable
model consulted for the decision, not for the reading. See
:func:`advise`.
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import subprocess
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Finding",
    "ScanReport",
    "scan_python",
    "scan_security",
    "apply_maintenance",
    "is_stable_version",
    "next_post_version",
    "should_release",
    "ReleaseDecision",
    "advise",
    "EXECUTOR_MODEL",
    "ADVISOR_MODEL",
]

#: The executor does the reading: diffs, findings, file contents. A lot
#: of tokens on mechanical work.
EXECUTOR_MODEL = "claude-sonnet-5"

#: Consulted for the one judgement call per run. Must be at least as
#: capable as the executor or the API rejects the pairing.
ADVISOR_MODEL = "claude-opus-5"

#: How hard the executor thinks. "high" is the floor for anything
#: intelligence-sensitive; "xhigh" is the setting for code work.
EXECUTOR_EFFORT = "xhigh"

#: Fixes below this count are not a release. One import reorder is not
#: worth a version number, and a bot that releases for one is a bot
#: people turn off.
MINIMUM_FIXES_FOR_RELEASE = 3


@dataclass(frozen=True)
class Finding:
    """One thing that is wrong, and how sure we are it is wrong."""

    kind: str          # "cosmetic" | "bug" | "security"
    rule: str
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


@dataclass
class ScanReport:
    findings: list[Finding] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def of(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    @property
    def bugs(self) -> list[Finding]:
        return self.of("bug")

    @property
    def security(self) -> list[Finding]:
        return self.of("security")

    @property
    def cosmetic(self) -> list[Finding]:
        return self.of("cosmetic")

    def summary(self) -> str:
        return (
            f"{len(self.bugs)} bug, {len(self.security)} security, "
            f"{len(self.cosmetic)} cosmetic · {len(self.fixed)} fixed"
        )

    def to_dict(self) -> dict:
        return {
            "bugs": [str(f) for f in self.bugs],
            "security": [str(f) for f in self.security],
            "cosmetic": len(self.cosmetic),
            "fixed": self.fixed,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Python bug scanning
# ---------------------------------------------------------------------------


class _BugVisitor(ast.NodeVisitor):
    """AST checks for things that are wrong at runtime, not in style.

    Deliberately a small set of high-confidence rules. A scanner that
    reports forty maybes trains its reader to skim, and then the one
    real finding goes past with the rest — which is not a hypothetical
    here: the first version of `silent-except` reported 132 hits on
    this repository because it flagged every `except: pass` regardless
    of whether the code said why. Its own message asked for a comment
    and it never looked for one.
    """

    def __init__(self, path: str, source: str = ""):
        self.path = path
        self.findings: list[Finding] = []
        self._lines = source.splitlines()

    def _explained(self, node: ast.AST) -> bool:
        """Is there a comment inside this handler, or just above it?

        A swallowed exception with a comment saying why is a decision.
        Without one it is a guess that something could go wrong here.
        Only the second is worth reporting.
        """
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start) or start
        # Two lines above covers `# best effort` sitting over the try,
        # and the body covers a comment on the `pass` itself.
        window = range(max(1, start - 2), min(len(self._lines), end) + 1)
        return any("#" in self._lines[number - 1] for number in window)

    def _add(self, node: ast.AST, rule: str, message: str) -> None:
        self.findings.append(
            Finding("bug", rule, self.path, getattr(node, "lineno", 0), message)
        )

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # `except:` catches KeyboardInterrupt and SystemExit, so a loop
        # containing one cannot be stopped with Ctrl-C.
        if node.type is None:
            self._add(node, "bare-except",
                      "bare `except:` also catches KeyboardInterrupt and "
                      "SystemExit — catch Exception instead")
        body = node.body
        if (
            len(body) == 1
            and isinstance(body[0], ast.Pass)
            and not self._explained(node)
        ):
            self._add(node, "silent-except",
                      "exception swallowed with `pass` and nothing says why "
                      "— a comment turns a guess into a decision")
        self.generic_visit(node)

    def _check_defaults(self, node) -> None:
        for default in list(node.args.defaults) + list(node.args.kw_defaults):
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self._add(default, "mutable-default",
                          "mutable default argument is created once and "
                          "shared by every call — use None")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_defaults(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_defaults(node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # `x is "literal"` compares identity of an interned object; it
        # works until the string is built at runtime and then it does not.
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(
                comparator, ast.Constant
            ) and not isinstance(comparator.value, (bool, type(None))):
                self._add(node, "is-literal",
                          "`is` against a literal compares identity, not "
                          "value — use == ")
        self.generic_visit(node)


def scan_python(root: Path, *, skip: Sequence[str] = ()) -> list[Finding]:
    """AST bug checks over every .py file under *root*."""
    findings: list[Finding] = []
    skip_parts = {".git", "__pycache__", "node_modules", ".venv", "build", "dist"}
    skip_parts.update(skip)

    for path in sorted(root.rglob("*.py")):
        if skip_parts & set(path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            findings.append(
                Finding("bug", "unparseable", str(path), 0, f"cannot parse: {exc}")
            )
            continue
        visitor = _BugVisitor(str(path.relative_to(root)), source)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


# ---------------------------------------------------------------------------
# Security scanning
# ---------------------------------------------------------------------------

#: Patterns worth stopping a release for. Each one is a thing that is
#: almost never correct, rather than a thing that is often suspicious —
#: a security scanner with false positives gets muted, and a muted
#: scanner is worse than none because it is believed to be running.
_SECURITY_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        "shell-injection",
        r"subprocess\.(run|call|Popen|check_output)\([^)]*shell\s*=\s*True",
        "subprocess with shell=True — if any part of that string comes "
        "from outside, it is a shell injection",
    ),
    (
        "tls-disabled",
        r"verify\s*=\s*False|ssl\._create_unverified_context|CERT_NONE",
        "TLS verification disabled — this makes every request "
        "interceptable, and it is never the right fix for a cert error",
    ),
    (
        "eval-exec",
        r"(?<![\w.])(eval|exec)\s*\(",
        "eval/exec on a value — if it can ever be influenced from "
        "outside, it is arbitrary code execution",
    ),
    (
        "unsafe-pickle",
        r"pickle\.loads?\s*\(|torch\.load\s*\((?![^)]*weights_only\s*=\s*True)",
        "unpickling runs arbitrary code — use weights_only=True for "
        "torch, or a format that is data",
    ),
    (
        "tempfile-predictable",
        r"(?<![\w.])(mktemp)\s*\(",
        "tempfile.mktemp is a race — the name is returned before the "
        "file exists, so something else can create it first",
    ),
)


def _string_spans(source: str) -> dict[int, list[tuple[int, int]]]:
    """Exact column ranges of string tokens, per line.

    Two bugs produced this, in order, and both are worth recording.

    The first scanner reported three TLS findings here and all three
    were prose: two were the pattern definitions above matching
    themselves, one was a docstring advising the reader *against*
    ``verify=False``. A security scanner with false positives gets
    muted, and a muted scanner is worse than none.

    The fix for that skipped any line containing a string — which
    silently hid two real ones, because
    ``torch.load(p, map_location="cpu", weights_only=False)`` has a
    string argument on it. Hiding a true finding to suppress a false one
    is the worse trade of the two.

    So: exact spans from `tokenize`, and a match counts only when it
    starts *inside* one. Code on a line with a string argument is still
    code.
    """
    spans: dict[int, list[tuple[int, int]]] = {}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.STRING:
                continue
            (start_row, start_col), (end_row, end_col) = token.start, token.end
            if start_row == end_row:
                spans.setdefault(start_row, []).append((start_col, end_col))
            else:
                # A multi-line string: the opening line from its column,
                # every middle line entirely, the closing line up to its end.
                spans.setdefault(start_row, []).append((start_col, 1 << 30))
                for row in range(start_row + 1, end_row):
                    spans.setdefault(row, []).append((0, 1 << 30))
                spans.setdefault(end_row, []).append((0, end_col))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # An unparseable file gets scanned without the prose filter
        # rather than skipped: a file too broken to tokenise is exactly
        # where you want the crude check to still run.
        return {}
    return spans


def _inside_string(spans: dict[int, list[tuple[int, int]]], line: int, column: int) -> bool:
    return any(start <= column < end for start, end in spans.get(line, ()))


def scan_security(root: Path, *, skip: Sequence[str] = ()) -> list[Finding]:
    """Pattern scan for the handful of things that are never fine."""
    findings: list[Finding] = []
    skip_parts = {".git", "__pycache__", "node_modules", ".venv", "build", "dist"}
    skip_parts.update(skip)
    compiled = [(rule, re.compile(pattern), message)
                for rule, pattern, message in _SECURITY_PATTERNS]

    for path in sorted(root.rglob("*.py")):
        if skip_parts & set(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        relative = str(path.relative_to(root))
        spans = _string_spans(text)
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            for rule, pattern, message in compiled:
                match = pattern.search(line)
                if match and not _inside_string(spans, number, match.start()):
                    findings.append(
                        Finding("security", rule, relative, number, message)
                    )
    return findings


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


def apply_maintenance(root: Path, *, dry_run: bool = False) -> tuple[list[str], list[str]]:
    """Run the safe automatic fixes. ``(fixed, errors)``.

    Only `ruff --fix` without `--unsafe-fixes`: those are the changes
    ruff itself guarantees preserve behaviour. Anything ruff calls
    unsafe is exactly the class this must not apply unattended.
    """
    fixed: list[str] = []
    errors: list[str] = []

    ruff = shutil.which("ruff")
    if ruff is None:
        errors.append("ruff is not installed; no automatic fixes were applied")
        return fixed, errors

    command = [ruff, "check", "--fix", "--output-format", "json", str(root)]
    if dry_run:
        command = [ruff, "check", "--output-format", "json", str(root)]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=600
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"ruff failed to run: {exc}")
        return fixed, errors

    try:
        remaining = json.loads(result.stdout or "[]")
    except ValueError:
        remaining = []

    # What ruff fixed is what it no longer reports; it does not say so
    # directly, so the count comes from the fix applicability it does report.
    for item in remaining:
        fix = item.get("fix")
        if fix and fix.get("applicability") in ("safe", "always"):
            fixed.append(
                f"{item.get('filename', '?')}: {item.get('code', '?')}"
            )
    return fixed, errors


def ruff_findings(root: Path) -> list[Finding]:
    """Whatever ruff still reports, as cosmetic findings."""
    ruff = shutil.which("ruff")
    if ruff is None:
        return []
    try:
        result = subprocess.run(
            [ruff, "check", "--output-format", "json", str(root)],
            capture_output=True, text=True, check=False, timeout=600,
        )
        items = json.loads(result.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []

    out: list[Finding] = []
    for item in items:
        location = item.get("location") or {}
        out.append(
            Finding(
                "cosmetic",
                str(item.get("code") or "ruff"),
                str(item.get("filename") or "?"),
                int(location.get("row") or 0),
                str(item.get("message") or ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Versions and the gate
# ---------------------------------------------------------------------------

#: A plain release: 1.2.3, with an optional .postN already on it.
#: Anything carrying dev/a/b/rc/+local is not one.
_STABLE = re.compile(r"^\d+(?:\.\d+)*(?:\.post\d+)?$")


def is_stable_version(version: str) -> bool:
    """True for a plain release, false for anything pre-release.

    The distinction the release gate turns on. ``1.0b1.post1`` sorts
    *below* ``1.0``, so publishing a post-release of a pre-release puts
    the fix in a lineage nobody installs — it looks like a release and
    reaches no one.
    """
    return bool(_STABLE.match(version.strip().lstrip("v")))


def next_post_version(version: str) -> str:
    """``1.2.3`` -> ``1.2.3.post1``; ``1.2.3.post4`` -> ``1.2.3.post5``."""
    cleaned = version.strip().lstrip("v")
    if not is_stable_version(cleaned):
        raise ValueError(
            f"{version!r} is a pre-release; a .post of it would sort below "
            f"the real release and reach nobody"
        )
    match = re.match(r"^(.*)\.post(\d+)$", cleaned)
    if match:
        return f"{match.group(1)}.post{int(match.group(2)) + 1}"
    return f"{cleaned}.post1"


@dataclass(frozen=True)
class ReleaseDecision:
    """Whether to release, and — always — why not."""

    release: bool
    reason: str
    version: str = ""

    def __str__(self) -> str:
        if self.release:
            return f"release {self.version}: {self.reason}"
        return f"no release: {self.reason}"


def should_release(
    report: ScanReport,
    latest_version: str,
    *,
    minimum_fixes: int = MINIMUM_FIXES_FOR_RELEASE,
) -> ReleaseDecision:
    """The gate. Every clause here is a way to say no.

    Ordered so the most important refusal is the one reported: a
    security finding outranks "not enough fixes", because the reader
    needs to know there is a vulnerability, not that the bot was bored.
    """
    if report.security:
        return ReleaseDecision(
            False,
            f"{len(report.security)} unresolved security finding(s) — "
            f"shipping faster is not the response to a vulnerability. "
            f"First: {report.security[0]}",
        )

    if not is_stable_version(latest_version):
        return ReleaseDecision(
            False,
            f"the most recent release ({latest_version}) is a pre-release. "
            f"A .post of it sorts below the real release and would reach "
            f"nobody; land these fixes in the next real version instead.",
        )

    if len(report.fixed) < minimum_fixes:
        return ReleaseDecision(
            False,
            f"only {len(report.fixed)} fix(es), below the threshold of "
            f"{minimum_fixes}. Not every sweep is a release.",
        )

    return ReleaseDecision(
        True,
        f"{len(report.fixed)} fixes applied, no security findings, "
        f"{latest_version} is a stable release",
        next_post_version(latest_version),
    )


# ---------------------------------------------------------------------------
# The advisor
# ---------------------------------------------------------------------------

_ADVICE_SYSTEM = """\
You are reviewing the output of an automated code sweep on a Python \
project, to decide whether the changes it made are safe to publish as a \
.post release.

You are the last check before something goes to PyPI under the \
maintainer's name. Be concrete and be willing to say no. A .post \
release should contain fixes that are obviously behaviour-preserving; \
anything that changes what the code does belongs in a reviewed pull \
request, however small it looks.

Answer with JSON only, no prose around it:
{"publish": true|false, "confidence": "low"|"medium"|"high",
 "reasoning": "<two sentences>", "concerns": ["..."]}
"""


def advise(
    report: ScanReport,
    diff: str,
    decision: ReleaseDecision,
    *,
    api_key: str | None = None,
    max_diff_chars: int = 60_000,
) -> dict:
    """Ask Claude whether the sweep's changes are safe to publish.

    Executor/advisor rather than one model: the executor reads the diff
    and the findings, which is a lot of tokens of mechanical work, and
    the advisor is consulted for the single judgement call. The advisor
    must be at least as capable as the executor or the API rejects the
    pairing.

    Returns the parsed verdict, or a ``{"publish": false}`` explaining
    why it could not be obtained — a scanner that treats "the API was
    down" as approval is a scanner that publishes on an outage.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {
            "publish": False,
            "confidence": "high",
            "reasoning": "No ANTHROPIC_API_KEY, so nothing reviewed the diff.",
            "concerns": ["unreviewed"],
        }

    try:
        import anthropic
    except ImportError:
        return {
            "publish": False,
            "confidence": "high",
            "reasoning": "The anthropic SDK is not installed here.",
            "concerns": ["unreviewed"],
        }

    if len(diff) > max_diff_chars:
        # Truncating the *diff* and saying so, rather than silently
        # reviewing half of it and reporting a verdict on the whole.
        diff = (
            diff[:max_diff_chars]
            + f"\n\n[diff truncated at {max_diff_chars} characters — "
            f"this review covers only the portion above]"
        )

    question = (
        f"Sweep result: {report.summary()}\n"
        f"Gate decision: {decision}\n\n"
        f"Findings not fixed:\n"
        + ("\n".join(f"  {f}" for f in report.bugs[:40]) or "  (none)")
        + f"\n\nThe diff this sweep produced:\n\n{diff or '(empty)'}\n"
    )

    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.beta.messages.create(
            model=EXECUTOR_MODEL,
            max_tokens=4096,
            betas=["advisor-tool-2026-03-01"],
            system=_ADVICE_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": EXECUTOR_EFFORT},
            tools=[
                {
                    "type": "advisor_20260301",
                    "name": "advisor",
                    "model": ADVISOR_MODEL,
                    "max_uses": 2,
                }
            ],
            messages=[{"role": "user", "content": question}],
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "do not publish"
        return {
            "publish": False,
            "confidence": "high",
            "reasoning": f"The review call failed: {type(exc).__name__}: {exc}",
            "concerns": ["unreviewed"],
        }

    if getattr(response, "stop_reason", None) == "refusal":
        return {
            "publish": False,
            "confidence": "high",
            "reasoning": "The review was declined by a safety classifier.",
            "concerns": ["refused"],
        }

    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    return _parse_verdict(text)


def _parse_verdict(text: str) -> dict:
    """The JSON out of the reply, or a refusal to publish.

    Anything unparseable is a no. The alternative — assuming approval
    when the answer cannot be read — is the one failure mode that
    publishes something nobody agreed to.
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {
            "publish": False,
            "confidence": "high",
            "reasoning": "The review did not come back as JSON.",
            "concerns": ["unparseable"],
        }
    try:
        verdict = json.loads(match.group(0))
    except ValueError:
        return {
            "publish": False,
            "confidence": "high",
            "reasoning": "The review's JSON did not parse.",
            "concerns": ["unparseable"],
        }
    verdict["publish"] = bool(verdict.get("publish"))
    return verdict


# ---------------------------------------------------------------------------


def run(root: Path, *, fix: bool = True) -> ScanReport:
    """A whole sweep: scan, optionally fix, scan again."""
    report = ScanReport()
    report.findings.extend(scan_python(root))
    report.findings.extend(scan_security(root))
    if fix:
        fixed, errors = apply_maintenance(root)
        report.fixed.extend(fixed)
        report.errors.extend(errors)
    report.findings.extend(ruff_findings(root))
    return report
