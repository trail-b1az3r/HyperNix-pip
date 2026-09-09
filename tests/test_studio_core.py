"""HyperNix Studio's security core, from the Python suite.

Studio is a Qt app and this repository's CI does not install Qt. That is
exactly why its security-critical half -- the tool policy and the runner
that stands between a language model and somebody's filesystem -- has no
Qt dependency: it can be compiled and checked with a C++ compiler and
nothing else, which means it can be checked *here*, on every run, rather
than only on a developer's machine with Qt installed.

The two suites cover different things and neither is redundant:

``tool_policy_test`` is pure. No filesystem, no clock. It covers the
lexical escapes -- ``../../etc/passwd``, a sibling directory that shares
a string prefix, an embedded NUL -- and the rule that nothing which
changes the machine happens on a model's word.

``tool_runner_test`` needs a real disk, because a symlink inside the
workspace pointing out of it passes every string test there is.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parent.parent / "desktop"
SRC = DESKTOP / "src"
TESTS = DESKTOP / "tests"

#: Studio targets Linux desktops -- Qt 6 on Ubuntu 24.04 and Debian 12 --
#: and ``ToolRunner`` is written against POSIX path semantics throughout.
#: ``IsTrulyInside`` compares ``fs::path`` components, and on Windows
#: those carry a root-name (``C:``) that compares as a case-sensitive
#: string, so a workspace given as ``C:\Users\...`` and a resolved path
#: that came back as ``c:\users\...`` are two different directories as
#: far as it is concerned. That is a real gap, and it is a gap in a
#: platform Studio does not ship on: the fix is a Windows path
#: comparison, not a tweak, and it should come with a Windows build to
#: test it against rather than be guessed at from here.
#:
#: So the compile-and-run halves are POSIX-only and say so. The *source*
#: guarantees below -- no way to run a command, no shell in the tool
#: list -- keep running on every platform, because those are the checks
#: that matter most and they read the file rather than the disk.
_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Studio ships on Linux; ToolRunner's path comparison is POSIX "
           "(fs::path components, no root-name or case folding). Needs a "
           "Windows path comparison and a Windows build to test it.",
)


def _compiler() -> str | None:
    for name in ("c++", "g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _build(name: str, tmp_path: Path) -> Path:
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++ compiler on this machine")
    out = tmp_path / name
    result = subprocess.run(
        [
            compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
            str(TESTS / f"{name}.cpp"),
            str(SRC / "ToolPolicy.cpp"), str(SRC / "ToolRunner.cpp"),
            "-o", str(out),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"{name} does not compile cleanly:\n{result.stderr}")
    return out


@_POSIX_ONLY
class TestTheToolPolicy:
    """Pure decisions: what a model may ask for, and what it may not."""

    def test_it_compiles_and_every_check_passes(self, tmp_path):
        binary = _build("tool_policy_test", tmp_path)
        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "0 failures" in result.stdout


@_POSIX_ONLY
class TestTheToolRunner:
    """The filesystem half, against a real filesystem."""

    def test_it_compiles_and_every_check_passes(self, tmp_path):
        binary = _build("tool_runner_test", tmp_path)
        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "0 failures" in result.stdout


class TestThereIsNoWayToRunACommand:
    """The guarantee that does not depend on a check being correct.

    Studio's tool list has no shell, no exec, no eval. Checked in the C++
    tests too, and again here from the outside, because this is the
    property that would be quietly lost by someone adding a convenient
    "run the tests" tool -- and losing it would make every other check in
    ToolPolicy beside the point.
    """

    def test_no_source_file_spawns_a_process(self):
        forbidden = (
            "system(", "popen(", "execv", "execl", "execp", "fork(",
            "QProcess", "posix_spawn",
        )
        for source in sorted(SRC.glob("*.cpp")) + sorted(SRC.glob("*.h")):
            text = source.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, (
                    f"{source.name} contains {token!r}. Studio must have no "
                    f"way to run a command -- see desktop/README.md."
                )

    def test_the_tool_list_names_only_files_and_search(self):
        policy = (SRC / "ToolPolicy.cpp").read_text(encoding="utf-8")
        block = policy.split("std::vector<std::string> ToolPolicy::ToolNames()")[1]
        block = block.split("}")[0]
        # Comments stripped first. The comment inside ToolNames() lists
        # the tools that are deliberately absent -- "no shell, no exec"
        # -- so a naive grep matches the explanation of why they are not
        # there and reports it as their presence. Exactly the same trap
        # caught the grep over `cmd_index` in bin/hypernix-t1.
        code = "\n".join(
            line for line in block.splitlines()
            if not line.strip().startswith("//")
        )

        assert '"read_file"' in code
        assert '"web_search"' in code
        for banned in ("shell", "exec", "bash", "eval", "command", "terminal"):
            assert banned not in code, f"a tool named for {banned!r} appeared"

    def test_qml_cannot_reach_the_runner_directly(self):
        """QML talks to StudioBridge and nothing else. A view that could
        call ToolRunner could call it without asking, and the entire
        design is that a tool runs only after a person agrees."""
        main = (SRC / "main.cpp").read_text(encoding="utf-8")

        assert "setContextProperty" in main
        # Exactly one object is exposed.
        assert main.count("setContextProperty") == 1
        assert '"studio"' in main


class TestTheApprovalDialogHasNoShortcuts:
    """Read out of the QML, because these are absences.

    Each of "approve all", "remember this", a timeout and
    click-outside-to-dismiss is the same feature under a different name:
    a way for a file to be written without anyone having looked. An
    absence cannot be tested by exercising it, so it is tested by
    reading for it -- which at least fails loudly when someone adds one.
    """

    def test_no_blanket_approval(self):
        qml = (DESKTOP / "qml" / "ToolApproval.qml").read_text(encoding="utf-8")
        lowered = qml.lower()

        for phrase in ("approve all", "allow all", "always allow", "remember"):
            # Only in prose that says why it is absent, never as a
            # property or a signal handler.
            for line in lowered.splitlines():
                if phrase in line:
                    assert line.strip().startswith("//"), (
                        f"{phrase!r} appears outside a comment: {line.strip()}"
                    )

    def test_nothing_times_out_into_approval(self):
        qml = (DESKTOP / "qml" / "ToolApproval.qml").read_text(encoding="utf-8")

        assert "Timer" not in qml, "a timer in the approval dialog"

    def test_escape_declines(self):
        qml = (DESKTOP / "qml" / "ToolApproval.qml").read_text(encoding="utf-8")

        assert "Keys.onEscapePressed: studio.rejectTool()" in qml

    def test_the_bridge_has_no_auto_approve_path(self):
        """approveTool() is the only route from the UI into a mutating
        tool, and it is only ever called from the dialog."""
        bridge = (SRC / "StudioBridge.cpp").read_text(encoding="utf-8")

        # runApproved is private and reached from exactly two places:
        # the Allow branch (reads only) and approveTool().
        assert bridge.count("runApproved(") == 3  # definition + 2 calls
        assert "void StudioBridge::approveTool()" in bridge
