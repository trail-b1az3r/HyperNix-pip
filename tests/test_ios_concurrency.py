"""Swift actor isolation in the on-device engine, checked without a Mac.

There is no Swift toolchain here, so these are structural checks on the
source. They exist because the compiler found something this
environment cannot:

    Call to actor-isolated instance method
    'generate(prompt:systemPrompt:maxTokens:)' in a synchronous main
    actor-isolated context

`ModelRunner` inherits `Actor`, which makes every requirement
actor-isolated by default. `LocalInference` is `@MainActor` and calls
`generate` synchronously, because it returns a stream immediately and
the isolated work happens inside that stream's `Task`.

Making the caller `async` would have been the easy fix and the wrong
one: every view starting a generation would await something that
returns straight away. `nonisolated` says what is actually true. The
cost is that an implementation may not touch isolated state in the
synchronous part of its body — `EchoRunner` did, and had to move the
check into the `Task`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ONDEVICE = REPO_ROOT / "ios" / "HyperLink" / "Sources" / "OnDevice"
LOCAL_RUNNER = ONDEVICE / "LocalRunner.swift"
LLAMA_RUNNER = ONDEVICE / "LlamaRunner.swift"


def code(path: Path) -> str:
    """Source with comments stripped.

    The comments quote the compiler error and name the old shape
    deliberately, so a check that read them would find exactly what it
    is supposed to be looking for the absence of.
    """
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("//")
    )


class TestGenerateIsNotActorIsolated:
    """The declaration that makes the synchronous call legal."""

    def test_the_protocol_requirement_is_nonisolated(self):
        body = code(LOCAL_RUNNER)
        found = re.search(
            r"protocol ModelRunner: Actor \{(.*?)\n\}", body, re.S
        )
        assert found, "ModelRunner protocol not found"
        assert re.search(r"nonisolated func generate", found.group(1)), (
            "generate is actor-isolated again — LocalInference.generate is a "
            "synchronous @MainActor method and cannot call it"
        )

    @pytest.mark.parametrize(
        ("path", "actor_name"),
        [(LOCAL_RUNNER, "EchoRunner"), (LLAMA_RUNNER, "LlamaRunner")],
    )
    def test_every_implementation_matches(self, path, actor_name):
        body = code(path)
        assert f"actor {actor_name}" in body, f"{actor_name} is gone"
        # Each file declares exactly one generate implementation.
        declarations = re.findall(r"^\s*(\w[\w ]*)func generate\(", body, re.M)
        implementations = [d for d in declarations if "nonisolated" in d]
        assert implementations, (
            f"{actor_name}.generate is not nonisolated, so it does not satisfy "
            f"the protocol requirement"
        )

    def test_the_caller_stays_synchronous(self):
        """Which is the point of the whole arrangement."""
        body = code(LOCAL_RUNNER)
        found = re.search(
            r"func generate\(prompt: String\)([^\{]*)\{", body
        )
        assert found, "LocalInference.generate not found"
        assert "async" not in found.group(1), (
            "LocalInference.generate became async — every view starting a "
            "generation now awaits something that returns immediately"
        )


class TestEveryOtherActorCallAwaits:
    """The class of error, not just the one instance.

    `generate` is the only member of `ModelRunner` that is not isolated.
    Every other call from `LocalInference` — a `@MainActor` class — has
    to await, and a missing one is the same compile error in a new
    place.
    """

    def test_all_runner_calls_are_awaited_except_generate(self):
        offenders = []
        for number, line in enumerate(code(LOCAL_RUNNER).splitlines(), 1):
            for call in re.finditer(r"(?<!\w)runner\.(\w+)", line):
                member = call.group(1)
                if member == "generate":
                    continue
                before = line[: call.start()]
                if "await" not in before:
                    offenders.append(f"{number}: {line.strip()}")
        assert not offenders, (
            "actor-isolated calls without await:\n  " + "\n  ".join(offenders)
        )

    def test_the_isolated_members_are_still_isolated(self):
        """If they all became nonisolated the test above proves nothing."""
        body = code(LOCAL_RUNNER)
        protocol = re.search(r"protocol ModelRunner: Actor \{(.*?)\n\}", body, re.S)
        assert protocol
        isolated = re.findall(r"^\s{4}func (\w+)", protocol.group(1), re.M)
        assert {"load", "unload", "cancel"} <= set(isolated), (
            f"expected load/unload/cancel to be isolated, found {isolated}"
        )


class TestNoSharedMutableStatics:
    """What strict concurrency exists to catch.

    `LlamaRunner` guarded `llama_backend_init()` with a mutable
    `static var`, which is shared mutable state across every instance of
    the actor. A global `let` with a side-effecting initialiser is
    Swift's once-only idiom: lazily initialised, and the runtime
    guarantees a single thread-safe initialisation.
    """

    def test_the_runner_has_no_mutable_static(self):
        body = code(LLAMA_RUNNER)
        found = re.findall(r"^\s*(?:private |internal |public )?static var \w+", body, re.M)
        assert not found, f"shared mutable state: {found}"

    def test_the_backend_is_initialised_once_through_a_global_let(self):
        body = code(LLAMA_RUNNER)
        assert re.search(r"^private let llamaBackendReady: Bool = \{", body, re.M)
        assert "llama_backend_init()" in body

    def test_it_is_actually_referenced(self):
        """A global `let` nobody touches is never initialised, so the
        backend would never come up."""
        body = code(LLAMA_RUNNER)
        assert "_ = llamaBackendReady" in body


class TestTheStreamBuilderTouchesNoIsolatedState:
    """`nonisolated` costs this, and the cost is easy to forget.

    The closure `AsyncThrowingStream` takes is escaping and Sendable, so
    reading an actor's stored property inside it is a concurrency error
    — which is what `EchoRunner` did before this. Anything isolated has
    to happen inside the `Task`.
    """

    @pytest.mark.parametrize("path", [LOCAL_RUNNER, LLAMA_RUNNER])
    def test_the_builder_opens_a_task_immediately(self, path):
        body = code(path)
        for match in re.finditer(
            r"nonisolated func generate\(.*?AsyncThrowingStream \{ continuation in\n(.*?)\n        \}",
            body, re.S,
        ):
            first = next(
                (line.strip() for line in match.group(1).splitlines() if line.strip()),
                "",
            )
            assert first.startswith("Task"), (
                f"{path.name}: the stream builder does work before opening a "
                f"Task — {first!r} — which cannot touch isolated state"
            )

    def test_echo_runner_checks_its_state_with_await(self):
        body = code(LOCAL_RUNNER)
        echo = body[body.index("actor EchoRunner"):]
        assert "await self.isLoaded" in echo, (
            "EchoRunner reads its own state without await again"
        )
        assert "guard model != nil else" not in echo, (
            "the synchronous state read is back"
        )


class TestNoMacOnlyAPIs:
    """iOS is not macOS, and the compiler is the only thing that knows.

    `SecTaskCreateFromSelf` and `SecTaskCopyValueForEntitlement` are
    macOS-only — private SPI on iOS — so using them is a compile error
    on a machine this repository does not have:

        error: cannot find 'SecTaskCreateFromSelf' in scope

    Everything on this list is real API that a reasonable person reaches
    for and that iOS does not offer. It is not exhaustive and cannot be;
    it holds the ones already paid for.
    """

    MAC_ONLY = [
        # Code-signing introspection: macOS only.
        "SecTaskCreateFromSelf",
        "SecTaskCopyValueForEntitlement",
        "SecCodeCopySelfSigningInformation",
        "SecStaticCodeCreateWithPath",
        # AppKit, and its usual companions.
        "NSWorkspace",
        "NSApplication",
        "NSPasteboard",
        "NSSavePanel",
        "NSOpenPanel",
        # Keychain access-control API that iOS does not expose.
        "SecKeychainCreate",
        "SecKeychainFindGenericPassword",
        # Process listing.
        "proc_listpids",
    ]

    @pytest.mark.parametrize("symbol", MAC_ONLY)
    def test_it_is_not_used(self, symbol):
        offenders = []
        root = REPO_ROOT / "ios" / "HyperLink" / "Sources"
        for path in sorted(root.rglob("*.swift")):
            if symbol in code(path):
                offenders.append(path.name)
        assert not offenders, (
            f"{symbol} is macOS-only and will not compile for iOS — {offenders}"
        )

    def test_the_keychain_apis_that_are_used_are_the_ios_ones(self):
        """SecItem* is available on both. SecKeychain* is not.

        The app's three keychain users all predate this and compile, so
        this is a guard rather than a fix.
        """
        root = REPO_ROOT / "ios" / "HyperLink" / "Sources"
        joined = "\n".join(code(p) for p in root.rglob("*.swift"))
        assert "SecItemCopyMatching" in joined, "no keychain use found at all"
        assert "SecKeychain" not in joined
