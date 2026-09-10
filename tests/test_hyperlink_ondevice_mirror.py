"""The Swift fit planner must agree with the Python one.

`ios/HyperLink/Sources/OnDevice/ModelFit.swift` is a deliberate
duplicate of `hypernix.hyperlink.ondevice`. The phone cannot run Python
and the decision has to be made on the phone — before a multi-gigabyte
download, and again immediately before a load, when there may be no
network at all.

Duplicated arithmetic drifts. It drifts quietly, and here the symptom of
drift is not a wrong number on a screen: it is the Swift side approving
a model the Python side would refuse, a four-gigabyte download over
cellular, and the OS killing the process partway through the first
reply.

So the constants and the quantisation table are parsed out of the Swift
and compared. There is no Swift toolchain in CI, so this cannot compile
it or run it — that limit is real and is why the checks below are about
the numbers, which are the part that decides whether someone's phone
survives.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from hypernix.hyperlink import ondevice

REPO_ROOT = Path(__file__).resolve().parent.parent
SWIFT = REPO_ROOT / "ios" / "HyperLink" / "Sources" / "OnDevice" / "ModelFit.swift"
MEMORY_SWIFT = REPO_ROOT / "ios" / "HyperLink" / "Sources" / "OnDevice" / "DeviceMemory.swift"


@pytest.fixture(scope="module")
def swift() -> str:
    assert SWIFT.is_file(), f"{SWIFT} is missing"
    return SWIFT.read_text(encoding="utf-8")


def _strip_comments(source: str) -> str:
    """Source with `//` lines removed.

    These files quote compiler errors and name retired APIs on purpose,
    to say what moved and why. A check that reads the comments finds
    exactly the thing it is looking for the absence of.
    """
    return "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith("//") and not line.strip().startswith("///")
    )


def swift_constant(source: str, name: str) -> float:
    """Read `static let <name> = <number>` out of the Swift."""
    match = re.search(
        rf"static let {re.escape(name)}\s*=\s*([0-9_]+(?:\.[0-9]+)?)"
        rf"(?:\s*\*\s*([0-9_]+(?:\s*\*\s*[0-9_]+)*))?",
        source,
    )
    assert match, f"no `static let {name}` in ModelFit.swift"
    value = float(match.group(1).replace("_", ""))
    if match.group(2):
        for factor in match.group(2).split("*"):
            value *= float(factor.strip().replace("_", ""))
    return value


def swift_quant_table(source: str) -> dict[str, float]:
    body = re.search(
        r"static let quantTable: \[String: Double\] = \[(.*?)\n    \]", source, re.S
    )
    assert body, "no quantTable in ModelFit.swift"
    return {
        name: float(bits)
        for name, bits in re.findall(r'"([A-Z0-9_]+)":\s*([0-9.]+)', body.group(1))
    }


class TestTheConstantsAgree:
    """One number differing is one model approved that cannot run."""

    @pytest.mark.parametrize(
        ("swift_name", "python_value"),
        [
            ("safetyMargin", ondevice.SAFETY_MARGIN),
            ("tightMargin", ondevice.TIGHT_MARGIN),
            ("baseOverheadBytes", ondevice.BASE_OVERHEAD_BYTES),
            ("overheadBytesPerContextToken", ondevice.OVERHEAD_BYTES_PER_CONTEXT_TOKEN),
            ("embeddingTensorBits", ondevice.EMBEDDING_TENSOR_BITS),
            ("sizeSafetyFactor", ondevice.SIZE_SAFETY_FACTOR),
        ],
    )
    def test_constant(self, swift, swift_name, python_value):
        assert swift_constant(swift, swift_name) == pytest.approx(python_value), (
            f"{swift_name} differs between ModelFit.swift and "
            f"hypernix/hyperlink/ondevice.py"
        )


class TestTheQuantTableAgrees:
    """A wrong bit width is a wrong size estimate for every model
    quantised that way."""

    def test_every_swift_entry_matches_python(self, swift):
        mismatched = {}
        for name, bits in swift_quant_table(swift).items():
            expected = ondevice.quant_bits(name)
            if expected and abs(expected - bits) > 1e-6:
                mismatched[name] = (bits, expected)
        assert not mismatched, f"swift vs python: {mismatched}"

    def test_no_swift_entry_is_unknown_to_python(self, swift):
        """A name only Swift knows sizes a model the server cannot."""
        unknown = [
            name for name in swift_quant_table(swift)
            if ondevice.quant_bits(name) == 0
        ]
        assert not unknown, f"unknown to hypernix.quant.formats: {unknown}"

    def test_the_common_names_are_in_both(self, swift):
        table = swift_quant_table(swift)
        for name in ("Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q2_K", "IQ4_NL"):
            assert name in table, f"{name} is missing from the Swift table"
            assert ondevice.quant_bits(name) > 0

    def test_the_tables_are_the_same_size(self, swift):
        """A name in one and not the other is drift by another route."""
        swift_names = set(swift_quant_table(swift))
        python_names = {
            f.name for f in ondevice.FORMATS.values() if f.family.value == "gguf"
        } | set(ondevice.SUPPLEMENTARY_GGUF_BITS)
        assert swift_names == python_names, (
            f"only in swift: {sorted(swift_names - python_names)}\n"
            f"only in python: {sorted(python_names - swift_names)}"
        )


class TestTheThingsBothMustRefuse:
    def test_neither_offers_a_neural_engine_backend(self, swift):
        """It cannot run a GGUF, and a settings toggle claiming
        otherwise would be a lie the user acts on."""
        backends = re.search(r"enum ComputeBackend[^{]*\{(.*?)\n\}", swift, re.S)
        assert backends
        cases = re.findall(r"case (\w+)", backends.group(1))
        assert "ane" not in cases and "neuralEngine" not in cases
        assert set(cases) == set(ondevice.ComputeBackend.ALL)

    def test_both_carry_the_explanation_instead(self, swift):
        assert "aneExplanation" in swift
        assert "Core ML" in swift and "Core ML" in ondevice.ANE_EXPLANATION

    def test_neither_subtracts_gpu_layers_from_the_memory_estimate(self, swift):
        """Unified memory: a Metal buffer and a malloc come from the
        same pool. Subtracting offloaded layers approves models that
        cannot run."""
        assert "unified memory" in swift
        # The Swift total is weights + kv + overhead, with no backend term.
        total_line = re.search(r"let total = ([^\n]+)", swift)
        assert total_line
        assert "backend" not in total_line.group(1)


class TestTheMemoryBudgetIsReadCorrectly:
    """The single most important call in on-device inference."""

    @pytest.fixture(scope="class")
    def memory_swift(self) -> str:
        assert MEMORY_SWIFT.is_file()
        return MEMORY_SWIFT.read_text(encoding="utf-8")

    def test_it_asks_the_process_not_the_device(self, memory_swift):
        assert "os_proc_available_memory()" in memory_swift

    def test_physical_memory_is_never_the_available_figure(self, memory_swift):
        """`physicalMemory` is the device's RAM. An app may not use it,
        and a check written against it tells the user a 5 GB model fits
        on a phone that will kill the process at 3 GB."""
        assignment = re.search(
            r"availableBytes:\s*Int\(([^)]+)\)", memory_swift
        )
        assert assignment, "cannot find where availableBytes is set"
        assert "physicalMemory" not in assignment.group(1)
        assert "os_proc_available_memory" in assignment.group(1)

    def test_the_budget_can_be_re_read(self, memory_swift):
        """It shrinks when other apps run, so a model that fit when the
        list was drawn may not fit when the user taps it."""
        assert "func refreshed()" in memory_swift

    def test_the_entitlement_is_read_not_assumed(self, memory_swift):
        """From the provisioning profile, which is the iOS-available route.

        This test used to assert `SecTaskCopyValueForEntitlement`
        appeared in the file — and it still does, in a comment
        explaining why that API cannot be used. Comments are stripped
        first, or this passes on the explanation for its own absence.
        """
        body = _strip_comments(memory_swift)
        assert "SecTaskCreateFromSelf" not in body, (
            "macOS-only API is back: 'cannot find SecTaskCreateFromSelf in scope'"
        )
        assert "SecTaskCopyValueForEntitlement" not in body
        assert "embedded" in body and "mobileprovision" in body
        assert "increased-memory-limit" in body

    def test_the_entitlement_never_changes_the_estimate(self, memory_swift):
        """`false` means "not found" — App Store builds and the
        simulator carry no profile — so it is never evidence of
        absence, and nothing may depend on it."""
        body = _strip_comments(memory_swift)
        assert "hasIncreasedLimit" in body
        # It is stored and reported. If it ever appears in arithmetic,
        # that is a planner trusting a signal that is missing exactly
        # where the app is most constrained.
        fit = (
            REPO_ROOT / "ios" / "HyperLink" / "Sources" / "OnDevice" / "ModelFit.swift"
        ).read_text(encoding="utf-8")
        arithmetic = [
            line for line in _strip_comments(fit).splitlines()
            if "hasIncreasedLimit" in line
            and any(op in line for op in ("*", "+", "-", "/", "safetyMargin"))
        ]
        assert not arithmetic, f"the entitlement is inflating an estimate: {arithmetic}"

    def test_it_counts_performance_cores_only(self, memory_swift):
        """Scheduling llama.cpp work onto efficiency cores costs more in
        scheduler churn than the cores contribute."""
        assert "hw.perflevel0.logicalcpu" in memory_swift


class TestTheMirrorIsDeclared:
    """A duplicate nobody knows is a duplicate is the dangerous kind."""

    def test_the_swift_names_the_python_reference(self, swift):
        assert "hypernix/hyperlink/ondevice.py" in swift

    def test_the_swift_names_this_test(self, swift):
        assert "test_hyperlink_ondevice_mirror" in swift

    def test_the_python_is_the_stated_authority(self, swift):
        assert "reference" in swift.lower()
