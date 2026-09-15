"""Regression tests for the ruff cleanup pass.

These tests are intentionally NOT pinned to a specific hypernix release
(unlike test_v0704b12_*.py, test_v052_6.py, etc). They exercise the
underlying behavior of the code paths that were touched to satisfy
ruff (I001, F841, F401, B007, B904, UP037) so that future refactors of
those modules keep working the same way, regardless of what version
number __init__.py reports.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Package-level sanity: the package still imports cleanly and exposes the
# conditional `tv`/`tvtop`/`tvtop_plus_plus`/`spinner` submodules regardless
# of the currently installed version string.
# ---------------------------------------------------------------------------

def test_package_imports_without_error() -> None:
    import hypernix

    assert hasattr(hypernix, "__version__")
    assert isinstance(hypernix.__version__, str)
    assert hypernix.__version__  # non-empty, but we don't pin the value


def test_conditional_submodules_are_importable() -> None:
    """The `from . import tv, tvtop, tvtop_plus_plus, spinner` branch in
    __init__.py must remain valid (it was reordered/reformatted by ruff's
    import-sort fix, so this guards against an accidental breakage)."""
    from hypernix import spinner, tv, tvtop, tvtop_plus_plus

    for mod in (spinner, tv, tvtop, tvtop_plus_plus):
        assert mod is not None


# ---------------------------------------------------------------------------
# hypernix.spinner — UP037 removed quotes from `-> "Spinner"` annotations.
# Confirm start()/__enter__() still return a usable Spinner instance and
# that the context manager protocol works end to end.
# ---------------------------------------------------------------------------

def test_spinner_start_returns_self() -> None:
    from hypernix.spinner import Spinner

    sp = Spinner("working")
    ret = sp.start()
    try:
        assert ret is sp
    finally:
        sp.stop()


def test_spinner_context_manager_returns_self_and_stops() -> None:
    from hypernix.spinner import Spinner

    with Spinner("working") as sp:
        assert sp.text == "working"
        sp.update("still working")
        assert sp.text == "still working"
    # After exiting the context, the background thread should be stopped.
    assert sp._stop.is_set()


def test_spinner_context_manager_propagates_no_exception() -> None:
    from hypernix.spinner import Spinner

    with Spinner("task"):
        pass  # no exception raised, __exit__ should return False/None cleanly


# ---------------------------------------------------------------------------
# hypernix.assistant — F841 removed the unused `status` binding from two
# `with console.status(...) as status:` blocks. The blocks themselves
# should still function as plain context managers.
# ---------------------------------------------------------------------------

def test_assistant_status_context_managers_present() -> None:
    import inspect

    from hypernix import assistant

    source = inspect.getsource(assistant)
    # The fixed lines should no longer bind an unused `as status` name.
    assert "as status:" not in source
    assert "console.status(" in source


# ---------------------------------------------------------------------------
# hypernix.fizzle — F401 removed the unused `AutoConfig` import, B007
# renamed the unused loop variable `cid` -> `_cid` in fuze_tokenizers().
# ---------------------------------------------------------------------------

def test_fuze_tokenizers_selects_llm_tokenizer_ignoring_key() -> None:
    """fuze_tokenizers() selects based on the *value's type name*
    containing "Tokenizer", never the dict key. Renaming the loop
    variable from `cid` to `_cid` (the B007 fix) must not change this."""
    from hypernix.fizzle import FuzedModelArch

    arch = FuzedModelArch(components=[])

    class FakeTokenizer:
        def __init__(self):
            self.added = None

        def add_special_tokens(self, mapping):
            self.added = mapping

    class FakeProcessor:  # name deliberately does NOT contain "Tokenizer"
        pass

    fake_tok = FakeTokenizer()
    # Use key names unrelated to the tokenizer's identity to prove
    # selection depends on the value's type, not the (unused) key.
    arch.tokenizers = {
        "asr-component-key": FakeProcessor(),
        "llm-component-key": fake_tok,
    }

    result = arch.fuze_tokenizers()
    assert result is fake_tok
    assert result.added == {
        "additional_special_tokens": [
            "<|asr_start|>", "<|asr_end|>", "<|tts_start|>", "<|tts_end|>",
            "<|image_start|>", "<|image_end|>",
        ]
    }


def test_fuze_tokenizers_with_only_non_llm_types_returns_none() -> None:
    from hypernix.fizzle import FuzedModelArch

    arch = FuzedModelArch(components=[])

    class FakeProcessor:
        pass

    arch.tokenizers = {"asr-component-key": FakeProcessor()}
    assert arch.fuze_tokenizers() is None


def test_fuze_tokenizers_returns_none_with_no_components() -> None:
    from hypernix.fizzle import FuzedModelArch

    arch = FuzedModelArch(components=[])
    assert arch.tokenizers == {}
    assert arch.fuze_tokenizers() is None


def test_fizzle_module_has_no_autoconfig_reference() -> None:
    import inspect

    from hypernix import fizzle

    source = inspect.getsource(fizzle)
    assert "AutoConfig" not in source


# ---------------------------------------------------------------------------
# hypernix.quantize — F401 removed the unused `import llama_cpp` (kept the
# `from llama_cpp import ...` names actually used), B904 added `from err`
# to the FileNotFoundError -> RuntimeError re-raise.
# ---------------------------------------------------------------------------

def test_quantize_missing_binary_raises_runtime_error_with_cause(tmp_path) -> None:
    from hypernix import quantize

    src = tmp_path / "model.gguf"
    src.write_bytes(b"\x00")
    out = tmp_path / "model-quant.gguf"

    with patch.object(quantize, "_find_llama_quantize", return_value="/nonexistent/llama-quantize"):
        with patch("subprocess.run", side_effect=FileNotFoundError("no such binary")):
            with pytest.raises(RuntimeError) as excinfo:
                quantize.quantize_gguf(src, out, "Q4_K_M")

    assert "not found or not executable" in str(excinfo.value)
    # B904 requires explicit exception chaining via `raise ... from err`.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_quantize_module_has_no_bare_llama_cpp_import() -> None:
    import inspect

    from hypernix import quantize

    source = inspect.getsource(quantize)
    assert "import llama_cpp\n" not in source
    assert "from llama_cpp import llama_model_quantize" in source


# ---------------------------------------------------------------------------
# Whole-repo lint gate: keeps this suite honest if new code reintroduces any
# of the same classes of issues, without hardcoding a version identifier.
# ---------------------------------------------------------------------------

def test_ruff_check_passes_on_src_and_tests() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_test_module_imports_fastapi_unguarded() -> None:
    """A bare ``from fastapi import …`` in a test module breaks collection.

    The base CI job installs the package without the ``[t1api]`` extra, so
    fastapi is absent there. An unguarded import is not a skip — it is a
    *collection error*, which aborts the whole run before any test
    executes, including every test that had nothing to do with the server.
    One file can take the entire suite down.

    Guarded means either ``pytest.importorskip("fastapi")`` at module
    level, or importing it inside the fixture or function that needs it
    behind a availability flag.
    """
    tests_dir = Path(__file__).resolve().parent
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        skipped_already = False
        for number, line in enumerate(lines, 1):
            # Matched without the closing paren: `importorskip("fastapi",
            # reason="…")` is the same guard and was being read as no
            # guard at all, which failed a file that had done it right.
            if 'importorskip("fastapi"' in line:
                # Everything after this point is unreachable without
                # fastapi, so a module-level import below it is fine.
                skipped_already = True
                continue
            stripped = line.strip()
            if not stripped.startswith(("import fastapi", "from fastapi")):
                continue
            # An indented import is either deferred into a function or
            # wrapped in a try/except, so it cannot break collection.
            # Whether the file mentions a flag elsewhere is irrelevant:
            # what matters is this line, on this line's indentation.
            if line[:1] in (" ", "\t"):
                continue
            if skipped_already:
                continue
            offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        "these modules import fastapi at collection time without a guard, "
        "which aborts the whole run when the [t1api] extra is absent: "
        + ", ".join(offenders)
    )
