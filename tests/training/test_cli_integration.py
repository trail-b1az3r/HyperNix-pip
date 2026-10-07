"""CLI integration tests: each subcommand with three flag-coverage tiers.

For every subcommand we verify:

* ``--help``                    — exits 0, shows a help banner, never tracebacks.
* no-args (or minimal args)     — either runs a happy-path or fails cleanly
                                  with a non-traceback message.
* "half"                        — a minimal but non-trivial flag set.
* "all"                         — the full non-mutually-exclusive flag set.

Commands that require network access (``download``, ``upload``,
``fetch-llama-quantize``) are only exercised via ``--help`` plus a
credential/connectivity failure path that should produce a clean error
string, not a bare Python traceback.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
TIMEOUT_SHORT = 30
TIMEOUT_LONG = 180


def _run(args: list[str], *, timeout: int = TIMEOUT_SHORT) -> subprocess.CompletedProcess:
    # Start from the real parent environment rather than a hand-rolled one.
    # A hardcoded Unix-style PATH (with no SystemRoot/etc.) breaks socket
    # initialization on Windows: subprocess.run(..., env=...) replaces the
    # child's entire environment, and without SystemRoot, Winsock can't
    # locate its provider, so anything touching sockets (asyncio, requests,
    # huggingface_hub, ...) dies with OSError: [WinError 10106]. Inheriting
    # os.environ and overriding only what the test actually needs avoids
    # that while keeping the run hermetic for the two knobs that matter.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["HYPERNIX_AUTO_INSTALL"] = "0"  # never touch pip during tests
    return subprocess.run(
        [sys.executable, "-m", "hypernix.cli", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
        check=False,
    )


def _no_traceback(cp: subprocess.CompletedProcess) -> None:
    combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
    assert "Traceback (most recent call last)" not in combined, (
        f"unexpected traceback:\n{combined}"
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def test_top_level_no_args_prints_usage() -> None:
    cp = _run([])
    assert cp.returncode == 0
    assert "Subcommands" in cp.stdout or "subcommand" in cp.stdout.lower()
    _no_traceback(cp)


def test_top_level_help() -> None:
    cp = _run(["--help"])
    assert cp.returncode == 0
    assert "hypernix" in cp.stdout.lower()
    _no_traceback(cp)


def test_top_level_version() -> None:
    cp = _run(["--version"])
    assert cp.returncode == 0
    assert cp.stdout.strip().startswith("hypernix ")


def test_unknown_subcommand_goes_to_all_parser() -> None:
    # `hypernix foo` used to be "not a subcommand => pass to `all` parser" which
    # would then fail on `foo` as unrecognized. Verify we don't traceback.
    cp = _run(["--this-flag-definitely-does-not-exist"])
    assert cp.returncode != 0
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# Per-subcommand --help coverage (13 subcommands)
# ---------------------------------------------------------------------------

ALL_SUBCOMMANDS = [
    "all",
    "download",
    "convert",
    "quantize",
    "verify",
    "info",
    "upload",
    "doctor",
    "fetch-llama-quantize",
    "train",
    "generate",
    "oven",
    "chat",
]


#: Subcommands that load a PyTorch model; on a Python with no torch wheel
#: (3.15, for now) they say so instead of running.
NEEDS_TORCH = {"all", "convert", "train", "generate", "oven", "chat"}
TORCH_MISSING = importlib.util.find_spec("torch") is None


def _refuses_for_missing_torch(cp: subprocess.CompletedProcess, cmd: str) -> None:
    assert cp.returncode == 2, cp.stderr
    assert f"hypernix {cmd} needs PyTorch" in cp.stderr
    _no_traceback(cp)


@pytest.mark.parametrize("cmd", ALL_SUBCOMMANDS)
def test_subcommand_help(cmd: str) -> None:
    cp = _run([cmd, "--help"])
    if TORCH_MISSING and cmd in NEEDS_TORCH:
        _refuses_for_missing_torch(cp, cmd)
        return
    assert cp.returncode == 0, f"{cmd} --help failed:\n{cp.stderr}"
    # argparse always writes "usage:" on --help
    assert "usage" in cp.stdout.lower() or "usage" in cp.stderr.lower()
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# info — the cheapest command, safe to run in full
# ---------------------------------------------------------------------------

def test_info_no_args() -> None:
    cp = _run(["info"])
    assert cp.returncode == 0
    assert "hypernix" in cp.stdout.lower()
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# doctor — runs locally; --fix is the "all flags" variant
# ---------------------------------------------------------------------------

def test_doctor_no_args() -> None:
    cp = _run(["doctor"])
    # doctor returns non-zero if a runtime dep is missing; either way no
    # traceback and something printed.
    _no_traceback(cp)
    assert cp.stdout or cp.stderr


def test_doctor_fix_flag_parses() -> None:
    # HYPERNIX_AUTO_INSTALL=0 means deps.ensure is a no-op; --fix should be
    # accepted without error and without triggering pip.
    cp = _run(["doctor", "--fix"])
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# convert / quantize / generate / oven / chat — need args, should fail cleanly
# without them rather than traceback.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cmd",
    ["convert", "quantize", "upload", "verify", "generate"],
)
def test_required_args_fail_cleanly(cmd: str) -> None:
    cp = _run([cmd])
    assert cp.returncode != 0
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# train — three tiers: init no-args, init happy-path (half), init all-flags
# ---------------------------------------------------------------------------

def test_train_no_args_fails_cleanly() -> None:
    cp = _run(["train"])
    assert cp.returncode != 0
    _no_traceback(cp)


def test_train_init_no_args_fails_cleanly() -> None:
    cp = _run(["train", "init"])
    assert cp.returncode != 0
    _no_traceback(cp)


def test_train_init_half_flags(tmp_path: Path) -> None:
    out_dir = tmp_path / "tiny"
    cp = _run([
        "train", "init",
        "--out-dir", str(out_dir),
        "--vocab-size", "256",
        "--hidden-size", "8",
        "--intermediate-size", "16",
        "--num-hidden-layers", "1",
        "--num-attention-heads", "2",
        "--max-position-embeddings", "16",
    ], timeout=TIMEOUT_LONG)
    if TORCH_MISSING:
        _refuses_for_missing_torch(cp, "train")
        return
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)
    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists()


def test_train_init_all_flags(tmp_path: Path) -> None:
    out_dir = tmp_path / "full"
    cp = _run([
        "train", "init",
        "--out-dir", str(out_dir),
        "--vocab-size", "256",
        "--hidden-size", "8",
        "--intermediate-size", "16",
        "--num-hidden-layers", "1",
        "--num-attention-heads", "2",
        "--num-key-value-heads", "1",
        "--max-position-embeddings", "16",
        "--rope-theta", "10000.0",
        "--tie-word-embeddings",
        "--seed", "42",
    ], timeout=TIMEOUT_LONG)
    if TORCH_MISSING:
        _refuses_for_missing_torch(cp, "train")
        return
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)
    assert (out_dir / "config.json").exists()


def test_train_run_no_args_fails_cleanly() -> None:
    cp = _run(["train", "run"])
    assert cp.returncode != 0
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# generate — half & all flags against a real tiny snapshot
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_snapshot(tmp_path: Path) -> Path:
    from hypernix import HyperNixConfig, init_from_scratch

    cfg = HyperNixConfig(
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16,
    )
    snap = tmp_path / "snap"
    init_from_scratch(str(snap), cfg, tokenizer_source=None, seed=0)
    return snap


def test_generate_half_flags(tiny_snapshot: Path) -> None:
    cp = _run([
        "generate",
        "--model-dir", str(tiny_snapshot),
        "--max-new-tokens", "2",
    ], timeout=TIMEOUT_LONG)
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)


def test_generate_all_flags(tiny_snapshot: Path) -> None:
    cp = _run([
        "generate",
        "--model-dir", str(tiny_snapshot),
        "--prompt", "hello",
        "--max-new-tokens", "2",
        "--temperature", "0.0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--seed", "0",
        "--device", "cpu",
        "--dtype", "float32",
    ], timeout=TIMEOUT_LONG)
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# oven — use --model-dir to avoid any network fetch
# ---------------------------------------------------------------------------

def test_oven_half_flags_no_prompt(tiny_snapshot: Path) -> None:
    # No --prompt / no --fill-* => just preheats and exits.
    cp = _run([
        "oven",
        "--model-dir", str(tiny_snapshot),
        "--device", "cpu",
    ], timeout=TIMEOUT_LONG)
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)


def test_oven_all_flags_with_prompt(tiny_snapshot: Path) -> None:
    cp = _run([
        "oven",
        "--model-dir", str(tiny_snapshot),
        "--device", "cpu",
        "--dtype", "float32",
        "--prompt", "def add(a, b):",
        "--max-new-tokens", "2",
        "--temperature", "0.0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--seed", "0",
        "--quiet",
    ], timeout=TIMEOUT_LONG)
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# chat — single-turn via --message
# ---------------------------------------------------------------------------

def test_chat_half_flags(tiny_snapshot: Path) -> None:
    cp = _run([
        "chat",
        "--model-dir", str(tiny_snapshot),
        "--message", "hi",
        "--max-new-tokens", "2",
    ], timeout=TIMEOUT_LONG)
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)


def test_chat_all_flags(tiny_snapshot: Path) -> None:
    cp = _run([
        "chat",
        "--model-dir", str(tiny_snapshot),
        "--device", "cpu",
        "--dtype", "float32",
        "--system", "You are terse.",
        "--message", "hi",
        "--max-new-tokens", "2",
        "--temperature", "0.0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--seed", "0",
        "--quiet",
    ], timeout=TIMEOUT_LONG)
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    _no_traceback(cp)


# ---------------------------------------------------------------------------
# verify — on a non-existent path must fail cleanly
# ---------------------------------------------------------------------------

def test_verify_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "not-there.gguf"
    cp = _run(["verify", str(missing)])
    assert cp.returncode != 0
    _no_traceback(cp)
    assert "not found" in cp.stderr.lower() or "not found" in cp.stdout.lower()


# ---------------------------------------------------------------------------
# fetch-llama-quantize — network; just check --help
# (full test lives in test_fetcher.py with its own mocks)
# ---------------------------------------------------------------------------

def test_fetch_llama_quantize_parses() -> None:
    cp = _run(["fetch-llama-quantize", "--help"])
    assert cp.returncode == 0
    _no_traceback(cp)
