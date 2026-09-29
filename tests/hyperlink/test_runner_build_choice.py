"""The runner picks a llama.cpp that can read the model, or says why not.

From a real session: ``hnx-t1 runner load hypernix-3-mini`` started a
stock llama-server on a GGUF with an INT3 layer. llama.cpp's own answer
was "tensor 'blk.2.ffn_up.weight' has invalid ggml type 210. should be
in [0, 43)", buried in the output of a refusal that said the model "did
not start within 300s" -- about a process that exited after 50 ms.

Three fixes, each tested here: a build is asked which HyperNix types it
decodes (one symbol per type, so a build patched before INT3 existed is
caught too); a patched build is preferred over a stock one found first;
and a model no build can read is refused before anything starts, in
words. A process that exits is reported as having exited.
"""
from __future__ import annotations

import logging
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from hypernix.hyperlink.managed import ManagedError, ManagedRunner
from hypernix.quant import runtime_bridge as bridge
from hypernix.quant.gguf import GGMLType, GGUFWriter


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No real home, no real builds, no stray override."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("HNX_LLAMA_BUILD", raising=False)
    # The checkout's own build, if a developer has one, is not this test's.
    monkeypatch.setattr(bridge, "candidate_build_dirs", _only_home_candidates)
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _only_home_candidates() -> list[Path]:
    paths = []
    recorded = bridge.recorded_build()
    if recorded is not None:
        paths.append(recorded)
    paths += [Path.home() / "llama.cpp" / "build", Path.home() / ".hypernix" / "llama.cpp" / "build"]
    return paths


def fake_build(root: Path, *, types: set[int] | None = None, server: str = "exit 0") -> Path:
    """A built llama.cpp whose base library exports *types*' decoders.

    ``None`` is a stock build: no HyperNix symbols at all.
    """
    bin_dir = root / "build" / "bin"
    bin_dir.mkdir(parents=True)
    blob = b"\x7fELF"
    for type_id in sorted(types or ()):
        blob += f"hnx_ggml_to_float_{bridge.HNX_TYPE_SYMBOLS[type_id]}".encode() + b"\0"
    for stem in bridge.CORE_LIBRARIES:
        (bin_dir / f"{stem}{bridge._library_suffix()}").write_bytes(blob)
    script = bin_dir / "llama-server"
    script.write_text(f"#!/bin/sh\n{server}\n")
    script.chmod(0o755)
    return root / "build"


ALL = set(bridge.HNX_TYPE_SYMBOLS)
BEFORE_INT3 = ALL - {int(GGMLType.HNX_INT3), int(GGMLType.HNX_FP8)}


def gguf_with(path: Path, *types: int) -> Path:
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    for i, kind in enumerate(types):
        writer.add_tensor(f"blk.{i}.ffn_up.weight", (256, 1), int(kind))
    writer.write(lambda t: b"\0" * t.nbytes)
    return path


class TestWhatABuildReads:
    def test_each_type_is_read_from_its_own_symbol(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "old", types=BEFORE_INT3))
        assert build.patched
        assert build.missing({210, 211, 205}) == [210, 211]

    def test_a_build_patched_before_a_type_existed_says_so(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "old", types=BEFORE_INT3))
        assert "rebuild" in build.note

    def test_a_current_build_misses_nothing(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "new", types=ALL))
        assert build.missing(ALL) == [] and build.note == ""

    def test_upstream_types_are_never_missing(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "stock"))
        assert build.missing({0, 1, 8, 12}) == []

    def test_the_model_types_are_read_from_the_header(self, tmp_path):
        path = gguf_with(tmp_path / "m.gguf", GGMLType.F32, GGMLType.HNX_INT3)
        assert bridge.model_types(path) == {0, 210}


class TestWhichBuildIsChosen:
    def test_a_patched_build_wins_over_a_stock_one_found_first(self, tmp_path, monkeypatch):
        """A stock ~/llama.cpp first in the list no longer shadows a
        patched build further down it."""
        stock = fake_build(Path.home() / "llama.cpp")
        patched = fake_build(tmp_path / "repo" / "llama.cpp", types=ALL)
        monkeypatch.setattr(bridge, "candidate_build_dirs", lambda: [stock, patched])
        assert bridge.find_build().bin_dir == patched / "bin"
        assert bridge.find_build(need={210}).bin_dir == patched / "bin"

    def test_one_that_reads_the_model_wins_over_an_older_patch(self, tmp_path, monkeypatch):
        old = fake_build(tmp_path / "old", types=BEFORE_INT3)
        new = fake_build(tmp_path / "new", types=ALL)
        monkeypatch.setattr(bridge, "candidate_build_dirs", lambda: [old, new])
        assert bridge.find_build(need={210}).bin_dir == new / "bin"

    def test_build_sh_records_where_it_built(self, tmp_path):
        patched = fake_build(tmp_path / "anywhere", types=ALL)
        (Path.home() / ".hypernix").mkdir()
        (Path.home() / ".hypernix" / "llama-build").write_text(f"{patched}\n")
        assert bridge.recorded_build() == patched
        assert bridge.find_build().bin_dir == patched / "bin"

    def test_the_override_is_used_as_named(self, tmp_path, monkeypatch):
        stock = fake_build(tmp_path / "stock")
        fake_build(Path.home() / "llama.cpp", types=ALL)
        monkeypatch.setenv("HNX_LLAMA_BUILD", str(stock))
        assert bridge.find_build(need={210}).bin_dir == stock / "bin"

    def test_the_build_script_writes_the_pointer(self):
        script = Path(__file__).resolve().parents[2] / "native" / "ggml-hnx" / "build.sh"
        text = script.read_text(encoding="utf-8")
        assert "llama-build" in text and '$TARGET/build' in text


@pytest.mark.skipif(sys.platform == "win32", reason="the fake server is a shell script")
class TestTheRunnerRefusesInWords:
    def _no_start(self, monkeypatch):
        def refuse(*a, **k):
            raise AssertionError("llama-server must not be started")
        monkeypatch.setattr(subprocess, "Popen", refuse)

    def test_a_stock_build_and_an_int3_model(self, tmp_path, monkeypatch):
        fake_build(Path.home() / "llama.cpp")
        model = gguf_with(tmp_path / "HyperNix.3-mini.gguf", GGMLType.F32, GGMLType.HNX_INT3)
        self._no_start(monkeypatch)
        with pytest.raises(ManagedError) as caught:
            ManagedRunner(port=18999).load(model)
        text = str(caught.value)
        assert "HyperNix.3-mini.gguf uses INT3 (210)" in text
        assert "stock build" in text and "build.sh" in text

    def test_a_build_patched_before_int3(self, tmp_path, monkeypatch):
        fake_build(Path.home() / "llama.cpp", types=BEFORE_INT3)
        model = gguf_with(tmp_path / "m.gguf", GGMLType.HNX_INT3, GGMLType.HNX_FP8)
        self._no_start(monkeypatch)
        with pytest.raises(ManagedError) as caught:
            ManagedRunner(port=18999).load(model)
        text = str(caught.value)
        assert "INT3 (210), FP8 (211)" in text and "patched before" in text

    def test_an_ordinary_model_on_a_stock_build_still_starts(self, tmp_path):
        """Stock llama.cpp is fine for a model with no HyperNix types."""
        fake_build(Path.home() / "llama.cpp", server="echo 'bad magic'; exit 3")
        model = gguf_with(tmp_path / "plain.gguf", GGMLType.F32)
        with pytest.raises(ManagedError) as caught:
            ManagedRunner(port=18999).load(model, timeout=20)
        # It got as far as starting the server, which then failed.
        assert "exited (code 3)" in str(caught.value)

    def test_an_exit_is_not_reported_as_a_timeout(self, tmp_path):
        fake_build(Path.home() / "llama.cpp", types=ALL, server="echo 'failed to load model'; exit 1")
        model = gguf_with(tmp_path / "m.gguf", GGMLType.HNX_INT3)
        with pytest.raises(ManagedError) as caught:
            ManagedRunner(port=18999).load(model, timeout=300)
        text = str(caught.value)
        assert "llama-server exited (code 1) while loading m" in text
        assert "did not start within" not in text
        assert "failed to load model" in text


def test_the_fixture_gguf_is_well_formed(tmp_path):
    path = gguf_with(tmp_path / "x.gguf", GGMLType.HNX_INT3)
    assert struct.unpack("<I", path.read_bytes()[:4])[0] == 0x46554747
