"""The C decoder for HyperNix sub-bit types, and the llama.cpp patcher.

Item 24: *make HyprSlug models load in LM Studio*. They cannot, and no
header rewriting changes that — ``IQ0.5_XXXL`` is not a llama.cpp
quantisation under a different name, it is different arithmetic. A
loader that believes a rewritten header reads a 30-byte block as though
it were a 210-byte Q3_K one.

So ``native/ggml-hnx`` is the decoder, in C, to be compiled into
llama.cpp — which is what LM Studio runs.

Why this is tested from Python
------------------------------
Because the thing worth testing is that the two implementations agree.
The Python encoder in ``hypernix.quant.subbit`` wrote every HyperNix
sub-bit file in existence; if the C decoder disagrees with it by one bit
of one byte, the model loads, runs at full speed, and emits fluent
nonsense. That failure looks like nothing at all from the outside, so
the only way to catch it is to have one side pack and the other unpack.

These tests compile the C with whatever compiler is on the box and skip
when there is not one. A skip is honest here: the C is also built and
checked by ``native/ggml-hnx``'s own ctest, and by CI where a compiler
is guaranteed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parent.parent / "native" / "ggml-hnx"
TOOLS = NATIVE / "tools"


def _compiler() -> str | None:
    for name in ("cc", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return found
    return None


@pytest.fixture(scope="module")
def selftest(tmp_path_factory) -> Path:
    """The compiled self-test binary."""
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C compiler on this machine")
    out = tmp_path_factory.mktemp("ggml-hnx") / "hnx_selftest"
    result = subprocess.run(
        [
            compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
            f"-I{NATIVE}",
            str(NATIVE / "ggml-hnx.c"), str(NATIVE / "hnx_selftest.c"),
            "-lm", "-o", str(out),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"ggml-hnx.c does not compile cleanly:\n{result.stderr}")
    return out


class TestTheDecoderCompilesAndPasses:
    def test_it_builds_with_no_warnings(self, selftest):
        """-Werror is the assertion. A warning in a tensor decoder is
        usually a sign conversion, and a sign conversion here is a model
        that reads as noise."""
        assert selftest.exists()

    def test_the_self_contained_checks_pass(self, selftest):
        result = subprocess.run(
            [str(selftest)], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "all checks passed" in result.stdout


class TestTheCAgreesWithPython:
    """The test this whole directory exists for."""

    def test_every_block_decodes_identically(self, selftest, tmp_path):
        vectors = tmp_path / "vectors.bin"
        generated = subprocess.run(
            [sys.executable, str(TOOLS / "gen_vectors.py"), str(vectors)],
            capture_output=True, text=True, check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert generated.returncode == 0, generated.stdout + generated.stderr
        assert vectors.stat().st_size > 0

        result = subprocess.run(
            [str(selftest), str(vectors)],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "decode identically" in result.stdout

    def test_the_vectors_cover_every_type(self, tmp_path):
        """A cross-check that silently skipped a type would be worse
        than no cross-check, because it would read as coverage."""
        import struct

        vectors = tmp_path / "vectors.bin"
        subprocess.run(
            [sys.executable, str(TOOLS / "gen_vectors.py"), str(vectors)],
            capture_output=True, check=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        data = vectors.read_bytes()

        seen = set()
        offset = 0
        while offset < len(data):
            type_id, block_bytes = struct.unpack_from("<ii", data, offset)
            seen.add(type_id)
            offset += 8 + block_bytes + 256 * 4

        assert seen == {200, 201, 202, 203, 204}
        assert offset == len(data), "the vector file does not parse cleanly"

    def test_the_block_sizes_match_the_registry(self):
        """``hypernix.quant.gguf`` publishes a block size per type and
        the C hard-codes one in a struct. Two copies of the same fact,
        so one can go stale -- and a struct one byte out reads every
        block after the first from the wrong offset, which produces a
        model that runs and lies rather than one that fails to load."""
        from hypernix.quant.gguf import _BLOCK_SHAPE, GGMLType

        source = (NATIVE / "ggml-hnx.c").read_text(encoding="utf-8")
        header = (NATIVE / "ggml-hnx.h").read_text(encoding="utf-8")

        for name, ggml_type, struct_name in (
            ("IQ0.9_L", GGMLType.HNX_IQ0_9, "iq0_9"),
            ("IQ0.75_M", GGMLType.HNX_IQ0_75, "iq0_75"),
            ("IQ0.5_XXXL", GGMLType.HNX_IQ0_5, "iq0_5"),
            ("IQ0.25_UXL", GGMLType.HNX_IQ0_25, "iq0_25"),
        ):
            block_size, block_bytes = _BLOCK_SHAPE[ggml_type]
            assert block_size == 256, f"{name} is not a 256-weight block"
            assert f'"{name}"' in source, f"{name} is missing from the C table"
            # The struct is an FP16 scale plus the payload, so the
            # payload is two bytes shorter than the block. Matched with
            # a pattern rather than a literal: the declarations are
            # column-aligned, and a test that failed on a space would
            # be a test people stop believing.
            declaration = re.search(
                rf"uint8_t\s+qs\[(\d+)\];\s*}}\s*hnx_block_{struct_name}\b",
                header,
            )
            assert declaration, f"no hnx_block_{struct_name} in ggml-hnx.h"
            assert int(declaration.group(1)) == block_bytes - 2, (
                f"{name} is {block_bytes} bytes in hypernix.quant.gguf but "
                f"{int(declaration.group(1)) + 2} in ggml-hnx.h"
            )


class TestTheShimCompilesAgainstGgmlsSignatures:
    def test_it_builds(self, tmp_path):
        compiler = _compiler()
        if compiler is None:
            pytest.skip("no C compiler on this machine")
        result = subprocess.run(
            [
                compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                f"-I{NATIVE}", "-c", str(NATIVE / "ggml-hnx-shim.c"),
                "-o", str(tmp_path / "shim.o"),
            ],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, result.stderr

    def test_every_type_has_both_entry_points(self):
        """ggml needs a to_float and a vec_dot per type. A registration
        naming a function that does not exist is a link error; one
        naming the *wrong* function is a model that runs and lies."""
        shim = (NATIVE / "ggml-hnx-shim.c").read_text(encoding="utf-8")
        header = (NATIVE / "ggml-hnx-shim.h").read_text(encoding="utf-8")

        for suffix in ("iq0_9", "iq0_75", "iq0_5", "iq0_25", "int1"):
            assert f"HNX_SHIM({suffix}," in shim.replace(" ", "")  or \
                   f"HNX_SHIM({suffix}," in shim
            assert f"hnx_ggml_to_float_{suffix}" in header
            assert f"hnx_ggml_vec_dot_{suffix}" in header


def _fake_llamacpp(root: Path) -> Path:
    """A tree shaped like llama.cpp, with the anchors the patcher wants.

    Deliberately not a real clone: the point is to test the patcher's
    edits, and a 200 MB download to do that would mean this test only
    ever ran somewhere with a network.
    """
    (root / "ggml" / "include").mkdir(parents=True)
    (root / "ggml" / "src").mkdir(parents=True)
    (root / "src").mkdir(parents=True)

    (root / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n"
        "    GGML_TYPE_F32 = 0,\n"
        "    GGML_TYPE_Q4_0 = 2,\n"
        "    GGML_TYPE_COUNT = 39,\n"
        "};\n",
        encoding="utf-8",
    )
    (root / "ggml" / "src" / "ggml.c").write_text(
        '#include "ggml-impl.h"\n'
        "\n"
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        "    [GGML_TYPE_F32] = { .type_name = \"f32\" },\n"
        "};\n",
        encoding="utf-8",
    )
    (root / "ggml" / "src" / "CMakeLists.txt").write_text(
        "add_library(ggml\n    ggml.c\n    ggml-alloc.c\n)\n", encoding="utf-8"
    )
    return root


class TestThePatcher:
    def test_check_reports_a_clean_tree_as_ready(self, tmp_path, capsys):
        sys.path.insert(0, str(TOOLS))
        try:
            import patch_llamacpp
        finally:
            sys.path.pop(0)

        root = _fake_llamacpp(tmp_path / "llama.cpp")
        assert patch_llamacpp.main([str(root), "--check"]) == 0
        assert "ready" in capsys.readouterr().out

    def test_it_applies_and_is_idempotent(self, tmp_path):
        sys.path.insert(0, str(TOOLS))
        try:
            import patch_llamacpp
        finally:
            sys.path.pop(0)

        root = _fake_llamacpp(tmp_path / "llama.cpp")
        assert patch_llamacpp.main([str(root)]) == 0

        header = (root / "ggml" / "include" / "ggml.h").read_text(encoding="utf-8")
        assert "GGML_TYPE_HNX_IQ0_5  = 202," in header
        # Inserted *before* the sentinel, not after: an enum value after
        # GGML_TYPE_COUNT would make the traits table one short.
        assert header.index("GGML_TYPE_HNX_IQ0_5") < header.index("GGML_TYPE_COUNT")

        source = (root / "ggml" / "src" / "ggml.c").read_text(encoding="utf-8")
        assert "[GGML_TYPE_HNX_IQ0_5] = {" in source
        assert '#include "ggml-hnx-shim.h"' in source

        cmake = (root / "ggml" / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
        assert "ggml-hnx.c" in cmake

        for name in ("ggml-hnx.c", "ggml-hnx.h", "ggml-hnx-shim.c", "ggml-hnx-shim.h"):
            assert (root / "ggml" / "src" / name).is_file(), name

        # Running it again must change nothing. It is meant to be safe
        # after every `git pull`, and a second run that duplicated the
        # enum would not compile.
        before = source
        assert patch_llamacpp.main([str(root)]) == 0
        after = (root / "ggml" / "src" / "ggml.c").read_text(encoding="utf-8")
        assert after == before

    def test_a_moved_anchor_changes_nothing(self, tmp_path, capsys):
        """The failure mode a .patch file has and this must not: a
        half-applied patch that still compiles. Everything is read and
        resolved before anything is written."""
        sys.path.insert(0, str(TOOLS))
        try:
            import patch_llamacpp
        finally:
            sys.path.pop(0)

        root = _fake_llamacpp(tmp_path / "llama.cpp")
        # Upstream renamed the traits table.
        source = root / "ggml" / "src" / "ggml.c"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "type_traits[GGML_TYPE_COUNT]", "type_traits_v2[GGML_TYPE_COUNT]"
            ),
            encoding="utf-8",
        )
        header_before = (root / "ggml" / "include" / "ggml.h").read_text(
            encoding="utf-8"
        )

        assert patch_llamacpp.main([str(root)]) == 1
        assert "Upstream moved" in capsys.readouterr().err
        assert (root / "ggml" / "include" / "ggml.h").read_text(
            encoding="utf-8"
        ) == header_before
        assert not (root / "ggml" / "src" / "ggml-hnx.c").exists()

    def test_revert_undoes_it(self, tmp_path):
        sys.path.insert(0, str(TOOLS))
        try:
            import patch_llamacpp
        finally:
            sys.path.pop(0)

        root = _fake_llamacpp(tmp_path / "llama.cpp")
        original = {
            path: path.read_text(encoding="utf-8")
            for path in (
                root / "ggml" / "include" / "ggml.h",
                root / "ggml" / "src" / "ggml.c",
                root / "ggml" / "src" / "CMakeLists.txt",
            )
        }

        assert patch_llamacpp.main([str(root)]) == 0
        assert patch_llamacpp.main([str(root), "--revert"]) == 0

        for path, content in original.items():
            assert path.read_text(encoding="utf-8") == content, path.name
        assert not (root / "ggml" / "src" / "ggml-hnx.c").exists()

    def test_a_directory_that_is_not_llamacpp_is_refused(self, tmp_path):
        sys.path.insert(0, str(TOOLS))
        try:
            import patch_llamacpp
        finally:
            sys.path.pop(0)

        with pytest.raises(SystemExit, match="does not look like"):
            patch_llamacpp.main([str(tmp_path)])
