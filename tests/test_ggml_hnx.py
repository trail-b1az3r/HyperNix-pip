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
    # The suffix is named rather than left off. MinGW's gcc appends
    # `.exe` when the -o argument has no extension, so asking for
    # `hnx_selftest` on Windows produced `hnx_selftest.exe` and the
    # existence check looked for a file the compiler had not been asked
    # to write. Naming it means -o and the check agree on every
    # platform, which is better than checking for both afterwards.
    name = "hnx_selftest.exe" if sys.platform == "win32" else "hnx_selftest"
    out = tmp_path_factory.mktemp("ggml-hnx") / name
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
    if not out.is_file():
        pytest.fail(
            f"{compiler} reported success but wrote no {out.name}. "
            f"Directory holds: {sorted(p.name for p in out.parent.iterdir())}"
        )
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

    It has to be shaped like the *current* upstream, though, and the
    first version of it was not. It modelled one ``type_traits`` table
    holding every field, which is how ggml looked once and has not
    looked for a while: upstream split it, leaving the format
    description (``type_name``, ``blck_size``, ``to_float``) in
    ``ggml.c`` and moving everything the CPU computes with
    (``from_float``, ``vec_dot``, ``vec_dot_type``, ``nrows``) into
    ``ggml_type_traits_cpu`` in ``ggml-cpu/ggml-cpu.c``. A fake with one
    table let the patcher write ``.vec_dot`` into ``ggml.c`` and every
    test here pass, while a real build failed with "no member named
    vec_dot" and "ggml_vec_dot_t undeclared".

    ``GGML_TYPE_COUNT`` is likewise a literal here, because it is one
    upstream -- not a value that falls out of the enum's length. Adding
    members before it does not grow it, so the patcher has to rewrite
    it, and a fake that let it grow on its own would hide that.
    """
    (root / "ggml" / "include").mkdir(parents=True)
    (root / "ggml" / "src" / "ggml-cpu").mkdir(parents=True)
    (root / "src").mkdir(parents=True)

    (root / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n"
        "    GGML_TYPE_F32 = 0,\n"
        "    GGML_TYPE_Q4_0 = 2,\n"
        "    GGML_TYPE_COUNT   = 43,\n"
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
    (root / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c").write_text(
        '#include "ggml-impl.h"\n'
        "\n"
        "static const struct ggml_type_traits_cpu "
        "type_traits_cpu[GGML_TYPE_COUNT] = {\n"
        "    [GGML_TYPE_F32] = { .vec_dot_type = GGML_TYPE_F32 },\n"
        "};\n",
        encoding="utf-8",
    )
    (root / "ggml" / "src" / "CMakeLists.txt").write_text(
        "add_library(ggml\n    ggml.c\n    ggml-alloc.c\n)\n", encoding="utf-8"
    )
    return root


def _patcher():
    """The patcher module, imported from tools/ without installing it."""
    sys.path.insert(0, str(TOOLS))
    try:
        import patch_llamacpp

        return patch_llamacpp
    finally:
        sys.path.pop(0)


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

    def test_the_type_count_is_raised_past_the_new_ids(self, tmp_path):
        """The bug that made a real build fail while every test passed.

        Upstream pins ``GGML_TYPE_COUNT`` to a literal instead of letting
        it fall out of the enum, and it sizes both trait tables. Insert
        ids at 200 without raising it and ``[GGML_TYPE_HNX_IQ0_9]`` is
        initialising element 200 of a 43-element array -- "array index in
        initializer exceeds array bounds", before the compiler even gets
        to the interesting error.
        """
        patcher = _patcher()
        root = _fake_llamacpp(tmp_path / "llama.cpp")
        assert patcher.main([str(root)]) == 0

        header = (root / "ggml" / "include" / "ggml.h").read_text(encoding="utf-8")
        # The declaring line, not the "was:" comment that records the
        # old one -- a plain search finds the comment first, which is
        # how the first version of this test managed to fail on a
        # correctly patched tree.
        declarations = [
            line for line in header.splitlines()
            if "GGML_TYPE_COUNT" in line and not line.lstrip().startswith("//")
        ]

        assert len(declarations) == 1, declarations
        count = re.search(r"GGML_TYPE_COUNT\s*=\s*(\d+)", declarations[0])
        assert count is not None
        assert int(count.group(1)) == patcher.HNX_TYPE_COUNT
        assert int(count.group(1)) > 204, "the highest HyperNix id"

    def test_vec_dot_goes_in_the_cpu_table_not_the_format_one(self, tmp_path):
        """The other half of the same failure.

        ``ggml_type_traits`` in ggml.c has no ``vec_dot`` member and
        cannot see ``ggml_vec_dot_t`` at all -- both moved to
        ``ggml_type_traits_cpu`` in ggml-cpu/ggml-cpu.c. Writing them
        into the wrong table is not a warning, it is four errors per
        type.
        """
        patcher = _patcher()
        root = _fake_llamacpp(tmp_path / "llama.cpp")
        assert patcher.main([str(root)]) == 0

        base = (root / "ggml" / "src" / "ggml.c").read_text(encoding="utf-8")
        cpu = (root / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c").read_text(
            encoding="utf-8")

        assert ".vec_dot" not in base, "vec_dot must not go in ggml.c"
        assert "ggml_vec_dot_t" not in base
        assert ".to_float" in base, "but to_float belongs there"

        assert ".vec_dot" in cpu
        assert ".vec_dot_type" in cpu
        assert ".to_float" not in cpu, "to_float must not go in the cpu table"
        for suffix in ("iq0_9", "iq0_75", "iq0_5", "iq0_25", "int1"):
            assert f"hnx_ggml_vec_dot_{suffix}" in cpu
            assert f"hnx_ggml_to_float_{suffix}" in base

    def test_both_files_get_the_shim_include(self, tmp_path):
        """Each table names functions the other file does not declare."""
        patcher = _patcher()
        root = _fake_llamacpp(tmp_path / "llama.cpp")
        assert patcher.main([str(root)]) == 0

        for path in (root / "ggml" / "src" / "ggml.c",
                     root / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"):
            assert '#include "ggml-hnx-shim.h"' in path.read_text(encoding="utf-8")

    def test_reverting_the_count_restores_upstreams_value(self, tmp_path):
        """Not a guess at what it used to be: the original line is
        carried in the marker comment, so a checkout whose count is 43
        goes back to 43 and one at 51 goes back to 51."""
        patcher = _patcher()
        root = _fake_llamacpp(tmp_path / "llama.cpp")
        header = root / "ggml" / "include" / "ggml.h"
        before = header.read_text(encoding="utf-8")

        assert patcher.main([str(root)]) == 0
        assert patcher.main([str(root), "--revert"]) == 0

        assert header.read_text(encoding="utf-8") == before

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


class TestTheCudaKernels:
    """Beta 1 shipped these types CPU-only. These are the GPU kernels.

    The one thing that can silently diverge is the geometry. The kernels
    take ``group``, ``kept`` and ``block_bytes`` as *template* parameters
    — they have to, so the inner loops unroll and the modulo arithmetic
    folds at compile time — which means each type's constants are written
    out a third time, next to the launchers.

    A wrong one there does not fail to compile and does not crash. It
    produces a model that runs at full speed on the GPU and talks
    nonsense, exactly as a wrong bit order on the CPU would. So the
    constants are checked against ``hypernix.quant.gguf`` here, and the
    emitted PTX is checked to confirm the templates instantiated with the
    values the source claims.
    """

    #: (suffix, group, kept) per type. block_bytes comes from the
    #: registry, so this table cannot quietly disagree about size.
    GEOMETRY = (
        ("iq0_9", 8, 7, 200),
        ("iq0_75", 4, 3, 201),
        ("iq0_5", 4, 2, 202),
        ("iq0_25", 16, 3, 203),
        ("int1", 1, 1, 204),
    )

    @pytest.fixture(scope="class")
    def source(self) -> str:
        return (NATIVE / "ggml-hnx-cuda.cu").read_text(encoding="utf-8")

    def test_the_file_exists_and_declares_all_five(self, source):
        for suffix, _group, _kept, _type_id in self.GEOMETRY:
            assert f"HNX_CUDA_LAUNCHERS({suffix}," in source.replace(" ", "") or \
                   f"HNX_CUDA_LAUNCHERS({suffix}," in source

    def test_the_launcher_constants_match_the_registry(self, source):
        """group, kept and block_bytes, against the CPU's own table."""
        from hypernix.quant.gguf import _BLOCK_SHAPE

        for suffix, group, kept, type_id in self.GEOMETRY:
            match = re.search(
                rf"HNX_CUDA_LAUNCHERS\(\s*{suffix}\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
                rf"\s*(\d+)\s*\)",
                source,
            )
            assert match, f"no launcher for {suffix}"
            got_group, got_kept, got_bytes = (int(g) for g in match.groups())

            assert got_group == group, f"{suffix} group"
            assert got_kept == kept, f"{suffix} kept"

            if type_id in _BLOCK_SHAPE:
                _, expected_bytes = _BLOCK_SHAPE[type_id]
                assert got_bytes == expected_bytes, (
                    f"{suffix} is {got_bytes} bytes in the CUDA launcher and "
                    f"{expected_bytes} in hypernix.quant.gguf"
                )

            # And the arithmetic has to close: the payload holds one bit
            # per stored sign, plus the two-byte FP16 scale.
            codes = 256 // got_group
            payload = (codes * got_kept + 7) // 8
            assert payload + 2 == got_bytes, (
                f"{suffix}: group {got_group} / kept {got_kept} needs "
                f"{payload + 2} bytes, launcher says {got_bytes}"
            )

    def test_it_uses_the_same_bit_order_as_the_cpu(self, source):
        """LSB-first within the byte, continuous across the payload. The
        expression is character-for-character the CPU's, which is the
        cheapest way to keep them from drifting."""
        cpu = (NATIVE / "ggml-hnx.c").read_text(encoding="utf-8")
        extraction = "(payload[bit >> 3] >> (bit & 7)) & 1"

        assert extraction in cpu
        assert extraction in source

    def test_cuda_is_opt_in(self):
        """The CPU decoder must build with a C compiler and nothing else.
        Requiring a CUDA toolkit to compile a 300-line C file would be a
        bad trade."""
        cmake = (NATIVE / "CMakeLists.txt").read_text(encoding="utf-8")

        assert 'option(GGML_HNX_CUDA' in cmake
        assert '"Build the CUDA kernels for the sub-bit types" OFF)' in cmake

    def test_pascal_is_in_the_architecture_list(self):
        """A GTX 1080 is exactly the card a sub-bit model exists for, and
        sm_61 is not in nvcc's default set."""
        cmake = (NATIVE / "CMakeLists.txt").read_text(encoding="utf-8")

        assert "CUDA_ARCHITECTURES 61" in cmake

    def test_the_row_stride_is_a_parameter_not_an_assumption(self):
        """ggml pads rows: nb[1] is not always nblocks * block_bytes, and
        computing it instead of being told reads every row after the
        first from the wrong offset."""
        header = (NATIVE / "ggml-hnx-cuda.h").read_text(encoding="utf-8")

        assert "row_stride_bytes" in header

    def test_it_compiles_if_nvcc_is_here(self, tmp_path):
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            pytest.skip("no CUDA toolkit on this machine")

        result = subprocess.run(
            [nvcc, "-std=c++17", "-O2", f"-I{NATIVE}", "-c",
             str(NATIVE / "ggml-hnx-cuda.cu"), "-o", str(tmp_path / "cuda.o")],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, result.stderr

    def test_every_kernel_is_instantiated_with_those_constants(self, tmp_path):
        """The check the source grep cannot make: that the templates
        actually instantiated with the values the launchers pass. The
        mangled PTX entry names carry them, so a launcher that passes one
        set while the template is stamped out with another shows up
        here."""
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            pytest.skip("no CUDA toolkit on this machine")

        ptx = tmp_path / "hnx.ptx"
        result = subprocess.run(
            [nvcc, "-std=c++17", "-O2", f"-I{NATIVE}", "-ptx",
             str(NATIVE / "ggml-hnx-cuda.cu"), "-o", str(ptx)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        text = ptx.read_text(encoding="utf-8")

        # Two kernels per type: the dot product and the expansion.
        assert text.count(".entry") == len(self.GEOMETRY) * 2

        from hypernix.quant.gguf import _BLOCK_SHAPE

        for suffix, group, kept, type_id in self.GEOMETRY:
            block_bytes = (
                _BLOCK_SHAPE[type_id][1] if type_id in _BLOCK_SHAPE else None
            )
            if block_bytes is None:
                continue
            # Itanium mangling for <int group, int kept, int bytes>.
            signature = f"ILi{group}ELi{kept}ELi{block_bytes}EE"
            assert f"hnx_mul_mat_vec{signature}" in text, (
                f"no mul_mat_vec instantiated for {suffix} at "
                f"({group}, {kept}, {block_bytes})"
            )
            assert f"hnx_dequantize{signature}" in text, (
                f"no dequantize instantiated for {suffix}"
            )

    def test_the_dot_product_reduces_across_the_warp(self, tmp_path):
        """One warp per row with a shuffle reduction, and no shared
        memory or barrier -- which is the whole reason the block size is
        32. If this stops appearing, the reduction has been replaced by
        something that needs synchronising."""
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            pytest.skip("no CUDA toolkit on this machine")

        ptx = tmp_path / "hnx.ptx"
        subprocess.run(
            [nvcc, "-std=c++17", "-O2", f"-I{NATIVE}", "-ptx",
             str(NATIVE / "ggml-hnx-cuda.cu"), "-o", str(ptx)],
            capture_output=True, check=True,
        )
        text = ptx.read_text(encoding="utf-8")

        assert "shfl.sync.down" in text
        assert "bar.sync" not in text, "a barrier appeared in a warp-level kernel"
