"""PEP 831 -- frame pointers -- in hyperNix-pip's native builds.

CPython 3.15 is built with ``-fno-omit-frame-pointer
-mno-omit-leaf-frame-pointer`` so profilers can walk stacks through C.
Native code built here has to inherit that from the interpreter's
``sysconfig`` rather than replace it, never switch it back off, and
never force it on a toolchain or interpreter that does not use it.
These tests check the flag logic for every case and, where a C compiler
exists, that the code it builds really keeps a frame pointer.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from hypernix import _native_flags as nf

ROOT = Path(__file__).resolve().parents[2]
FP = list(nf.FRAME_POINTER_FLAGS)
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_315 = pytest.mark.skipif(sys.version_info < (3, 15), reason="PEP 831 builds are 3.15")


def _vars(cflags: str | None) -> dict[str, str | None]:
    return {"CFLAGS": cflags}


def test_flags_are_inherited_not_invented():
    assert nf.frame_pointer_flags(_vars("-O3 -Wall")) == []
    assert nf.frame_pointer_flags(_vars(None)) == []                      # MSVC: no CFLAGS
    assert nf.frame_pointer_flags(_vars("-O3 " + " ".join(FP))) == FP
    assert nf.frame_pointer_flags(_vars("-fno-omit-frame-pointer")) == FP[:1]


def test_an_interpreter_built_without_them_gets_none():
    assert nf.frame_pointer_flags(_vars("-O3 -fomit-frame-pointer")) == []


def test_extra_flags_cannot_switch_them_off():
    merged = nf.merge_flags(FP, ["-O3", "-fomit-frame-pointer", "-momit-leaf-frame-pointer", "-O3"])
    assert merged == [*FP, "-O3"]


def test_extra_flags_pass_through_when_there_is_nothing_to_protect():
    assert nf.merge_flags([], ["-O3", "-fomit-frame-pointer"]) == ["-O3", "-fomit-frame-pointer"]


def test_extension_args_on_a_frame_pointer_interpreter():
    args = nf.extension_compile_args(["-std=c++17", "-O3"], config_vars=_vars(" ".join(FP)))
    assert args == [*FP, "-std=c++17", "-O3"]


def test_msvc_gets_its_own_spelling_and_no_gcc_flags():
    args = nf.extension_compile_args(["-std=c++17", "-O3", "-fno-omit-frame-pointer"],
                                     compiler="msvc", config_vars=_vars(" ".join(FP)))
    assert args == ["/std:c++17", "/O2"]


def test_cmake_flags():
    assert nf.cmake_flags(_vars("-O2")) == []
    assert nf.cmake_flags(_vars(" ".join(FP))) == [
        f"-DCMAKE_C_FLAGS={' '.join(FP)}", f"-DCMAKE_CXX_FLAGS={' '.join(FP)}"]


def test_it_reads_this_interpreter():
    cflags = sysconfig.get_config_var("CFLAGS") or ""
    assert nf.frame_pointer_flags() == [f for f in FP if f in cflags.split()]


@needs_315
@pytest.mark.skipif(sys.platform == "win32", reason="MSVC builds have no CFLAGS")
def test_this_3_15_interpreter_was_built_with_them():
    """The premise. A 3.15 configured --without-frame-pointers is legal;
    if this fails, check how this Python was built before anything else."""
    assert "-fno-omit-frame-pointer" in (sysconfig.get_config_var("CFLAGS") or "")


def test_setup_py_uses_it():
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "_native_flags" in text and "extension_compile_args" in text
    assert 'cmdclass={"build_ext": _BuildExt}' in text


def test_the_cmake_build_keeps_them_and_checks_the_compiler():
    text = (ROOT / "native" / "ggml-hnx" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "GGML_HNX_FRAME_POINTERS" in text and "check_c_compiler_flag" in text
    assert "NOT MSVC" in text


@pytest.mark.skipif(CC is None, reason="needs a C compiler")
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="reads x86-64 prologues")
def test_the_flags_really_keep_a_frame_pointer(tmp_path):
    """Not just that the flags are on the command line: that the code the
    compiler emits with them sets up %rbp, at -O2, where it otherwise
    would not."""
    source = tmp_path / "f.c"
    source.write_text("int g(int); int f(int x) { return g(x) + 1; }\n", encoding="utf-8")

    def assembly(flags: list[str]) -> str:
        out = tmp_path / "f.s"
        subprocess.run([CC, "-O2", *flags, "-S", "-o", str(out), str(source)], check=True)
        return out.read_text(encoding="utf-8")

    kept = assembly(nf.merge_flags(FP, ["-fomit-frame-pointer"]))
    assert re.search(r"push[q]?\s+%rbp", kept) and re.search(r"mov[q]?\s+%rsp,\s*%rbp", kept), kept
    omitted = assembly(["-fomit-frame-pointer"])
    assert not re.search(r"mov[q]?\s+%rsp,\s*%rbp", omitted), "the compiler keeps one anyway"


@needs_315
@pytest.mark.skipif(sys.platform == "win32", reason="checks an ELF/Mach-O C extension build")
@pytest.mark.skipif(CC is None, reason="needs a C compiler")
def test_a_setuptools_extension_is_built_with_them(tmp_path):
    """The path setup.py takes, end to end, on this interpreter."""
    pytest.importorskip("setuptools")
    (tmp_path / "m.c").write_text(
        '#include <Python.h>\nstatic struct PyModuleDef d = {PyModuleDef_HEAD_INIT, "m"};\n'
        "PyMODINIT_FUNC PyInit_m(void) { return PyModuleDef_Init(&d); }\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text(
        "import importlib.util, sys\n"
        "from setuptools import Extension, setup\n"
        "from setuptools.command.build_ext import build_ext\n"
        f"spec = importlib.util.spec_from_file_location('nf', {str(ROOT / 'src/hypernix/_native_flags/__init__.py')!r})\n"
        "nf = importlib.util.module_from_spec(spec); spec.loader.exec_module(nf)\n"
        "class B(build_ext):\n"
        "    def build_extensions(self):\n"
        "        for e in self.extensions:\n"
        "            e.extra_compile_args = nf.extension_compile_args(e.extra_compile_args,"
        " compiler=self.compiler.compiler_type)\n"
        "        super().build_extensions()\n"
        "setup(name='m', ext_modules=[Extension('m', ['m.c'], extra_compile_args=['-O3',"
        " '-fomit-frame-pointer'])], cmdclass={'build_ext': B})\n", encoding="utf-8")
    done = subprocess.run([sys.executable, "setup.py", "-v", "build_ext", "--inplace"],
                          cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    compile_line = next(line for line in (done.stdout + done.stderr).splitlines()
                        if " -c " in line and "m.c" in line)
    tail = compile_line.split("m.c", 1)[1]
    assert "-fno-omit-frame-pointer" in tail and "-fomit-frame-pointer" not in tail, compile_line
