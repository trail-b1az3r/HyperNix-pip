"""Legacy setup.py shim.

All real configuration lives in ``pyproject.toml`` (PEP 621). This file
exists so ``python setup.py <cmd>`` and ``pip install .`` work with
older tooling and with integrators that still look for ``setup.py``
in the project root.
"""
import importlib.util
import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _native_flags():
    """hypernix/_native_flags/__init__.py, loaded by path: the package is not
    importable while it is being built."""
    path = Path(__file__).resolve().parent / "src" / "hypernix" / "_native_flags" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_hypernix_native_flags", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BuildExt(build_ext):
    """Compile with the interpreter's own flags, frame pointers kept (PEP 831).

    setuptools passes sysconfig's CFLAGS first; the extension's own flags
    go after them, translated for MSVC, with anything that would turn
    frame pointers back off dropped.
    """

    def build_extensions(self):
        flags = _native_flags()
        for ext in self.extensions:
            ext.extra_compile_args = flags.extension_compile_args(
                list(ext.extra_compile_args or []), compiler=self.compiler.compiler_type)
        super().build_extensions()

# By default, do not build the C++ extension during standard wheel builds
# to ensure we produce a universal py3-none-any.whl for PyPI.
# Users installing from source or those who explicitly set BUILD_CCTVTOP=1
# will get the compiled C++ cctvtop dashboard.
build_cctvtop = os.environ.get("BUILD_CCTVTOP", "0") == "1"

ext_modules = []
if build_cctvtop:
    cctvtop_ext = Extension(
        "hypernix.cctvtop_ext",
        sources=["src/hypernix/cctvtop.cpp"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    )
    ext_modules.append(cctvtop_ext)

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": _BuildExt},
    )
