"""Compiler flags for HyperNix's native code that keep Python's (PEP 831).

From 3.15, CPython is built with frame pointers by default
(``-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer``), so perf,
py-spy, bpftrace and ``profiling.sampling --native`` can walk a stack
through C. An extension compiled with ``-fomit-frame-pointer`` -- or with
its own flags in place of the interpreter's -- puts a hole in every
stack that passes through it.

So native builds here start from the interpreter's own configuration
(``sysconfig``'s ``CFLAGS``) and add to it; they never replace it, and
never let a later flag switch frame pointers back off. Nothing is forced
on an interpreter built without them (3.12-3.14 usually are, and so is a
3.15 configured ``--without-frame-pointers``): the flags are inherited,
not invented, and MSVC -- which has no such flags -- gets none.

Stdlib only, and importable on its own: ``setup.py`` loads this file by
path before the package exists.
"""
from __future__ import annotations

import shlex
import sysconfig

__all__ = [
    "FRAME_POINTER_FLAGS",
    "OMIT_FLAGS",
    "cmake_flags",
    "extension_compile_args",
    "frame_pointer_flags",
    "merge_flags",
    "python_cflags",
]

#: What PEP 831 builds CPython with, in the order it passes them.
FRAME_POINTER_FLAGS: tuple[str, ...] = ("-fno-omit-frame-pointer", "-mno-omit-leaf-frame-pointer")

#: The flags that would undo them, each mapped to the one it undoes.
OMIT_FLAGS: dict[str, str] = {
    "-fomit-frame-pointer": "-fno-omit-frame-pointer",
    "-momit-leaf-frame-pointer": "-mno-omit-leaf-frame-pointer",
}


def python_cflags(config_vars: dict[str, str | None] | None = None) -> list[str]:
    """The ``CFLAGS`` this interpreter was built with, as a list (empty on MSVC)."""
    raw = (config_vars or {}).get("CFLAGS") if config_vars is not None \
        else sysconfig.get_config_var("CFLAGS")
    return shlex.split(raw or "")


def frame_pointer_flags(config_vars: dict[str, str | None] | None = None) -> list[str]:
    """The frame-pointer flags the interpreter itself was built with.

    Inherited, never invented: an interpreter built without them gets
    none back, so a toolchain that does not know them is never handed
    them.
    """
    cflags = python_cflags(config_vars)
    if any(flag in cflags for flag in OMIT_FLAGS) and not any(
            flag in cflags for flag in FRAME_POINTER_FLAGS):
        return []
    return [flag for flag in FRAME_POINTER_FLAGS if flag in cflags]


def merge_flags(base: list[str], extra: list[str]) -> list[str]:
    """*base*, then *extra*, without letting *extra* undo frame pointers.

    A flag in *extra* that turns off a frame pointer *base* keeps on is
    dropped; duplicates are kept once, at their first position.
    """
    kept = set(base)
    out: list[str] = []
    for flag in [*base, *extra]:
        if flag in OMIT_FLAGS and OMIT_FLAGS[flag] in kept:
            continue
        if flag not in out:
            out.append(flag)
    return out


def extension_compile_args(extra: list[str], *, compiler: str = "unix",
                           config_vars: dict[str, str | None] | None = None) -> list[str]:
    """``extra_compile_args`` for a setuptools ``Extension``.

    setuptools already passes the interpreter's ``CFLAGS`` first; these
    are appended after them, so they restate the frame-pointer flags
    (last one wins on gcc and clang) and drop any *extra* flag that
    would switch them off. *compiler* is setuptools' compiler type:
    ``msvc`` translates ``-std=`` and gets no frame-pointer flags.
    """
    if compiler == "msvc":
        return [_msvc(flag) for flag in extra if _msvc(flag)]
    return merge_flags(frame_pointer_flags(config_vars), extra)


def _msvc(flag: str) -> str:
    """A gcc/clang flag in MSVC's spelling, or "" for one it has no use for."""
    if flag.startswith("-std="):
        return "/std:" + flag[len("-std="):]
    if flag in ("-O2", "-O3"):
        return "/O2"
    return "" if flag.startswith("-") else flag


def cmake_flags(config_vars: dict[str, str | None] | None = None) -> list[str]:
    """``-D`` arguments that carry the inherited frame-pointer flags into a
    CMake build (llama.cpp, ggml-hnx). Empty when there are none to carry."""
    flags = " ".join(frame_pointer_flags(config_vars))
    if not flags:
        return []
    return [f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}"]
