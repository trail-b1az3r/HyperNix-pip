#!/usr/bin/env python3
"""Teach a llama.cpp checkout the HyperNix sub-bit types.

Not a .patch file, on purpose. llama.cpp changes fast and a diff against
line numbers rots within weeks — you get a rejected hunk, no idea which
half applied, and a half-patched tree that compiles. This finds the
registration points by pattern, edits them, and says exactly which ones
it could not find, so a failed run leaves the tree untouched and names
what upstream moved.

    python tools/patch_llamacpp.py /path/to/llama.cpp
    python tools/patch_llamacpp.py /path/to/llama.cpp --check
    python tools/patch_llamacpp.py /path/to/llama.cpp --revert

Idempotent: a tree that is already patched is reported as such and left
alone, so it is safe to run after every `git pull`.

What it changes
---------------
1. ``ggml/include/ggml.h`` — five values on ``enum ggml_type``, and
   ``GGML_TYPE_COUNT`` raised to cover them.
2. ``ggml/src/ggml.c`` — five entries in ``type_traits``: the *format*
   of each type (block size, size in bytes, how to turn one back into
   floats).
3. ``ggml/src/ggml-cpu/ggml-cpu.c`` — five entries in
   ``type_traits_cpu``: the *arithmetic* (``vec_dot``, and the F32
   activations it wants).
4. Both of those files — one ``#include`` of the shim header.
5. ``ggml/src/CMakeLists.txt`` — ggml-hnx.c and the shim added to the
   ggml-base sources, which ggml-cpu links against.

Two of those are not where they used to be, and both cost a broken
build to learn:

**The traits table is two tables.** ``ggml_type_traits`` in ggml.c has
no ``vec_dot`` member and cannot see ``ggml_vec_dot_t`` — upstream moved
everything the CPU computes with into ``ggml_type_traits_cpu``, in a
different file, behind a different header. Writing ``.vec_dot`` into
ggml.c is four errors per type.

**``GGML_TYPE_COUNT`` is a literal, not a count.** It is written
``GGML_TYPE_COUNT = 43``, so adding enum members before it does not grow
it — and it sizes both tables, so ``[GGML_TYPE_HNX_IQ0_9]`` at 200 is an
initialiser for element 200 of a 43-element array. This rewrites the
line and records the original in the marker comment, so ``--revert``
restores whatever that checkout had rather than a number baked in here.

The decoder itself is never patched in: ggml-hnx.c is *copied* into the
tree. That is the whole design — the arithmetic lives in one file that
upstream never touches, and only the registration is fragile.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
#: One marker per edit, and each must be *unique*, not merely present.
#: A single shared marker was a substring of three different first
#: lines, so `--revert` matched the wrong block and deleted sixty lines
#: of somebody else's code -- caught by the round-trip test, which is
#: why there is one.
MARK_ENUM = "GGML_HNX_ENUM"
MARK_COUNT = "GGML_HNX_COUNT"
MARK_TRAITS = "GGML_HNX_TRAITS"
MARK_TRAITS_CPU = "GGML_HNX_TRAITS_CPU"
MARK_INCLUDE = "GGML_HNX_INCLUDE"
MARK_INCLUDE_CPU = "GGML_HNX_INCLUDE_CPU"
MARK_CMAKE = "GGML_HNX_SOURCES"

#: One past the highest HyperNix type id. ``GGML_TYPE_COUNT`` sizes both
#: trait tables, and upstream pins it to a literal (``= 43`` at the time
#: of writing) rather than letting it fall out of the enum -- so adding
#: members before it does *not* grow it, and ``[GGML_TYPE_HNX_IQ0_9]``
#: at index 200 lands outside a 43-entry array. That is the "array index
#: in initializer exceeds array bounds" a build hits if only the enum is
#: patched.
#:
#: The gap costs a couple of kilobytes of mostly-empty table and buys id
#: stability, which is not negotiable: the numeric type id is written
#: into every GGUF file hyprslug produces, so renumbering these to sit
#: densely after upstream would make yesterday's models unreadable the
#: next time upstream adds a type. Holes are already normal here --
#: upstream's own 36, 37 and 38 are commented out and their table slots
#: sit empty -- and every read goes through a bounds-checked lookup by
#: an id some tensor actually carries, never a sweep of the range.
HNX_TYPE_COUNT = 205

# The enum values. Kept as one string so the marker and the entries
# cannot be added separately -- a tree with the marker but not the
# entries would report as patched and fail to build.
ENUM_ADDITION = f"""\
        // {MARK_ENUM} — HyperNix sub-1-bit types. See ggml-hnx.h.
        //
        // 200+ deliberately: upstream's own ids are well under 100 and
        // have room to grow before they reach here. A collision would
        // mean a stock build silently reinterpreting HyperNix tensors as
        // whatever upstream added at that number.
        GGML_TYPE_HNX_IQ0_9  = 200,
        GGML_TYPE_HNX_IQ0_75 = 201,
        GGML_TYPE_HNX_IQ0_5  = 202,
        GGML_TYPE_HNX_IQ0_25 = 203,
        GGML_TYPE_HNX_INT1   = 204,
"""

TRAITS_ADDITION = f"""\
    // {MARK_TRAITS}
    //
    // The *format* half of a type: how big a block is, and how to
    // turn one back into floats. The arithmetic half lives in the CPU
    // table -- see TRAITS_CPU_ADDITION for why they are separate.
    //
    // from_float_ref is NULL for all five: these are decode-only here.
    // Quantising to a sub-bit tier is hyprslug's job, which has the
    // importance matrix that decides where the surviving signs go --
    // and without one the result is meaningfully worse. A NULL
    // from_float makes `llama-quantize -> IQ0.5_XXXL` refuse cleanly
    // instead of producing a file that is the right size and wrong
    // inside.
    [GGML_TYPE_HNX_IQ0_9] = {{
        .type_name                = "IQ0.9_L",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 30,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_iq0_9,
        .from_float_ref           = NULL,
    }},
    [GGML_TYPE_HNX_IQ0_75] = {{
        .type_name                = "IQ0.75_M",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 26,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_iq0_75,
        .from_float_ref           = NULL,
    }},
    [GGML_TYPE_HNX_IQ0_5] = {{
        .type_name                = "IQ0.5_XXXL",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 18,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_iq0_5,
        .from_float_ref           = NULL,
    }},
    [GGML_TYPE_HNX_IQ0_25] = {{
        .type_name                = "IQ0.25_UXL",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 8,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_iq0_25,
        .from_float_ref           = NULL,
    }},
    [GGML_TYPE_HNX_INT1] = {{
        .type_name                = "INT1",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 34,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_int1,
        .from_float_ref           = NULL,
    }},
"""

TRAITS_CPU_ADDITION = f"""\
    // {MARK_TRAITS_CPU}
    //
    // vec_dot lives here, not beside type_name, because upstream
    // split ggml_type_traits in two: the format description stays in
    // ggml.c and everything the CPU backend needs to compute with --
    // from_float, vec_dot, vec_dot_type, nrows -- moved to
    // ggml_type_traits_cpu in ggml-cpu/ggml-cpu.c. Writing vec_dot
    // into the ggml.c table, as this used to, fails with "no member
    // named vec_dot" and "ggml_vec_dot_t undeclared" -- the type is
    // not even visible from there.
    //
    // vec_dot_type is F32 rather than Q8_0: every weight in these
    // types is +/-scale, so a block reduces to scale * sum(+/-y) and
    // the activations are wanted unquantised. from_float is NULL for
    // the same reason from_float_ref is.
    [GGML_TYPE_HNX_IQ0_9] = {{
        .from_float               = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_9,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_IQ0_75] = {{
        .from_float               = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_75,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_IQ0_5] = {{
        .from_float               = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_5,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_IQ0_25] = {{
        .from_float               = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_25,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_INT1] = {{
        .from_float               = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_int1,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
"""

INCLUDE_ADDITION = f'#include "ggml-hnx-shim.h" // {MARK_INCLUDE}\n'
# A second spelling for ggml-cpu.c, only so the two edits carry distinct
# markers. One shared marker is how an earlier version of this deleted
# sixty lines of somebody else's code on --revert.
INCLUDE_CPU_ADDITION = f'#include "ggml-hnx-shim.h" // {MARK_INCLUDE_CPU}\n'

CMAKE_ADDITION = f"""\
    # {MARK_CMAKE}
    ggml-hnx.c
    ggml-hnx-shim.c
"""


@dataclass
class Replace:
    """One line, rewritten, carrying the original so it can be put back.

    Insertion is not enough for ``GGML_TYPE_COUNT``. Upstream pins it to
    a literal (``GGML_TYPE_COUNT = 43``) rather than letting it fall out
    of the enum's length, so adding members before it leaves it at 43 --
    and ``type_traits[GGML_TYPE_COUNT]`` stays a 43-entry array that
    ``[GGML_TYPE_HNX_IQ0_9]`` (200) cannot be initialised into. The
    compiler says "array index in initializer exceeds array bounds" and
    then, confusingly, keeps going and reports the *next* problem.

    The original text is written into the marker comment rather than
    remembered anywhere else, so ``--revert`` restores exactly what was
    there without a stored copy of the file or a guess at upstream's
    current value.
    """

    candidates: tuple[str, ...]
    #: Matches the whole line to rewrite. One group: the part replaced.
    pattern: str
    #: What that group becomes.
    value: str
    marker: str
    what: str


@dataclass
class Edit:
    """One file, one anchor, one insertion."""

    #: Candidate paths, tried in order. Upstream has moved several of
    #: these between releases, so more than one spelling is offered
    #: rather than pinning a single layout.
    candidates: tuple[str, ...]
    #: The literal text to insert before or after.
    anchor: str
    addition: str
    #: True to insert after the anchor, False for before.
    after: bool
    what: str
    #: The unique token on the addition's first line. Matched exactly,
    #: so two edits to one file cannot be confused for each other.
    marker: str


REPLACEMENTS = (
    Replace(
        candidates=("ggml/include/ggml.h", "include/ggml.h", "ggml.h"),
        pattern=r"^(\s*GGML_TYPE_COUNT\s*=\s*)(\d+)(\s*,?)\s*$",
        value=str(HNX_TYPE_COUNT),
        marker=MARK_COUNT,
        what="GGML_TYPE_COUNT",
    ),
)

EDITS = (
    Edit(
        candidates=("ggml/include/ggml.h", "include/ggml.h", "ggml.h"),
        # The sentinel every version of this enum has ended with.
        anchor="GGML_TYPE_COUNT",
        addition=ENUM_ADDITION,
        after=False,
        what="the ggml_type enum",
        marker=MARK_ENUM,
    ),
    Edit(
        candidates=("ggml/src/ggml.c", "src/ggml.c", "ggml.c"),
        anchor="static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {",
        addition=TRAITS_ADDITION,
        after=True,
        what="the type_traits table",
        marker=MARK_TRAITS,
    ),
    Edit(
        candidates=(
            "ggml/src/ggml-cpu/ggml-cpu.c",
            "ggml/src/ggml-cpu.c",
            "src/ggml-cpu/ggml-cpu.c",
        ),
        anchor="static const struct ggml_type_traits_cpu "
               "type_traits_cpu[GGML_TYPE_COUNT] = {",
        addition=TRAITS_CPU_ADDITION,
        after=True,
        what="the type_traits_cpu table",
        marker=MARK_TRAITS_CPU,
    ),
    Edit(
        candidates=("ggml/src/ggml.c", "src/ggml.c", "ggml.c"),
        anchor='#include "ggml-impl.h"',
        addition=INCLUDE_ADDITION,
        after=True,
        what="the ggml.c includes",
        marker=MARK_INCLUDE,
    ),
    Edit(
        candidates=(
            "ggml/src/ggml-cpu/ggml-cpu.c",
            "ggml/src/ggml-cpu.c",
            "src/ggml-cpu/ggml-cpu.c",
        ),
        anchor='#include "ggml-impl.h"',
        addition=INCLUDE_CPU_ADDITION,
        after=True,
        what="the ggml-cpu.c includes",
        marker=MARK_INCLUDE_CPU,
    ),
    Edit(
        candidates=("ggml/src/CMakeLists.txt", "src/CMakeLists.txt"),
        anchor="ggml.c",
        addition=CMAKE_ADDITION,
        after=True,
        what="the ggml source list",
        marker=MARK_CMAKE,
    ),
)

#: Copied in rather than patched. The shim is what bridges ggml's
#: calling convention to the plain-C decoder, and it is separate from
#: ggml-hnx.c so that the arithmetic can be built and tested without
#: ggml's headers anywhere in sight.
COPIES = ("ggml-hnx.c", "ggml-hnx.h", "ggml-hnx-shim.c", "ggml-hnx-shim.h")


def _resolve(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    return None


def _target_dir(root: Path) -> Path:
    """Where ggml's sources live in this checkout."""
    for name in ("ggml/src", "src"):
        if (root / name / "ggml.c").is_file():
            return root / name
    raise SystemExit(
        f"patch_llamacpp: {root} does not look like a llama.cpp checkout "
        f"(no ggml/src/ggml.c or src/ggml.c)"
    )


def _apply_replacement(text: str, rep: Replace) -> str | None:
    """The rewritten text, or None if nothing matched.

    The original line is preserved verbatim in the marker comment above
    it, which is what makes --revert exact rather than a guess at what
    upstream's value used to be.
    """
    if rep.marker in text:
        return text                             # already applied
    out = []
    done = False
    for line in text.splitlines(keepends=True):
        match = re.match(rep.pattern, line.rstrip("\n"))
        if match and not done:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(
                f"{indent}// {rep.marker} was: {line.strip()}\n"
                f"{match.group(1)}{rep.value}{match.group(3)}\n"
            )
            done = True
            continue
        out.append(line)
    return "".join(out) if done else None


def _revert_replacement(text: str, rep: Replace) -> str:
    """Put the original line back, from the marker comment."""
    out = []
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        token = f"// {rep.marker} was: "
        if token in line:
            original = line.split(token, 1)[1].rstrip("\n")
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{original}\n")
            index += 2                          # skip the rewritten line
            continue
        out.append(line)
        index += 1
    return "".join(out)


def check(root: Path) -> int:
    """Report what is present, patched, or missing. Changes nothing."""
    problems = 0
    patched = 0
    for rep in REPLACEMENTS:
        path = _resolve(root, rep.candidates)
        if path is None:
            print(f"  MISSING  {rep.what}: none of {', '.join(rep.candidates)}")
            problems += 1
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root)
        if rep.marker in text:
            print(f"  patched  {rep.what} ({rel})")
            patched += 1
        elif _apply_replacement(text, rep) is not None:
            print(f"  ready    {rep.what} ({rel})")
        else:
            print(f"  ANCHOR   {rep.what} ({rel}): no line matching "
                  f"{rep.pattern!r}")
            problems += 1
    for edit in EDITS:
        path = _resolve(root, edit.candidates)
        if path is None:
            print(f"  MISSING  {edit.what}: none of {', '.join(edit.candidates)}")
            problems += 1
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root)
        if edit.marker in text:
            print(f"  patched  {edit.what} ({rel})")
            patched += 1
        elif edit.anchor in text:
            print(f"  ready    {edit.what} ({rel})")
        else:
            print(f"  ANCHOR   {edit.what} ({rel}): {edit.anchor!r} not found")
            problems += 1
    return problems if problems else (0 if patched else 0)


def apply(root: Path) -> int:
    target = _target_dir(root)

    # Everything is resolved and edited in memory before anything is
    # written, so a missing anchor leaves the tree exactly as it was
    # rather than half-patched. A half-patched llama.cpp compiles, which
    # is the problem.
    #
    # Keyed by path, and that is load-bearing: two edits target ggml.c,
    # and reading it fresh for each while writing both meant the second
    # write discarded the first. The traits table was never registered
    # and nothing said so -- the tree looked patched, the enum was
    # there, and the build failed much later with an unrelated-looking
    # error.
    pending: dict[Path, str] = {}
    changed: set[Path] = set()

    for rep in REPLACEMENTS:
        path = _resolve(root, rep.candidates)
        if path is None:
            print(
                f"patch_llamacpp: cannot find {rep.what} "
                f"(tried {', '.join(rep.candidates)})",
                file=sys.stderr,
            )
            return 1
        text = pending.get(path)
        if text is None:
            text = path.read_text(encoding="utf-8")
        rewritten = _apply_replacement(text, rep)
        if rewritten is None:
            print(
                f"patch_llamacpp: {path.relative_to(root)} has no line "
                f"matching {rep.pattern!r}. Upstream moved {rep.what}; "
                f"nothing was changed.",
                file=sys.stderr,
            )
            return 1
        if rewritten != text:
            changed.add(path)
        pending[path] = rewritten

    for edit in EDITS:
        path = _resolve(root, edit.candidates)
        if path is None:
            print(
                f"patch_llamacpp: cannot find {edit.what} "
                f"(tried {', '.join(edit.candidates)})",
                file=sys.stderr,
            )
            return 1
        text = pending.get(path)
        if text is None:
            text = path.read_text(encoding="utf-8")
        if edit.marker in text:
            pending[path] = text
            continue                            # already applied
        if edit.anchor not in text:
            print(
                f"patch_llamacpp: {path.relative_to(root)} no longer contains "
                f"{edit.anchor!r}. Upstream moved {edit.what}; nothing was "
                f"changed.",
                file=sys.stderr,
            )
            return 1
        index = text.index(edit.anchor)
        if edit.after:
            # To the end of the anchor's line, so the insertion lands
            # between statements rather than inside one.
            cut = text.index("\n", index) + 1
        else:
            # To the start of the anchor's line, same reason.
            cut = text.rfind("\n", 0, index) + 1
        pending[path] = text[:cut] + edit.addition + text[cut:]
        changed.add(path)

    for path in sorted(changed):
        path.write_text(pending[path], encoding="utf-8")
        print(f"  patched  {path.relative_to(root)}")

    for name in COPIES:
        source = HERE / name
        if not source.is_file():
            print(f"patch_llamacpp: {source} is missing", file=sys.stderr)
            return 1
        shutil.copy2(source, target / name)
        print(f"  copied   {name} -> {target.relative_to(root)}/")

    if not changed:
        print("  (registration was already in place)")
    return 0


def revert(root: Path) -> int:
    """Remove the additions and the copied files.

    Line-based rather than a reverse diff: the additions are all
    contiguous blocks introduced by a marker comment, so they can be
    identified without knowing what the file looked like before.
    """
    target = _target_dir(root)
    for rep in REPLACEMENTS:
        path = _resolve(root, rep.candidates)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        if rep.marker not in text:
            continue
        path.write_text(_revert_replacement(text, rep), encoding="utf-8")
        print(f"  reverted {rep.what} ({path.relative_to(root)})")

    for edit in EDITS:
        path = _resolve(root, edit.candidates)
        if path is None:
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        matches = [i for i, line in enumerate(lines) if edit.marker in line]
        if not matches:
            continue
        if len(matches) > 1:
            print(
                f"patch_llamacpp: {path.relative_to(root)} contains "
                f"{edit.marker} {len(matches)} times. Refusing to guess which "
                f"to remove -- revert it with git instead.",
                file=sys.stderr,
            )
            return 1
        start = matches[0]
        expected = edit.addition.splitlines(keepends=True)
        end = start + len(expected)
        # Verified rather than counted. Deleting a line count from a
        # marker position is how an earlier version of this removed
        # sixty lines of ggml.c: the marker had matched a different
        # edit's first line. If what is there is not what was added,
        # something else has edited it and git is the right tool.
        if [line.rstrip("\n") for line in lines[start:end]] != [
            line.rstrip("\n") for line in expected
        ]:
            print(
                f"patch_llamacpp: the block at {path.relative_to(root)}:"
                f"{start + 1} is not what was inserted. It has been edited "
                f"since; revert it with git instead.",
                file=sys.stderr,
            )
            return 1
        del lines[start:end]
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  reverted {path.relative_to(root)}")

    for name in COPIES:
        copied = target / name
        if copied.exists():
            copied.unlink()
            print(f"  removed  {target.relative_to(root)}/{name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patch_llamacpp.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="a llama.cpp checkout")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="report only; change nothing.")
    mode.add_argument("--revert", action="store_true",
                      help="undo the additions and remove the copied files.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"patch_llamacpp: {root} is not a directory", file=sys.stderr)
        return 1

    if args.check:
        return check(root)
    if args.revert:
        return revert(root)
    return apply(root)


if __name__ == "__main__":
    raise SystemExit(main())
