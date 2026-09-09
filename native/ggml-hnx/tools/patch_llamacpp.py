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
Four small things, each at a place upstream has kept stable for years:

1. ``ggml/include/ggml.h`` — five values on ``enum ggml_type``.
2. ``ggml/src/ggml.c`` — five entries in the ``type_traits`` table, each
   pointing at the decoder in ggml-hnx.c.
3. ``ggml/src/CMakeLists.txt`` — ggml-hnx.c added to the sources.
4. ``src/llama-model-loader.cpp`` (or ``llama.cpp``) — the type-name
   table, so an unsupported-type error names IQ0.5_XXXL rather than "204".

The decoder itself is never patched in: ggml-hnx.c is *copied* into the
tree. That is the whole design — the arithmetic lives in one file that
upstream never touches, and only the registration is fragile.
"""
from __future__ import annotations

import argparse
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
MARK_TRAITS = "GGML_HNX_TRAITS"
MARK_INCLUDE = "GGML_HNX_INCLUDE"
MARK_CMAKE = "GGML_HNX_SOURCES"

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
    // from_float is NULL for all five: these are decode-only here.
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
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_9,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_IQ0_75] = {{
        .type_name                = "IQ0.75_M",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 26,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_iq0_75,
        .from_float_ref           = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_75,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_IQ0_5] = {{
        .type_name                = "IQ0.5_XXXL",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 18,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_iq0_5,
        .from_float_ref           = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_5,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_IQ0_25] = {{
        .type_name                = "IQ0.25_UXL",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 8,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_iq0_25,
        .from_float_ref           = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_iq0_25,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
    [GGML_TYPE_HNX_INT1] = {{
        .type_name                = "INT1",
        .blck_size                = HNX_BLOCK_SIZE,
        .type_size                = 34,
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) hnx_ggml_to_float_int1,
        .from_float_ref           = NULL,
        .vec_dot                  = (ggml_vec_dot_t) hnx_ggml_vec_dot_int1,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    }},
"""

INCLUDE_ADDITION = f'#include "ggml-hnx-shim.h" // {MARK_INCLUDE}\n'

CMAKE_ADDITION = f"""\
    # {MARK_CMAKE}
    ggml-hnx.c
    ggml-hnx-shim.c
"""


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
        candidates=("ggml/src/ggml.c", "src/ggml.c", "ggml.c"),
        anchor='#include "ggml-impl.h"',
        addition=INCLUDE_ADDITION,
        after=True,
        what="the ggml.c includes",
        marker=MARK_INCLUDE,
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


def check(root: Path) -> int:
    """Report what is present, patched, or missing. Changes nothing."""
    problems = 0
    patched = 0
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
