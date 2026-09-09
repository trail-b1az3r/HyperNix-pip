#!/usr/bin/env bash
# Build a llama.cpp that can load HyperNix sub-bit models.
#
#   ./build.sh                        clone, patch, build into ./llama.cpp
#   ./build.sh /path/to/llama.cpp     patch and build an existing checkout
#   ./build.sh --check /path          report what would change; touch nothing
#
# What comes out is a normal llama.cpp -- llama-cli, llama-server, the
# lot -- that additionally understands IQ0.9_L, IQ0.75_M, IQ0.5_XXXL,
# IQ0.25_UXL and INT1. LM Studio ships its own llama.cpp build, so
# pointing it at this one is a matter of replacing the runtime it
# loads; see README.md, which also says plainly where that is fragile.
#
# The GPU backends are the upstream ones, untouched: -DGGML_CUDA=ON and
# -DGGML_HIPBLAS=ON work exactly as they do upstream. The sub-bit types
# themselves are CPU-only for now -- there is a CUDA kernel to write and
# it is not written -- so a sub-bit tensor is dequantised on the CPU and
# the rest of the graph runs wherever you sent it. That is slower than a
# native kernel and still much faster than not loading at all.
set -euo pipefail

# Pinned rather than tracking master. A build that silently follows
# upstream is a build that breaks on a day you were not looking. Move it
# deliberately: bump this, run --check, fix what moved.
#
# Moved from b4585, which no longer compiles. Not because of anything
# here -- b4585's src/llama-mmap.h uses uint32_t without including
# <cstdint>, and got away with it only because older libstdc++ headers
# happened to pull <cstdint> in behind <vector> and <memory>. GCC 15 and
# 16 stopped doing that, so the pinned build failed on a current
# toolchain with "'uint32_t' does not name a type" and, once the
# compiler had guessed `int` for it, a pile of no-declaration-matches
# errors after it. Upstream fixed that file; the pin was nine months
# behind it.
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-b10883}"

# The same class of bug, defused rather than chased. Upstream still has
# hundreds of files that get their fixed-width integer types from
# somebody else's header, and the next compiler that tightens its
# transitive includes will break a different one. Forcing <cstdint>
# ahead of every translation unit costs nothing, edits no upstream
# source, and means a stale checkout -- or a LLAMA_REF you pinned
# yourself -- still builds.
#
# Overridable: HNX_FORCE_STDINT=0 turns it off if your compiler does not
# take -include (MSVC wants /FI).
if [ "${HNX_FORCE_STDINT:-1}" = "1" ]; then
  STDINT_CXX="-include cstdint"
  STDINT_C="-include stdint.h"
else
  STDINT_CXX=""
  STDINT_C=""
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0

if [ "${1:-}" = "--check" ]; then
  CHECK_ONLY=1
  shift
fi
TARGET="${1:-$HERE/llama.cpp}"

python_bin() {
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then echo "$candidate"; return; fi
  done
  echo "build.sh: need python3 to run the patcher" >&2
  exit 1
}
PY="$(python_bin)"

if [ ! -d "$TARGET" ]; then
  if [ "$CHECK_ONLY" = "1" ]; then
    echo "build.sh: $TARGET does not exist (nothing to check)" >&2
    exit 1
  fi
  echo "==> cloning llama.cpp $LLAMA_REF"
  # Shallow, at the pinned tag: the full history is ~1 GB and nothing
  # here needs it.
  git clone --depth 1 --branch "$LLAMA_REF" "$LLAMA_REPO" "$TARGET"
else
  # An existing checkout is reused as-is, and that is the right default
  # -- it may be one you are working in. But it is also how somebody
  # pulls a fix to the patcher, re-runs this, and rebuilds the same
  # stale tree they had before, which is a confusing way to spend an
  # afternoon. So: say what is there, rather than deciding for them.
  if [ -d "$TARGET/.git" ] && command -v git >/dev/null 2>&1; then
    have="$(git -C "$TARGET" describe --tags --exact-match 2>/dev/null \
            || git -C "$TARGET" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    if [ "$have" != "$LLAMA_REF" ]; then
      echo "==> using the checkout already at $TARGET ($have)"
      echo "    this script pins $LLAMA_REF. To move to it:"
      echo "      rm -rf $TARGET && $0"
      echo "    (or set LLAMA_REF=$have to pin what you have)"
    fi
  fi
fi

if [ "$CHECK_ONLY" = "1" ]; then
  echo "==> checking $TARGET"
  exec "$PY" "$HERE/tools/patch_llamacpp.py" "$TARGET" --check
fi

echo "==> registering the HyperNix types"
"$PY" "$HERE/tools/patch_llamacpp.py" "$TARGET"

echo "==> checking the decoder against hypernix.quant.subbit"
# Before building 200 MB of upstream, not after. If the C and the Python
# disagree the resulting binary loads models and produces nonsense, and
# finding that out from a chat transcript is much worse than finding it
# out here.
if "$PY" -c "import hypernix.quant.subbit" 2>/dev/null; then
  cmake -S "$HERE" -B "$HERE/build-selftest" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$HERE/build-selftest" --target hnx_selftest -j"$(nproc 2>/dev/null || echo 4)" >/dev/null
  "$PY" "$HERE/tools/gen_vectors.py" "$HERE/build-selftest/vectors.bin"
  "$HERE/build-selftest/hnx_selftest" "$HERE/build-selftest/vectors.bin"
else
  echo "    hypernix is not importable; running the self-contained checks only."
  echo "    (pip install -e . to cross-check against the Python encoder)"
  cmake -S "$HERE" -B "$HERE/build-selftest" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$HERE/build-selftest" --target hnx_selftest -j"$(nproc 2>/dev/null || echo 4)" >/dev/null
  "$HERE/build-selftest/hnx_selftest"
fi

echo "==> building llama.cpp"
# GGML_NATIVE=OFF so the binary runs on any machine of the same
# architecture rather than only the one that built it -- these get
# copied onto other boxes, and an illegal-instruction crash on startup
# is a confusing way to learn about -march=native.
cmake -S "$TARGET" -B "$TARGET/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_CXX_FLAGS="$STDINT_CXX ${CMAKE_CXX_FLAGS:-}" \
  -DCMAKE_C_FLAGS="$STDINT_C ${CMAKE_C_FLAGS:-}" \
  "${@:2}"
cmake --build "$TARGET/build" -j"$(nproc 2>/dev/null || echo 4)"

cat <<DONE

==> done

  binaries   $TARGET/build/bin
  try it     $TARGET/build/bin/llama-cli -m model-IQ0.5_XXXL.gguf -p "hello"

A sub-bit model is a much worse model than the one it came from. Below
about 1.5 bits per weight it stops being a slightly degraded version of
itself and becomes a different, far weaker one. This makes such a file
load and run correctly; it cannot make it good.
DONE
