#!/usr/bin/env bash
#
# build_llama_xcframework.sh — the inference engine HyperLink links against.
#
# Produces ios/vendor/llama.xcframework: llama.cpp built for iOS device
# and simulator, with Metal, as a framework Swift can `import llama`.
#
# It does not hand-list source files into an Xcode target. llama.cpp
# restructures its build between releases -- ggml-metal.m became
# ggml-metal.cpp, the Metal backend moved directory, `use_mmap` became
# `load_mode` -- and a hand-maintained file list breaks on every bump in
# a way that reads as a compiler error rather than as "the list is
# stale". Upstream ships build-xcframework.sh for exactly this, so that
# is what runs; this script's job is to fetch the right commit, apply
# the HyperNix patch, and put the result where the project expects it.
#
# Requires macOS with Xcode. There is no cross-compilation path: an
# xcframework needs xcodebuild.
#
#   ./build_llama_xcframework.sh                 # patched, device + simulator
#   HNX_PATCH=0 ./build_llama_xcframework.sh     # stock upstream
#   LLAMA_REF=b11000 ./build_llama_xcframework.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$IOS_DIR/.." && pwd)"

# Pinned to the same commit native/ggml-hnx/build.sh uses. The two must
# agree: a phone running a different llama.cpp from the desktop would
# disagree about the HyperNix tensor types, and the failure would look
# like a corrupt model rather than a version skew.
LLAMA_REF="${LLAMA_REF:-b10883}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
WORK="${HNX_LLAMA_WORK:-$IOS_DIR/.build/llama.cpp}"
VENDOR="$IOS_DIR/vendor"
OUTPUT="$VENDOR/llama.xcframework"

# The HyperNix patch adds tensor types 200-204 so the phone can read the
# sub-bit models hyprslug produces. Without it the app runs stock GGUF
# only, which is a legitimate build and the reason this is a switch.
HNX_PATCH="${HNX_PATCH:-1}"

PY="${PYTHON:-python3}"

say()  { printf '\033[38;5;253m%s\033[0m\n' "$*"; }

# The platform slices inside the xcframework, as a JSON array body.
# A glob rather than `ls | grep`: the directory names come from
# xcodebuild and a `for` over a glob copes with whatever it chose.
slice_list() {
  local first=1 entry name
  for entry in "$OUTPUT"/*; do
    name="$(basename "$entry")"
    [ "$name" = "Info.plist" ] && continue
    [ -d "$entry" ] || continue
    [ "$first" = 1 ] || printf ', '
    printf '"%s"' "$name"
    first=0
  done
}
ok()   { printf '\033[38;5;71m  ✓\033[0m %s\n' "$*"; }
err()  { printf '\033[38;5;160m  ✗\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Preconditions, checked before a twenty-minute build rather than during
# ---------------------------------------------------------------------------

[ "$(uname -s)" = "Darwin" ] || die \
  "This needs macOS: an xcframework is produced by xcodebuild, which does not
  cross-compile. On Linux, build the desktop engine instead with
  native/ggml-hnx/build.sh."

command -v xcodebuild >/dev/null 2>&1 || die \
  "xcodebuild not found. Install Xcode (not just the Command Line Tools) and
  run: sudo xcode-select -s /Applications/Xcode.app"

command -v cmake >/dev/null 2>&1 || die \
  "cmake not found. brew install cmake"

command -v git >/dev/null 2>&1 || die "git not found."

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$WORK")" "$VENDOR"

if [ -d "$WORK/.git" ]; then
  current="$(git -C "$WORK" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  say "Reusing $WORK (at $current)"
  # A checkout left patched from a previous run would be patched twice.
  # Reverting first is cheaper than reasoning about idempotence, and the
  # patcher is written to be reversible for exactly this.
  if [ -f "$WORK/.hnx-patched" ]; then
    say "  reverting the previous HyperNix patch first"
    "$PY" "$REPO_ROOT/native/ggml-hnx/tools/patch_llamacpp.py" "$WORK" --revert \
      || die "could not revert the previous patch; delete $WORK and re-run"
    rm -f "$WORK/.hnx-patched"
  fi
  git -C "$WORK" fetch --depth 1 origin "$LLAMA_REF" 2>/dev/null \
    || die "could not fetch $LLAMA_REF"
  git -C "$WORK" checkout -q FETCH_HEAD
else
  say "Cloning llama.cpp $LLAMA_REF"
  rm -rf "$WORK"
  git clone --depth 1 --branch "$LLAMA_REF" "$LLAMA_REPO" "$WORK" \
    || die "could not clone $LLAMA_REPO at $LLAMA_REF"
fi
ok "llama.cpp at $(git -C "$WORK" rev-parse --short HEAD)"

[ -f "$WORK/build-xcframework.sh" ] || die \
  "$LLAMA_REF has no build-xcframework.sh. Upstream moved or removed it; pin a
  ref that has it, or the Apple build has to be reworked."

# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

if [ "$HNX_PATCH" = "1" ]; then
  say "Applying the HyperNix tensor types (200-204)"
  "$PY" "$REPO_ROOT/native/ggml-hnx/tools/patch_llamacpp.py" "$WORK" \
    || die "the patch did not apply — see native/ggml-hnx/README.md"
  touch "$WORK/.hnx-patched"
  ok "patched"
else
  say "HNX_PATCH=0 — building stock llama.cpp (no sub-bit HyperNix types)"
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
#
# GGML_METAL_EMBED_LIBRARY is upstream's default and is load-bearing on
# iOS: with it off, the Metal shaders ship as a separate .metallib that
# the app has to find at runtime inside its own bundle, and the failure
# when it cannot is a silent fall back to CPU rather than an error.

say "Building llama.xcframework (this takes a while)"
(
  cd "$WORK"
  GGML_METAL_EMBED_LIBRARY=ON ./build-xcframework.sh ios-device ios-sim
) || die "build-xcframework.sh failed"

BUILT="$WORK/build-apple/llama.xcframework"
[ -d "$BUILT" ] || die "expected $BUILT and it is not there"

rm -rf "$OUTPUT"
cp -R "$BUILT" "$OUTPUT"

# Record what this was built from. `xcodebuild` gives no way to ask an
# xcframework which commit produced it, and "why does the phone disagree
# with the desktop about a tensor type" is unanswerable without this.
cat > "$VENDOR/llama.xcframework.json" <<JSON
{
  "llama_ref": "$LLAMA_REF",
  "commit": "$(git -C "$WORK" rev-parse HEAD)",
  "hnx_patched": $([ "$HNX_PATCH" = "1" ] && echo true || echo false),
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slices": [$(slice_list)]
}
JSON

# Turn the engine on for the Xcode build. Written rather than decided
# at generate time: XcodeGen cannot evaluate disk state into a build
# setting, and a setting that claimed the framework was present when it
# was not would surface as a link error instead of a clear message.
cat > "$VENDOR/LocalLlama.xcconfig" <<'CFG'
// Generated by ios/scripts/build_llama_xcframework.sh.
// Reset to empty by passing --unlink, or by `git checkout` on this file.
HNX_LOCAL_LLAMA = HNX_LOCAL_LLAMA
CFG

ok "$OUTPUT"
ok "local inference enabled (vendor/LocalLlama.xcconfig)"
say ""
say "  Next:  cd $IOS_DIR && xcodegen generate && open HyperLink.xcodeproj"
say "  The app links it automatically; project.yml already references it."
