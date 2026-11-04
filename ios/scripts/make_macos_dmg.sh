#!/usr/bin/env bash
#
# HyperLink for the Mac, as a .dmg -- one architecture per run (0.73.0).
#
#     ios/scripts/make_macos_dmg.sh ARCH VERSION [BUILD] [OUT_DIR]
#
#   ARCH     arm64 (Apple silicon) or x86_64 (Intel)
#   VERSION  the marketing version to stamp (ios/scripts/app_version.py)
#   BUILD    CFBundleVersion, default 1
#   OUT_DIR  where the .dmg goes, default ios/build/dmg
#
# The Mac app is the iPhone app built with Mac Catalyst: the same
# sources, the same SwiftUI, a native macOS app rather than an iPad app
# in a window. It is built for exactly one architecture per run, so the
# Intel and Apple-silicon downloads are each the size of one slice.
#
# Signing: ad hoc. An Apple-silicon Mac will not run code that carries
# no signature at all, and an ad hoc signature (`codesign -s -`) is the
# least that satisfies it without an Apple Developer identity. Gatekeeper
# still does not know the developer, so the first launch is Control-click
# > Open (or System Settings > Privacy & Security > Open Anyway) -- the
# release notes say so.
#
# The on-device llama.cpp engine is not in the Mac build: the framework
# CI builds has iOS slices only, and on a Mac the HyperNix server itself
# is the thing to run models with. The app falls back to EchoRunner,
# which says so, exactly as an iPhone build without the engine does.
#
# Run from the repository root or from ios/; expects `xcodegen` and
# Xcode with the Mac Catalyst SDK.
set -euo pipefail

arch="${1:?usage: make_macos_dmg.sh ARCH VERSION [BUILD] [OUT_DIR]}"
version="${2:?usage: make_macos_dmg.sh ARCH VERSION [BUILD] [OUT_DIR]}"
build_number="${3:-1}"

case "$arch" in
  arm64)  label="apple-silicon" ;;
  x86_64) label="intel" ;;
  *) echo "make_macos_dmg: ARCH must be arm64 or x86_64, not '$arch'" >&2; exit 2 ;;
esac
# The version becomes part of a file name and an xcodebuild argument.
if ! printf '%s' "$version" | grep -Eq '^[0-9][0-9A-Za-z.+-]{0,40}$'; then
  echo "make_macos_dmg: '$version' does not look like a version" >&2
  exit 2
fi
if ! printf '%s' "$build_number" | grep -Eq '^[0-9]{1,9}$'; then
  echo "make_macos_dmg: BUILD must be a whole number" >&2
  exit 2
fi

ios_dir="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="${4:-$ios_dir/build/dmg}"
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
cd "$ios_dir"

# The project is generated, never committed (see project.yml). No engine:
# prepare_project.py leaves the llama.xcframework dependency out when the
# framework is not on disk, and a Mac build must not pick up iOS slices.
if [ ! -d HyperLink.xcodeproj ] || [ project.yml -nt HyperLink.xcodeproj ]; then
  python3 scripts/prepare_project.py
  xcodegen generate --spec project.generated.yml
fi

derived="$ios_dir/build/mac-$arch"
rm -rf "$derived"
xcodebuild \
  -project HyperLink.xcodeproj \
  -scheme HyperLink \
  -configuration Release \
  -destination "generic/platform=macOS,variant=Mac Catalyst" \
  -derivedDataPath "$derived" \
  ARCHS="$arch" \
  ONLY_ACTIVE_ARCH=NO \
  MARKETING_VERSION="$version" \
  CURRENT_PROJECT_VERSION="$build_number" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY="" \
  build

app="$derived/Build/Products/Release-maccatalyst/HyperLink.app"
if [ ! -d "$app" ]; then
  echo "make_macos_dmg: the build produced no $app" >&2
  exit 1
fi

# One slice, the one asked for -- a universal binary here would mean the
# ARCHS setting was ignored and both downloads are the same file.
binary="$app/Contents/MacOS/HyperLink"
slices="$(lipo -archs "$binary")"
if [ "$slices" != "$arch" ]; then
  echo "make_macos_dmg: expected a $arch binary, got: $slices" >&2
  exit 1
fi

codesign --force --deep --sign - "$app"
codesign --verify --deep --strict "$app"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -R "$app" "$stage/"
ln -s /Applications "$stage/Applications"

dmg="$out_dir/HyperLink-$version-macos-$label.dmg"
rm -f "$dmg"
hdiutil create -volname "HyperLink $version" -srcfolder "$stage" -ov -format UDZO "$dmg"
hdiutil verify "$dmg"

(cd "$out_dir" && shasum -a 256 "$(basename "$dmg")" >> SHA256SUMS.macos)
echo "$dmg"
