#!/usr/bin/env bash
# install_macos_legacy.sh — set up hypernix on an old Intel Mac that
# can't run PyTorch 2.x.
#
# Targets:
#   * macOS 10.15 Catalina / 11 Big Sur on Intel
#   * Python 3.8 / 3.9 / 3.10 (torch 1.13 does not support 3.11+)
#   * hypernix 0.72.6.post4, the last release for those Pythons: from
#     0.72.7 it requires Python 3.12+, and pip on 3.10 skips it on its own
#   * No Apple Silicon / no MPS (this is the Intel-only path)
#
# What it does:
#   1. Creates (or reuses) a virtualenv at .venv/
#   2. Pins torch==1.13.1 from the PyPI CPU wheel
#   3. Installs hypernix with the [legacy-torch] extra, which loosens
#      numpy / safetensors / huggingface-hub / tqdm / sentencepiece
#      to versions known to work alongside torch 1.13.
#   4. Runs `python -c "import hypernix; print(hypernix.torch_compat.describe())"`
#      as a smoke check.
#
# Caveats:
#   * GGUF quantization still needs `llama-quantize`; the auto-fetch
#     path downloads an x86_64 mac binary from ggml-org/llama.cpp.
#     If that's missing on your macOS version, build it from source
#     first: `brew install llama.cpp`.
#   * `torch.compile`, `torch.nn.functional.scaled_dot_product_attention`,
#     and `nn.RMSNorm` are NOT in torch 1.13 — hypernix.torch_compat
#     provides fallbacks for the latter two.  `torch.compile` is
#     disabled on legacy torch.

set -euo pipefail

PY="${PY:-python3}"
VENV=".venv"

if ! command -v "$PY" >/dev/null; then
    echo "error: $PY not found on PATH" >&2
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "[legacy] creating venv at $VENV"
    "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade "pip<24" "wheel"

# torch 1.13.1 CPU — pinned.  Do NOT let pip resolve torch from the
# main hypernix install line; install it first and reuse.
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==1.13.1"

# Core hypernix + the legacy-torch extra.  As of 0.47.1 the main
# install_requires accepts torch>=1.13, so pip will honour the pin
# from the previous step and reuse torch 1.13.1 — no --no-deps
# hack required.  The [legacy-torch] extra adds looser numpy /
# huggingface-hub / sentencepiece pins that co-install cleanly
# with torch 1.13.
python -m pip install "hypernix[legacy-torch]"

echo
echo "[legacy] smoke-testing hypernix.torch_compat:"
python - <<'PY'
import hypernix
from hypernix import torch_compat

desc = torch_compat.describe()
print(desc)
assert desc["is_legacy_torch"], "torch_compat should report legacy mode"
print("hypernix", hypernix.__version__, "on torch", desc["torch_version"])
PY

echo
echo "[legacy] done.  Activate with:  source $VENV/bin/activate"
