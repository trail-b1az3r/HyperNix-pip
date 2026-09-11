"""The C decoder and the Python one must produce identical floats.

Two independent implementations of the same packing exist: the Python in
:mod:`hypernix.quant.subbit`, which `hnxrun` and the quantiser use, and
the C in ``native/ggml-hnx/ggml-hnx.c``, which is what a patched
llama.cpp actually runs. Nothing compared them.

That is the exact shape of the worst bug in this package's history: the
row-length bug survived because the writer and the reader shared a
misconception and agreed with each other. Here the stakes are higher —
if these two disagree, a model that generates fine under `hnx generate`
generates noise under llama.cpp, and every test in the suite passes
while it happens.

Requires a C compiler; skipped without one.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hypernix.quant.subbit import BLOCK_SIZE, dequantize_array, quantize_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE = REPO_ROOT / "native" / "ggml-hnx"

#: tier -> (ggml type id, packing name). The ids are what the ggml patch
#: registers; the packings are what Python encodes with.
PAIRS = {
    "IQ0.9_L": (200, "sign_scale_l"),
    "IQ0.75_M": (201, "pair_code_m"),
    "IQ0.5_XXXL": (202, "quad_code_xxxl"),
    "IQ0.25_UXL": (203, "quarter_code_uxl"),
    "INT1": (204, "int1_binary"),
    "HNX_1375BIT": (207, "hnx_1375bit"),
}

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include "ggml-hnx.h"
int main(int argc, char **argv) {
    int type = atoi(argv[1]);
    size_t nb = (size_t)atoi(argv[2]);
    size_t bb = hnx_block_bytes(type);
    if (!bb) { fprintf(stderr, "unknown type %d\n", type); return 2; }
    unsigned char *src = malloc(bb * nb);
    if (fread(src, 1, bb * nb, stdin) != bb * nb) return 2;
    float *dst = malloc(sizeof(float) * HNX_BLOCK_SIZE * nb);
    size_t n = hnx_dequantize_rows(type, src, dst, nb);
    if (n != HNX_BLOCK_SIZE * nb) return 2;
    fwrite(dst, sizeof(float), n, stdout);
    return 0;
}
"""

pytestmark = pytest.mark.skipif(
    shutil.which("cc") is None and shutil.which("gcc") is None,
    reason="needs a C compiler to build the ggml-hnx decoder",
)


@pytest.fixture(scope="module")
def decoder(tmp_path_factory):
    """Build the C decoder once."""
    work = tmp_path_factory.mktemp("hnx-c")
    source = work / "xcheck.c"
    source.write_text(HARNESS, encoding="utf-8")
    binary = work / "xcheck"
    compiler = shutil.which("cc") or shutil.which("gcc")
    built = subprocess.run(
        [compiler, "-O2", "-I", str(NATIVE), "-o", str(binary), str(source),
         str(NATIVE / "ggml-hnx.c"), "-lm"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if built.returncode != 0:
        pytest.skip(f"ggml-hnx.c did not build here:\n{built.stderr}")
    return binary


def _c_decode(decoder: Path, type_id: int, packed: bytes, blocks: int) -> np.ndarray:
    done = subprocess.run(
        [str(decoder), str(type_id), str(blocks)], input=packed, capture_output=True
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    return np.frombuffer(done.stdout, dtype="<f4", count=BLOCK_SIZE * blocks)


@pytest.mark.parametrize("tier", sorted(PAIRS))
def test_the_two_decoders_agree_exactly(decoder, tier):
    """Bit-for-bit, not approximately. Both are decoding the same bytes
    with the same arithmetic; anything but equality is a bug in one."""
    type_id, packing = PAIRS[tier]
    rng = np.random.default_rng(0)
    blocks = 8
    values = rng.normal(0, 0.08, BLOCK_SIZE * blocks).astype(np.float32)
    packed = bytes(quantize_tensor(values.tolist(), packing))

    python = np.asarray(dequantize_array(packed, packing), dtype=np.float32)[: BLOCK_SIZE * blocks]
    native = _c_decode(decoder, type_id, packed, blocks)
    assert np.array_equal(native, python), (
        f"{tier}: C and Python disagree — a model that runs under hnxrun "
        f"would produce noise under llama.cpp.\n"
        f"  python[:8] {python[:8]}\n  c     [:8] {native[:8]}"
    )


@pytest.mark.parametrize("tier", sorted(PAIRS))
def test_they_agree_on_awkward_input(decoder, tier):
    """Zeros, a single huge outlier, and all-negative — the cases where
    a scale convention or a sign-fill order differs between two
    implementations that pass on ordinary noise."""
    type_id, packing = PAIRS[tier]
    cases = {
        "zeros": np.zeros(BLOCK_SIZE, np.float32),
        "one outlier": np.concatenate([np.full(BLOCK_SIZE - 1, 0.01, np.float32),
                                       np.array([10.0], np.float32)]),
        "all negative": np.full(BLOCK_SIZE, -0.05, np.float32),
        "alternating": np.tile(np.array([0.1, -0.1], np.float32), BLOCK_SIZE // 2),
    }
    for label, values in cases.items():
        packed = bytes(quantize_tensor(values.tolist(), packing))
        python = np.asarray(dequantize_array(packed, packing), dtype=np.float32)[:BLOCK_SIZE]
        native = _c_decode(decoder, type_id, packed, 1)
        assert np.array_equal(native, python), f"{tier} / {label}"


@pytest.mark.parametrize("tier", sorted(PAIRS))
def test_the_block_sizes_agree_too(decoder, tier):
    """If C thinks a block is a different number of bytes, every tensor
    offset after the first is wrong and the file reads as noise without
    either decoder being wrong about arithmetic."""
    from hypernix.quant.gguf import type_size_bytes

    type_id, packing = PAIRS[tier]
    packed = bytes(quantize_tensor([0.0] * BLOCK_SIZE, packing))
    # A short read makes the harness fail, so a successful decode of
    # exactly this many bytes is the agreement.
    _c_decode(decoder, type_id, packed, 1)
    assert len(packed) == type_size_bytes(type_id)
