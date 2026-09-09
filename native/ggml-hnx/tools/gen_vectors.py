#!/usr/bin/env python3
"""Generate cross-check vectors for hnx_selftest.

The C decoder in ``ggml-hnx.c`` and the Python one in
``hypernix.quant.subbit`` have to mean exactly the same thing by
"IQ0.5_XXXL", down to which bit of which byte holds which sign. If they
do not, a model loads, runs at full speed and produces fluent nonsense —
a failure that looks like nothing at all from the outside.

So the Python side, which is what actually wrote every HyperNix sub-bit
file in existence, packs blocks here and records what it decodes them
to. The C side reads that back and compares element by element, exactly,
with no tolerance: both implementations multiply the same FP16 scale by
±1, so there is no rounding to forgive, and a tolerance would hide the
off-by-one bit errors this exists to catch.

    python tools/gen_vectors.py vectors.bin
    ./hnx_selftest vectors.bin

Record layout, little-endian:

    int32   type id
    int32   block byte count
    bytes   the packed block
    float32 x 256, Python's decoding of it
"""
from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

# The four sub-bit tiers plus INT1, keyed by the GGML type id the C side
# uses. Kept as a literal rather than imported from hypernix.quant.gguf
# so that a divergence between the two shows up as a failure here rather
# than being defined away.
TYPES = {
    200: "sign_scale_l",       # IQ0.9_L
    201: "pair_code_m",        # IQ0.75_M
    202: "quad_code_xxxl",     # IQ0.5_XXXL
    203: "quarter_code_uxl",   # IQ0.25_UXL
    204: "int1_binary",        # INT1
}


def _blocks(rng: random.Random) -> list[list[float]]:
    """Weight blocks worth checking, not just random ones.

    Each of these has caught something or would: all-positive and
    all-negative pin the sign convention, the alternating one pins the
    bit order (a reversed one still passes on a uniform block), and the
    zero block pins the tie-break on ``w >= 0``. Random blocks are there
    for the rest.
    """
    size = 256
    cases = [
        [1.0] * size,                                    # every sign +
        [-1.0] * size,                                   # every sign -
        [1.0 if i % 2 == 0 else -1.0 for i in range(size)],   # bit order
        [1.0 if i % 8 < 4 else -1.0 for i in range(size)],    # group stride
        [0.0] * size,                                    # scale 0, sign +
        [0.0 if i % 3 else 1.0 for i in range(size)],    # zeros and ones
        # A realistic weight row: small, mixed, and none of it exactly
        # representable in FP16, so the scale exercises rounding.
        [rng.gauss(0.0, 0.02) for _ in range(size)],
        [rng.gauss(0.0, 0.02) for _ in range(size)],
        # One with a large outlier, which is what drags a mean-based
        # scale around and is where the two implementations would
        # disagree if one used the maximum instead.
        [rng.gauss(0.0, 0.02) for _ in range(size - 1)] + [3.5],
    ]
    return cases


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(f"usage: {Path(argv[0]).name} OUTPUT.bin", file=sys.stderr)
        return 2

    try:
        from hypernix.quant import subbit
    except ImportError as exc:
        print(f"gen_vectors: needs hypernix on the path ({exc})", file=sys.stderr)
        return 1

    rng = random.Random(20260909)
    out = bytearray()
    records = 0

    for type_id, packing in TYPES.items():
        block_bytes = subbit.packed_block_bytes(packing)
        for weights in _blocks(rng):
            packed = subbit.quantize_block(weights, packing)
            if len(packed) != block_bytes:
                print(
                    f"gen_vectors: {packing} packed {len(packed)} bytes, "
                    f"expected {block_bytes}",
                    file=sys.stderr,
                )
                return 1
            decoded = subbit.dequantize_block(packed, packing)
            out += struct.pack("<ii", type_id, block_bytes)
            out += packed
            out += struct.pack(f"<{len(decoded)}f", *decoded)
            records += 1

    Path(argv[1]).write_bytes(bytes(out))
    print(f"wrote {records} records ({len(out)} bytes) to {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
