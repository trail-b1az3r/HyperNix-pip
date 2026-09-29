# The extension types — sub-bit, one-bit and codebook

```bash
hypernix quantize --source model.f32.gguf --output model.fp2.gguf \
    --type FP2 -hnx --quantize-embeddings --quantize-output
steamroller model.f32.gguf IQ0.25_UXL -hnx
hyprslug --list-tiers
```

Every type at GGML id 200 and above, in families that fail
differently and are worth keeping apart.

## The whole table

| Tier | id | family | block | bpw | what it stores |
|---|---|---|---|---|---|
| `FP8` | 211 | fixed codebook | 258 B | **8.062** | E4M3 floats, stored as E4M3 bytes |
| `INT8` | 208 | fixed codebook | 258 B | **8.062** | 8-bit codes over `-128..127` |
| `INT4` | 205 | fixed codebook | 130 B | **4.062** | 4-bit codes over `-8..7` |
| `INT3` | 210 | fixed codebook | 98 B | **3.062** | 3-bit codes over `-4..3` |
| `INT2` | 209 | fixed codebook | 66 B | **2.062** | 2-bit codes over `-2, -1, 0, +1` |
| `FP2` | 206 | fixed codebook | 66 B | **2.062** | 2-bit codes over `±1, ±2` |
| `HNX_1375BIT` | 207 | sign and sub-scale | 44 B | **1.375** | every sign, plus a 5-bit magnitude per 16 weights |
| `INT1` | 204 | sign and scale | 34 B | **1.062** | every sign, no magnitude |
| `IQ0.9_L` | 200 | sign and scale | 30 B | **0.938** | 7 signs of every 8 |
| `IQ0.75_M` | 201 | sign and scale | 26 B | **0.812** | 3 of every 4 |
| `IQ0.5_XXXL` | 202 | sign and scale | 18 B | **0.562** | 2 of every 4 |
| `IQ0.25_UXL` | 203 | sign and scale | 8 B | **0.250** | 3 of every 16 |

A block is 256 weights throughout, matching the K-quant family, so a
tensor that divides evenly for `Q4_K` divides evenly for all of these.

## The rate is the name plus the scale

`INT4` is 4.062 bits per weight, not 4. `INT1` is 1.062, not 1. The FP16
block scale has to live somewhere, and 16 bits over 256 weights is
0.0625 of a bit each.

This is llama.cpp's own convention — `Q4_0` is 4.5 bits per weight and
nobody calls it `Q4.5_0` — and it is stated here rather than left to be
discovered from a file size. The `IQ0.x` names are the exception and are
exact: `IQ0.25_UXL` really is a quarter of a bit, scale included, because
8 bytes covering 256 weights is 0.25 exactly.

## Sign and scale

The family [subbit](HyprSlug.md) already had. One FP16 scale per block,
the magnitude gone entirely, and `kept` signs stored per `group` of
weights — the rest reconstructed by repeating the last stored one, which
is right about half the time.

Two new members, and neither needed new arithmetic:

**`INT1`** is the `k == g` case. Every sign kept, nothing reconstructed,
so the only loss is magnitude — a binary net with a block scale. It sits
in the same table as the others because it *is* the same code with
nothing dropped, and reimplementing it separately would be a second thing
to keep correct for no gain.

**`IQ0.25_UXL`** is the far end: three signs of every sixteen, the other
thirteen repeating the third. About **59% of signs survive**, against the
50% a coin gets. That is the honest number and it is on the tier's
warning. Treat it as a measurement of how far the packing goes, not as
something to deploy.

| tier | signs stored | signs correct |
|---|---|---|
| `INT1` | 16 of 16 | 100% |
| `IQ0.9_L` | 14 of 16 | 93.8% |
| `IQ0.75_M` | 12 of 16 | 87.5% |
| `IQ0.5_XXXL` | 8 of 16 | 75% |
| `IQ0.25_UXL` | 3 of 16 | **59.4%** |

**`HNX_1375BIT`** is the step above `INT1`: every sign, plus a 5-bit
magnitude index for each 16-weight sub-block on top of the block scale,
so a block reconstructs at sixteen magnitudes instead of one. It is what
hyprslug's hybrids call **IQ1_XS**: llama.cpp has no such type (its
one-bit types are `IQ1_S` at 1.56 and `IQ1_M` at 1.75 bits per weight,
both needing an importance matrix), and this is the HyperNix one-bit
type below `IQ1_S`.

All five sign-only tiers take the folded matmul in [HnxRun](HnxRun.md) — the dropped
positions repeat a stored sign, so the input can be folded to match and
the dot product runs against the stored signs alone.

## Fixed codebook

New arithmetic, in `hypernix.quant.lowbit`. These carry magnitude: one
FP16 block scale and a code per weight indexing a table that never
changes. No importance weighting — at these widths there is nothing to
allocate.

**`INT4`**: sixteen levels, `-8..7`. Asymmetric because two's complement
is — `-8` exists and `+8` does not, and pretending otherwise wastes an
eighth of the range on every block.

**`FP2`**: four levels, `-2, -1, +1, +2`. One sign bit and one exponent
bit, which is what two bits of float buys. **There is no zero**: a 2-bit
type *with* a zero needs five levels and therefore three bits, and the
version that rounds a third of a normal distribution to zero is
measurably worse than the version that does not.

**`INT8`** is not `Q8_0`. Same code width, but `Q8_0` pays a scale every
32 weights (8.5 bits per weight) and `INT8` every 256 (8.0625): a
smaller file, a coarser scale. Take `Q8_0` when the file has to load in
stock llama.cpp.

**`INT2`** is `FP2`'s counterpart with a zero in it, the two's-complement
range `-2..+1`. It was expected to lose to `FP2` and ties it (0.384
against 0.397 relative RMS error); the reason to pick it is that around
40% of a Gaussian tensor lands on the zero, which a sparse kernel can
skip.

**`INT3`** (0.72.6.post3) fills the gap between them: eight levels,
`-4..3`. Three bits do not divide a byte, so a block's 768 code bits run
across byte boundaries; the packers were written for that before a codec
needed it.

**`FP8`** (0.72.6.post3) is the OCP E4M3 float: a sign, four exponent
bits biased by 7, three mantissa bits, no infinities, and `S.1111.111`
as NaN. That is 253 distinct values from −448 to 448, dense near zero,
against an FP16 block scale. Each code is stored as **its own E4M3
byte**, not an index into a sorted table, so a kernel with native FP8
arithmetic can read the codes as FP8. The encoder never writes a NaN
byte, and a decoder given one reads it as zero.

Relative RMS error on Gaussian weights, all six:

| codec | error |
|---|---|
| `FP8` | 0.026 |
| `INT8` | 0.007 |
| `INT4` | 0.104 |
| `INT3` | 0.198 |
| `INT2` | 0.384 |
| `FP2` | 0.397 |

`INT8` beats `FP8` on this: a Gaussian block has no outliers to spend
FP8's exponent on. FP8 is the one to take for weights with a heavy tail.

Because they carry magnitude there is nothing to fold, so `HnxRun` runs
the ordinary matmul against them.

### The scale search, which was not the original plan

The first version fitted the scale to each block's largest magnitude, the
way `Q4_0` and `Q8_0` do, on the reasoning that a fixed codebook has no
budget for a search. Measured on Gaussian weights, relative RMS error:

| codec | peak fit | 17-step search |
|---|---|---|
| `INT4` | 0.113 | **0.104** |
| `FP2` | 0.944 | **0.396** |

`FP2`'s peak fit is not merely worse — it is worse than **one bit**.
`INT1` scores 0.599 at half the size. With four levels and the scale
pinned to a 3.5σ outlier, the levels land at 1.75σ and 3.5σ and almost
every weight rounds to the larger of two numbers that are both too big.

A 2-bit format that loses to a 1-bit format is not a format. So there is
a 17-step search over shrink factors, per block, picking the scale with
the lowest squared error — the same shape as upstream's
`make_qx_quants`. It is cheap because the codebook is fixed: the levels
are sorted, so nearest-level is a `searchsorted` against their midpoints
rather than an argmin over a broadcast.

The candidate is rounded to FP16 *inside* the loop. Scoring a scale the
file cannot hold picks a winner on a number that never reaches disk.

## What they cost, measured

For the tiers that existed at 0.72.3. A 15.8M-parameter llama, everything quantised including the embedding
and head, cosine similarity of the logits against the float original:

| tier | file | resident | bpw | agreement |
|---|---|---|---|---|
| `Q4_K_M` | 75 KB | 71 KB | 5.294 | 0.990 |
| `INT4` | 60 KB | 55 KB | 4.146 | 0.955 |
| `FP2` | 33 KB | 29 KB | 2.152 | 0.596 |
| `INT1` | 20 KB | 15 KB | 1.155 | 0.244 |
| `IQ0.9_L` | 18 KB | 14 KB | 1.031 | 0.145 |
| `IQ0.5_XXXL` | 14 KB | 9 KB | 0.657 | 0.131 |
| `IQ0.25_UXL` | 9 KB | 5 KB | 0.345 | −0.033 |

Monotone down to about a bit, then noise — which is the expected story
and the reason the tests assert *ordering among the wider codecs* rather
than quality anywhere. Below roughly 1.5 bits a model stops being a worse
version of itself and starts being a different model; no packing changes
that, and a test that demanded good output from `IQ0.25_UXL` could only
be passed by not quantising.

`INT4` is worth a specific caveat: **`Q4_K_M` is better at this rate and
any llama.cpp can read it.** `INT4` exists for when the codebook has to
be exactly the integers — a target that wants plain int4 tensors, a
comparison, a model that was already integer. It is not a better `Q4_K_M`.

## Where they can go

**Not in stock llama.cpp.** The ids are outside upstream's range on
purpose, so a stock loader refuses the file by name rather than reading
a quarter-bit tensor as `Q4_K` and returning noise.

**llama.cpp built with [`native/ggml-hnx`](../native/ggml-hnx/README.md)
runs all eleven.** Its patch registers ids 200 to 211 with a decoder and
a dot product for each, checked exactly against the Python decoders.
One limit: llama.cpp's CPU `get_rows` has no case for them, so a *token
embedding* in one of these types aborts on the first token. hyprslug
leaves the embedding alone for these tiers by default, and its hybrids
always keep it a stock K-quant.

They also run in [HnxRun](HnxRun.md), HyperNix's own runtime, on CPU
through numpy and on a GPU through `hypernix.models.hnxtorch`, and
[hyprslug-headers](HyprSlug-Headers.md) has the three ways of getting
one somewhere else.

## Q4_M, while we are here

Not a new type. `Q4M` is the spelling people type for `Q4_K_M`, and
squashing separators does not reach it — the missing character is the
`K`, not an underscore — so it fell through to "unknown target", which is
a confusing way to reject the most common request there is. `Q4M`,
`Q4_M`, `Q3L`, `Q5M`, `Q4S` and friends all resolve now.

## See also

- [HyprSlug](HyprSlug.md) — the quantiser, including the hybrids that
  mix these with llama.cpp's types
- [HnxRun](HnxRun.md) — the runtime, and the folded matmul
- [HyprSlug-Headers](HyprSlug-Headers.md) — getting one into LM Studio
