# hyprslug — quantise a GGUF without llama.cpp

Also answers to **doomslug**, **doomslugthedestroyer** and **dstd**.

```bash
hyprslug model.f16.gguf Q4_K_M      -o model.q4km.gguf
hyprslug model.f16.gguf IQ0.5_XXXL  -o model.iq05.gguf
hyprslug model.q8_0.gguf Q4_K_M --imatrix model.imatrix   # requantise
steamroller model.f16.gguf Q3_K_L -hnx
hypernix quantize --source model.f16.gguf --output out.gguf --type IQ0.75_M -hnx
```

## What this fixes

Everything that quantised in this package used to shell out to
`llama-quantize`. That has two problems.

The small one: a machine that has not built llama.cpp cannot quantise.

The large one: the sub-bit tiers — `IQ0.9_L`, `IQ0.75_M`, `IQ0.5_XXXL` —
are HyperNix types that `llama-quantize` has **never heard of**, so for
those it was never going to be the answer. What
[`steamroller`](Quantization.md) actually did was copy the Q3_K_L staging
file and write a sidecar JSON naming a tier.

So a "0.5-bit model" was byte-identical to the 3-bit model it came from,
the same size on disk, and no more quantised than its input. The tier was
a label on an unchanged file.

## How you get below one bit

Not by storing a fraction of a bit per weight — you cannot — but by
storing **fewer signs than weights** and reconstructing the rest. Every
tier keeps one FP16 scale per 256-weight block; the magnitude is gone
entirely and the scale stands in for all of it.

| Tier | Signs kept | Block | bits/weight | GGML type |
|---|---|---|---|---|
| `IQ0.9_L` | 7 of every 8 | 30 B | 0.938 | 200 |
| `IQ0.75_M` | 3 of every 4 | 26 B | 0.812 | 201 |
| `IQ0.5_XXXL` | 2 of every 4 | 18 B | 0.562 | 202 |

The signs kept are the **first k of each group**, and that is forced
rather than chosen. Selecting them by magnitude was tried first and is
wrong: the decoder has no bits telling it which positions were stored, so
it fills left to right regardless, and a cleverer encoder only lands the
signs on the wrong weights. It showed up as the widest tier having the
*worst* reconstruction error — a test now pins that error is monotonic in
bit rate.

Where an importance matrix earns its place is the **scale**, which once
the magnitudes are gone is most of what is left to get right. The scale
is the weighted *mean* absolute value, not the maximum: the maximum
minimises a different quantity and biases every reconstruction high.

## Type ids at 200

Deliberately far above anything llama.cpp has allocated. A stock loader
hits an unknown type id and refuses the file **by name** instead of
reading a 0.5-bit tensor as Q4_K and producing noise. Refusing loudly is
the whole point of picking a number that cannot collide.

Even the reference `gguf` Python reader refuses one:

```
ValueError: np.uint32(202) is not a valid GGMLQuantizationType
```

That is the design working. It also used to mean the file had nowhere to
run, which made a correct 0.5-bit quantisation useless — see
[HnxRun](HnxRun.md), which is the runtime that reads them.

`hypernix chat` and `hypernix generate` route on this automatically: an
upstream tier goes to llama.cpp, which is better at `Q4_K_M` than
anything here will be, and a sub-bit tier goes to HyperNix's own runtime.

## What it will and will not touch

Two different arguments, and it is worth keeping them apart.

**Not worth the bits.** Normalisation weights, biases and anything
one-dimensional are copied at source precision — all of the damage, none
of the size. Token embeddings and the output head are configurable,
because which dominates a file depends on the model.

**Not a weight in the sense a quantiser assumes.** A handful of small
2-D tensors are never packed at any tier, because their values are not
summed over a long reduction — the averaging that makes a 4-bit dot
product survivable never happens to them.

The clearest case, and the one that sent this list looking, is the
mixture-of-experts router. `ffn_gate_inp` is `[n_embd, n_expert]`: a few
hundred kilobytes in a model of tens of gigabytes, and its output is an
**argmax**. Quantising it moved the router's logits by a few percent,
which is nothing right up until two experts are within a few percent of
each other — and then it is a *different expert*, one never trained for
this token. The model does not degrade gracefully when that happens; it
stops being language. That was what "hyprslug models fall apart" turned
out to be: not the attention weights, which were fine, but eight numbers
per token deciding the wrong thing.

The same reasoning covers Mamba's `ssm_conv1d`, RWKV's time-mixing
constants (`time_mix_first`, `time_mix_w1`, the decay pair — but *not*
`time_mix_key`/`time_mix_value`, which are full-sized projections and
quantise like anything else), and the positional and token-type tables.
llama.cpp refuses exactly this list, for exactly this reason.

A tensor whose row length does not divide into 256 is copied and
**reported**. A run that quietly left half a model at F16 while reporting
`IQ0.5` would be the same failure this module exists to fix.

```
IQ0.5_XXXL  (quad_code_xxxl)
  4821.3 MB -> 512.7 MB (9.4x)
  226/291 tensors packed, 96.8% of weights
  65 copied at source precision:
    blk.0.attn_norm.weight: a norm: all of the damage, none of the size
    blk.0.ffn_gate_inp.weight: a mixture-of-experts router: its output is
      an argmax, and a quantised argmax picks a different expert
```

## The file says what it is

`general.file_type` used to be copied from the source and never
rewritten, so a `Q4_K_M` made from an F16 announced itself as **F16** —
to llama.cpp's load banner, to a hub listing, and to `hypernix-t1
index`. The tensor table was right and the field everybody actually
reads was wrong, which is the more expensive way round: nobody re-reads
a tensor table to check a field that is right there.

Every run now writes the target's real `general.file_type` and
`general.quantization_version`, as u32 like every other writer makes
them. The sub-bit tiers get numbers of their own from
`HNX_FILE_TYPE_BASE` rather than borrowing an upstream one: a half-bit
file labelled `Q2_K` would be claiming four times the precision it has,
and an unrecognised number is the correct thing to say about a file
stock llama.cpp cannot load anyway.

## `-hnx` means never

`resolve_binary()` downloads a llama.cpp build when it cannot find one,
so a lookup that happens and goes unused still leaves llama.cpp on the
machine. In `-hnx` mode the lookup does not happen, and a test replaces
`resolve_binary` with an assertion to keep it that way.

There is no staging pass in this mode either: the packer reads the
unquantised source directly, and going through Q3_K_L first would only
throw away precision it is about to use.

`-hnx` with an *upstream* tier now writes a file rather than skipping the
only step it had. That it did not was the same confusion in the other
direction: "hnx mode does not run llama-quantize" had been implemented as
"hnx mode does not quantise".

## Honestly

Below about 1.5 bits per weight a model stops being a slightly worse
version of itself and becomes a different, much worse model. No packing
changes that, and every tier carries the warning. What changed here is
that the file now genuinely is what its header says it is.

## The upstream types, too

The sub-bit tiers *had* to be written here — `llama-quantize` has never
heard of them. `Q4_K_M` did not, and so the one quantisation everybody
actually wants still needed a binary the machine might not be able to
build. It does not any more.

| Written by hyprslug | bits/weight | Block |
|---|---|---|
| `Q8_0` | 8.50 | 32 |
| `Q6_K` | 6.56 | 256 |
| `Q5_1` / `Q5_0` | 6.00 / 5.50 | 32 |
| `Q5_K`, `Q5_K_S`, `Q5_K_M` | 5.50 | 256 |
| `Q4_1` / `Q4_0` | 5.00 / 4.50 | 32 |
| `Q4_K`, `Q4_K_S`, `Q4_K_M` | 4.50 | 256 |
| `Q3_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L` | 3.44 | 256 |
| `Q2_K`, `Q2_K_S` | 2.62 | 256 |

`hyprslug --list-tiers` prints the live table.

Two things llama.cpp's names conflate are kept apart here. **`Q4_K` is a
block format**; **`Q4_K_M` is a mix** — most tensors at `Q4_K`, `attn_v`
and `ffn_down` a step wider, the output head at `Q6_K`. So there is a
table of recipes over the formats rather than ten more encoders.

The block layouts are exact: every struct matches `ggml-common.h` field
for field, the byte counts are asserted against the same table the GGUF
writer sizes tensors from, and the scale searches are ports of
`make_qx_quants` and `make_qkx2_quants` including the 19- and 21-step
searches that do most of the quality work. What is **not** claimed is
byte-for-byte reproduction of upstream's *mix* policy: llama.cpp picks
per layer index as well as per tensor role, and a file claiming to match
that exactly would be claiming something nobody has checked.

## Requantising

A `Q8_0` GGUF is the only copy of the model most people have, and
"quantise from the unquantised weights" is advice they cannot take. So
an already-quantised source is read back through the decoders:

```bash
hyprslug model.q8_0.gguf Q4_K_M -o model.q4km.gguf
```

The report names the type it came from, because requantising compounds
whatever the first pass lost and whether that matters is your call:

```
  ! requantised from an already-quantised source: Q8_0 x291
```

## Importance matrices

`--imatrix` takes either format — llama.cpp's binary `.imatrix` or JSON —
decided by content rather than by suffix. [Make one](Imatrix.md) with
`hnx-imatrix measure`.

An imatrix carries one number per input *channel*; the quantiser needs
one per weight, so the vector tiles across the tensor's rows. A width
that does not divide is a different model's imatrix, and it is ignored
with a warning rather than stretched to fit.

## See also

- [HnxRun](HnxRun.md) — running what llama.cpp cannot
- [Imatrix](Imatrix.md) — measuring importance, and reading anyone else's
- [Dflash2](Dflash2.md) — a draft model inside the model, for speculative decoding
- [Quantization](Quantization.md) — the GGUF pipeline and the upstream quant ladder
- [CLI](CLI.md) — `hypernix quantize`, `steamroller`, `hyprslug`
