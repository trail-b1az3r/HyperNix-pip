# ggml-hnx — HyperNix sub-1-bit types for llama.cpp

## The problem this solves

A HyperNix sub-bit model does not load in stock llama.cpp, and no amount
of header rewriting changes that. `IQ0.5_XXXL` is not a llama.cpp
quantisation under a different name — it is different arithmetic.
`hyprslug-headers` can make a file *identify* as something llama.cpp
knows, and that is genuinely useful for tools that only inspect
metadata, but the tensors inside are still packed the HyperNix way. A
loader that believes the rewritten header reads a 30-byte block as
though it were a 210-byte Q3_K one, and what comes out is noise.

So: the decoder, in C, compiled into llama.cpp. LM Studio runs
llama.cpp, so a llama.cpp that understands these types is the route to
LM Studio understanding them too.

## The format, briefly

Below one bit per weight you cannot store a fraction of a bit. What you
can do is store **fewer signs than weights** and reconstruct the rest.
Every type keeps one FP16 scale per 256-weight block and discards the
magnitudes entirely — the scale stands in for all of them. What
separates the types is how many signs survive:

| type | group | kept | block bytes | bits/weight |
|---|---|---|---|---|
| `IQ0.9_L` | 8 | 7 | 30 | 0.938 |
| `IQ0.75_M` | 4 | 3 | 26 | 0.812 |
| `IQ0.5_XXXL` | 4 | 2 | 18 | 0.562 |
| `IQ0.25_UXL` | 16 | 3 | 8 | 0.250 |
| `INT1` | 1 | 1 | 34 | 1.062 |

Within each group the **first** `kept` signs are the stored ones, and the
positions after them repeat the last stored sign. That mapping is fixed
and has to be — there are no bits left to describe a cleverer choice, so
encoder and decoder can only agree on "the first k, always". Repeating
rather than alternating is deliberate too: adjacent weights in a row
correlate, so a repeat is right more often than a coin.

Bit order is LSB-first within each byte and continuous across the block
payload.

## Build it

```bash
./build.sh                      # clone llama.cpp, patch, build
./build.sh /path/to/llama.cpp   # patch and build an existing checkout
./build.sh --check /path        # report what would change, touch nothing
```

GPU backends are upstream's, untouched: `-DGGML_CUDA=ON` and
`-DGGML_HIPBLAS=ON` are passed straight through.

Test the decoder on its own, without cloning 200 MB of upstream:

```bash
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

## How it is put together

| file | what |
|---|---|
| `ggml-hnx.h` / `.c` | the decoder. Plain C99, no ggml headers. |
| `ggml-hnx-shim.h` / `.c` | ggml's calling convention over the decoder. |
| `ggml-hnx-cuda.cu` / `-cuda.h` | the GPU kernels. Opt-in; see below. |
| `hnx_selftest.c` | the checks, including the cross-check against Python. |
| `tools/gen_vectors.py` | packs blocks with `hypernix.quant.subbit` and records what it decodes them to. |
| `tools/patch_llamacpp.py` | registers the types in a checkout. |
| `build.sh` | clone, patch, verify, build. |

Two decisions worth knowing before changing anything.

**The decoder has no ggml dependency.** That is what let the bit order be
verified against the Python encoder before any of this touched upstream,
and it is what keeps a llama.cpp rebase from ever needing the arithmetic
rewritten. Only `ggml-hnx-shim.c` speaks ggml, and all it does is adapt
signatures.

**The registration is a patcher, not a `.patch` file.** llama.cpp moves
fast and a diff against line numbers rots within weeks: you get a
rejected hunk, no idea which half applied, and a half-patched tree that
compiles. `patch_llamacpp.py` finds each registration point by pattern,
edits them all in memory, and writes nothing at all if any anchor has
moved — so a failed run leaves the tree untouched and names what
upstream changed. It is idempotent, so it is safe after every `git pull`,
and `--revert` undoes it.

## The test that matters

If the C decoder and the Python encoder disagree by one bit of one byte,
the model loads, runs at full speed, and emits fluent nonsense. Nothing
about that looks like a failure from the outside.

So `tools/gen_vectors.py` has the Python side — which wrote every
HyperNix sub-bit file in existence — pack blocks and record its own
decoding of them, and `hnx_selftest` compares element by element,
**exactly**. No tolerance: both implementations multiply the same FP16
scale by ±1, so there is no rounding to forgive, and a tolerance would
hide precisely the errors this exists to catch. The vectors include
all-positive, all-negative, alternating and group-aligned blocks,
because a uniform block passes with the bit order reversed.

`tests/test_ggml_hnx.py` runs all of it from the Python suite, so it is
covered by CI rather than by remembering to run `ctest`.

## Using it with LM Studio

LM Studio ships its own llama.cpp runtime and swapping it is
version-specific and unsupported by them. In outline: build here, find
the runtime directory LM Studio loads (`~/.lmstudio/extensions/backends`
on Linux at the time of writing), and place the built libraries where it
looks. Expect this to break when LM Studio updates, and expect no help
from them if it does.

The supported route is HyperNix's own stack, which loads these types
directly: `hnx run`, the T1 API's `/inference/*`, and HyperNix Studio.

## What this does not do

**It does not make a sub-bit model good.** Below roughly 1.5 bits per
weight a model stops being a slightly worse version of itself and
becomes a different, much worse model. This makes such a file loadable
and *correct*; accuracy is not something a decoder can give back.

**CUDA is opt-in.** `-DGGML_HNX_CUDA=ON` builds
`ggml-hnx-cuda.cu`: two kernels per type, one warp per row, a shuffle
reduction and no shared memory. The dot product never materialises a
row — every weight is ±scale, so a block reduces to `scale · Σ(±y)`,
which means a row that would be 1 KB in F32 is 30 bytes of L2 and the
kernel is bound by reading the activations rather than the weights. That
is the compensation for discarding the magnitudes, and the reason these
types are worth having beyond the file size.

Off by default, because the CPU decoder has to build with a C compiler
and nothing else. `sm_61` is first in the architecture list: Pascal is
this project's floor, a GTX 1080 is exactly the card a sub-bit model
exists for, and 61 is not in nvcc's default set.

Without CUDA a sub-bit tensor is dequantised on the CPU and the rest of
the graph runs wherever you sent it — slower, and much faster than not
loading at all.

**Quantising to these types is not exposed to `llama-quantize`.** The
`from_float` slots are deliberately NULL, so `llama-quantize -> IQ0.5_XXXL`
refuses cleanly rather than producing a file that is the right size and
wrong inside. Which signs survive is the whole decision at these rates,
and it needs the importance matrix that `hyprslug` has and llama.cpp
does not.
