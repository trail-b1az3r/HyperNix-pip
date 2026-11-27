# HnxRun — running the models llama.cpp cannot

```bash
hypernix generate --model-dir model.iq05.gguf --prompt "hello"
hypernix chat     --model-dir model.iq05.gguf --cache-bytes 2G
```

```python
from hypernix.models import hnxrun

model = hnxrun.load_model("model.iq05.gguf")   # weights stay packed
print(model.describe())
print(model.resident_bits_per_weight)          # 0.572
tokens = hnxrun.generate_tokens(model, [1, 5, 9], max_new_tokens=32)
text   = hnxrun.generate_text("model.iq05.gguf", "hello", max_new_tokens=32)

# Spend memory on speed only where there is memory to spend.
faster = hnxrun.load_model("model.iq05.gguf", cache_bytes=2 << 30)
```

## The gap this closes

The [IQ0.x tiers](HyprSlug.md) have been real quantisations since
0.72.3 pt 2. The tensors genuinely carry 0.56 bits per weight and the
container is a well-formed GGUF. What they were **not** was runnable.

The type ids are at 200 and above — deliberately outside anything
upstream has allocated, so a stock loader refuses the file by name
instead of reading a 0.5-bit tensor as `Q4_K` and producing noise. Every
llama.cpp refuses them. So does the reference reader:

```
>>> gguf.GGUFReader("model.iq05.gguf")
ValueError: np.uint32(202) is not a valid GGMLQuantizationType
```

Which left a file that was correct, 30× smaller, and had nowhere to go.
"It is a real quantisation" is not much comfort when nothing will load
it.

## What this is

A reference implementation of the llama-family graph, in torch, that
reads every type this package writes: F32/F16/BF16, the llama.cpp block
types (including `Q8_K`) through [llamaquants](HyprSlug.md), the
HyperNix sub-bit types through `subbit`, and the fixed codebooks
(`INT8`, `INT4`, `INT3`, `INT2`, `FP2`, `FP8`) through `lowbit`. The
codebook list is read from `lowbit` itself: until 0.72.6.post3 it was
written out here and stopped at `INT4` and `FP2`, so `INT8` and `INT2`
files quantised and then could not be run. Quantised weights stay packed in memory and are
unpacked a slice at a time inside each matmul (see below); the graph is
RMSNorm → RoPE → grouped-query causal attention with a KV cache →
SwiGLU → output head.

Correctness first. No kernels, no fused anything, and it will not beat
llama.cpp — **it is not trying to.** For an upstream quant type
`hypernix generate` still routes to llama.cpp, which is better at
`Q4_K_M` than this ever will be. The point is that a 0.5-bit model has
somewhere to run at all.

## Sub-bit in memory too, not just on disk

The obvious way to write this runtime is to dequantise everything to
float32 at load time. That turns 0.56 bits into 32 — *larger* than the
F16 model the quantisation was made from — and hands back every byte the
tier existed to save. The file would still be small and the tier would
be pointless, which is the failure that is easy to ship because
everything still works.

So the packed bytes are what is held. Rows are unpacked a group at a
time inside each matmul, into a buffer that is thrown away — bytes, on
the small model the tests build, so the whole ladder fits on one page:

| Tier | On disk | Resident | bits/weight resident |
|---|---|---|---|
| `IQ0.5_XXXL` | 10,816 | 8,768 | **0.657** |
| `IQ0.75_M` | 14,112 | 12,096 | **0.906** |
| `IQ0.9_L` | 15,776 | 13,760 | 1.031 |
| `Q4_K_M` | 72,672 | 70,688 | 5.294 |
| `Q8_0` | 116,384 | 114,432 | 8.570 |
| *(materialised)* | — | 427,264 | 32.0 |

Slightly above each packing's own rate because the norms are
one-dimensional and stay in float32. They are a rounding error of a real
model's size and most of the gap on a toy one — which is why `IQ0.9_L`
reads 1.031 here and would read about 0.94 on anything real.

`load_model(path, materialize=True)` takes the other trade: float32 up
front and the memory profile of the model it was made from.
`cache_bytes` is the dial between the two — weights are pinned
largest-first until the budget is spent, because every forward pass
touches every tensor exactly once, so there is no locality to exploit
and the only question is how much decode work a byte of budget buys.

### And it has to be fast enough to be worth running

Sub-bit memory that costs 30× the time is a different way of not
shipping the tier. Measured on a 15.8M-parameter llama (4 layers, 512
wide), best of eight interleaved runs, against the same file loaded
`materialize=True`:

| Tier | On disk | Resident | bits/weight | ms/token | vs float32 |
|---|---|---|---|---|---|
| float32 | — | 63.2 MB | 32.0 | 2.7 | 1.0× |
| `IQ0.9_L` | 1.87 MB | 1.87 MB | 0.947 | 16.4 | 6.2× |
| `IQ0.75_M` | 1.63 MB | 1.62 MB | 0.822 | 15.6 | 5.9× |
| `IQ0.5_XXXL` | 1.13 MB | 1.13 MB | **0.572** | 10.6 | **4.0×** |

56× the memory for 4× the time, and the tier that saves the most memory
is also the fastest — because the fold below makes the arithmetic
proportional to the signs actually stored.

### The fold: never widen the weight

The obvious packed matmul unpacks a chunk of rows to one float32 per
weight and multiplies. That expansion is the single largest thing on the
hot path — for a 0.5-bit tensor it is 57× what the tensor costs — and it
is avoidable, because a dropped sign is not arbitrary: it *repeats* its
group's last stored sign. So the group's contribution factors:

```
sum_j sign[j]*x[j]  ==  sum_{j<k-1} sign[j]*x[j]  +  sign[k-1] * sum_{j>=k-1} x[j]
```

Fold `x` that way — each group's last stored position absorbing the
dropped ones — and the dot product runs against the `kept` signs alone.
At `IQ0.5_XXXL` that is half the arithmetic and none of the expansion.
Packed generation went from 7.1× float32 to 4.0×, and the resident cost
did not move.

Decoding those signs is one gather off an 8 KiB byte→signs table, which
replaced `unpackbits` plus a uint8→float32 conversion plus the `2b - 1`
mapping: three passes over the widest array became one.

### Logits agree to float32 rounding, not exactly

The fold changes which terms are added together, and chunking changes
the order they are added in. Float32 addition is not associative, so the
packed and materialised paths differ in the last bits — about 5e-7 on a
15.8M model. The *weights* are bit-identical, and that is what the tests
assert exactly; the logits are asserted `allclose`. An earlier version of
this page claimed bit-identical logits, which was true only while every
tensor fitted in a single chunk.

### Slicing granularity

The packed stream is blocks over the *flattened* tensor, so a slice has
to land on a block boundary. One row is not always a whole number of
blocks: a 64-wide layer packed in 256-element blocks puts four rows in
one block. The unit is therefore the smallest run of rows that is whole,
`block // gcd(block, columns)` — one row for any real model, a handful
for a narrow one. Getting this wrong reads the wrong bytes and gives a
plausible, wrong model rather than an error, so it is tested against the
materialised weights row by row.

## The RoPE convention, which is where this goes wrong

llama.cpp's converter **permutes** Q and K on the way into a GGUF so
that rotary embedding applies to *adjacent pairs* rather than to split
halves. Read those tensors back and apply the half-split form — which is
what every Hugging Face implementation does — and you get a model that
loads, runs, and generates confident nonsense.

Adjacent pairs, matching the file. The KV cache is checked against a
full recompute in the tests, because a cache that drifts from the
uncached path gives output that is plausible and not what the model
would have said.

## Two different questions

The tests keep these apart deliberately, because confusing them is how a
quantiser ends up secretly not quantising.

**Does the quantiser work?** Measured at the weights, where the answer is
arithmetic. Each tier stores `kept` signs of every `group` and
reconstructs the rest by repeating the last stored one — right half the
time — so the fraction of surviving signs is fixed by the design:

| Tier | Stored | Signs that should survive | Measured |
|---|---|---|---|
| `IQ0.9_L` | 7 of 8 | 93.75% | 93.6% |
| `IQ0.75_M` | 3 of 4 | 87.5% | 87.4% |
| `IQ0.5_XXXL` | 2 of 4 | 75% | 74.8% |

50% would mean the stored signs are landing on the wrong weights — a
fault invisible in file size and fatal to the model.

**Is the model any good?** Measured at the logits, where the answer at
half a bit is *no*, and is supposed to be. A test that demanded good
output from a 0.5-bit model would be demanding the impossible, and the
only way to pass it would be to stop quantising. What is asserted is
**ordering**: more bits must not produce a worse model.

## Tokenizers

A GGUF describes its own vocabulary — `tokenizer.ggml.tokens`, plus
`merges` for BPE or `scores` for SentencePiece — so no second file has
to be kept in sync. Both families are implemented in the straightforward
way rather than the fast way: rank-ordered merging for BPE, Viterbi over
the scores for SPM. Greedy longest-match is the classic SPM shortcut and
it silently produces a different segmentation.

A file with no tokenizer metadata gets `None` and `generate_text` says
so, because inventing an encoding produces output that reads as a broken
model rather than as a missing tokenizer. `generate_tokens` still works
with ids you already have.

## Refusals

- An architecture it does not implement. Running the llama graph over a
  model that is not one produces confident nonsense rather than an
  error, so the name is checked against what the forward pass matches.
- A file with no `token_embd.weight` — a fragment, not a model.
- A token id outside the vocabulary.

## How you actually reach it

Nothing here needs to be imported to be used. `hypernix generate` and
`hypernix chat` read the file's own metadata, and a GGML type at 200 or
above routes here instead of to llama.cpp:

```bash
hypernix quantize --source model.f32.gguf --output model.iq05.gguf \
    --type IQ0.5_XXXL -hnx \
    --quantize-embeddings --quantize-output
hypernix generate --model-dir model.iq05.gguf --prompt "hello"
hypernix chat     --model-dir model.iq05.gguf --cache-bytes 2G
```

`steamroller <src> IQ0.5_XXXL -hnx` writes the same file through the
same encoder, so either route arrives here.

Those two `--quantize-*` flags are the difference between a file the
tables above describe and one they do not. A sub-bit tier leaves
`token_embd` and the output head in float by default — at half a bit the
embedding table is the model, and that is a defensible call — but on a
7B the untouched pair is then most of the file, and the resident cost
lands nearer 1.7 bits per weight than 0.572. The default is a quality
decision with a size consequence; both halves are now reachable from the
command line, which they were not.

`--cache-bytes` is `load_model`'s budget, and only these tiers can use
it — llama.cpp has its own answer to how much to keep resident.

`chat` holds the model open for the whole session. That is worth stating
because it was not true: `load_gguf` handed back a bare `LoadedModel`,
which has no `.chat()`, so `hypernix chat` on a 0.5-bit model loaded
successfully and then died with `AttributeError` on the first message —
while every test passed, because they checked that the CLI *mentions*
`load_gguf` and that the routing avoids llama.cpp, and both were true of
the broken version. `load_gguf` now returns an `HnxSession` that speaks
the same `.chat()` as every other backend, and the tests run a real
model through `main()` rather than reading the source.

## See also

- [Devices](Devices.md) — running it on CUDA, ROCm, Metal or Intel, and
  why `torch.cuda.is_available()` is not the question
- [HyprSlug](HyprSlug.md) — the quantiser that writes these files
- [Dflash2](Dflash2.md) — a draft model inside the model
- [Imatrix](Imatrix.md) — deciding which weights the quantiser protects
