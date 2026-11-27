# Imatrix — measuring importance, and reading anyone else's

```bash
hnx-imatrix measure ./Llama-3.2-1B -t calibration.txt -o model.imatrix
hnx-imatrix show    model.imatrix
hnx-imatrix convert model.imatrix -o model.json

hyprslug model.f16.gguf Q3_K_M --imatrix model.imatrix
```

## What an importance matrix is

One number per **input channel** of every matmul in the model: the mean
of that channel's activation squared, measured over calibration text.

It is not a property of the weights. Two models with identical weights
and different training data want different imatrices, and no amount of
staring at a weight tensor produces one — which is why "derive it from
the weights" is not something this module offers, however convenient that
would be. A weight-derived number is not an approximation of an imatrix;
it is a different quantity wearing its name.

## What it buys

Where the quantiser spends its error. Below about four bits a block
cannot represent everything in it, and the difference between a usable
`Q3_K` and an unusable one is almost entirely which channels it decided
to protect.

For the [sub-bit tiers](HyprSlug.md) it is not a refinement at all but
the whole mechanism: at half a bit there is no magnitude left to
allocate, and all an imatrix can do — all there *is* to do — is decide
which signs survive.

## How it is measured

Forward hooks on every linear layer, accumulating `sum(x²)` per input
feature across calibration tokens. That is what llama.cpp's `imatrix`
tool does, and doing the same thing means the numbers mean the same
thing: an imatrix from here works in `llama-quantize`, and one from the
community works in `hyprslug`.

```bash
hnx-imatrix measure ./checkpoint \
  -t wiki.txt -t code/ \
  --chunk-tokens 512 \
  -o model.imatrix
```

`-t` takes a file, a directory (`.txt`, `.md`, `.jsonl` inside it), or a
literal string. Text is concatenated and cut into fixed-size chunks,
because a matmul's channel statistics are a property of the activations
and short ragged inputs give the padding a vote.

## Both formats, decided by content

llama.cpp's binary `.imatrix` is the one people share; JSON is the one
you can look at when the answer is wrong and you need to know why.
`Imatrix.load()` sniffs the file rather than trusting the suffix, because
people rename these.

The binary stores **sums** beside their call count, not means — that is
what llama.cpp writes, and its loader divides by `ncall` itself. A file
of means with `ncall` set would be divided twice, and every weight would
be wrong by the same factor: exactly the kind of error that looks like a
bad calibration set.

## Two details that bite

**Keys are GGUF tensor names, not torch module names.** An imatrix keyed
by `model.layers.7.self_attn.q_proj` looks right, loads fine, and matches
nothing. The mapping is explicit — `blk.7.attn_q.weight` — and returns
empty rather than guessing for a module with no GGUF counterpart, because
a guessed name silently weights the wrong tensor.

**One value per channel, not per weight.** The quantiser needs one per
weight, and a GGUF weight tensor is rows of `n_input` elements, so the
vector tiles across the rows. A width that does not divide is a different
model's imatrix and is ignored with a warning rather than stretched.

## The honest limit

Measuring means running the model, and running a GGUF means an inference
engine this package does not carry. So point it at the Hugging Face
checkpoint the GGUF was converted from — the tensor names match, so the
result applies to either:

```
An imatrix is measured from activations, which means running the model —
and a GGUF needs an inference engine to run. Point this at the Hugging
Face checkpoint the GGUF was converted from; the tensor names match, so
the result applies to either.
```

Needs the `[train]` extra (`transformers`) for `measure`. Reading,
writing and converting need nothing beyond the base install.

## See also

- [HyprSlug](HyprSlug.md) — the quantiser that consumes it
- [Dflash2](Dflash2.md) — a draft model inside the model
- [Quantization](Quantization.md) — the GGUF pipeline
