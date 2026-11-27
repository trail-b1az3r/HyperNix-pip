# Dflash2 — a draft model, inside the model it drafts for

```bash
dflash2 attach  model.Q4_K_M.gguf -o model.dflash2.gguf
dflash2 info    model.dflash2.gguf
dflash2 extract model.dflash2.gguf -o draft.gguf
dflash2 strip   model.dflash2.gguf -o model.gguf
```

## What speculative decoding is

Two models. A small fast one proposes the next few tokens; the large one
checks all of them in a **single** forward pass and keeps the longest
prefix it agrees with. Every token that survives cost one draft step
instead of one full step.

The part that makes it safe rather than a quality trade: the output is
*identical* to what the large model would have produced on its own. A
proposal is kept only where the large model independently chose the same
token, and at the first disagreement the large model's token wins and the
rest of the proposal is thrown away. **A bad draft costs time. It cannot
cost correctness.**

## Why nobody does it

Logistics. It needs two files that share a tokenizer, and the small one
has to come from somewhere. So the speed-up sits behind "find or train a
compatible draft model", which for most people is where it stops.

Dflash2 removes that step. `attach` derives a draft from the base model
and writes it into the **same GGUF**, under a namespaced tensor prefix
with its own metadata block. One file, one download, one path to pass
around.

## What the draft actually is

Layers, dropped and requantised. Not trained, not distilled — this runs
in the time a quantisation takes, on a machine with no GPU, from nothing
but the base model.

```
dflash2 draft: 3/8 layers (38%) at Q4_0, proposing 4 tokens per round
  keeping layers: 0, 3, 7
  4821.3 MB -> 5108.9 MB (+6.0% for the draft)
  24 draft tensor(s) beside 67 base tensor(s)
  sharing the base's token_embd.weight, output.weight, output_norm.weight
```

**The first and last layers are always kept.** A pruned model that lost
its first block does not produce merely worse tokens, it produces tokens
from a different distribution entirely, and its proposals are rejected at
a rate that makes the whole exercise negative.

The embedding and output tensors are **shared** with the base rather than
copied: the draft has to speak the same vocabulary as the model it drafts
for, and a second copy would be the largest thing in the draft and
identical to one already in the file.

| Option | Default | What it changes |
|---|---|---|
| `--depth` | `0.25` | Fraction of the base's layers the draft keeps |
| `--layers` | — | Explicit indices instead of a depth |
| `--quant` | `Q4_0` | Block format for the draft's weights |
| `--draft-tokens` | `4` | Tokens proposed per round |
| `--no-share-embeddings` | off | Give the draft its own vocabulary tensors |

Past about four draft tokens per round, one rejection throws away more
work than the extra acceptances win back.

## Whether it helped

A draft that agrees 15% of the time makes generation **slower**, and the
only honest way to know is to count. `speculate()` returns both numbers:

```
28 token(s) in 9 target call(s) (3.11 per call)
  draft proposed 32, 24 accepted (75%)
```

`tokens_per_target_call` is the one that decides it. One means the draft
bought nothing and cost its own runtime; below the ratio of draft cost to
target cost, it made things worse.

## It still runs everywhere it ran before

The extra tensors are namespaced under `dflash2.` and every `dflash2.*`
metadata key sits outside the `general.` / `<arch>.` namespaces upstream
uses. A llama.cpp that has never heard of Dflash2 loads the file, finds
every tensor it expects, ignores the ones it does not, and runs the base
model exactly as before.

`dflash2 strip` reproduces the original file **byte for byte** — which is
the strongest statement available that attaching one is non-destructive,
and a test asserts it.

## Handing the draft to a runtime

Every llama.cpp that speculates wants the draft as a second *path*
(`--model-draft`). Carrying it inside the model means you still only
download and pass around one file; `extract` is where it becomes two:

```bash
dflash2 extract model.dflash2.gguf -o draft.gguf
llama-server -m model.dflash2.gguf --model-draft draft.gguf
```

`hypernix.models.ggufrun.materialize_draft()` does the same thing from
Python, caching beside the model so it is written once.

The block count in the extracted draft's architecture metadata is
rewritten to the draft's own. A draft that claimed the base's layer count
would describe tensors it does not have, and a loader would fail on the
first missing block rather than on the metadata.

## See also

- [HyprSlug](HyprSlug.md) — the quantiser the draft's weights go through
- [Imatrix](Imatrix.md) — measuring importance for the base quantisation
- [Quantization](Quantization.md) — the GGUF pipeline
