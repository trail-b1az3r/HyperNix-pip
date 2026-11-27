# `hypernix-t1 index` — a folder of GGUFs into a model registry

```bash
hypernix-t1 index                          # ./hypernix/models -> models.json
hypernix-t1 index --dir /srv/models --dry-run
hypernix-t1 index --refresh
hypernix-t1 index --plan pro --input-price 0.5 --output-price 1.5
```

## Why the registry has to be filled at all

The registry is the **only** place the T1 API looks up what a model can
do — its context limit, its token limits, whether it is routable. Every
route calls `ModelRegistry.require` rather than trusting a
client-supplied `model_id`. That is the right design, and it also means
a server whose registry is empty serves nothing.

Until this command the two ways to fill it were the installer's
one-entry template — placeholders, marked *"Edit before serving
traffic"* — and writing the JSON by hand. Both ask an operator to
transcribe numbers that are already in the files. Transcription is where
those numbers go wrong, and **a context limit that is wrong in the
registry is not caught anywhere**: it is simply the number the server
enforces.

## Measured, assumed, and decided

The distinction the whole command turns on.

| | comes from | example |
|---|---|---|
| **measured** | the GGUF's metadata and tensor table | architecture, context length, parameter count |
| **assumed** | a default, *and said to be* | `context_limit` when the file carries none |
| **decided** | you, via flags | price, plan, availability, routing priority |

`total_parameters` is summed from the tensor table. Three quantisations
of one model therefore report the same figure, and none of them is read
off a filename — a file called `7B` whose tensors say 6.74 billion is
reported as 6.74.

An assumed value goes in the entry's `notes` as `assumed: context_limit`
and is printed in the report. Defaulting is fine; defaulting *silently*
is what turns a guess into a limit the server enforces without anyone
knowing it was invented.

Nothing about price can be read from a GGUF, so nothing about price is
guessed. The default is free-and-counted, which is accounting rather
than a claim that the model is free to run.

## Running it twice is safe

This is the property that decides whether the command is usable at all.

**An entry already in the registry is left exactly as it is.** `--refresh`
re-reads the *measured* fields for those too — and still leaves
`pricing`, `minimum_plan`, `routing_priority`, `availability`, `status`,
`fallback_model`, `license` and `notes` as you set them. An indexer that
reset a hand-tuned entry on every run would be worse than no indexer,
because the loss is silent.

**An unchanged registry is not even rewritten.** Re-indexing an unchanged
folder is what running the command twice does, and rewriting identical
bytes moves the mtime — which is what a file watcher or a config-reload
hook is looking at. "Nothing changed" should look like nothing changed.

**A registry that is not valid JSON is refused, not overwritten.** If
something hand-written is in there and malformed, the fix is to look at
it, not to lose it.

```
  toy-iq0-9-l
             1M  IQ0.9_L      hnxrun     ctx 8192  llama
      assumed: context_limit (not in the file's metadata)
  broken.gguf: unreadable — does not start with the GGUF magic

  models.json
    added     1: toy-iq0-9-l
    left      2 already-registered entries unchanged (pass --refresh to re-read them)
    3 model(s) in the registry
```

An unreadable file is reported and the walk continues — the point of
scanning a folder of models is to find out which one is the problem. The
exit code is non-zero, because the registry that was written is missing
a model someone put there on purpose.

## What runs each model

`local_available` records whether this machine can execute it: a
HyperNix extension type needs [HnxRun](HnxRun.md) or llama.cpp built
with [`native/ggml-hnx`](../native/ggml-hnx/README.md), and an upstream
quant can go to any llama.cpp. The tier and bit rate land in `notes`, so
a registry listing says which entries are the cheap ones.

## Symlinked folders

The walk follows symlinks (0.72.6.post3). A model kept on another disk
and linked in, `ln -s /data/qwen ./hypernix/models/qwen`, is indexed
under the name it has here. `Path.rglob`, which the index used before,
never descends into a symlinked folder, so such a model was silently
left out. A link back up the tree is walked once. A link whose target
has gone is reported as unreadable, with the rest of the unreadable
files, rather than skipped.

## Flags

| | |
|---|---|
| `--dir D` | where the models are (default `./hypernix/models`) |
| `-o FILE` | registry to write (default `models.json` beside the config) |
| `--refresh` | re-read measured fields of entries already present |
| `--dry-run` | report and write nothing |
| `--plan`, `--input-price`, `--output-price`, `--currency` | policy for *new* entries |
| `--availability`, `--priority` | ditto |
| `--json` | machine-readable, including the per-file report |

It needs neither a configured server nor the `[t1api]` extra: someone who
has just dropped models in a folder has not run `create` yet.

## See also

- [CLI](CLI.md) — the rest of `hypernix-t1`
- [HyprSlug-Headers](HyprSlug-Headers.md) — where the tier and bit rate come from
- [HnxRun](HnxRun.md) — what executes the sub-bit entries
