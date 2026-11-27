# Ovens — `hypernix.old_oven` / `hypernix.new_oven`

The oven is the user-facing object for generation, training, and
serialization. A HuggingFace snapshot (`config.json` + weights +
tokenizer files) is the raw dough; the oven bakes it into a working
language model with a stable Python API.

## Loading: `old_oven.preheat`

```python
from hypernix import old_oven

# Short name from KNOWN_MODELS — downloads on first call.
oven = old_oven.preheat(repo_id="nix2.5", device="cuda", dtype="float16")

# Or a local path.
oven = old_oven.preheat(local_dir="./my-snapshot", device="cpu")
```

Supported arguments:

| Arg | Default | What |
|---|---|---|
| `repo_id` | `"ray0rf1re/hyper-nix.1"` | HF repo id OR short name from KNOWN_MODELS |
| `revision` | `None` | git ref / tag |
| `local_dir` | `None` | Skip download if a snapshot is already on disk |
| `token` | `None` | HF token for gated repos |
| `device` | auto (`cuda` if available) | `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`, … |
| `dtype` | `"float32"` | `"float32"` / `"float16"` / `"bfloat16"` |
| `quiet` | `False` | Suppress download progress |

## Generation

```python
# Prompt completion.
oven.complete(
    "def fibonacci(n):",
    max_new_tokens=128,
    temperature=0.2, top_k=40, top_p=0.95,
    stop=("\nclass ", "\ndef "),   # `()` disables trimming
    seed=0,                        # deterministic when temperature=0
)

# Fill-in-the-middle (falls back to prefix-only continuation if the
# tokenizer has no FIM tokens).
oven.fill(
    prefix="def add(a, b):\n    return ",
    suffix="\n\nresult = add(1, 2)",
    max_new_tokens=32,
)

# Chat — multi-turn, uses the tokenizer's chat template if present.
oven.chat([
    {"role": "system", "content": "You are terse."},
    {"role": "user",   "content": "Capital of France?"},
], max_new_tokens=32, temperature=0.7)
```

`CodeOven.complete` accepts a `stop` tuple of substrings. The sampler
runs until `max_new_tokens` or an EOS, then the output is post-trimmed
at the first occurrence of any stop string. Pass `stop=()` to disable.

## Serializing

```python
# Self-contained torch.load-able bundle:
oven.save_pt("./hypernix.pt")
restored = old_oven.load_pt("./hypernix.pt", device="cpu")

# Standard HF snapshot directory (config.json + model.safetensors):
from hypernix import save_snapshot
save_snapshot(oven.model, "./snapshot")
```

`save_pt` captures: the state dict, the config, the tokenizer kind
(`"byte"`, `"sentencepiece"`, `"tokenizers"`), and enough metadata to
reconstruct the whole oven. `load_pt` reverses it. Size is ~1× the
weight size.

## Training from an oven

```python
oven.train(
    dataset_path="./corpus.txt",  # raw-text file
    out_dir="./trained",
    steps=1000, batch_size=2, context_length=512,
    lr=3e-4, weight_decay=0.1,
    log_every=10, save_every=500,
    seed=0, quiet=False,
)
```

`oven.train` is a thin wrapper around `hypernix.train.train` that
skips the round trip through disk: the model is already in memory, so
training mutates it in place and then calls `save_snapshot` at the end.

When using custom optimizers (`PressureCookerV3`, `StovetopV3CookerPlus`),
parameters are taken from `old_fridge.unwrap_model(model)` so DDP/FSDP
wrappers do not double-wrap the optimizer state.

Pass `use_abbicus=True` for automatic curriculum length regulation.
Pass `compute_framework=...` to route backward/step through a
`ComputeFramework` instance.

For multi-round dataset splits see [Tupperware.md](Tupperware.md).

## Fresh-init: `new_oven`

```python
from hypernix import new_oven

# Any arch in ARCH_PRESETS works.
oven = new_oven(
    "./fresh", arch="qwen2.5",
    vocab_size=151936, hidden_size=2048, intermediate_size=11008,
    num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2,
    max_position_embeddings=32768,
    device="cuda", seed=0,
)
```

See [Architectures.md](Architectures.md) for the full preset list and
the HF-arch mapping.

## `bake_code` shortcut

For a one-shot code completion without preheating separately:

```python
from hypernix import bake_code

print(bake_code(
    "./snapshot",                 # path or short name
    "def quicksort(xs):",
    max_new_tokens=128, temperature=0.2, stop=("\nclass ", "\ndef "),
))
```

`bake_code` preheats internally and discards the oven afterward. Handy
for CLI-style scripts; for interactive use preheat once and call
`oven.complete` repeatedly.

## FAQ

**Q. I get "no tokenizer.json" warnings.** The snapshot shipped without a
tokenizer. `preheat` falls back to a byte tokenizer and clamps
`vocab_size=256` on fresh inits; load any HF snapshot with tokenizer
files to get full coverage.

**Q. `.fill()` just continues the prefix, ignoring the suffix.** Your
tokenizer has no `<fim_prefix>` / `<fim_middle>` / `<fim_suffix>` tokens,
so FIM falls back to prefix-only continuation. This is the byte
tokenizer's behavior; SentencePiece + BPE with a FIM-aware tokenizer
picks up the three tokens automatically.

**Q. How do I load a Gemma 4 / Qwen 3.5 / GLM 5 checkpoint?** Just pass
the short name or repo id to `preheat`. `load_snapshot` routes those
through `transformers.AutoModelForCausalLM` — you don't need a preset
to load them, only to spin a fresh-init model in that shape.
