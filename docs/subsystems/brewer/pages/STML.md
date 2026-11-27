# STML — `hypernix.stml` (v0.70.4)

**Short Term Memory Loss** is a context management tool with two capabilities:

1. **`calculate_vram_context`** — a CLI-friendly function that estimates the
   maximum safe trained context length given your GPU VRAM, model size, and
   training configuration.
2. **`STML`** — a training-time context manager that enforces a hard untrained
   cap, and folds sequences into the batch dimension as named "segments" so
   the model trains on all the context (not just a truncated slice).

---

## VRAM context calculator

```python
from hypernix import calculate_vram_context

ctx = calculate_vram_context(
    vram_gb=24.0,          # total GPU VRAM (e.g. 24.0 for an RTX 3090)
    model_size_params=7.0, # model parameter count in billions
    batch_size=2,
    precision="fp16",      # fp32 | fp16 | int8 | int4
    num_layers=32,         # transformer layer count
    num_heads=32,          # attention head count
    head_dim=128,          # per-head dimension
)
print(f"Max safe trained context: {ctx} tokens")
```

The estimate accounts for:
- Model weights + gradients + AdamW optimizer states
- KV cache per-token memory
- Activation memory (heuristic: `10 × layers × hidden_dim × precision_bytes × batch`)
- 15% VRAM safety margin (min 1 GB) for PyTorch/CUDA overhead

Result is always a multiple of 128 with a floor of 128.

### CLI

```bash
hypernix stml --vram 24.0 --params 7.0 --batch-size 2

# Full options:
hypernix stml --vram 16.0 --params 4.0 --batch-size 4 \
    --precision int4 --num-layers 32 --num-heads 32 --head-dim 128
```

| Flag | Default | Description |
|---|---|---|
| `--vram` | *(required)* | Available VRAM in GB |
| `--params` | `4.0` | Model parameters in billions |
| `--batch-size` | `2` | Training batch size |
| `--precision` | `fp16` | `fp32`, `fp16`, `int8`, or `int4` |
| `--num-layers` | `32` | Number of transformer layers |
| `--num-heads` | `32` | Number of attention heads |
| `--head-dim` | `128` | Dimension per head |

---

## STML context manager

```python
from hypernix import STML

mgr = STML(
    trained_context=2048,        # the context length you're training at
    untrained_max_context=8192,  # hard cap — sequences longer than this are truncated
    segment_length=512,          # fold sequences into segments of this length
    regulator=None,              # optional: Abbicus or TurboAbbicus instance
)

for step in range(steps):
    batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    batch = mgr.regulate(batch)
    # batch["input_ids"] now has shape (batch_size * num_segments, segment_length)
    loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
    ...
```

### What `regulate` does (in order)

1. **Curriculum regulator** — if a `regulator` (`Abbicus` / `TurboAbbicus`)
   was provided, it is called first to apply its own truncation/regulation.
2. **Hard truncation** — sequences longer than `untrained_max_context` are
   truncated. This is the "untrained" cap — the model never sees tokens beyond
   this point.
3. **Segment folding** — sequences longer than `segment_length` are padded to
   the nearest multiple of `segment_length`, then reshaped from
   `(batch, seq)` → `(batch × num_segments, segment_length)`. This lets the
   model train on all the context in properly-sized chunks.

Tensors that are processed: `input_ids`, `attention_mask`, `labels`.
Padding uses `0` for `input_ids`/`attention_mask` and `-100` (ignore index)
for `labels`.

### STML with TurboAbbicus

```python
from hypernix import TurboAbbicus, TurboAbbicusConfig, STML

ta = TurboAbbicus(TurboAbbicusConfig(hard_cap=8192))
mgr = STML(
    trained_context=2048,
    untrained_max_context=8192,
    segment_length=512,
    regulator=ta,
)

for step in range(steps):
    ta.step(step)               # update TurboAbbicus curriculum
    batch = mgr.regulate(batch) # TurboAbbicus runs first, then STML folds
    ...
```

### CLI integration

```bash
hypernix train run \
    --model-dir ./snap \
    --dataset corpus.txt \
    --out-dir ./out \
    --use-stml \
    --untrained-max-context 8192 \
    --segment-length 512 \
    --use-turbo-abbicus
```

### `old_oven.CodeOven.train()` integration

```python
oven = old_oven.preheat("nix2.5", device="cuda")
oven.train(
    "corpus.txt", "./out",
    steps=5000,
    use_stml=True,
    untrained_max_context=8192,
    segment_length=512,
    use_turbo_abbicus=True,
)
```

---

See also: [Abbicus.md](Abbicus.md), [Kitchen.md](Kitchen.md).
