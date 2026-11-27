# Abbicus — `hypernix.abbicus`

Automatic token regulation and curriculum tuning during training. Dynamically
adjusts max sequence length and padding based on model size, dataset type,
and global step.

Two regulators are available: the original `Abbicus` (linear growth) and the
new `TurboAbbicus` (exponential growth with sine-wave oscillation at the hard cap).

---

## Abbicus (linear curriculum)

```python
from hypernix import Abbicus, AbbicusConfig

cfg = AbbicusConfig(
    model_size="7B",           # 0.5B … 72B — sets size multiplier
    base_context_length=4096,
    dataset_type="text",       # text | math | code
    curriculum_steps=10000,
    dynamic_padding=True,
)
abb = Abbicus(cfg)
```

### Training loop

```python
for step in range(steps):
    abb.step(step)
    batch = {"input_ids": ids, "labels": labels}
    batch = abb.regulate(batch)
    loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
    ...
```

### Behaviour

- **Curriculum length** — starts at 25% of `base_context_length`, grows to
  `100% × size_multiplier` over `curriculum_steps` (linear).
- **Math / code padding** — pads sequences to multiples of 8 for tensor-core
  efficiency when `dynamic_padding=True`.
- **Size multipliers** — smaller models (≤1B) use 0.5×; 70B+ uses 2.0×.

---

## TurboAbbicus (v0.70.4 — exponential curriculum)

`TurboAbbicus` replaces linear growth with an **exponential** ramp up to a
configurable `hard_cap`, then oscillates around the cap using a sine wave
adjusted by CPU utilisation (never GPU).

```python
from hypernix import TurboAbbicus, TurboAbbicusConfig

cfg = TurboAbbicusConfig(
    model_size="7B",              # sets size multiplier (same as AbbicusConfig)
    base_context_length=4096,
    hard_cap=16384,               # absolute maximum context length
    curriculum_steps=10000,
    oscillation_enabled=True,     # enable sine-wave oscillation at hard_cap
    oscillation_frequency=0.01,   # radians per step
    oscillation_amplitude=0.10,   # fraction of hard_cap (+/- 10%)
    cpu_factor_scale=0.05,        # system-load adjustment scale
    vram_safety_threshold=0.90,   # scale back if VRAM > 90% used
    dynamic_padding=True,
    dataset_type="text",
)
ta = TurboAbbicus(cfg)
```

### Training loop

```python
for step in range(steps):
    ta.step(step)               # updates context length + VRAM safeguard
    batch = {"input_ids": ids, "labels": labels}
    batch = ta.regulate(batch)  # truncates to current_max_length
    loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
    ...

print(ta.current_max_length)    # current allowed context in tokens
```

### Behaviour

- **Exponential growth** — starts at 25% of base, grows as
  `base_len × exp(k × progress)` where `k = ln(hard_cap / base_len)`.
- **Sine oscillation at cap** — once the ramp reaches `hard_cap`, context
  oscillates as `hard_cap × (1 + sin(step × freq) × amp + cpu_adj)`.
- **CPU adjustment** — the oscillation is shifted slightly by how busy the
  host CPU is (`(cpu_percent - 50) / 50 × cpu_factor_scale`). GPU is never
  used as a change factor.
- **VRAM safeguard** — `step()` checks `torch.cuda.memory_allocated / total`.
  If usage exceeds `vram_safety_threshold`, `current_max_length` is scaled
  down by 10% (min 50%). It recovers by +5% per step when pressure eases.

### TurboAbbicusConfig reference

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `"7B"` | Sets the size multiplier (0.5B→0.5×, 7B→1.0×, 70B→2.0×) |
| `base_context_length` | `4096` | Base context to ramp from |
| `hard_cap` | `16384` | Maximum context length |
| `curriculum_steps` | `10000` | Steps to reach the hard cap |
| `oscillation_enabled` | `True` | Enable sine-wave at cap |
| `oscillation_frequency` | `0.01` | Sine frequency in radians/step |
| `oscillation_amplitude` | `0.10` | Amplitude as fraction of hard_cap |
| `cpu_factor_scale` | `0.05` | CPU-load adjustment scale |
| `vram_safety_threshold` | `0.90` | VRAM fraction that triggers scale-down |
| `dynamic_padding` | `True` | Pad math/code batches to multiples of 8 |
| `dataset_type` | `"text"` | `text` / `math` / `code` |

---

## Using with `train run` (CLI)

```bash
# Linear Abbicus
hypernix train run --model-dir ./snap --dataset data.txt --out-dir ./out \
    --use-abbicus

# Turbo Abbicus
hypernix train run --model-dir ./snap --dataset data.txt --out-dir ./out \
    --use-turbo-abbicus --untrained-max-context 16384

# Turbo Abbicus + STML segmentation
hypernix train run --model-dir ./snap --dataset data.txt --out-dir ./out \
    --use-turbo-abbicus --use-stml --segment-length 512
```

---

## Pairing

- Pair with **STML** (see [STML.md](STML.md)) to fold long segments into the
  batch dimension instead of hard-truncating.
- Use with `PressureCookerV3` / `instant_pot.brew()` and
  [Compute Framework](Frameworks.md) for multi-GPU runs.
- For multi-round datasets see [Tupperware](Tupperware.md).

See also: [Training.md](Training.md), [STML.md](STML.md).
