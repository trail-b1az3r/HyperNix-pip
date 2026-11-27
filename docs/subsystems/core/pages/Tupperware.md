# Tupperware — `hypernix.tupperware`

Automated dataset round splitting for multi-phase fine-tunes. Splits a corpus
into N rounds with automatic step budgets, per-round learning rates, and
optional evaluation after each round.

## Quick start

```python
from hypernix import Tupperware, TupperwareConfig

box = Tupperware(TupperwareConfig(num_rounds=4, eval_each_round=True))
paths = box.split_file("./corpus.txt", out_dir="./rounds")
plan = box.plan(num_tokens=120_000, param_count=80_000_000)

for rnd, (path, cfg) in enumerate(zip(paths, plan)):
    print(f"round {rnd}: {cfg.steps} steps @ lr={cfg.lr:.2e}")
    # train_on(path, steps=cfg.steps, lr=cfg.lr, ...)
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `num_rounds` | 3 | How many dataset slices |
| `total_steps` | auto | Override total step budget |
| `base_lr` | auto | Scale-aware LR from param count |
| `eval_each_round` | False | Run eval hook after each round |
| `eval_final_only` | False | Only eval on last round |
| `warmup_ratio` | 0.05 | Warmup fraction per round |
| `cooldown_ratio` | 0.03 | Cooldown fraction per round |
| `lr_decay_per_round` | 0.85 | LR multiplier per successive round |

## Plotting

Use `new_fridge.plot_round_losses()` to chart per-round loss curves after
training.

See also: [Training.md](Training.md), [Fridges.md](Fridges.md), [Abbicus.md](Abbicus.md).
