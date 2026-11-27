# Whisk — `hypernix.whisk`

A whisk blends. In ML terms: take N saved snapshots/state dicts and
produce a single merged set of weights (checkpoint averaging). All modes
work in place on plain `dict[str, Tensor]`, so they compose with
anything, and accept either state dicts directly or paths to `.pt` /
`.safetensors` files.

## Modes

| Function | Mode name | Formula |
|---|---|---|
| `swa_average(items, *, strict=False)` | `"swa"` / `"average"` / `"mean"` | Uniform mean across all N inputs. |
| `ema(items, *, decay=0.99, strict=False)` | `"ema"` | Exponential moving average — earlier inputs weighted `decay ** (N-1-i)`, last input weighted `1`. |
| `geometric_mean(items, *, strict=False, eps=1e-12)` | `"geometric-mean"` / `"geo-mean"` | Element-wise geometric mean, values clamped to `eps` first so `log()` stays finite. |

For all three: non-floating-point tensors (e.g. integer id buffers) are
**not** averaged — `swa_average`/`geometric_mean` take them from the
first input, `ema` takes them from the *last* input. Averaging happens
in `float32` internally, then cast back to the original dtype.

`strict=False` (default) intersects keys across all inputs, silently
dropping any key not present in every one. `strict=True` raises
`ValueError` on any key mismatch instead. Key intersection does not validate
tensor shapes or checkpoint architecture compatibility; mismatched shapes
can still fail during arithmetic. Check that all checkpoints share a
compatible model layout before averaging. For path inputs, `.pt` files are
loaded on CPU with `weights_only=True`; state dictionaries passed directly
are used as supplied.

## One-shot helpers

```python
from hypernix.whisk import whisk, whisk_to_snapshot

merged = whisk(["ckpt-1.pt", "ckpt-2.pt", "ckpt-3.pt"], mode="swa")
model.load_state_dict(merged)

# Or write a full HF-style snapshot directory in one call:
whisk_to_snapshot(
    ["ckpt-1.pt", "ckpt-2.pt"],
    out_dir="merged-snapshot/",
    tokenizer_source="ckpt-1.pt",
    mode="ema", decay=0.95,
)
```

`whisk(items, *, mode="swa", decay=0.99, strict=False)` — dispatches to
one of the three functions above by name (case-insensitive, `_`→`-`).
Raises `ValueError` listing valid modes if `mode` doesn't match.

`whisk_to_snapshot(items, out_dir, tokenizer_source=None, *, mode="swa", decay=0.99, strict=False) -> Path`:
1. Calls `whisk()` to merge.
2. Attempts to recover a model config via `_try_load_config()` — looks
   for a `config.json` sibling (`p.parent / "config.json"` or
   `p.with_suffix(".json")`) next to each **path**-shaped input (dict
   inputs are skipped since they have no associated path), returns the
   first one that parses successfully as a `HyperNixConfig`.
3. **If no config is found:** emits a `UserWarning` and instead writes a
   bare `model.safetensors` file directly into `out_dir` (no config,
   no tokenizer) — you finish the snapshot manually.
4. **If a config is found:** builds a `HyperNixModel`, loads the merged
   state dict (respecting `strict`), and calls
   `hypernix.train.save_snapshot()` to write a full snapshot, copying
   tokenizer files from `tokenizer_source` if given.

`MODES: tuple[str, ...] = ("swa", "ema", "geometric-mean")` — the
canonical mode name list.

### Required modules

- `torch` (hard dependency)
- `safetensors.torch` — only imported when reading/writing `.safetensors` files
- `hypernix.train` (`HyperNixModel`, `HyperNixConfig`, `save_snapshot`) — internal, imported lazily inside `whisk_to_snapshot`/`_try_load_config`
- Standard library: `warnings`, `json`, `pathlib`, `collections.abc`, `typing`


> **Canonical source:** `src/hypernix/models/whisk.py`. The `hypernix.whisk` import spelling used in examples is the compatibility-facing API; the source path above is the implementation to check when verifying behavior. `strict=False` intersects keys but does not make differently shaped tensors compatible. Shape disagreements can still fail during arithmetic; validate checkpoint architecture/config compatibility before averaging.

---

## See also

- [Training](Training.md) — `save_snapshot`, `HyperNixModel`, `HyperNixConfig`
- `hypernix.tupperware` — round-based training that pairs naturally with checkpoint averaging between rounds
