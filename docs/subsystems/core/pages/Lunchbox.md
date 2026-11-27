# Lunchbox — `hypernix.lunchbox`

The kitchen-idiom equivalent of bringing home-cooked food to work: you
pack records into a single box with a consistent layout, and whoever
unpacks it at the other end gets something coherent. `Lunchbox` packages
an eval/judge dataset with **one guaranteed schema** across every row —
specifically to avoid the classic HuggingFace `CastError: Couldn't cast
... because column names don't match`, which fires when Parquet shards
on disk disagree about the column set (e.g. you added a `latency_s`
field to your evaluator halfway through a run and appended newer shards
next to older ones).

```python
from hypernix.lunchbox import Lunchbox

box = Lunchbox()
for result in evaluation_run():
    box.add(
        id=result.id, category=result.category, difficulty=result.difficulty,
        tier=result.tier, prompt=result.prompt, reference=result.reference,
        model_response=result.response, keyword_score=result.kw_score,
        latency_s=result.latency, variant=result.variant, pipeline_meta=result.meta,
    )

box.pack("eval.parquet")                                   # write locally
box.push_to_hub("ray0rf1re/eval", token=os.environ["HF_TOKEN"])  # or push directly
```

## How it guarantees one schema

1. Collects every record as a plain Python dict (`.add(**fields)`).
2. Computes the **superset** of keys across all records (`.columns()`).
3. Fills missing cells with `None` (→ Arrow null on write) via `.normalize()`.
4. Writes through `datasets.Dataset` so the embedded `huggingface`
   metadata key in the Parquet file matches the actual column layout.

## `Lunchbox` (dataclass)

| Field | Type | Default | Notes |
|---|---|---|---|
| `records` | `list[dict[str, Any]]` | `[]` | |
| `required_columns` | `tuple[str, ...] \| None` | `None` | When set: `.add()` raises `ValueError` on any unknown key, and output columns follow this exact order. When `None`: columns are the alphabetically-sorted superset of all record keys. |

### Adding records

| Method | Notes |
|---|---|
| `.add(**fields)` | Append one record. Extra keys allowed (unless `required_columns` is set); missing keys tolerated (filled with `None` at pack time). |
| `.extend(iterable)` | Calls `.add(**r)` for each dict in `iterable`. |
| `len(box)` | `__len__` — record count. |

### Normalization / validation

| Method | Notes |
|---|---|
| `.columns()` | `required_columns` if set, else the sorted superset of all record keys. |
| `.normalize()` | Every row gets every column, missing cells as `None`. |
| `.validate()` | Raises `ValueError` if any column has mixed **incompatible** types across rows (e.g. `float` in some rows, `str` in others). `int` and `float` are treated as compatible (Arrow unifies them) — only genuinely incompatible type pairs raise. Runs automatically before every write. |

### Writing / pushing

| Method | Signature | Notes |
|---|---|---|
| `.pack(path)` | `(Path\|str) -> Path` | Writes a single Parquet file via `datasets.Dataset.to_parquet`. |
| `.pack_jsonl(path)` | `(Path\|str) -> Path` | One JSON object per line — same normalization, but no Parquet/pyarrow dependency needed. |
| `.push_to_hub(repo_id, *, token=None, private=False, commit_message="Update dataset via hypernix.lunchbox", split="train")` | `-> str` | Pushes via `datasets.Dataset.push_to_hub` (the actual `CastError` fix — per-shard metadata stays coherent). Returns the dataset's Hub URL. |

`datasets` is a **lazy dependency** — imported on first use inside
`_build_hf_dataset()`, and if missing, auto-installed via
`hypernix.deps.ensure(["datasets>=2.14"], reimport=["datasets"])`
(respects `HYPERNIX_AUTO_INSTALL=0`). `pack_jsonl()` avoids this
dependency entirely.

## `EVAL_SCHEMA` and `Lunchbox.for_eval()`

```python
from hypernix.lunchbox import Lunchbox
box = Lunchbox.for_eval()   # required_columns=EVAL_SCHEMA
```

`EVAL_SCHEMA = ("id", "category", "difficulty", "tier", "prompt",
"reference", "model_response", "keyword_score", "latency_s", "variant",
"pipeline_meta")` — the recommended (not enforced unless you use
`for_eval()`) column set for HyperNix-emitted eval datasets; it's what
`industrial_range`, `espresso_maker`, and `smoker` can emit naturally,
and what the Hub viewer expects for `ray0rf1re/eval`-style repos.

## Other constructors

```python
from hypernix.lunchbox import Lunchbox, lunchbox

box = Lunchbox.from_records(my_records, required_columns=EVAL_SCHEMA)
box2 = lunchbox(my_records)   # module-level shortcut, required_columns=None by default
```

`Lunchbox.from_records(records, required_columns=None)` — classmethod,
preloads via `.extend()`. `lunchbox(records=None, *, required_columns=None)`
— module-level constructor, optionally preloaded.

### Required modules

- `hypernix.deps` (internal — lazy `datasets` install)
- `datasets` — lazy, only needed for `.pack()`/`.push_to_hub()` (not `.pack_jsonl()`)
- Standard library: `json` (lazy, inside `pack_jsonl`), `dataclasses`, `pathlib`, `collections.abc`, `typing`


> **Canonical source:** `src/hypernix/data/lunchbox.py`. The `hypernix.lunchbox` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. `pack_jsonl()` avoids the optional Hugging Face `datasets` dependency; Parquet packing and Hub pushes need the appropriate optional dependencies and credentials.

---

## See also

- [Espresso Maker](EspressoMaker.md) — a natural source of eval records for `Lunchbox.for_eval()`
- `hypernix.industrial_range` — LLM-as-judge scoring, another common `Lunchbox` record source
- `hypernix.deps.ensure` — the lazy-install mechanism `lunchbox` relies on
