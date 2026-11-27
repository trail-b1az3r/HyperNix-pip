# Fridges — cold storage for weights, data, and graphs

Three small modules living under `hypernix.*`, named in the kitchen
idiom that started with the oven: each keeps something cold so the
training loop can stay hot.

| Module | Keeps cold | Main entry points |
|---|---|---|
| `old_fridge` | **Weights** (freeze, stats, offload, unwrap) | `freeze`, `unfreeze`, `parameter_stats`, `offload_to_cpu`, `chill_cache`, `unwrap_model` |
| `mediocre_fridge` | **Datasets** (judge training examples) | `synthesize_judge_corpus`, `collect_responses_from`, `JudgeExample` |
| `new_fridge` | **Logs → graphs** | `parse_training_log`, `plot_loss_curve`, `plot_score_distribution`, `plot_round_losses` |

## `old_fridge` — weight housekeeping

### `freeze(model, patterns)`

Sets `requires_grad=False` on parameters whose names match any glob or
substring pattern in `patterns`. Returns the number of frozen params.

```python
from hypernix import old_fridge

frozen = old_fridge.freeze(model, patterns=("embed_tokens",))
# => frozen = 32000*768 = 24_576_000

# Multiple patterns.  Matches "embed_tokens.*", "layers.0.*", "layers.1.*"
old_fridge.freeze(model, patterns=("embed_tokens", "layers.0", "layers.1"))
```

Idempotent — re-freezing an already-frozen parameter is free.

### `unfreeze(model, patterns)`

Inverse. Default unfreezes everything (`patterns=("*",)`).

### `parameter_stats(model)`

Returns a `ParamStats(total, trainable, frozen, bytes)` dataclass with
a `.megabytes` convenience property.

```python
stats = old_fridge.parameter_stats(model)
print(f"{stats.trainable:,} trainable / {stats.total:,} total")
print(f"{stats.megabytes:.1f} MB on device")
```

### `offload_to_cpu(model, patterns)`

Moves named submodules to CPU. Useful for the "huge embedding + small
active set" pattern. Returns the number of modules moved.

### `chill_cache()`

`gc.collect()` + `torch.cuda.empty_cache()` (when CUDA is available).
Safe to call on CPU-only hosts.

### `unwrap_model(model)`

Peels DDP (`.module`), FSDP (`_fsdp_wrapped_module`), and
`torch.compile` (`_orig_mod`) wrappers so freeze/stats/optimizer binding
see the inner parameters. Used by `CodeOven.train` when attaching
PressureCookerV3 on distributed models.

## `mediocre_fridge` — judge-training corpus synthesis

The "mediocre" name refers to the output quality: the corpus is
intentionally mixed (half good, half mangled) so a pointwise or pairwise
judge model has a contrast set to learn from.

### Output format

One example per line:

```
<JUDGE_PROMPT>Capital of France?<JUDGE_RESPONSE>Paris<JUDGE_LABEL>GOOD
<JUDGE_PROMPT>2+2=?<JUDGE_RESPONSE>4444444<JUDGE_LABEL>BAD
```

Delimiter tokens are plain ASCII strings (not tokenizer specials) so any
byte / BPE tokenizer can consume them without modification.

### `synthesize_judge_corpus(n, out_path, *, seed=0, good_ratio=0.5)`

Cycles through a 16-entry seed corpus of common-knowledge Q/A pairs.
`good_ratio` of examples keep the reference answer; the remainder are
mangled (truncated, shuffled, replaced with plausible wrong answers,
emptied, repeated).

```python
from hypernix import mediocre_fridge

mediocre_fridge.synthesize_judge_corpus(
    n=2048, out_path="judge.txt", seed=0, good_ratio=0.5,
)
```

Deterministic for a given seed.

### `collect_responses_from(oven, prompts, *, max_new_tokens=32, label_rule=None)`

Sample real responses from an existing oven and wrap them as
`JudgeExample` tuples. Pass `label_rule=lambda p, r: ...` to label
according to your own heuristic; omit it to tag every response as
`GOOD` (teacher-sampled positives, to be paired with mangled negatives
from `synthesize_judge_corpus`).

```python
prompts = ["Capital of France?", "2+2=?", "Largest planet?"]
positives = mediocre_fridge.collect_responses_from(teacher_oven, prompts)
mediocre_fridge.write_examples(positives, "positives.txt")
```

## `new_fridge` — graphing and analytics

### `parse_training_log(stdout) -> [(step, loss)]`

Extract training progress from the text `hypernix.train.train` /
`oven.train` print on stdout:

```python
pairs = new_fridge.parse_training_log(open("train.log").read())
# => [(10, 2.3456), (20, 1.987), (30, 1.5012), ...]
```

Regex matches `step N/M  loss=X` — works with both `[hypernix.train]` and
`[old_oven.train]` prefixes.

### `plot_loss_curve(pairs, out_path, *, title)`

Save a PNG of the training curve:

```python
new_fridge.plot_loss_curve(pairs, "loss.png", title="HyperNix 1.5 judge")
```

### `plot_score_distribution(scores, out_path, *, title, bins=20)`

Histogram over a list of scores.

### Lazy matplotlib

matplotlib is not a `hypernix` dependency. The first plotting call uses
`hypernix.deps.ensure(["matplotlib>=3.7"])` to install it on demand,
respecting `HYPERNIX_AUTO_INSTALL=0` as a kill switch.

On a machine where you can't pip-install, call
`parse_training_log` to get the pairs and plot them any way you like.

## Putting them together

The canonical "train an evaluator" flow exercised by
[`examples/train_hypernix_0_1_5_evaluator.py`](../examples/train_hypernix_0_1_5_evaluator.py):

```python
import contextlib, io
from hypernix import old_oven, old_fridge, mediocre_fridge, new_fridge

# 1. Synthesize a judge corpus.
mediocre_fridge.synthesize_judge_corpus(n=2048, out_path="judge.txt")

# 2. Freeze embeddings on a fresh oven (cheap finetune).
oven = old_oven.preheat(local_dir="./scratch", device="cuda", dtype="float16")
old_fridge.freeze(oven.model, patterns=("embed_tokens",))
print(old_fridge.parameter_stats(oven.model))

# 3. Train, capturing stdout so we can plot the curve.
log = io.StringIO()
with contextlib.redirect_stdout(log):
    oven.train("judge.txt", "./trained", steps=500, batch_size=1, log_every=10)

# 4. Plot.
pairs = new_fridge.parse_training_log(log.getvalue())
new_fridge.plot_loss_curve(pairs, "judge_loss.png", title="Judge training")
```
