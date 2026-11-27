# Kitchen — pans, microwave, table, sink, instant pot, coffee maker, pressure cooker

Seven modules added in v0.44, all following the same kitchen idiom
that started with the oven. They cover the bits of the lifecycle the
oven / fridge / freezer trio didn't: data preprocessing, one-shot
inference, training-log inspection, file output, end-to-end pipelines,
scheduled repetition, and a custom training optimizer.

| Module | What it does |
|---|---|
| [`pans`](#pans) | 5-tier preprocessing pipeline: FryingPan → SaucePan → Skillet → GrillPan → Wok. |
| [`microwave`](#microwave) | 5-tier throwaway inference: defrost → low_zap → zap → high_zap → chat_zap. |
| [`table`](#table) | Dead-simple tabular viewer over training logs and judge corpora. |
| [`sink`](#sink) | Append-only file output with optional rotation + dedupe. |
| [`instant_pot`](#instant_pot) | One-call end-to-end pipeline: preheat → train → optional GGUF. |
| [`coffee_maker`](#coffee_maker) | 3 tiers (drip / french-press / percolator) + cold-brew type. |
| [`espresso_maker`](#espresso_maker) | 4 tiers of prompt-battery evaluation: ristretto / single / double / lungo. |
| [`blender`](#blender) | 4 tiers of multi-source mixing. |
| [`toaster`](#toaster) | 4 tiers of per-line formatting. |
| [`food_processor`](#food_processor) | 4 tiers of bulk chunking / slicing / shredding. |
| [`smoker`](#smoker) | 4 tiers of training quality — useable / good / commercial / high-quality. |
| [`pressure_cooker`](#pressure_cooker) | AdamW + warmup / plateau / cooldown + lookahead. |

## pans

Five tiers of text preprocessing, progressively more opinionated.
Each pan reads from a file, path, or any iterable of strings and
yields processed strings.

```python
from hypernix import pans

# Tier 1 — verbatim, trim trailing whitespace.
for line in pans.FryingPan("corpus.txt"):
    ...

# Tier 2 — collapse internal whitespace, drop empties.
pans.SaucePan("corpus.txt")

# Tier 3 — wrap each line with chat or instruct tags.
pans.Skillet("corpus.txt", mode="chat")
pans.Skillet("corpus.txt", mode="instruct")

# Tier 4 — SaucePan + SHA1 deduplication + min-length filter.
pans.GrillPan("corpus.txt", min_chars=8)

# Tier 5 — buffer + shuffle + optional reverse-order augmentation.
pans.Wok("corpus.txt", seed=0, reverse_ratio=0.1)
```

Pick by name when the tier is a runtime variable:

```python
pans.pick_pan("grill-pan", source="corpus.txt", min_chars=16)
```

Pair with the sink to persist the output:

```python
from hypernix import pans, sink
sink.Sink("clean.txt").pour(pans.GrillPan("raw.txt", min_chars=8))
```

## microwave

Five power-level tiers of throwaway inference:

```python
from hypernix import microwave

# Tier 1 — defrost: preheat only, return the oven for reuse.
oven = microwave.defrost("nix2.5", device="cuda")

# Tier 2 — low_zap: 16 tokens, temp 0, top_k 1 (deterministic one-liner).
slug = microwave.low_zap("nix2.5", "Filename for: Annual revenue report 2024")

# Tier 3 — zap: 64 tokens, temp 0.2 (default).
out = microwave.zap("nix2.5", "def fib(n):")

# Tier 4 — high_zap: 512 tokens, temp 0.7 (draft a paragraph).
para = microwave.high_zap("nix2.5", "Explain RoPE in two paragraphs.")

# Tier 5 — chat_zap: one-turn chat with system prompt.
reply = microwave.chat_zap("nix2.5", "Capital of France?", system="Be terse.")

# Continuation without a rebuild:
more = microwave.reheat(oven, prior_output=out, max_new_tokens=32)
```

`microwave.TIERS` maps short names (`"defrost"`, `"low"`, `"standard"`,
`"high"`, `"chat"`) to the callable. All accept KNOWN_MODELS short
names, full HF repo ids, or a local snapshot path.

## table

Minimal tabular viewer over a list of dicts. Two canned constructors
load the two log types `hypernix` itself produces; the rest is
`.head()` / `.filter()` / `.select()` / `.sort_by()` / `.show()`.

```python
from hypernix.table import Table

# Training log
t = Table.from_training_log("train.log")
print(t.show(10))

# Judge corpus (mediocre_fridge.synthesize_judge_corpus output)
c = Table.from_judge_corpus("judge.txt")
bad = c.filter(lambda r: r["label"] == "BAD")
print(bad.select("prompt", "response").show())
```

No external deps; column widths auto-size from the data.

## sink

Append-only text sink. Optional rotation (one file per N bytes) and
dedupe (SHA1 hash set). The companion to `pans`: whatever the pan
yields, `sink.pour()` writes to disk.

```python
from hypernix.sink import Sink

s = Sink(path="out.txt", rotate_bytes=10_000_000, dedupe=True)
s.write("first line")
s.write("first line")                  # False — skipped
s.write_json({"event": "step", "i": 1})
s.close()

# Pipeline terminal:
Sink("clean.txt").pour(SaucePan("raw.txt"))
```

`Sink` is also a context manager (no buffering though — writes open /
close the file on every call for crash safety).

## instant_pot

One call, one trained snapshot. Recipe is a plain dict so it can live
in JSON:

```python
from hypernix import instant_pot

trained = instant_pot.brew({
    "repo_id": "nix2.5",               # or "local_dir": "./my-snap"
    "dataset": "./corpus.txt",
    "out_dir": "./trained",
    "steps": 500, "batch_size": 1, "context_length": 1024,
    "lr": 3e-4, "device": "cuda", "dtype": "float16",
    "freeze_embed": True,              # old_fridge.freeze embed_tokens
    "quants": ["fp16", "q4_k_m"],      # optional: also emit GGUFs
})
```

Required keys: `dataset`, `out_dir`. Everything else has a sensible
default. When `quants` is set, the trained snapshot is converted to
GGUF and each quant is produced via `llama-quantize` under
`out_dir/gguf/`.

## coffee_maker

Three tiers plus a separate "cold brew" type.

**Tier 1 — `CoffeeMaker` (drip).** The original: scheduled repetition.

```python
from hypernix import coffee_maker

def nightly_pretrain():
    ...  # pull fresh data, continue pretrain

maker = coffee_maker.coffee_maker(nightly_pretrain, interval_seconds=86400)
maker.run(cycles=7)                 # one week of nightly brews
maker.summary()                      # cycles / failed / mean_duration_s / last_ok
```

Exceptions don't stop the loop — they're captured as failed `Brew`
records. `.stop()` for a cooperative cancel; `.serve()` installs a
SIGINT handler.

**Tier 2 — `FrenchPressMaker` (batch).** Run a list of zero-arg
callables in sequence, collect all results.

```python
results = coffee_maker.french_press([
    lambda: train_lora("dataset_a"),
    lambda: train_lora("dataset_b"),
    lambda: train_lora("dataset_c"),
]).plunge()
```

**Tier 3 — `PercolatorMaker` (cyclic refinement).** The output of
cycle N feeds cycle N+1. Optional `convergence(old, new) -> bool`
short-circuits the loop.

```python
def draft_then_critique(prior):
    critique = judge.complete(f"critique this draft: {prior}")
    return oven.complete(f"revise given: {prior}\ncritique: {critique}")

final = coffee_maker.percolator(
    draft_then_critique, seed_input="First draft here.", max_cycles=5,
).percolate()
```

**New type — `ColdBrewMaker`.** Long single brew with mandatory disk
checkpoints. `brew_fn(state, phase)` reads the last state from
`checkpoint_path` and returns the next state.

```python
def phase_fn(state, phase):
    # Phase 0: download; Phase 1: convert; Phase 2: train; Phase 3: quantize; …
    state[f"phase_{phase}_done"] = True
    return state

cb = coffee_maker.cold_brew(phase_fn, phases=4,
                            checkpoint_path="./run/ckpt.json")
final_state = cb.brew()
```

Crashing mid-run? Just call `brew()` again — it picks up from the
last persisted phase.

## pressure_cooker

AdamW + three-phase LR schedule + lookahead, all unified within the `PressureCookerV3`
subclass. No separate scheduler object — the schedule is driven by
`PressureCookerV3._step`.

```python
from hypernix.pressure_cooker_v3 import PressureCookerV3, StovetopV3Cooker
from hypernix.optimizers.pressure_cooker_v4 import CookerLite   # a V4 now

opt = PressureCookerV3(
    model.parameters(),
    lr=3e-4,                   # (formerly peak_lr)
    warmup_steps=200,          # 0 -> peak, linear
    plateau_steps=1000,        # peak, constant
    cooldown_steps=200,        # peak -> 0, cosine
    betas=(0.9, 0.95),
    weight_decay=0.1,
    lookahead_k=5,             # pressure seal every 5 inner steps
    lookahead_alpha=0.5,
)

for batch in loader:
    out = model(batch["input"], labels=batch["labels"])
    out["loss"].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Introspection
    print(opt.phase(), opt.scheduled_lr())   # "warmup", 1.5e-4 …
```

### Why a custom optimizer?

Stock `AdamW` + `torch.optim.lr_scheduler.CosineAnnealingLR` covers
most cases. `PressureCookerV3` is a single-object answer when you want:

1. **One step call owns the whole schedule** — no risk of forgetting
   `scheduler.step()`, no order-of-operations bugs.
2. **Three-phase schedule baked in** — warmup / plateau / cosine are
   common enough together that having them in one place saves a few
   lines per training script.
3. **Lookahead's "slow weights" seal** — the Zhang et al. 2019 trick
   that empirically stabilizes AdamW on narrow networks. Off by
   default (`lookahead_k=0`); set to 5 or 10 to enable.

### Hardware-Specific Variants
If you are running on specialized hardware, use the explicit variants:
- **`StovetopV3Cooker`**: Automatically disables fused and foreach kernels to run safely on older CUDA 6.1 (Pascal) GPUs like the GTX 1080.
- **`CookerLite`**: Strips out lookahead and EMA by default and forces safe CPU-only operations for a much faster training loop on non-GPU hardware. It moved to `pressure_cooker_v4` and is a `PressureCookerV4` subclass now; `from hypernix import CookerLite` finds it either way.

Introspection helpers:

```python
opt.phase()                    # "warmup" | "plateau" | "cooldown" | "done"
opt.scheduled_lr(step=1200)    # what LR is used at step 1200
repr(opt)
# PressureCooker(peak_lr=0.0003, warmup=200, plateau=1000, cooldown=200,
#                lookahead=k=5, alpha=0.5)
```

## espresso_maker

Four tiers of prompt-battery evaluation. All four share a `pull`
method: given a list of prompts (+ optional references), run the
oven against each, score, return a list of `Shot(prompt, output,
score, reference)`.

| Tier | Tokens | Temp | Samples/prompt | Use |
|---|---|---|---|---|
| `Ristretto` | 16 | 0.0 | 1 | deterministic spot-check |
| `SingleShot` | 64 | 0.2 | 1 | standard eval |
| `DoubleShot` | 96 | 0.4 | 2 | scorer picks winner |
| `Lungo` | 256 | 0.8 | 4 | show-me-what-the-model-thinks |

```python
from hypernix import espresso_maker

maker = espresso_maker.double_shot(
    oven,
    scorer=lambda prompt, output, reference: sum(
        w in output.lower() for w in reference.lower().split()
    ),
)
shots = maker.pull(prompts=["Q1", "Q2"], references=["ref1", "ref2"])
print(maker.mean_score)
```

## blender

Four tiers of multi-source data mixing.

```python
from hypernix import blender, sink

# Straight concatenation:
blender.HandBlender(sources=["a.txt", "b.txt"])

# Round-robin interleave:
blender.PersonalBlender(sources=["a.txt", "b.txt"])

# Weighted sampling (70% source A, 30% source B):
blender.CountertopBlender(sources=["a.txt", "b.txt"], weights=[0.7, 0.3])

# Full buffer + shuffle (RAM-resident):
blender.HighPowerBlender(sources=["a.txt", "b.txt"], seed=0)

# Any blender pairs with a sink:
sink.Sink("mixed.txt").pour(blender.CountertopBlender(
    sources=["raw.txt", "curated.txt"], weights=[0.7, 0.3],
))
```

## toaster

Four tiers of per-line formatting.

```python
from hypernix import toaster

# Tier 1 — pair every 2 lines as (prompt, response):
toaster.TwoSliceToaster(source="pairs.txt",
                        prompt_tag="Q: ", response_tag="A: ")

# Tier 2 — four lines to one 2-turn chat:
toaster.FourSliceToaster(source="turns.txt")

# Tier 3 — streaming per-line template:
toaster.ConveyorToaster(source="stream.txt", template="<T>{line}</T>")

# Tier 4 — whole-document wrap (blank lines separate docs):
toaster.ToasterOven(source="docs.txt",
                    header="<DOCUMENT>", footer="</DOCUMENT>")
```

## food_processor

Four blade tiers for bulk text chunking.

```python
from hypernix import food_processor as fp

# Chop on blank lines:
fp.ChopBlade(source="big.txt", separator="\n\n")

# Fixed-length character slices with optional overlap:
fp.SliceBlade(source="big.txt", slice_chars=1024, overlap_chars=128)

# Whitespace-tokenized sliding window:
fp.ShredBlade(source="big.txt", window_tokens=256, stride_tokens=128)

# Whole file as one blob, whitespace collapsed:
fp.PureeBlade(source="big.txt")
```

## smoker

Four tiers of training quality, low-and-slow. Each wraps
`oven.train` with progressively more machinery; all return the
trained snapshot path.

| Tier | Adds to the previous tier |
|---|---|
| `UseableSmoker` | minimum viable — kwargs pass-through to `oven.train` |
| `GoodSmoker` | + linear warmup / plateau / cosine-cooldown LR schedule |
| `CommercialSmoker` | + EMA (exponential moving average of weights) blend at end |
| `HighQualitySmoker` | + curriculum (progressive context length) |

```python
from hypernix import smoker, old_oven

oven = old_oven.preheat(repo_id="nix2.5", device="cuda", dtype="float16")

# Useable — fastest, roughest output:
smoker.useable_smoker(oven=oven, steps=1000).smoke("corpus.txt", "./u")

# Good — warmup + cooldown:
smoker.good_smoker(oven=oven, steps=2000,
                   warmup_frac=0.1, cooldown_frac=0.2).smoke("corpus.txt", "./g")

# Commercial — + EMA:
smoker.commercial_smoker(oven=oven, steps=4000, ema_decay=0.95).smoke(
    "corpus.txt", "./c",
)

# High-quality — + curriculum:
smoker.high_quality_smoker(
    oven=oven, steps=8000, base_context_length=128, context_length=1024,
).smoke("corpus.txt", "./hq")
```

Each call records a one-line entry in `smoker.history` for
provenance. The tier choice is ultimately customization — swap
`UseableSmoker` for `HighQualitySmoker` without changing anything
else in your pipeline.

## CLI: `hypernix brew`

The `instant_pot.brew(recipe)` call is also wired to the CLI:

```bash
hypernix brew recipe.json                    # plain
hypernix brew recipe.json --set steps=2000   # typed override
hypernix brew recipe.json --set device='"cuda"' --set batch_size=2
```

`--set KEY=VALUE` accepts a JSON literal for the value (strings must
be quoted with double quotes inside single quotes on the shell).
Non-JSON values are used as plain strings.

## salt_shaker

Three tiers of **gentle data augmentation**.  Every shaker reads
lines from `source`, perturbs them, and yields strings — same
iterator shape as a pan, so they plug into `sink.Sink.pour(...)`
identically.

| Tier | Class | What it does |
|---|---|---|
| t1 | `FromTheBag` | per-character substitution at `rate`, preserves length |
| t2 | `HandCrusher` | adjacent-token swaps at `rate` |
| t3 | `PoshSaltDish` | independent drop / duplicate / swap rates at word granularity |

```python
from hypernix import salt_shaker, sink

# t1 — coarsest: random char substitutions
sink.Sink("noisy.txt").pour(
    salt_shaker.FromTheBag(source="clean.txt", rate=0.02, seed=0)
)

# t3 — finest: word-level drop/dup/swap
shaker = salt_shaker.PoshSaltDish(
    source="clean.txt",
    drop_rate=0.01, duplicate_rate=0.005, swap_rate=0.01, seed=0,
)
```

`salt_shaker.salt_shaker(tier, source, **kw)` picks a tier by short
name (`"from-the-bag"`, `"hand-crusher"`, `"posh-salt-dish"`).

## pepper_shaker

Three tiers of **sharp perturbations** — mask-language-model
training, typo robustness, negation-aware classifiers.

| Tier | Class | What it does |
|---|---|---|
| t1 | `SmallShaker` | random word masking; configurable `mask_token` (default `[MASK]`) |
| t2 | `Dish` | typo injection (drop / duplicate an internal char); preserves first + last |
| t3 | `TallHandmade` | negation injection; prepends `negator` (default `"NOT"`) at `rate` |

```python
from hypernix import pepper_shaker

masked = pepper_shaker.SmallShaker(
    source="cleaned.txt", rate=0.15, mask_token="[MASK]", seed=0,
)
typos = pepper_shaker.Dish(source="cleaned.txt", rate=0.05, seed=0)
negated = pepper_shaker.TallHandmade(source="cleaned.txt", rate=0.1, seed=0)
```

`pepper_shaker.pepper_shaker(tier, source, **kw)` picks a tier.

## Chaining shakers

Shakers compose by chaining their output iterators:

```python
from hypernix import pepper_shaker, salt_shaker, sink

salted = salt_shaker.HandCrusher(source="clean.txt", rate=0.02, seed=0)
peppered = pepper_shaker.SmallShaker(source=list(salted), rate=0.1, seed=1)
sink.Sink("augmented.txt").pour(peppered)
```

Each shaker takes a file path **or** any iterable of strings as
`source`, so the output of one becomes the input of the next without
a temp file.

## End-to-end example

```python
from hypernix import pans, sink, instant_pot, pressure_cooker

# 1. Preprocess raw scraped text into a training corpus.
sink.Sink("clean.txt").pour(pans.GrillPan("raw.txt", min_chars=16))

# 2. Brew a trained snapshot in one call.
trained = instant_pot.brew({
    "repo_id": "nix2.5",
    "dataset": "clean.txt",
    "out_dir": "./out",
    "steps": 1000, "batch_size": 1, "context_length": 1024,
    "quants": ["fp16", "q4_k_m"],
})

# 3. Push the GGUFs (hypernix.upload.upload_gguf or hypernix upload CLI).
```

Or, for anything with more moving parts, stitch together the individual
subsystems (oven, freezer, alarms, fridges, ranges) directly — instant
pot is for the cases where the defaults are right.

---

## qa (v0.70.4)

`QAProcessor` turns structured datasets into raw text strings for causal
language model training. It works with `salt_shaker` and `pepper_shaker` for
optional seasoning, and supports two training modes: **question-answer**
(predicts an answer given a question) and **predict-next** (concatenates the
two fields for next-token prediction).

```python
from hypernix.qa import QAProcessor

# From a list of dicts
data = [
    {"question": "What is AI?", "answer": "Artificial Intelligence."},
    {"question": "What is ML?", "answer": "Machine Learning."},
]
for text in QAProcessor(data, format_mode="question_answer"):
    print(text)
# Question: What is AI?
# Answer: Artificial Intelligence.

# From a JSONL file
for text in QAProcessor("dataset.jsonl"):
    ...
```

### Supported input formats

| Format | How it's parsed |
|---|---|
| `list[dict]` with `question`/`answer` keys | Direct extraction |
| `list[dict]` with `instruction`/`completion` keys | Automatic fallback |
| `list[dict]` with `prompt`/`response`/`input`/`output` keys | Automatic fallback |
| JSONL file path | Parsed line-by-line |
| Plain text file with tab/`::` / ` \| ` delimiter | Split on first delimiter |
| Any `Iterable[str]` | JSON-parse attempted, then delimiter split |

### Format modes

```python
# Question-Answer mode (default)
proc = QAProcessor(data, format_mode="question_answer")
# → "Question: {q}\nAnswer: {a}"

# Predict-next mode (plain concatenation)
proc = QAProcessor(data, format_mode="predict_next")
# → "{q} {a}"
```

### Seasoning (salt_shaker / pepper_shaker)

Shakers are applied **to the raw question/answer fields before templating**,
which keeps `Question:` and `Answer:` template keywords intact and maximises
efficiency.

```python
from hypernix import salt_shaker, pepper_shaker
from hypernix.qa import QAProcessor

salt = salt_shaker.FromTheBag(source=[], rate=0.02, seed=0)
pepper = pepper_shaker.SmallShaker(source=[], rate=0.05, seed=0)

proc = QAProcessor(
    data,
    salt_shaker=salt,
    pepper_shaker=pepper,
    season_target="both",    # "question" | "answer" | "both"
)
for text in proc:
    ...
```

`pepper_shaker` is applied first, then `salt_shaker`.

### QAProcessor reference

```python
QAProcessor(
    source,                        # path, list[dict], list[str], or any Iterable
    salt_shaker=None,              # optional salt_shaker.Shaker instance
    pepper_shaker=None,            # optional pepper_shaker.Shaker instance
    format_mode="question_answer", # "question_answer" or "predict_next"
    question_key="question",       # dict key for question field
    answer_key="answer",           # dict key for answer field
    season_target="both",          # which field(s) to season
)
```

