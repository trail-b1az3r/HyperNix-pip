# Training a model with HyperNix — what to use, and when

HyperNix ships about forty modules that touch training. That is the
problem this page exists to solve: `wiki/Training.md` documents the
`hypernix.train` API, `wiki/Optimizers.md` documents the optimizers,
`wiki/Ovens.md` documents the ovens — and none of them answers "I want to
train a model, which of these do I use?"

This one does. It is a decision guide, not an API reference; every
section links to the reference for the thing it tells you to use.

---

## Start here: three questions

**1. Are you making a new model, or changing an existing one?**

|                                   | Use                                              |
| --------------------------------- | ------------------------------------------------ |
| A new architecture, from nothing  | [`brewer`](Workshop.md) — presets, or your own shape |
| An existing checkpoint, adapted   | [`train`](Training.md) — the causal-LM loop      |
| An existing checkpoint, made bigger | [`expand_checkpoint`](Training.md) — warm-start a larger model |
| A GGUF you downloaded             | You want [quantisation](HyprSlug.md), not training |

**2. What are you training on?**

|                | Use                                                    |
| -------------- | ------------------------------------------------------ |
| One GPU        | [`pressure_cooker_v6`](Pressure-Cooker-V6.md) + [`freezer`](VRAM.md) |
| Several GPUs, no NVLink | [`lazy_suzan`](Training.md#lazy_suzan-decentralized-multi-gpu) |
| CPU only       | A `cpu-*` brewer preset. Not a GPU preset with a smaller batch |
| A laptop, on battery | Don't. Start on the laptop, move the run elsewhere |

**3. How much VRAM?**

|            | Preset                       | Optimizer                            |
| ---------- | ---------------------------- | ------------------------------------ |
| No GPU     | `cpu-nano` … `cpu-small`     | `pressure_cooker_v6`                 |
| 6–8 GB     | `33m`, `micro`               | `pressure_cooker_v5s` (memory-first) |
| 10–12 GB   | `small`                      | `pressure_cooker_v5`                 |
| 16–24 GB   | `medium`                     | `pressure_cooker_v6` (speed-first)   |
| 40 GB+     | `large`                      | `pressure_cooker_v6`                 |

If you are not sure, run `hnx doctor` — it reports what the machine has
and what that supports.

---

## The fast path

If you already know what you want and just want a GGUF at the end, that
is what [`instant_pot`](Kitchen.md) is for. It is the one module in the
package that is not a named tool with a narrow job: a single `brew(recipe)`
that runs download → preheat → train → save, and optionally convert →
quantize on the end.

```python
from hypernix.training.instant_pot import brew

brew({
    "model": "Qwen/Qwen2.5-0.5B",
    "data": "corpus.txt",
    "steps": 2000,
    "out": "./my-model",
    "export": {"format": "gguf", "quant": "Q4_K_M"},
})
```

The recipe is a plain dict so it loads from JSON or YAML. Use the instant
pot when the defaults fit. When they do not, peel it back to the
individual subsystems below — which is the whole reason they exist
separately.

---

## Building a model from nothing

[`brewer`](Workshop.md) builds `hyperNix0x-v2`: a custom PyTorch
transformer with RMSNorm, RoPE, grouped-query attention, SwiGLU FFNs and
optional sliding-window attention. Seven presets ship:

| Preset       | Layers | Params  | Context | Trains on         |
| ------------ | -----: | ------: | ------: | ----------------- |
| `cpu-nano`   |      4 |   2.07 M |     512 | A laptop CPU      |
| `cpu-tiny`   |      6 |   9.21 M |   1 024 | A laptop CPU      |
| `cpu-small`  |      8 |  26.45 M |   2 048 | A desktop CPU     |
| `33m`        |      6 |  33.64 M |   4 096 | 6 GB              |
| `small`      |      9 |    458 M |  20 482 | 10–12 GB          |
| `medium`     |     18 |    918 M |  40 964 | 16–24 GB          |
| `large`      |     36 |    3.5 B | 103 724 | 40 GB+, or several |

```bash
hnx brew new    --preset small --name my-model        # ./brewer_models/my-model
hnx brew train  --name my-model --data corpus.txt --steps 2000
hnx brew export --name my-model --format gguf --out my-model.gguf
```

Each model is a folder (`config.json`, `model.safetensors`, and for a
character-level model `char_vocab.json`), so the three commands can run
days apart. With `--save-dir D` on `new`, point the others at the folder
with `--dir D/my-model`.

**Pick the smallest preset that fits your data, not the largest that fits
your GPU.** A 458 M model on 10 MB of text memorises it; a 9 M model on
the same text learns something. The presets' context lengths are a
stronger constraint than their parameter counts — `large` at 103 724
tokens of context needs the VRAM for that KV cache before it needs any
for weights.

The `cpu-*` presets are their own family rather than the GPU presets
scaled down. `cpu-nano` drops the vocabulary to 8 000 and `cpu-tiny` to
16 000, where every GPU preset is at 32 000 — an embedding table is
`vocab x hidden`, so on a 2 M-parameter model the vocabulary *is* most of
the model, and halving it matters more than any other single change.
Their contexts are 512–2 048 against the GPU presets' 4 096–103 724.

---

## Data: the part that decides whether any of this works

More care here is worth more than any optimizer choice.

**Collecting it** — [`gather`](Gather.md) crawls documentation, wikis and
blogs into files the trainer reads:

```bash
hnx gather crawl -W https://example.com -Q 2 -T 4 -f parquet -o ./out
```

**Mixing it** — [`blender`](Blender.md) interleaves several sources,
which is what you want when the corpus is 90 % one thing. Four tiers,
each an iterable of lines: `HandBlender` concatenates, `PersonalBlender`
round-robins, and `CountertopBlender` / `HighPowerBlender` weight the
streams. `blender("personal", sources=[...])` constructs one by name.

**Cleaning it** — the [fridges](Fridges.md) are data hygiene. Run them.
Deduplication in particular: a corpus with the same paragraph forty times
trains a model that is very confident about that paragraph.

**Regulating it** — [`abbicus`](Training.md#abbicus-dynamic-token-regulation)
adjusts token flow against the model's size, context length and the
dataset's complexity. Useful when the corpus is uneven; unnecessary when
it is not.

**Building preference data** — [`mediocre_fridge`](Fridges.md) generates
`(prompt, response, GOOD|BAD)` triples, which is the signal a reward
model needs and the thing nobody has when they decide they want one.

---

## Optimizers: which pressure cooker

Six versions ship. They are not a progression where the highest number
wins — V5 and V6 optimise different things and the right one depends on
what you are short of.

| Version | Optimises for | Take it when                                   |
| ------- | ------------- | ---------------------------------------------- |
| V3      | Simplicity    | You want something you can read end to end     |
| V4      | Stability     | V5/V6 diverge on your model and you need to know why |
| V5      | **Memory**    | The optimizer state is what does not fit       |
| V5S     | Memory, more  | 6–8 GB, and V5 still does not fit              |
| V6      | **Speed**     | The default. Memory is not the binding constraint |
| V6V     | Speed, tuned  | V6 works and you are chasing the last few percent |

V5 is memory-first: quantised momentum, factored curvature, uint8 age
counters — every trick available to shrink optimizer state, at the cost
of several small Python-level ops per parameter per step. V6 throws all
of that away and keeps one momentum buffer updated with fused
multi-tensor ops. What makes V6 faster than AdamW is not a smarter update
rule; it is a smaller op count and far fewer host↔device synchronisation
points.

**Default to V6.** Move to V5 or V5S when you actually run out of memory,
and expect it to be slower — that is the trade you are making, not a
regression.

---

## Training loops: which smoker

[`smoker`](Smoker.md) wraps the oven's training loop in four ascending
tiers:

| Tier              | Adds                                              |
| ----------------- | ------------------------------------------------- |
| `UseableSmoker`   | Nothing. Forwards straight to `oven.train`        |
| `GoodSmoker`      | A pressure cooker with warmup / plateau / cooldown |
| `GreatSmoker`     | EMA and validation                                 |
| `CompetitionSmoker` | Everything, with checkpointing                   |

Anything you intend to keep should be at `GoodSmoker` or above. A run
without a learning-rate schedule is a run you cannot reproduce or reason
about, and `UseableSmoker` is there for smoke tests.

---

## Not losing the run

Four modules exist because four specific things go wrong.

[`apron`](Apron.md) captures every RNG source your script might touch —
Python's `random`, NumPy, PyTorch CPU and CUDA — and restores it on exit.
Use it as a context manager around the whole run. Without it,
"reproducible" means "reproducible until something else in the process
draws a random number".

[`cake_pan`](CakePan.md) checks the loss (and optionally the gradients)
after every step, reverts to the last known-good snapshot on a NaN or
Inf, and raises `BakeOff`. It also does per-layer device placement and
has a memory watchdog that empties the cache when free VRAM crosses a
threshold. A long run without it is a long run that can spend six hours
producing NaN.

[`freezer`](VRAM.md) budgets VRAM. Three variants behind one interface:
`OldFreezer` for 8–10 GB cards, `NewFreezer` for 11 GB+, and a third for
larger ones. It picks the batch and context defaults so you do not have
to discover them by OOM.

[`fusebox`](FuseBox.md) is the one to reach for when a run dies at 3 a.m.
and you want to know why.

---

## Watching it

```bash
tvtoppro --log train.log --modules disk,swap
```

[`tvtoppro`](TvTopPro.md) is the dashboard: per-core CPU, the memory
breakdown, the GPU row, and the training log's step / loss / throughput /
ETA. Two of its modules matter more than they look on a training box —
`disk`, because "the GPU is at 30 %" and "the dataloader is reading
900 MB/s off a spinning disk" are the same problem and only one of them
shows in the CPU box; and `swap`, because a run that starts swapping does
not slow down by 20 %, it slows down by two orders of magnitude, while
the memory box shows RAM at a comfortable 85 % throughout.

If the dashboard sits on a log nobody is writing to, `--find-run` finds
the busiest Python process on the machine and the log it is actually
writing.

---

## Getting a file out

Training produces a HuggingFace-style snapshot directory. Shipping needs
a GGUF.

```bash
# The usual: 4-bit, runs anywhere llama.cpp runs
hyprslug ./my-model/model.f16.gguf Q4_K_M -o my-model.q4km.gguf

# Everything in one file, so people pick a tier without a second download
hyprslug ./my-model/model.f16.gguf --multi Q8_0,Q4_K_M,IQ0.5_XXXL -o my-model.gguf

# A speculative-decoding draft, inside the same file
hyprslug my-model.gguf --draft dflash2 -o my-model.with-draft.gguf
```

[`hyprslug`](HyprSlug.md) does all of it without llama.cpp on the machine
at any point. See [Quantization](Quantization.md) for what each tier
costs and [Dflash2](Dflash2.md) for the draft.

### Running a brewed model in llama.cpp

A `hyperNix0x-v2` model is a Llama-shaped transformer, and `brew export`
and `brew gguf` write it as llama.cpp's `llama` architecture, so it runs
in every llama.cpp build: stock, LM Studio's, and the
[`native/ggml-hnx`](../native/ggml-hnx/README.md) patched one.

```bash
hnx brew gguf ~/.hypernix/models/HyperNix.3-mini          # f16, beside the weights
llama-cli -m ~/.hypernix/models/HyperNix.3-mini/hypernix.3-mini.f16.gguf -p "The river"
```

Before this, `--format gguf` wrote an `HNXG` file, a JSON header and raw
F32 tensors that no llama.cpp could open. What the export does now:

- **RoPE.** Brewer rotates the two halves of each head; llama.cpp rotates
  adjacent pairs. Q and K are permuted to match, the same transform
  llama.cpp's own converter applies to every Llama checkpoint.
- **The tokenizer goes inside the file.** A byte-level BPE
  `tokenizer.json` (HyperNix.3-mini's kind) becomes llama.cpp's `gpt2`
  vocabulary. A model `brew train` trained on characters uses the
  `char_vocab.json` it now saves, rebuilt as a byte-level vocabulary so
  accented and other multi-byte characters work. Before this, `brew
  train` kept its character table in memory only, so its models could
  never turn their output back into text, anywhere.
- **Sliding-window models** get a context capped at the window, where
  llama.cpp's attention is identical to Brewer's (`--context full` keeps
  the trained length and says what that costs).
- **The FFN is zero-padded to a multiple of 256.** HyperNix.3-mini's
  `d_ff` is 2203; a row that is not a whole number of k-quant blocks
  falls back to F16 in `llama-quantize` and `hyprslug`. Zero rows and
  columns add exactly nothing to the output.

Checked against llama.cpp built from source, stock and with ggml-hnx
patched in: logits matched PyTorch to within 0.0025 at every position in
f32 and f16, and tokenization matched exactly. The runner uses the same
conversion: with a llama.cpp build it serves a brewed model through it,
which gives it a KV cache, quantised weights and a GPU layer split.

**If you know at the start that the model is shipping at 4 bits or
below,** train it that way: [`qat`](LowBit.md) puts the quantisation in
the forward pass so the model is told what is going to happen to it.
Post-training quantisation to a sub-bit tier measures out about as you
would expect — IQ0.5_XXXL keeps 75 % of signs and correlates +0.40 with
the weights it came from, which is to say most of the information is gone
and the model never knew.

---

## Two things that will waste your week

**Evaluate before you ship.** [`espresso_maker`](EspressoMaker.md) runs a
model against a small prompt battery and scores it — no warmup, no
schedule, just a pull — and the [ranges](Ranges.md) label
`(prompt, response)` pairs `GOOD`/`BAD`: `new_range` with zero-dependency
heuristics, `industrial_range` with a stronger model as the judge.

Below roughly 1.5 bits per weight a model stops being a worse version of
itself and starts being a different model. The loss curve will not tell
you that, and neither will a handful of prompts you tried by hand.

(`hnx vera` is not this. It checks *HyperNix module* syntax and runs
smoke tests over the package — useful, and not a model evaluation.)

**A draft model is only worth it if its proposals are accepted.** Below
about 30 % acceptance, speculative decoding makes generation *slower* —
the base model spends forward passes checking tokens it then throws away.
`dflash2 speculate` measures the rate. Measure it.

---

## A complete run

Small model, one GPU, from raw text to a shipped GGUF.

```bash
# 1. Data
hnx gather crawl -W https://docs.example.com -Q 2 -o ./raw
python -c "
from hypernix.data.blender import PersonalBlender
blend = PersonalBlender(sources=['./raw/docs.jsonl', './extra.txt'])
open('corpus.txt', 'w').writelines(line + chr(10) for line in blend)
"

# 2. Model
hnx brew new --preset 33m --name my-model --save-dir ./models

# 3. Train
hnx brew train --name my-model --data corpus.txt --steps 4000 \
    --batch-size 8 2>&1 | tee train.log

# 4. Watch it, from another terminal
tvtoppro --log train.log --modules disk,swap

# 5. Evaluate — a short prompt battery, scored
python -c "
from hypernix.evaluation.espresso_maker import EspressoMaker
from hypernix.models.neo_oven import preheat
maker = EspressoMaker(preheat('./models/my-model'))
maker.pull(['What is 2 + 2?', 'Name three colours.'])
print(maker.mean_score())
"

# 6. Ship
hnx brew export --name my-model --format gguf --outtype f16 --out my-model.f16.gguf
hyprslug my-model.f16.gguf --multi Q8_0,Q4_K_M -o my-model.gguf
hyprslug my-model.gguf --draft dflash2 -o my-model.final.gguf
```

In Python, with the run protected:

```python
from pathlib import Path

from hypernix.system.freezer import auto_freezer, flash_freezer
from hypernix.training.apron import apron
from hypernix.training.brewer import Brewer, hypernix0x_v2_33m

with apron(seed=1234):                      # reproducible, and stays that way
    cfg = hypernix0x_v2_33m()
    brewer = Brewer(cfg, name="my-model")
    brewer.build()

    fridge = flash_freezer(base=auto_freezer())   # picks Old/New by VRAM,
    batch = fridge.suggest_batch_size(hint=8)     # then adds OOM retries

    brewer.train(
        data_path="corpus.txt",
        steps=4000,
        batch_size=batch,
        log_callback=lambda step, loss: print(f"step {step} loss={loss:.4f}"),
    )
    brewer.save()                                   # the folder, vocabulary included
    brewer.export(out_path=Path("my-model.f16.gguf"), fmt="gguf", outtype="f16")
```

`Brewer.train` runs its own loop, so `cake_pan` does not slot into it —
`CakePan` wraps a step you control:

```python
from hypernix.training.cake_pan import BakeOff, CakePan

pan = CakePan(model, optimizer, free_gb_trip=1.0)
pan.save_pristine()

for batch in loader:
    try:
        loss = pan.bake(lambda: train_step(model, batch))
    except BakeOff:
        # NaN, Inf, or a step that overran step_timeout_s. The model has
        # already been rolled back to the last pristine state; skip the
        # batch rather than spending six more hours producing NaN.
        continue
```

---

## Where to go next

| You want                        | Read                                      |
| ------------------------------- | ----------------------------------------- |
| The `hypernix.train` API        | [Training](Training.md)                   |
| Architecture details            | [Architectures](Architectures.md), [Workshop](Workshop.md) |
| Optimizer internals             | [Optimizers](Optimizers.md), [Pressure-Cooker-V6](Pressure-Cooker-V6.md) |
| Multi-GPU                       | [Frameworks](Frameworks.md)               |
| VRAM arithmetic                 | [VRAM](VRAM.md), [Pascal](Pascal.md)      |
| Quantisation tiers              | [Quantization](Quantization.md), [HyprSlug](HyprSlug.md), [LowBit](LowBit.md) |
| Multi-token prediction          | [MTP](MTP.md)                             |
| Model evaluation                | [Ranges](Ranges.md), [EspressoMaker](EspressoMaker.md) |
| Every CLI command               | [CLI](CLI.md)                             |
| A script written for you        | `hnx-scriptgen`                           |
