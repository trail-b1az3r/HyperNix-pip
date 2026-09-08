<p align="center">
  <a href="https://trail-b1az3r.github.io/HyperNix-pip/" target="_blank" rel="noopener">
    <img src="https://raw.githubusercontent.com/trail-b1az3r/HyperNix-pip/main/assets/logo-new/hypernix-lockup-light.svg" alt="hypernix logo" width="360" />
  </a>
</p>

# hypernix

[![PyPI](https://img.shields.io/pypi/v/hypernix.svg)](https://pypi.org/project/hypernix/)
[![Python](https://img.shields.io/pypi/pyversions/hypernix.svg)](https://pypi.org/project/hypernix/)
[![License](https://img.shields.io/pypi/l/hypernix.svg)](https://github.com/trail-b1az3r/hypernix-pip/blob/main/LICENSE)

**End-to-end toolkit for training ai models on modern or old devices, originaly for converting hypernix.1 into gguf, now for all around training**

## What's fixed in this update
See [Changelog.md](/wiki/Changelog.md)
for most updates

## Table of contents

- [What's fixed in this update](#whats-fixed-in-this-update)
- [What's new: 0.72.3.post2 — sub-bit models you can actually run](#whats-new-0723post2--sub-bit-models-you-can-actually-run)
- [What's new: 0.72.3 — T1 v1.0.2026.8.1.1](#whats-new-0723--t1-v102026811)
- [Package layout](#package-layout)
- [Module reference](#module-reference)
- [What's new in v0.70.5](#whats-new-in-v0705)
- [What's new in v0.70.4](#whats-new-in-v0704)
- [Install](#install)
- [Quickstart](#quickstart)
- [Python API tour](#python-api-tour)
- [CLI reference](#cli-reference)
- [Supported model families](#supported-model-families)
- [Examples](#examples)
- [Wiki / deep dives](#wiki--deep-dives)
- [How the GGUF pipeline works](#how-the-gguf-pipeline-works)
- [Platform notes](#platform-notes)
- [CI autofix](#ci-autofix)
- [Build / release](#build--release)
- [Usage & Documentation](#usage--documentation)
- [License](#license)


Cross-platform: Linux, macOS, Windows. Python 3.10 - 3.14.

## What's new: 0.72.3.post2 — sub-bit models you can actually run

**The error this release is about.** LM Studio, opening a model this
package produced:

```
llama_model_loader: failed to load model from Qwen3.8-2B-IQ0.9_L.gguf
```

That error is correct, and no header fixes it: the GGML type id at 200 is
how the loader noticed, but the missing dequantisation kernel is why it
stopped. So there are three real ways out, and
`hypernix hyprslug-headers` leads with which is which.

```bash
hypernix hyprslug-headers install                        # find what needs it
hypernix hyprslug-headers serve model.iq09.gguf          # keep the tier, talk HTTP
hypernix hyprslug-headers wrap  model.iq09.gguf -o ok.gguf   # open anywhere, bigger
hypernix hyprslug-headers stamp model.iq09.gguf          # make it self-describing
```

`serve` puts [HnxRun](wiki/HnxRun.md) behind `/v1/chat/completions`, so
LM Studio and Bionic reach a 0.9-bit model without converting it.
`wrap` re-encodes to a stock type and says, in the report, that the
result is a `Q2_K` copy of a `IQ0.9_L` model rather than the original —
verified against the reference `gguf` reader. See
[HyprSlug-Headers](wiki/HyprSlug-Headers.md).

**Five new quant types.** `IQ0.25_UXL` at a quarter of a bit exactly,
`INT1`, `FP2`, `INT4`, and `Q4M` as a spelling of `Q4_K_M` that used to
be rejected. `FP2`'s scale is searched, not fitted to the block peak —
the obvious fit scored *worse than one bit*, and a 2-bit format that
loses to a 1-bit format is not a format. Full table and the measurement
in [LowBit](wiki/LowBit.md).

**`tvtoppro`** — tvtop++'s stats under a btop++ presentation, with
themes. Braille graphs, gradient meters, titles in the box border, and
btop's own `.theme` files loading unchanged.

```bash
tvtoppro --theme gruvbox-dark
tvtoppro --theme ~/.config/btop/themes/nord.theme
```

Seven themes built in and exported to `examples/tvtoppro/`. See
[TvTopPro](wiki/TvTopPro.md).

**`hypernix-t1` fixes.** `create --host/--port` failed from a checkout
(the documented flags reached `install-t1.sh`, which had never heard of
them), and `hypernix-t1 start` started uvicorn on a different port from
the one the installer configured, because the bind address only ever went
into `start-t1.sh`. Both paths agree now.

## What's new: 0.72.3 — T1 v1.0.2026.8.1.1

**A new server can be set up without knowing anything.** It issues itself
one admin key on first start, prints it once, and that key works only
from the machine that made it and only for three days. That is enough to
point `waiter` at it and mint a real one.

```bash
./install-t1.sh                 # interactive setup, or:
hypernix-t1 start               # start / stop / restart / status / logs
hypernix-t1 test                # health, status, and a real end-to-end probe
hypernix-t1 autostart on        # a systemd user service
```

**HyperLink connects.** Three separate bugs each produced the same
symptom — the app times out and the server log is empty, because nothing
ever arrived. The server advertised port 8000 whatever port it was on;
iOS blocked tailnet addresses before sending, since Tailscale's
100.64.0.0/10 is not one of the RFC 1918 ranges ATS exempts; and when
Tailscale was missing the server said nothing about why. All three fixed,
and the app now takes a **T2S key** as well as a pairing code.

**Keys can pay for themselves.** A **T2P** key carries a billing binding —
provider references, a spend cap, a currency — so it can be issued to
someone who pays for their own usage. No card data reaches the server,
and the binding is not in the credential. A server can refuse them
outright and point at its own payment page, or require payment on a
*separate* key, so the credential that identifies a caller and the one
that spends money have different lifetimes.

**`gkey` mints every format.** `-v v1|v2|v2short`, plus `gkey version`
for what this build can issue.

```bash
gkey create -v v2 --level 5            # T2_…-5
gkey create -v v2short                 # T2S_…  for HyperLink
gkey version                           # package, T1 API, key formats
```

**The AI agent asks before it runs anything.** Tool calls are parsed out
of the model's own reply, so anything that can influence that reply — a
file it read, a web result, a page it fetched — could previously execute
shell commands with no prompt. Side-effecting tools now require consent;
`HYPERNIX_TOOL_POLICY=ask|deny|allow`, and "ask" with no terminal means
deny.

**More fits on the same card.** `hypernix.system.vram` — allocator
tuning so a long run stops fragmenting, activation checkpointing so
sequence length stops costing activation memory, an optimizer that steps
during backward so the gradients are never all held at once, and a way to
measure whether any of it worked. Nothing is applied for you and nothing
changes a default.

```bash
hypernix train run ... --gradient-checkpointing --tune-allocator
```

**Releases are gated on a live server.** After the tests, two jobs each
mint a T2 key, start a real API, chat through a fake model, drive an
iPhone simulator, then delete every key they made. Nothing publishes
until both pass.

Full detail in the [Changelog](wiki/Changelog.md).

## Package layout

Modules are grouped by what they do rather than sitting in one flat
directory:

| Directory | Modules | Contents |
|---|---|---|
| `hypernix/chat/` | 5 | Chat templating, prompt presets and multi-turn session state. |
| `hypernix/data/` | 15 | Datasets: collection, cleaning, splitting, packing and augmentation. |
| `hypernix/evaluation/` | 6 | Scoring, rubric labelling, judging and module verification. |
| `hypernix/interfaces/` | 11 | Human-facing front ends: CLIs, TUIs, GUIs and launchers. |
| `hypernix/models/` | 11 | Architectures, snapshot loading, generation and model utilities. |
| `hypernix/monitoring/` | 9 | Live dashboards, logging, telemetry and hardware sampling. |
| `hypernix/optimizers/` | 8 | The Pressure Cooker optimizer family and optimizer plumbing. |
| `hypernix/quant/` | 4 | The GGUF pipeline: convert, quantize, fetch tooling and upload. |
| `hypernix/security/` | 3 | API keys, quotas and request gating. |
| `hypernix/system/` | 15 | Environment, dependencies, hardware and housekeeping. |
| `hypernix/timing/` | 5 | Timers, alarms, cadence control and progress animation. |
| `hypernix/training/` | 14 | Training entry points, schedules and weight perturbation. |
| `hypernix/t1api/` | — | The [T1 API](wiki/T1-API.md) server: registry, routing, quota, billing, audit, rate limiting, mTLS, deployment. |
| `hypernix/t1sdk/` | — | The T1 API client SDK — typed, stdlib-only, no server extra needed. |
| `hypernix/waiter/` | — | [`waiter`](wiki/Waiter-TUI.md), the official T1 API TUI/CLI. |

**Every module keeps its old import path.** `hypernix.timer` and
`hypernix.timing.timer` return the same module object, so nothing that
imported a module before the move needs to change:

```python
from hypernix.timer import KitchenTimer          # always worked, still works
from hypernix.timing.timer import KitchenTimer   # where the file actually is
import hypernix; hypernix.timer is hypernix.timing.timer   # True
```

`hypernix.MODULE_CATEGORIES` (and its reverse, `hypernix.CATEGORY_OF`) is the
one place the layout is written down — the lazy loader, the alias finder, the
`hnx wiki` browser and the `scripts/autofix-*` tooling all read it, so moving a
module between categories is a one-line change.

## Module reference

Click a category below to expand it.

<details>
<summary><strong>Models & Training</strong> &nbsp;(12 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.download` | Pull snapshots from the Hub (short-name resolution, gated repos, offline cache). |
| `hypernix.train` | `HyperNixConfig`, `HyperNixModel`, `init_from_scratch`, `expand_checkpoint`, `train`. Non-HyperNix archs route through `AutoModelForCausalLM`. |
| `hypernix.brewer` | `hyperNix0x-v2` architecture preset family — `Brewer(config).build()` for a from-scratch `BrewerModel`. GPU-oriented presets `33m` / `micro` / `small` / `medium` / `large`[...]|
| `hypernix.instant_pot` | `brew(recipe)` — one-shot end-to-end pipeline. Also available as `hypernix brew recipe.json`. |
| `hypernix.coffee_maker` | 3 tiers (drip / french-press / percolator) + `cold_brew` type for long checkpointed runs. |
| `hypernix.deep_fryer` | 2-tier model-weight perturbation: `LightFry` (regulariser) / `HeavyFry` (severe, for bad-model negatives). In-place, reversible via snapshot. |
| `hypernix.abbicus` | Automatic token regulation and curriculum tuning. **`Abbicus`** (linear) dynamically modifies max sequence length based on model size (0.5B-72B), global step, and dataset t[...]
| `hypernix.compute_framework` | Hardware-agnostic multi-device training. Abstracts CUDA, MPS, CPU, TPU backends with automatic DDP/ZeRO wrapping. `ComputeFramework` handles PyTorch DDP initializ[...]
| `hypernix.workshop` | Model frameworks and TTS/ASR pipelines. `WorkshopFramework` base class with `FrameworkConfig` for TTS, ASR, LLM, Vision models. Pre-built templates for the ray0rf1re/nano-[...]
| `hypernix.whisk` | Checkpoint averaging — `swa_average` (uniform mean), `ema` (exponential), `geometric_mean`. Accepts state dicts or paths to `.pt` / `.safetensors`. `whisk_to_snapshot` writ[...]
| `hypernix.recipe_book` | Named-config registry. `RecipeBook` with `add` / `get` / `save` / `load` / `cook(name, **overrides)`. `cook` dispatches by `kind` (`instant_pot` / `cold_brew` / `espresso`) [...]
| `hypernix.mtp` | *(v0.70.5)* Multi-Token Prediction — predict multiple future tokens for 1.5-3x training efficiency + speculative decoding. `MTPConfig`, `MTPHead`, `MTPTrainer`. |

</details>

<details>
<summary><strong>Optimizers</strong> &nbsp;(3 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.pressure_cooker` | Custom AdamW optimizer in 5 tiers: base `PressureCooker` + CPU (`StovetopCooker`, `ElectricCooker`) + GPU (`InductionCooker`, `ProCooker`) + `universal_cooker` sele[...]
| `hypernix.pressure_cooker_v3` | ZeRO-optimized V3 optimizer with FP8 support. `QuantDtype` enum (FP8/FP16/FP32/FP64/Q8/Q6/Q5_5/Q4M) and `QuantConfig` dataclass. `PressureCookerV3` / `PressureCo[...]
| `hypernix.pressure_cooker_v5` | *(v0.70.5 / v0.70.6)* ORCP optimizer family with int8-quantized momentum, factored curvature, QAT (Q4/Q5/Q6/Q8), Multi-Token Prediction, and EMA shadowing. `Pres[...]

</details>

<details>
<summary><strong>Memory / VRAM</strong> &nbsp;(4 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.old_fridge` | Memory housekeeping: `freeze`, `unfreeze`, `parameter_stats`, `offload_to_cpu`, `chill_cache`. |
| `hypernix.freezer` | VRAM manager: `OldFreezer` (8-10 GB, conservative batches, bf16/fp16), `NewFreezer` (11 GB+, fp32-preferred), `FlashFreezer` (OOM-safe retry wrapper around either). Pascal [...]
| `hypernix.cake_pan` | Hybrid CPU + GPU training guard with NaN/Inf detection, wall-time watchdog, memory-pressure offload, and pristine-state rollback via `BakeOff`. |
| `hypernix.vram` | *(v0.72.3)* **VRAM optimizations** — `configure_allocator()` (`expandable_segments`, so a long run stops fragmenting; must run before the first CUDA allocation, which is why[...]
| `hypernix.stml` | *(v0.70.4)* **Short Term Memory Loss** — two tools. `calculate_vram_context(vram_gb, params, batch_size, precision)` estimates the max safe trained context given your hardwa[...]

</details>

<details>
<summary><strong>Data Pipeline</strong> &nbsp;(9 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.pans` | 5-tier data preprocessing: `FryingPan` → `SaucePan` → `Skillet` → `GrillPan` → `Wok`. Pair with `sink.Sink.pour` to write the output to disk. |
| `hypernix.blender` | 4-tier multi-source mixing: `HandBlender` / `PersonalBlender` / `CountertopBlender` / `HighPowerBlender`. |
| `hypernix.toaster` | 4-tier per-line formatting: `TwoSliceToaster` / `FourSliceToaster` / `ConveyorToaster` / `ToasterOven`. |
| `hypernix.food_processor` | 4-tier bulk chunking: `ChopBlade` / `SliceBlade` / `ShredBlade` / `PureeBlade`. |
| `hypernix.salt_shaker` | 3-tier gentle data augmentation: `FromTheBag` / `HandCrusher` / `PoshSaltDish`. |
| `hypernix.pepper_shaker` | 3-tier sharp perturbations: `SmallShaker` (MLM-style mask) / `Dish` (typos) / `TallHandmade` (negation). |
| `hypernix.qa` | *(v0.70.4)* **`QAProcessor`** — turns structured datasets (JSONL, `list[dict]`, plain text) into causal LM training strings. Two modes: `question_answer` (`Question: {q}\nAnsw[...]
| `hypernix.cutting_board` | Train / val / test splitting. `CuttingBoard` (deterministic random) + `StratifiedBoard` (preserves class distribution on labelled records). Renormalises ratios that d[...]
| `hypernix.lunchbox` | Consistent-schema dataset packager. `Lunchbox.for_eval()` pre-loads the recommended eval-results columns; `pack(path)` / `push_to_hub(repo_id)` routes through `datasets.Da[...]

</details>

<details>
<summary><strong>Inference & Chat</strong> &nbsp;(7 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.old_oven` | `CodeOven` — ready-to-use wrapper around a snapshot: `.complete()`, `.chat()`, `.fill()`, `.save_pt()`. `new_oven()` spins a fresh one from the [ARCH_PRESETS](#arch_pres[...]
| `hypernix.microwave` | 5-tier throwaway inference: `defrost` → `low_zap` → `zap` → `high_zap` → `chat_zap`, plus `reheat` for continuing a prior output. |
| `hypernix.cookbook` | Chat-template registry. Built-in templates for `chatml` / `hyper-nix.2` / `llama3` / `llama2` / `alpaca` / `vicuna` / `plain`. `for_model(repo_id)` picks the right one aut[...]
| `hypernix.countertop` | Multi-turn chat session. `Countertop(oven, system=…)` with `say(user)` / `reset()` / `save(path)` / `load(path)`. Auto-trims long histories; optional `bell=` for token[...]
| `hypernix.menu` | Named system-prompt registry: `default` / `concise` / `code-helper` / `judge` / `creative` / `chef` / `hyper-nix`. Pair with `countertop(oven, persona="…")` to pick a system prom[...]
| `hypernix.bell` | Streaming-token + done-notification primitive. `Bell.iter_chat(oven, messages)` yields tokens; `stream_chat` collects and fires callbacks. `stdout_bell()` / `file_bell(path)` ship [...]
| `hypernix.flour` | Chat-quality logits processor — repetition penalty, frequency / presence penalty, no-repeat n-gram, bad-word suppression, role-leak suppression (cuts hallucinated `user:`-style [...]

</details>

<details>
<summary><strong>Monitoring & CLI</strong> &nbsp;(5 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.smoke_alarm` | Training-step planner & monitor. `RadsAlarm` (constants, lightest), `GasAlarm` (CPU/GPU presets), `ModernAlarm` (warmup-measured), `AutoAlarm` (selector). Plus `storage[...]
| `hypernix.table` | Dead-simple tabular viewer: `from_training_log`, `from_judge_corpus`, `filter`, `select`, `show`. |
| `hypernix.tvtop` | Backwards-compatibility shim — all functionality moved to `hypernix.tv`. Re-exports everything so `import hypernix.tvtop` continues to work. Console script `tvtop` now laun[...]
| `hypernix.wiki_cli` | *(v0.70.5)* `hnx` / `hypenix` command — auto-generating wiki from source docstrings. `hnx`, `hnx -q`, `hnx -b`. |
| `hypernix.vera` | *(v0.70.5)* Module verification — syntax, docstrings, types, smoke tests. `hnx vera <file>` / `hnx vera --all`. |

</details>

<details>
<summary><strong>Datasets & Judging</strong> &nbsp;(6 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.mediocre_fridge` | Judge-training dataset generation — `synthesize_judge_corpus`, `collect_responses_from`. |
| `hypernix.new_fridge` | Training-curve graphing — `parse_training_log`, `plot_loss_curve`, `plot_score_distribution`. Matplotlib installed lazily. |
| `hypernix.new_range` / `old_range` / `industrial_range` | Labeling rubrics for `mediocre_fridge.collect_responses_from`: `new_range` is a zero-dep first-fail rubric, `old_range` is a scored rub[...]
| `hypernix.espresso_maker` | 4-tier evaluation: `Ristretto` / `SingleShot` / `DoubleShot` / `Lungo` — run a prompt battery, score, return shots. |
| `hypernix.smoker` | 4-tier training quality: `UseableSmoker` / `GoodSmoker` / `CommercialSmoker` / `HighQualitySmoker`. |
| `hypernix.scavenger` | *(v0.70.5)* HuggingFace dataset discovery engine. Keyword search, storage budgets, quality filtering, relevance scoring. `ScavengerCriteria` + `Scavenger.hunt()`. |

</details>

<details>
<summary><strong>Quantize & Export</strong> &nbsp;(3 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.convert` | Safetensors → GGUF at fp32/fp16. Architecture-agnostic tensor naming. |
| `hypernix.quantize` | `llama-quantize` driver. v0.51.3 ships a 30-type `QUANT_CATALOG` (`QuantSpec` dataclass per type with bits-per-weight, category, recommendation) covering floats (`F32` / `F16` [...]
| `hypernix.upload` | Push the produced artifacts back to a HuggingFace repo. |

</details>

<details>
<summary><strong>Utilities</strong> &nbsp;(3 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.sink` | Append-only file sink with optional rotation + dedupe. |
| `hypernix.apron` | RNG-state guard. `apron(seed=…)` context manager snapshots Python `random`, NumPy (if installed), torch CPU and every CUDA device's RNG, optionally seeds all of them, and r[...]
| `hypernix.torch_compat` | Portability shim (RMSNorm + SDPA) for running on old Intel Macs with torch 1.13. See [`wiki/macOS-legacy.md`](wiki/macOS-legacy.md). |

</details>

---

## What's new in v0.70.5

Eleven major additions:

- **`hnx` / `hypenix` Wiki CLI** — Auto-generating documentation browser. `hnx` shows all modules; `hnx <module>` shows docs; `hnx -q <module>` streams quick mode; `hnx -b` opens in browser. Do[...]