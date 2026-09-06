<p align="center">
  <img src="https://raw.githubusercontent.com/trail-b1az3r/hypernix-pip/2d5eb37/assets/logo1.png" alt="hypernix logo" width="240" />
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
| `hypernix.brewer` | `hyperNix0x-v2` architecture preset family — `Brewer(config).build()` for a from-scratch `BrewerModel`. GPU-oriented presets `33m` / `micro` / `small` / `medium` / `large` (33.6M-3.5B params), plus `cpu-nano` / `cpu-tiny` / `cpu-small` (2.1M/9.2M/26.5M params) sized for CPU-only training and inference. `custom_arch(**kwargs)` for a fully bespoke config. Also available as `hypernix brew new --preset <name>`. |
| `hypernix.instant_pot` | `brew(recipe)` — one-shot end-to-end pipeline. Also available as `hypernix brew recipe.json`. |
| `hypernix.coffee_maker` | 3 tiers (drip / french-press / percolator) + `cold_brew` type for long checkpointed runs. |
| `hypernix.deep_fryer` | 2-tier model-weight perturbation: `LightFry` (regulariser) / `HeavyFry` (severe, for bad-model negatives). In-place, reversible via snapshot. |
| `hypernix.abbicus` | Automatic token regulation and curriculum tuning. **`Abbicus`** (linear) dynamically modifies max sequence length based on model size (0.5B-72B), global step, and dataset type. `TurboAbbicus` (exponential) adds sine-wave oscillation and a hard VRAM safeguard. |
| `hypernix.compute_framework` | Hardware-agnostic multi-device training. Abstracts CUDA, MPS, CPU, TPU backends with automatic DDP/ZeRO wrapping. `ComputeFramework` handles PyTorch DDP initialization, device placement, and gradient sync without manual `torch.distributed` boilerplate. |
| `hypernix.workshop` | Model frameworks and TTS/ASR pipelines. `WorkshopFramework` base class with `FrameworkConfig` for TTS, ASR, LLM, Vision models. Pre-built templates for the ray0rf1re/nano-nano collection plus 30+ third-party architectures. |
| `hypernix.whisk` | Checkpoint averaging — `swa_average` (uniform mean), `ema` (exponential), `geometric_mean`. Accepts state dicts or paths to `.pt` / `.safetensors`. `whisk_to_snapshot` writes the merged weights back out as a loadable HyperNix snapshot. |
| `hypernix.recipe_book` | Named-config registry. `RecipeBook` with `add` / `get` / `save` / `load` / `cook(name, **overrides)`. `cook` dispatches by `kind` (`instant_pot` / `cold_brew` / `espresso`) so a saved recipe runs the matching pipeline directly. |
| `hypernix.mtp` | *(v0.70.5)* Multi-Token Prediction — predict multiple future tokens for 1.5-3x training efficiency + speculative decoding. `MTPConfig`, `MTPHead`, `MTPTrainer`. |

</details>

<details>
<summary><strong>Optimizers</strong> &nbsp;(3 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.pressure_cooker` | Custom AdamW optimizer in 5 tiers: base `PressureCooker` + CPU (`StovetopCooker`, `ElectricCooker`) + GPU (`InductionCooker`, `ProCooker`) + `universal_cooker` selector that picks a tier automatically from the detected device. |
| `hypernix.pressure_cooker_v3` | ZeRO-optimized V3 optimizer with FP8 support. `QuantDtype` enum (FP8/FP16/FP32/FP64/Q8/Q6/Q5_5/Q4M) and `QuantConfig` dataclass. `PressureCookerV3` / `PressureCookerV3Plus` classes with ZeRO-1/2 sharding, plus `StovetopV3Cooker` / `StovetopV3CookerPlus` CPU-tuned variants. |
| `hypernix.pressure_cooker_v5` | *(v0.70.5 / v0.70.6)* ORCP optimizer family with int8-quantized momentum, factored curvature, QAT (Q4/Q5/Q6/Q8), Multi-Token Prediction, and EMA shadowing. `PressureCookerV5` + `PressureCookerV5Plus`, plus the ground-up 3D-ORCP `PressureCookerV5S`. Pascal-safe variants: `Agedcookerv5`, `ULTRAagedcookerv5`, `Agedcookerv5s`. See the [efficiency paper](pressure_cooker_v5_v5s_paper.md). |

</details>

<details>
<summary><strong>Memory / VRAM</strong> &nbsp;(4 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.old_fridge` | Memory housekeeping: `freeze`, `unfreeze`, `parameter_stats`, `offload_to_cpu`, `chill_cache`. |
| `hypernix.freezer` | VRAM manager: `OldFreezer` (8-10 GB, conservative batches, bf16/fp16), `NewFreezer` (11 GB+, fp32-preferred), `FlashFreezer` (OOM-safe retry wrapper around either). Pascal (sm_61 / CUDA 6.1) helpers + 60 CPU presets (Intel i5/i7/i9 7th-14th gen, Core Ultra Series 1/2, AMD Ryzen 5000/7000/9000 series) via `auto_freezer()`. |
| `hypernix.cake_pan` | Hybrid CPU + GPU training guard with NaN/Inf detection, wall-time watchdog, memory-pressure offload, and pristine-state rollback via `BakeOff`. |
| `hypernix.vram` | *(v0.72.3)* **VRAM optimizations** — `configure_allocator()` (`expandable_segments`, so a long run stops fragmenting; must run before the first CUDA allocation, which is why importing the module does not import torch), `checkpoint_blocks(model, every=N)` (recompute activations instead of storing them), `fuse_optimizer_into_backward()` (step and free each gradient as it is produced, so they are never all held at once), `offload_optimizer_state(opt)` (a context manager for the duration of an eval pass), `measure_peak()` and `recommend()`. Each opt-in, each reversible, each refusing rather than silently doing nothing. See [VRAM](wiki/VRAM.md). |
| `hypernix.stml` | *(v0.70.4)* **Short Term Memory Loss** — two tools. `calculate_vram_context(vram_gb, params, batch_size, precision)` estimates the max safe trained context given your hardware. The `STML` context manager folds long sequences into batch segments to keep the *untrained* context length bounded during training. |

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
| `hypernix.qa` | *(v0.70.4)* **`QAProcessor`** — turns structured datasets (JSONL, `list[dict]`, plain text) into causal LM training strings. Two modes: `question_answer` (`Question: {q}\nAnswer: {a}`) and plain completion, with optional integrated `salt_shaker` / `pepper_shaker` seasoning. |
| `hypernix.cutting_board` | Train / val / test splitting. `CuttingBoard` (deterministic random) + `StratifiedBoard` (preserves class distribution on labelled records). Renormalises ratios that don't sum to 1; writes per-split files or returns in-memory lists. |
| `hypernix.lunchbox` | Consistent-schema dataset packager. `Lunchbox.for_eval()` pre-loads the recommended eval-results columns; `pack(path)` / `push_to_hub(repo_id)` routes through `datasets.Dataset` so column-schema mismatches fail fast instead of at upload time. |

</details>

<details>
<summary><strong>Inference & Chat</strong> &nbsp;(7 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.old_oven` | `CodeOven` — ready-to-use wrapper around a snapshot: `.complete()`, `.chat()`, `.fill()`, `.save_pt()`. `new_oven()` spins a fresh one from the [ARCH_PRESETS](#arch_presets-seeds-for-new_oven) seed list instead of downloading a snapshot. |
| `hypernix.microwave` | 5-tier throwaway inference: `defrost` → `low_zap` → `zap` → `high_zap` → `chat_zap`, plus `reheat` for continuing a prior output. |
| `hypernix.cookbook` | Chat-template registry. Built-in templates for `chatml` / `hyper-nix.2` / `llama3` / `llama2` / `alpaca` / `vicuna` / `plain`. `for_model(repo_id)` picks the right one automatically from the repo's config; wired into `old_oven` and `countertop` by default. |
| `hypernix.countertop` | Multi-turn chat session. `Countertop(oven, system=…)` with `say(user)` / `reset()` / `save(path)` / `load(path)`. Auto-trims long histories; optional `bell=` for token-by-token streaming, `flour=` for output cleanup, `t1_key=` for HNX1/T1-backed remote models. |
| `hypernix.menu` | Named system-prompt registry: `default` / `concise` / `code-helper` / `judge` / `creative` / `chef` / `hyper-nix`. Pair with `countertop(oven, persona="…")` to pick a system prompt by name instead of writing one out each time. |
| `hypernix.bell` | Streaming-token + done-notification primitive. `Bell.iter_chat(oven, messages)` yields tokens; `stream_chat` collects and fires callbacks. `stdout_bell()` / `file_bell(path)` ship as ready-made done-callbacks; `silent_bell()` disables notifications. |
| `hypernix.flour` | Chat-quality logits processor — repetition penalty, frequency / presence penalty, no-repeat n-gram, bad-word suppression, role-leak suppression (cuts hallucinated `user:`-style follow-on turns a base-model-flavoured checkpoint sometimes emits). |

</details>

<details>
<summary><strong>Monitoring & CLI</strong> &nbsp;(5 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.smoke_alarm` | Training-step planner & monitor. `RadsAlarm` (constants, lightest), `GasAlarm` (CPU/GPU presets), `ModernAlarm` (warmup-measured), `AutoAlarm` (selector). Plus `storage_warning()` for disk-space checks before a long run. |
| `hypernix.table` | Dead-simple tabular viewer: `from_training_log`, `from_judge_corpus`, `filter`, `select`, `show`. |
| `hypernix.tvtop` | Backwards-compatibility shim — all functionality moved to `hypernix.tv`. Re-exports everything so `import hypernix.tvtop` continues to work. Console script `tvtop` now launches the `tvtop_plus_plus` dashboard by default; use `tvtop-old` for the classic view. |
| `hypernix.wiki_cli` | *(v0.70.5)* `hnx` / `hypenix` command — auto-generating wiki from source docstrings. `hnx`, `hnx -q`, `hnx -b`. |
| `hypernix.vera` | *(v0.70.5)* Module verification — syntax, docstrings, types, smoke tests. `hnx vera <file>` / `hnx vera --all`. |

</details>

<details>
<summary><strong>Datasets & Judging</strong> &nbsp;(6 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.mediocre_fridge` | Judge-training dataset generation — `synthesize_judge_corpus`, `collect_responses_from`. |
| `hypernix.new_fridge` | Training-curve graphing — `parse_training_log`, `plot_loss_curve`, `plot_score_distribution`. Matplotlib installed lazily. |
| `hypernix.new_range` / `old_range` / `industrial_range` | Labeling rubrics for `mediocre_fridge.collect_responses_from`: `new_range` is a zero-dep first-fail rubric, `old_range` is a scored rubric with per-rule weights and explainable [0, 1] scores, and `industrial_range` uses any `CodeOven`-compatible model as an LLM judge (including pairwise comparison for preference pairs). |
| `hypernix.espresso_maker` | 4-tier evaluation: `Ristretto` / `SingleShot` / `DoubleShot` / `Lungo` — run a prompt battery, score, return shots. |
| `hypernix.smoker` | 4-tier training quality: `UseableSmoker` / `GoodSmoker` / `CommercialSmoker` / `HighQualitySmoker`. |
| `hypernix.scavenger` | *(v0.70.5)* HuggingFace dataset discovery engine. Keyword search, storage budgets, quality filtering, relevance scoring. `ScavengerCriteria` + `Scavenger.hunt()`. |

</details>

<details>
<summary><strong>Quantize & Export</strong> &nbsp;(3 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.convert` | Safetensors → GGUF at fp32/fp16. Architecture-agnostic tensor naming. |
| `hypernix.quantize` | `llama-quantize` driver. v0.51.3 ships a 30-type `QUANT_CATALOG` (`QuantSpec` dataclass per type with bits-per-weight, category, recommendation) covering floats (`F32` / `F16` / `BF16`), legacy k-quants (`Q4_0`…`Q5_1`), K-quants (`Q2_K`…`Q6_K`), and importance-matrix quants (`IQ1_S`…`IQ4_XS`); see the alias table below. |
| `hypernix.upload` | Push the produced artifacts back to a HuggingFace repo. |

</details>

<details>
<summary><strong>Utilities</strong> &nbsp;(3 modules)</summary>

| Subsystem | What it does |
|---|---|
| `hypernix.sink` | Append-only file sink with optional rotation + dedupe. |
| `hypernix.apron` | RNG-state guard. `apron(seed=…)` context manager snapshots Python `random`, NumPy (if installed), torch CPU and every CUDA device's RNG, optionally seeds all of them, and restores the original state on exit. |
| `hypernix.torch_compat` | Portability shim (RMSNorm + SDPA) for running on old Intel Macs with torch 1.13. See [`wiki/macOS-legacy.md`](wiki/macOS-legacy.md). |

</details>

---

## What's new in v0.70.5

Eleven major additions:

- **`hnx` / `hypenix` Wiki CLI** — Auto-generating documentation browser. `hnx` shows all modules; `hnx <module>` shows docs; `hnx -q <module>` streams quick mode; `hnx -b` opens in browser. Docs regenerate from source docstrings, so they can't drift out of sync with the code.
- **`hnx vera`** — Module verification: syntax check, docstring coverage, type annotations, smoke test. `hnx vera <file>` or `hnx vera --all`.
- **`pressure_cooker_v5`** — ORCP optimizer family with int8-quantized momentum (~75% smaller than fp32, ~87% smaller total optimizer state than AdamW -- see the [efficiency paper](pressure_cooker_v5_v5s_paper.md)), QAT (Q4/Q5/Q6/Q8), Multi-Token Prediction, EMA shadowing, and the ground-up 3D-ORCP `PressureCookerV5S` variant (v0.70.6).
- **`mtp`** — Multi-Token Prediction for 1.5-3x training efficiency. Sequential/independent modes, shared/independent heads, native workshop integration.
- **`scavenger`** — HuggingFace dataset discovery with keyword search, storage budgets, quality filtering (likes/downloads/age), and relevance scoring.
- **Freezer QAT support** — `suggest_qat_batch_size()`, `prepare_for_qat()`, per-bit-width VRAM multiplier profiles.
- **Workshop native MTP** — `attach_mtp_head()` and `compute_mtp_loss()` built into WorkshopFramework.
- **tvtop++ fixes** — Eliminated border flicker (layout built once), added `_block_history_bar` re-export, implemented `small_mode`, fixed self-process filtering.
- **New wiki pages** — [Pressure-Cooker-V5](wiki/Pressure-Cooker-V5.md), [MTP](wiki/MTP.md), [Scavenger](wiki/Scavenger.md)
- **Kitchen.md updated** — Added scavenger, MTP, and QAT sections
- **Training benefits chart** — See below

### Training Benefits vs Complexity

![HyperNix Training Features](docs/public/training-benefits-chart.png)

> **Key insight**: MTP + Speculative Decoding offer the highest benefit-to-cost ratio. Int8-quantized momentum cuts the momentum buffer's own memory by 75% versus fp32, and PressureCookerV5/V5S's factored curvature keeps the rest of the optimizer state small too -- measured optimizer-state memory lands around 12-13% of AdamW's (see the [efficiency paper](pressure_cooker_v5_v5s_paper.md) for the exact numbers and methodology). The trade-offs -- including step-time overhead on some hardware -- are real and are covered in the paper rather than summarized as a single percentage here.

## What's new in v0.70.4

Seven additions in the 0.70.4 series:

- **`qa`** — `QAProcessor` formats Q&A datasets into causal LM training strings with optional salt/pepper seasoning
- **`stml`** — Short Term Memory Loss: `STML` context manager (segment folding, untrained hard cap) + `calculate_vram_context` VRAM calculator with CLI
- **`TurboAbbicus`** — exponential curriculum regulator with configurable hard cap, sine-wave oscillation (CPU-adjusted, never GPU), and VRAM safeguard
- **`tvtop++` fixes** — layout tree bug (border shifting on refresh), colors matching original tvtop (CPU=green, RAM=magenta, GPU=red), dynamic console resizing, dynamic graph/log widths
- **`hypernix stml`** CLI subcommand — VRAM context calculator from the shell
- **`hypernix train run`** new flags — `--use-abbicus`, `--use-turbo-abbicus`, `--use-stml`, `--untrained-max-context`, `--segment-length`
- **`CodeOven.train()`** new kwargs — `use_turbo_abbicus`, `use_stml`, `untrained_max_context`, `segment_length`

### Earlier: v0.70.0

Five new modules + major optimizer rewrites:

- **`abbicus`** — Automatic token regulation and curriculum tuning for model sizes 0.5B–72B
- **`compute_framework`** — Hardware-agnostic multi-device training with auto DDP/ZeRO wrapping (CUDA/MPS/CPU/TPU)
- **`pressure_cooker` V2** — Quantization-aware training with fp16/bf16/fp64 mixed-precision, QAT hooks for Q8/Q6/Q5.5/Q4M, plus 10 upgrades (mixed-precision autodetect, QAT hooks, gradient-checkpointing integration, adaptive per-layer gradient clipping, EMA weight shadowing, DDP/FSDP-aware distributed training, dynamic loss scaling with overflow backoff, parameter freeze/unfreeze callbacks, an LR finder, and metrics streaming to tvtop)
- **`pressure_cooker_v3`** — ZeRO-1/2 optimizations, FP8 support, `QuantDtype` enum + `QuantConfig` dataclass
- **`workshop`** — Model frameworks for TTS/ASR/LLM/Vision with pre-built templates, nano-nano collection support, 30+ architectures (LiquidAI LFM2.5, MiniCPM5, Gemma 4, Qwen3.5, Phi-4, DeepSeek-V2.x, and others)
- **`tvtop`** — Now launches the premium `tvtop_plus_plus` dashboard by default; use `tvtop-old` for the classic view

## Install

From PyPI:

```bash
pip install "hypernix[llama-cpp]"     # + bundled llama-cpp-python
pip install "hypernix[train]"         # + transformers, accelerate
pip install hypernix                  # core only
```

Setting up the **T1 API** server specifically? `./install-t1.sh` is a
guided installer — it asks what kind of deployment this is (bind
address, key policy, allowlist, rate limits, cost accounting, models,
HyperLink, the `waiter` manager TUI) and writes a matching
configuration, an admin key, and a start script. `--dry-run` shows what
it would do without writing anything. See
[T1-API.md](wiki/T1-API.md#the-installer).

Need a specific torch build? Install torch **first**; pip will reuse
it rather than replace it:

```bash
# CUDA 11.8 — old drivers, Pascal GPUs (GTX 1080 et al.)
pip install --index-url https://download.pytorch.org/whl/cu118 torch
pip install hypernix

# CUDA 12.x — modern default
pip install --index-url https://download.pytorch.org/whl/cu124 torch
pip install hypernix

# CPU-only
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install hypernix

# Old Intel Mac / torch 1.13 — the compat shim takes over.
pip install --index-url https://download.pytorch.org/whl/cpu 'torch==1.13.1'
pip install 'hypernix[legacy-torch]'
```

### `hypernix: command not found`

The console scripts land in your interpreter's scripts directory, which on
a lot of systems isn't on `PATH` — `pip install --user` puts them in
`~/.local/bin`, and Debian/Ubuntu only add that at login *if it already
existed*. HyperNix fixes this itself the first time you run it, printing
what it changed. To do it explicitly:

```bash
python -m hypernix path            # what would change (writes nothing)
python -m hypernix path --apply    # write the block into your shell profile
python -m hypernix path --undo     # take it back out
```

It writes one marked, reversible block into the startup file your shell
actually reads, and refuses to do anything inside a virtualenv or conda
env — that directory belongs to the environment and is only meant to be on
`PATH` while it's activated. Set `HYPERNIX_NO_PATH_SETUP=1` to turn the
automatic version off entirely.

The main `install_requires` is `torch>=1.13,<3` — 2.7+ is the
recommended version (native `nn.RMSNorm`, fused SDPA), but 1.13+
works via `hypernix.torch_compat`. See
[`wiki/macOS-legacy.md`](wiki/macOS-legacy.md) for the full story.

Sanity-check the environment:

```bash
hypernix doctor          # report
hypernix doctor --fix    # install missing runtime deps
```

Automatic dependency management can be disabled with
`HYPERNIX_AUTO_INSTALL=0`.

## Quickstart

### Chat with any supported model

```bash
hypernix chat --repo-id nix2.5 --message "hello"
hypernix chat --repo-id qwen3.5-4b --message "explain rotary embeddings"
hypernix chat --repo-id gemma-4-e4b --message "write a haiku"
```

Short names resolve via `KNOWN_MODELS`; see
[Supported model families](#supported-model-families).

### Convert a snapshot to GGUF

```bash
# Default: fp32 + fp16
hypernix --repo-id ray0rf1re/hyper-nix.1 --output-dir ./out

# Opt in to k-quants (needs llama-quantize)
hypernix --repo-id ray0rf1re/hyper-nix.1 --output-dir ./out \
    --quants fp32 fp16 q8_0 q6_k q4_k_m
```

### Train HyperNix 1.5 (~92.1 M params) on a GTX 1080

```bash
python examples/train_hypernix_1_5_gtx1080.py \
    --dataset corpus.txt \
    --tokenizer-source ./hyper-nix-v1 \
    --out-dir ./hypernix-1.5 \
    --steps 2000 --batch-size 1 --context-length 1024
```

Auto-detects compute capability 6.x, forces fp16 (Pascal has no native
bf16), disables TF32 / SDPA / `torch.compile`, and wraps the training
loop in a `FlashFreezer` so OOMs pause-and-halve rather than crash. See
[`wiki/Pascal.md`](wiki/Pascal.md) for the full Pascal playbook.

### Build a HyperNix 0.1.5 evaluator

```bash
python examples/train_hypernix_0_1_5_evaluator.py --out-dir ./eval
```

Synthesizes a judge-training corpus with `mediocre_fridge`, freezes
embeddings with `old_fridge`, trains via `oven.train`, reloads with the
other oven, plots the loss curve with `new_fridge`. Self-contained
smoke test for every subsystem.

## Python API tour

```python
import hypernix
from hypernix import freezer, old_oven, old_fridge, mediocre_fridge, new_fridge

# 1) Auto-pick a VRAM strategy.  On a GTX 1080 this returns OldFreezer(fp16);
#    on a 3090 it returns NewFreezer(fp32 / bf16 on Ampere).
fz = freezer.flash_freezer(base=freezer.auto_freezer(), slow=True)

# 2) Preheat an oven from a short name (downloads on first call).
oven = old_oven.preheat(repo_id="nix2.5", device="cuda", dtype="float16")

# 3) Memory hygiene.
old_fridge.freeze(oven.model, patterns=("embed_tokens",))
print(old_fridge.parameter_stats(oven.model))

# 4) Training data.
dataset = mediocre_fridge.synthesize_judge_corpus(n=1024, out_path="judge.txt")

# 5) Train inside a FlashFreezer so OOMs don't blow up the run.
fz.guard(lambda: oven.train(dataset, "./trained", steps=500, batch_size=1))

# 6) Graph.
import pathlib
log = pathlib.Path("./trained/train.log").read_text()
new_fridge.plot_loss_curve(new_fridge.parse_training_log(log), "loss.png")
```

## CLI reference

```
hypernix <subcommand> [options]

  all                   download -> convert -> [quantize]   (default)
  download              fetch a HuggingFace snapshot
  convert               produce fp32 / fp16 GGUF from a snapshot
  quantize              run llama-quantize on an fp16 / fp32 GGUF
  verify                read-validate a GGUF and print headers
  info                  package + optional GGUF header summary
  upload                push files to a HuggingFace repo
  doctor                environment diagnostic  (pass --fix to install deps)
  path                  put the console scripts on your PATH  (--apply / --undo)
  fetch-llama-quantize  pre-seed the llama-quantize cache
  train init            create a fresh HyperNix snapshot
  train expand          warm-start a bigger model from a smaller one
  train run             minimal causal-LM training loop
  generate              sample text from a local snapshot
  oven                  code-generation wrapper (preheat + complete / fill)
  chat                  interactive chat REPL against any supported model
  hyped+ / hyped-pro    Node.js TUI agent CLI w/ real cloud+local model dispatch, /gui desktop mode
                        (/t1api routes through a local or remote HyperNix T1 API server)
  stml                  VRAM trained context length calculator
```

`train run` accepts curriculum / context management flags:

```bash
hypernix train run --model-dir ./snap --dataset data.txt --out-dir ./out \
    --use-turbo-abbicus \        # exponential curriculum (--use-abbicus for linear)
    --use-stml \                 # fold long sequences into batch segments
    --untrained-max-context 16384 \
    --segment-length 512
```

Quant aliases accepted by `--quants` and `hypernix quantize` (v0.51.3
ships 49 aliases mapping to 30 distinct quant types — the table below
shows the headline subset; `hypernix.quant_list_types()` returns the
full list at runtime, and `hypernix.QUANT_CATALOG[name]` gives you the
full `QuantSpec` for any one):

| Alias | llama.cpp enum | bpw | Recommended? |
|---|---|---|---|
| `fp32`, `f32` | F32 | 32.0 | reference |
| `fp16`, `f16` | F16 | 16.0 | ✓ baseline |
| `bf16` | BF16 | 16.0 |  |
| `q4_0`, `q4_1`, `q5_0`, `q5_1` | Q4_0 / Q4_1 / Q5_0 / Q5_1 | 4.5 – 6.0 | legacy |
| `q8`, `q8_0` | Q8_0 | 8.5 | ✓ near-lossless |
| `q2_k`, `q2_k_s`, `q3_k_s`, `q3_k_m`, `q3_k_l` | Q2_K … Q3_K_L | 2.5 – 4.0 |  |
| `q4_k_s`, `q4km`, `q4_k_m` | Q4_K_S, Q4_K_M | 4.5, 4.83 | ✓ chat sweet spot |
| `q5_k_s`, `q5km`, `q5_k_m` | Q5_K_S, Q5_K_M | 5.5, 5.83 | ✓ |
| `q6`, `q6_k` | Q6_K | 6.56 | ✓ near-fp16 |
| `iq1_s`, `iq1_m`, `iq2_*`, `iq3_*`, `iq4_nl`, `iq4_xs` | IQ1_S … IQ4_XS | 1.56 – 4.5 | imatrix-friendly |

## Supported model families

### Short names (CLI & Python)

Pass any of these to `hypernix chat --repo-id`, `old_oven.preheat`,
`download_model`, etc.

| Family | Short names |
|---|---|
| **HyperNix** | `hyper-nix.1`, `hyper-nix`, `hypernix`, `nano-nano-v4`, `nano-mini-6.99-v2`, `nano-nano-927-v3` |
| **Nix** (ray0rf1re/nix collection) | `nix`, `nix2.5`, `nix2.6-m`, `nix2.6-mm`, `nix-2.7a`, `nix2.7`, `nix2.6` |
| **Llama 3.x** | `llama-3.1-8b`, `llama-3.1-8b-instruct`, `llama-3.2-1b`, `llama-3.2-3b`, `llama-3.3-70b-instruct` |
| **Qwen 2.5 / 3 / 3.5 / 3.6** | `qwen2.5-*`, `qwen3-0.6b`, `qwen3-8b`, `qwen3.5-{0.8b,2b,4b,9b,27b,35b-a3b,122b-a10b,397b-a17b}`, `qwen3.6-35b-a3b` |
| **Gemma 2 / 3 / 4** | `gemma-2-{2b,9b,27b}`, `gemma-3-{1b,4b}`, `gemma-4-{e2b,e4b,26b-a4b,31b}` |
| **Phi 3 / 3.5 / 4** | `phi-3-mini`, `phi-3.5-mini`, `phi-4` |
| **DeepSeek** | `deepseek-r1-distill-llama-8b`, `deepseek-r1-distill-qwen-7b`, `deepseek-v2-lite`, `deepseek-v3` |
| **GLM 4 / 5 / 5.1** | `glm-4-9b-chat`, `glm-4.1v`, `glm-5`, `glm-5.1`, `glm-5.1-fp8` |
| **Mistral / Mixtral** | `mistral-7b-instruct`, `mixtral-8x7b-instruct` |
| **NVIDIA** | `nemotron-4-15b`, `llama-3.1-nemotron-70b-instruct`, `mistral-nemo-12b` |
| **OpenAI gpt-oss** | `gpt-oss-20b`, `gpt-oss-120b` |

The full registry lives in `hypernix.KNOWN_MODELS`.

### ARCH_PRESETS (seeds for `new_oven`)

`new_oven(arch="...", ...)` spins a fresh, parametric model in the
shape of any of these families:

- `hypernix`, `llama`, `llama3`, `llama3.1`, `llama3.2`, `llama3.3`, `llama4`
- `qwen2`, `qwen2.5`, `qwen3`, `qwen3.5`, `qwen3.6`
- `gemma`, `gemma2`, `gemma3`, `gemma4`
- `mistral`, `phi3`, `phi4`
- `glm4`, `glm5`, `glm5.1`
- `deepseek`, `deepseek-r1`, `nemotron`, `gpt-oss` / `gptoss`
- `nix`, `nix2`

Presets are seeds for brand-new parametric models. **Loading** a
pretrained checkpoint for any of these families works without a matching
preset because non-HyperNix `model_type` values route through
`transformers.AutoModelForCausalLM`.

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) — 5-line Python API demo.
- [`examples/custom_arch.py`](examples/custom_arch.py) — arbitrary-size HyperNix.
- [`examples/upload_to_hub.py`](examples/upload_to_hub.py) — publish to the Hub.
- [`examples/train_hypernix_0_1_5_evaluator.py`](examples/train_hypernix_0_1_5_evaluator.py) — tiny evaluator demo wiring ovens + all three fridges.
- [`examples/train_hypernix_1_5_gtx1080.py`](examples/train_hypernix_1_5_gtx1080.py) — production-shape 92.1 M model trained on an 8 GB Pascal card.

## Wiki / deep dives

Topic-focused reference guides live in the `wiki/` directory:

- [`wiki/Home.md`](wiki/Home.md) — index
- [`wiki/Ovens.md`](wiki/Ovens.md) — `old_oven` / `new_oven` reference
- [`wiki/Fridges.md`](wiki/Fridges.md) — `old_fridge` / `mediocre_fridge` / `new_fridge`
- [`wiki/Ranges.md`](wiki/Ranges.md) — `new_range` / `old_range` / `industrial_range` (labeling rubrics)
- [`wiki/Freezer.md`](wiki/Freezer.md) — VRAM manager (OldFreezer / NewFreezer / FlashFreezer)
- [`wiki/Alarms.md`](wiki/Alarms.md) — smoke alarms (Rads / Gas / Modern / Auto) + CPU / GPU preset tables
- [`wiki/Kitchen.md`](wiki/Kitchen.md) — pans / microwave / table / sink / instant pot / coffee maker / pressure cooker / qa
- [`wiki/Abbicus.md`](wiki/Abbicus.md) — Abbicus (linear) and TurboAbbicus (exponential + sine oscillation)
- [`wiki/STML.md`](wiki/STML.md) — Short Term Memory Loss context manager and VRAM calculator
- [`wiki/Pascal.md`](wiki/Pascal.md) — CUDA 6.1 / GTX 1080 playbook
- [`wiki/Architectures.md`](wiki/Architectures.md) — ARCH_PRESETS and KNOWN_MODELS
- [`wiki/Training.md`](wiki/Training.md) — scratch training, expansion, and fine-tuning flows
- [`wiki/CLI.md`](wiki/CLI.md) — full CLI cheat sheet
- [`wiki/Quantization.md`](wiki/Quantization.md) — GGUF conversion + k-quant pipeline
- [`wiki/Changelog.md`](wiki/Changelog.md) — full per-release version history
- [`wiki/Pressure-Cooker-V5.md`](wiki/Pressure-Cooker-V5.md) — Pressure Cooker V5 / V5+ / V5S with QAT and MTP
- [`wiki/MTP.md`](wiki/MTP.md) — Multi-Token Prediction guide
- [`wiki/Scavenger.md`](wiki/Scavenger.md) — HuggingFace dataset discovery

## How the GGUF pipeline works

1. `huggingface_hub.snapshot_download` pulls weights + tokenizer files.
2. The converter loads the state dict, infers dimensions from tensor
   shapes (so any HyperNix size works), and maps tensor names onto
   llama.cpp's canonical GGUF layout when a recognizable pattern matches
   (Llama, GPT-NeoX, GPT-2, nanoGPT). Unknown names round-trip verbatim.
3. `llama-quantize` consumes the fp16 GGUF to produce each k-quant.

The CLI emits exactly one fp16 intermediate and reuses it for every
k-quant in the plan.

## Platform notes

- **Linux**: full support, every distro tested on: (Ubuntu, Debian, Arch.)
- **macOS**: Metal for inference, Homebrew for `llama-quantize`. (untested)
- **Windows**: native support; doctor accepts Windows; `llama-quantize` auto-downloads Windows binaries; use scoop / chocolatey for system deps. (untested)
- **Pascal (GTX 1080 / 1080 Ti / Titan Xp)**: install torch from the CUDA 11.8 index first (see above). Use `OldFreezer` or `auto_freezer()`; `pascal_safe_dtype()` picks fp16. `hypernix.freezer.pascal_mode_hints()` returns a dict of recommended settings (batch size, dtype, TF32/SDPA toggles) for the detected card.

## CI autofix

Three scripts in `scripts/`, each owning one failure class, plus a router
that reads a CI log and runs the right one:

| Script | Owns |
|---|---|
| `autofix-B` | ruff diagnostics |
| `autofix-E` | imports, syntax, anything that stops collection |
| `autofix-F` | failing tests for a module category (timing by default) |

```bash
scripts/autofix                      # reproduce the failure, classify, fix
scripts/autofix --log ci-output.txt  # classify an existing CI log
scripts/autofix-F --dry-run          # timer-test repair, without writing
```

`autofix-F` engages only when *some but not all* of the timing tests fail —
the signature of a wall-clock assertion that lost a race, which is the one
thing it can fix. It widens the margins in those tests by scaling every time
constant in them uniformly, re-runs only what it changed, and commits with an
`Autofix-Scope:` trailer. CI reads that trailer and verifies just those tests
instead of re-running the 4-OS x 4-Python matrix. Failures it can't honestly
fix — a renamed symbol, a changed signature, a real regression — are reported
and left alone.

See [`scripts/README.md`](scripts/README.md) for the full picture.

## Build / release

```bash
pip install build twine
python -m build
twine check --strict dist/*
```

Release tags (`vX.Y.Z`) fire `.github/workflows/release.yml` which
publishes to PyPI via Trusted Publishing and attaches the wheel +
sdist + an `examples-scripts` tarball + `SHA256SUMS` to a GitHub
Release.

## Usage & Documentation

Comprehensive performance analysis and training guides available in the following PDFs:

| Document | Description |
|---|---|
| [**pressure_cooker_v5_v5s_paper.md**](pressure_cooker_v5_v5s_paper.md) | PressureCookerV5 / V5S architecture, math, and measured efficiency numbers (memory + step-time), reproducible from the scripts below. |
| [**01_ram_and_training_time.pdf**](01_ram_and_training_time.pdf) | RAM usage patterns and training time impact analysis. |
| [**02_optimizer_speed_and_memory.pdf**](02_optimizer_speed_and_memory.pdf) | Optimizer performance comparison and memory optimization strategies. |
| [**03_gpu_utilization_and_vram.pdf**](03_gpu_utilization_and_vram.pdf) | GPU utilization patterns and VRAM management best practices. |
| [**04_architecture_and_pipeline.pdf**](04_architecture_and_pipeline.pdf) | Detailed architecture documentation and training pipeline information. |

```bash
# Reproduce the optimizer benchmarks yourself:
python scripts/benchmark_v5.py              # AdamW vs PressureCookerV5, step time + peak mem
python scripts/benchmark_v5s.py             # AdamW vs V5 vs V5S, step time + peak mem
python scripts/measure_optimizer_memory.py  # exact optimizer-state bytes per parameter tensor
```

## License

HyperNix is **dual-licensed** — recipients choose one of the two options
below (see the full text in [`LICENSE`](LICENSE)):

- **LLU-0.1** — the HyperNix OpenCode Light Limited Use License, Version
  0.1. Source-available, with a same-license requirement for forks and a
  §12 field-of-use restriction (it does not meet the OSI Open Source
  Definition because of that restriction). This is the **default** if
  you don't make an active choice.
- **HOS-1.0** — the HyperNix Open Source License, Version 1.0. An
  OSI-compliant open-source license with no field-of-use restrictions.

You must pick one license and follow its terms — you can't mix terms
from both. Large trained models (29.1B+ parameters) that are shared
publicly carry a transparency requirement (disclosing training-data
sources or a data composition summary) under either license; see
`LICENSE` §7–8 (LLU-0.1) / the equivalent HOS-1.0 sections for specifics.
`hypernix` is **not** Apache-2.0, MIT, or any other stock license.
