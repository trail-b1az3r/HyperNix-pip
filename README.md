<div align="center" style="display:block; width:100%; max-width:950px; margin:0 auto;">
  <a href="https://trail-b1az3r.github.io/HyperNix-pip/" target="_blank" rel="noopener"
     style="display:inline-block; vertical-align:middle;">
    <img src="https://github.com/trail-b1az3r/HyperNix-pip/raw/main/assets/logo-new/icon-512.png?raw=true"
         alt="hypernix icon" width="140"
         style="border-radius:16px; border:6px solid #0c0c0c; box-shadow:0 14px 36px rgba(0,0,0,0.6);" />
  </a>

  <img alt="decor bar" src="https://img.shields.io/badge/--/--/--?style=flat-square&color=0b0b0e&label=%20"
       style="height:84px; width:700px; vertical-align:middle; margin-left:14px; border-radius:12px; border:6px solid #111; box-shadow:0 14px 36px rgba(0,0,0,0.6);" />

  <div style="margin-top:12px; text-align:center;">
    <img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.72.3-ff2d55?style=for-the-badge&logo=pypi&logoColor=white"
         style="border-radius:10px; border:2px solid #24000a; box-shadow:0 8px 20px rgba(255,45,85,0.12);" />
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.14-00c9ff?style=for-the-badge&logo=python&logoColor=white"
         style="border-radius:10px; border:2px solid #00212a; box-shadow:0 8px 20px rgba(0,201,255,0.10); margin-left:8px;" />
    <img alt="License" src="https://img.shields.io/badge/License-HOS%20/%20LLU-00c853?style=for-the-badge"
         style="border-radius:10px; border:2px solid #05240a; box-shadow:0 8px 20px rgba(0,200,83,0.08); margin-left:8px;" />
  </div>
</div>

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
