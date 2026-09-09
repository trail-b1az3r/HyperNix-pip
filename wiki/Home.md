# hypernix wiki

Deep-dive reference for the `hypernix` package. For the headline tour
see the [top-level README](../README.md).

## Topic guides

Every page in this wiki, grouped by theme. (Previously this table only
indexed 19 of the 55+ pages that actually exist — the rest were only
reachable if you already knew the filename. Fixed.)

**Start here / whole-system**

| Guide | Covers |
|---|---|
| [CLI](CLI.md) | Every subcommand (all 34), every flag, typical invocations. |
| [HyperNix Studio](../desktop/README.md) | The Qt desktop client: models, chat, a workspace the model may edit, and why it cannot run a command. |
| [ggml-hnx](../native/ggml-hnx/README.md) | Building a llama.cpp that reads HyperNix sub-bit models. |
| [Kitchen](Kitchen.md) | pans / microwave / table / sink / instant_pot / coffee_maker / pressure_cooker / pressure_cooker_v3. |
| [CoffeeMaker](CoffeeMaker.md) | Scheduled training/eval runs — brew on a timer instead of babysitting a run. |
| [Blender](Blender.md) | Interleaves/mixes multiple input data streams into one. |
| [Architectures](Architectures.md) | `ARCH_PRESETS` seed registry and `KNOWN_MODELS` short-name registry. |
| [HuggingFace Models](HuggingFace-Models.md) | Every model currently published under the `ray0rf1re` HF account. |
| [Roadmap](Roadmap.md) | Planned features and releases. |
| [Changelog](Changelog.md) | Full per-release notes — features, fixes, UX papercuts. |
| [Release Timeline](Release-Timeline.md) | Auto-updated commit-by-commit timeline of public releases. |

**Downloading, converting, quantizing**

| Guide | Covers |
|---|---|
| [Download](Download.md) | `download_model`, short-name resolution, offline cache, gated repos. |
| [Convert](Convert.md) | Safetensors/PyTorch checkpoint → GGUF (fp32/fp16), architecture-agnostic tensor naming. |
| [Quantization](Quantization.md) | GGUF pipeline, k-quants, `HyperNixQuantizer`, `pressure_cooker_v3` QAT. |
| [HyprSlug](HyprSlug.md) | Quantise a GGUF with no llama.cpp at all — the upstream types (`Q4_K_M` and friends) *and* the sub-bit tiers (`IQ0.9_L`, `IQ0.75_M`, `IQ0.5_XXXL`), which `llama-quantize` cannot produce. |
| [Imatrix](Imatrix.md) | Measure an importance matrix from activations, and read anyone else's — both llama.cpp's binary format and JSON. |
| [Dflash2](Dflash2.md) | A draft model carried inside the model it drafts for. Speculative decoding from one file, with the same tokens out. |
| [HnxRun](HnxRun.md) | The runtime for the models llama.cpp cannot read — loads and runs a sub-bit GGUF, so a 0.5-bit quantisation has somewhere to go. |
| [Devices](Devices.md) | CUDA (including the sm_61 / GTX 1080 trap), ROCm, Metal, Intel, Vulkan and CPU — what runs where, and which torch wheel. |
| [LowBit](LowBit.md) | Every GGML type at 200 and above: `IQ0.25_UXL`, `INT1`, `FP2`, `INT4` — the two families, and what each rate really costs. |
| [HyprSlug-Headers](HyprSlug-Headers.md)
- [ModelIndex](ModelIndex.md) — `hypernix-t1 index`, a folder of GGUFs into a model registry | `hypernix hyprslug-headers` — self-describing headers, the compat export, and the server that lets LM Studio reach a sub-bit model without converting it. |
| [PipelineMechanics](PipelineMechanics.md) | The small support modules gluing download → convert → quantize together. |

**Training core**

| Guide | Covers |
|---|---|
| [Training](Training.md) | `init_from_scratch`, `expand_checkpoint`, `train`, AutoModel fallback, `compute_framework`, `abbicus`. |
| [Abbicus](Abbicus.md) | Automatic token regulation and curriculum tuning by model size / step. |
| [Optimizers](Optimizers.md) | The custom AdamW-family optimizer modules (`pressure_cooker` and relatives). |
| [Pressure Cooker v3](Pressure-Cooker-V3.md) | `PressureCookerV3`, V3Plus QAT, `StovetopV3CookerPlus`, `CookerLite`. |
| [Pressure Cooker v4](Pressure-Cooker-V4.md) | Optimized quantization-aware training mechanism and optimizer wrapper. |
| [Pressure Cooker v5](Pressure-Cooker-V5.md) | Flagship ORCP optimizer — quantized momentum, QAT, MTP support, V5+/V5S variants. |
| [Pressure Cooker v6](Pressure-Cooker-V6.md) | Speed-first optimizer — single fused momentum buffer, `torch._foreach_*` multi-tensor updates; `PressureCookerV6V` adds CUDA graph capture + optional `torch.compile`. |
| [MTP](MTP.md) | Multi-Token Prediction — predict several future tokens at once for sample efficiency + speculative decoding. |
| [STML](STML.md) | Short Term Memory Loss — `STML` context manager and `calculate_vram_context` VRAM calculator. |
| [Frameworks](Frameworks.md) | `ComputeFramework` multi-device training and `workshop` TTS/ASR pipelines. |
| [Freezer](Freezer.md) | VRAM manager — `OldFreezer`, `NewFreezer`, `FlashFreezer`, `auto_freezer`, CPU/GPU preset registries. |
| [VRAM](VRAM.md) | Making more fit without changing what the model learns — allocator tuning, activation checkpointing, optimizer-in-backward, optimizer-state offload, and how to measure whether any of it worked. |
| [Alarms](Alarms.md) | Smoke alarms — step planners from lightest heuristic up to hardware-calibrated. |
| [Pascal](Pascal.md) | GTX 1080 / CUDA 6.1 / sm_61 training playbook. |
| [macOS-legacy](macOS-legacy.md) | Running on old Intel Macs with torch 1.13 via `hypernix.torch_compat`. |

**Data**

| Guide | Covers |
|---|---|
| [Scavenger](Scavenger.md) | Dataset discovery engine — searches the HF Hub under storage/quality/recency constraints. |
| [Lunchbox](Lunchbox.md) | Dataset packager — fixes HF-Hub `CastError` column-mismatch issues. |
| [Tupperware](Tupperware.md) | Automated dataset round splitting with optimal LR and optional eval. |
| [CuttingBoard](CuttingBoard.md) | Train/val/test dataset splitting, deterministic + stratified. |
| [Shakers](Shakers.md) | `salt_shaker` / `pepper_shaker` — training-example perturbation/augmentation. |

**Evaluation & labeling**

| Guide | Covers |
|---|---|
| [Ranges](Ranges.md) | `new_range` / `old_range` / `industrial_range` — labeling rubrics from cheap heuristics up to LLM-as-judge. |
| [EspressoMaker](EspressoMaker.md) | Prompt-battery evaluation tiers (Ristretto/SingleShot/DoubleShot/Lungo). |
| [Fridges](Fridges.md) | `old_fridge` (memory housekeeping), `mediocre_fridge` (judge-data synthesis), `new_fridge` (graphing). |
| [Dashboards](Dashboards.md) | The three live btop++-style terminal training dashboards (`tvtop`/`cctvtop`/`tvtop++`). |
| [TvTopPro](TvTopPro.md) | `tvtoppro` — tvtop++'s stats under a btop++ presentation, with themes. Not built on cctvtop. |
| [HyperLog](HyperLog.md) | Premium training TUI logger, compatible with tvtop. |

**Audio**

| Guide | Covers |
|---|---|
| [WakeUp](WakeUp.md) | Train a wake word on your voice, a folder of recordings, or TTS overnight — then listen for it. What openWakeWord does, without using it. |
| [Quantization](Quantization.md) | …and see [HyprSlug](HyprSlug.md) for quantising without llama.cpp. |

**Inference, chat, and serving**

| Guide | Covers |
|---|---|
| [Ovens](Ovens.md) | `CodeOven`, `old_oven.preheat`, `new_oven`, `bake_code`, `fill_middle`, `save_pt`/`load_pt`. |
| [Cookbook](Cookbook.md) | Chat-template registry (chatml / hyper-nix.2 / llama3 / llama2 / alpaca / vicuna / plain). |
| [Countertop](Countertop.md) | Multi-turn chat session management with persisted history and auto-trim. |
| [Menu](Menu.md) | System-prompt presets (default / code-helper / judge / creative / chef / hyper-nix). |
| [Bell](Bell.md) | Streaming-token callback + done notification for chat/completion loops. |
| [Flour](Flour.md) | Chat-quality logits processor — repetition penalty, no-repeat n-gram, stop-sequence detection. |
| [Vera](Vera.md) | The AI assistant built into HyperNix, and its `hypernix vera` CLI. |
| [Camouflage](Camouflage.md) | RLHF/RLAF alignment scaffolding, optional AI-assisted evaluator scoring. |

**Post-training model utilities**

| Guide | Covers |
|---|---|
| [Whisk](Whisk.md) | Checkpoint averaging — SWA / EMA / geometric-mean over N saved snapshots. |
| [RecipeBook](RecipeBook.md) | Named-config registry (`cook(name, **overrides)`), the backing store for `hypernix brew`. |
| [DeepFryer](DeepFryer.md) | Reversible weight-noise perturbation — a regularizer or a severe stress-test. |
| [CakePan](CakePan.md) | Hybrid CPU+GPU training guard — NaN/Inf detection, wall-time watchdog, memory-pressure rollback. |
| [Apron](Apron.md) | RNG-state snapshot/restore context manager. |

**Data-tier processing (the "kitchen appliance" family)**

| Guide | Covers |
|---|---|
| [Pans](Pans.md) | Five-tier data preprocessing, from `FryingPan` up to `Wok`. |
| [Microwave](Microwave.md) | Five-tier throwaway inference — quick zaps without keeping an oven preheated. |
| [FoodProcessor](FoodProcessor.md) | Bulk chunking of ingredients/data into ready-to-use pieces, four blade tiers. |
| [Toaster](Toaster.md) | Per-line formatting, four toaster tiers. |
| [Smoker](Smoker.md) | Slow, high-quality training — the opposite tradeoff from `Microwave`. |

**Serving: the T1 API, its keys, and its clients**

| Guide | Covers |
|---|---|
| [T1-API](T1-API.md) | `hypernix.t1api` — controlled HTTP gateway into HyperNix-pip (model registry, auth, routing, modules, servers, jobs, events, billing, the LM Studio bridge, HyperLink). Released: **T1 v1.0.26.8.1.1**. |
| [T1-API Security Checklist](T1-API-Security-Checklist.md) | What to check before exposing a deployment — one page, per-item, with the config field each maps to. |
| [Waiter-TUI](Waiter-TUI.md) | `waiter` — the official T1 API client CLI/TUI (`waiter serv`, `models`, `usage`, `hyperlink`, `version`, `help`). Beta 3 complete: every spec flag wired, full curses TUI (`-G`). |

Surfaces documented inside those pages rather than on their own:

| Surface | Where |
|---|---|
| `hypernix-t1` — start / stop / restart / status / logs / create / configure / test / key / autostart | [CLI](CLI.md#hypernix-t1) |
| `gkey` — minting T1, T2, T2S and T2P keys (`-v v1\|v2\|v2short`), `gkey version` | [T1-API § Minting keys in each format](T1-API.md#minting-keys-in-each-format) |
| The bootstrap key a new server issues itself (loopback-only, 3 days, once) | [T1-API § The first key a new server has](T1-API.md#the-first-key-a-new-server-has) |
| T2P billing keys, and a server refusing or separating them | [T1-API § Billing keys](T1-API.md#billing-keys-t2p-and-refusing-them) |
| The HyperLink iOS app and pairing | [T1-API § HyperLink](T1-API.md#hyperlink) |
| `HYPERNIX_TOOL_POLICY` — consent before the AI agent runs a side-effecting tool | [CLI § Environment variables](CLI.md#environment-variables) |

**Reference / meta**

| Guide | Covers |
|---|---|
| [Workshop](Workshop.md) | Model frameworks for TTS/ASR/LLM/Vision, nano-nano collection, 30+ architectures. |

## The subsystem map

```
                 ┌──────────────┐
                 │  download    │  huggingface-hub + KNOWN_MODELS
                 └──────┬───────┘
                        ▼
          ┌───────────────────────────┐
          │        train              │  HyperNixConfig / Model
          │  (init, expand, loop,     │  AutoModel fallback for
          │   load_snapshot)          │  Gemma 4 / Qwen 3.5+ / GLM 5 / …
          └──────┬────────────┬───────┘
                 │            │
                 ▼            ▼
          ┌───────────┐  ┌───────────┐
          │ old_oven  │  │  new_oven │  new_oven = fresh init in
          │ (preheat) │  │           │  one of 20+ ARCH_PRESETS
          └─────┬─────┘  └─────┬─────┘
                └──────┬──────┘
                       ▼
         ┌──────────────────────────────┐
         │        CodeOven              │  .complete, .chat, .fill,
         │                              │  .train, .save_pt
         └──────────────────────────────┘

                           Assist modules
                                │
            ┌──────────────┬────┴────┬──────────────┐
            ▼              ▼         ▼              ▼
       freezer         old_fridge  mediocre_   new_fridge
       (VRAM mgr)      (memory)    fridge       (graphing)
                                   (datasets)

       convert → quantize → upload     (GGUF pipeline)
```

## Design principles

- **Small, inspectable modules.** Every subsystem is <~300 LOC and
  usable in isolation.
- **No hard dependencies on the big stuff.** `transformers`, `matplotlib`,
  and `llama-cpp-python` are all loaded lazily when first needed.
  `HYPERNIX_AUTO_INSTALL=0` disables runtime pip calls.
- **Degrade on CPU / old hardware.** Every VRAM and dtype decision has a
  CPU-safe fallback; the test suite exercises CPU-only paths directly.
- **One name, one thing.** `preheat` loads, `bake_code` generates,
  `freeze` freezes, `chill_cache` frees cache, `suggest_batch_size`
  suggests. Names are verbs where it makes sense.

## Version history

Recent releases (see [Changelog](Changelog.md) for the full per-release
notes going all the way back to 0.2.0):

- **0.72.4** (in progress) — **Training administration**: `hypernix.training.monitor` reports what a run is doing (progress, loss history, checkpoints, ETA) and reconciles it against the actual process, so a crashed trainer is reported *stale* rather than left claiming to run. `GET /training/*` and stop/pause/resume over HTTP, plus `hypernix-t1 training` for when the API is the thing in trouble — admin-only unless trusted-network mode is on, with the destructive half behind a second opt-in. **HyperLink knows which machine it is talking to**: each installation gets a stable fingerprint the app pins at pairing time and re-checks on every reconnection, because an address can end up pointing at a different machine and a server *name* authenticates nothing. Tailnet peer discovery (`GET /hyperlink/peers`, admin-only, every row explicitly unverified), keyless connections on a trusted network, an admin credential store that forgets on restart, and the app finally has an icon — the current HyperNix mark, generated from the SVG with the geometry pinned by a test. **`native/ggml-hnx`**: a llama.cpp that can actually read a sub-bit model, cross-checked block-for-block against the Python encoder that wrote every such file. **HyperNix Studio**, a Qt 6 desktop client — model switching, chat, a workspace the model may edit with per-call approval, and no way whatsoever to run a command. **[`hnx gather`](Gather.md)**: a crawler that turns a site into a corpus — `-W`/`-L` sites, `-T` threads, `-Q` depth, seven output formats including a merged single-document one, compression, and optional upload to GitHub or Hugging Face; robots.txt and a per-host delay on by default, a host's own `Crawl-delay` overriding yours, and `-f js` that *saves* JavaScript without running any of it. **hyperNix0x-v2 in Neo oven**: `preheat_brewed()` / `new_brewed()`, and plain `preheat()` recognises a Brewer checkpoint by looking inside it rather than by unpickling it. See the [Changelog](Changelog.md) for the whole of it.
- **0.72.3** — T1 v1.0.2026.8.1.1. A new server can be set up without knowing anything: first start mints itself a **bootstrap admin key** that works only from that machine, expires after three days, and is minted once — which is what `waiter hyperlink pair` had no way to do on a fresh install. **`hypernix-t1`**, one dependency-free executable for the whole server lifecycle (start/stop/kill/restart/status/logs/create/configure/test/key/autostart/remove). **HyperLink connects**: three separate bugs each produced the same silent timeout — the server advertised port 8000 whatever port it was on, iOS ATS blocked Tailscale's 100.64.0.0/10 (shared address space, not RFC 1918), and a missing tailnet said nothing about why — all fixed, and the app takes a T2S key as well as a pairing code. **T2P billing keys** carry a spend cap and provider references, never administrator rights and never card data; `T1_BILLING_KEY_POLICY` lets a server accept them, refuse them at its own payment page, or require payment on a *separate* key. The **AI agent asks before it runs anything** (`HYPERNIX_TOOL_POLICY`) — tool calls are parsed out of the model's own reply, so a file it read could previously run shell commands. `gkey` gains `-v v1|v2|v2short` and `gkey version`, and finally honours `T1_KEYMASTER_DIR`, which the server was already reading. New **`hypernix.system.vram`**: allocator tuning (`expandable_segments`), activation checkpointing, optimizer-in-backward, optimizer-state offload, and peak measurement — each opt-in, each reversible, and each refusing rather than silently doing nothing, since every one of them is invisible when it quietly fails. Reachable from `hypernix train run`. CI and the public release gate on a **live server**: two jobs mint their own T2 keys, drive a real API and a booted iPhone simulator against a fake model, and delete every key they made — nothing publishes until both are green.
- **0.72.2** (`.post1`–`.post5`) — **`install-t1.sh`**, an interactive T1 API installer that writes a configuration matching the deployment kind you describe (bash 3.2, so a stock macOS runs it). Then five fix bumps: the admin key the installer printed was invisible to the server it configured (`T1_KEYMASTER_DIR`); `waiter` explains an unreachable server instead of restating the errno, and gains `waiter help <topic>`; T2S keys were refused as malformed by a length check that predated them; and — the last one — **every key `gkey` ever minted carried server ID `00001-A1`**, because the counter lived in memory and each `gkey` run is its own process. `POST /t1/auth/undo` could never undo anything either, for two independent reasons.
- **0.72.1** — T1 v1.0.26.8.1.0. **The T2 key family**: T1's structure plus an access level (1–9), an optional admin password, and an SSPKID. A T2 key converts to a valid T1 key and authenticates against the store that already holds it, so there is no migration. Round-tripping 3000 keys caught a real conversion bug — the T2 special alphabet excluded `-`, so any T1 key whose specials contained one converted to a *different* key that then failed to authenticate.
- **0.72.0** — T1 v1.0.26.8.0.1. The betas end. The **LM Studio bridge**, **HyperLink** (pairing, sessions, attachments, endpoint advertisement), Hugging Face link merging, and the **HyperLink iOS app**. The T1 API stops tracking the pip version and takes the six-part `api.major.year.month.feature.fix` scheme — the two ship together but answer different questions.
- **0.71.5b2** — T1 API Beta 2: the model routing/quota-cascade engine (`POST /models/route`, plan-scoped cascades loaded from data, never silently substitutes an exhausted manual selection), a server registry with trust-gating (`untrusted` by default, admin-only promotion), the module system (create/upload/version/sync-tracking, never executes what it stores), async jobs (`queued→running→succeeded|failed|cancelled`, real thread-pool execution, cooperative cancellation), an in-process event bus (`GET /events` poll + `GET /events/stream` SSE), and an internal billing ledger (payment tokens, balances, transactions — explicitly not a payment-processor integration). New SSRF and path-traversal guards (`hypernix.t1api.security`) shared by the server/module systems, with an explicit `allow_private_address` opt-in for Tailscale/local deployments. See [T1-API](T1-API.md) for the full endpoint reference and roadmap.
- **0.71.5b1** — New `hypernix.t1api`: a controlled HTTP gateway into HyperNix-pip (mountable FastAPI module, `hypernix.t1api.create_app`), wrapping the existing `Keymaster`/`Gatekeeper` for auth rather than duplicating them. Model registry enforces "unregistered model → `MODEL_NOT_SUPPORTED`, always" as a hard rule; the nine example models from the T1 API spec ship as seed data but stay invisible unless `T1_ENABLE_EXAMPLE_MODELS=1`. Per-key/per-model usage metering on SQLite. New `waiter` console script — the official T1 API TUI/CLI, zero hard deps beyond core `hypernix` — implements the spec's one-shot `waiter serv -A -I <server> -K <token> -E` setup plus `models`/`status`/`usage`/`whoami` subcommands. Beta 1 of a 3(+1)-beta rollout — see [T1-API](T1-API.md) and [Waiter-TUI](Waiter-TUI.md) for the full roadmap.
- **0.71.5a1** — New speed-first optimizer generation: `PressureCookerV6` (single fused momentum buffer, `torch._foreach_*` multi-tensor updates, one host↔device sync per step instead of several per parameter) and `PressureCookerV6V` (adds CUDA graph capture + optional `torch.compile`). Measured (not estimated) 0.5x AdamW's optimizer-state bytes and 1.55x AdamW's CPU step time in this repo's own benchmark script — see [Pressure Cooker v6](Pressure-Cooker-V6.md). Docs site gains a full holiday/observance calendar (Trans Day of Visibility and many more) alongside the existing Christmas/Halloween/Pride banners.
- **0.71.4b10** — Public release workflow now diffs against the last *stable* tag when cutting a stable release (skipping over betas in between), not just the immediately preceding tag. Fixed several stale/false claims: README said "16 CPU presets... Ryzen" when the real count was 48 and zero were actually Ryzen — now 60 real presets including 12 verified Ryzen entries, plus 3 new CPU-sized `hypernix.brewer` architecture presets. GitHub Pages docs site and `hypernix.brewer`'s missing README row both fixed. 125 new tests covering everything shipped since 0.71.4b6 (was previously validated only by hand).
- **0.71.4b9** — Two more catalog fixes (`qwable-3.6-27b-mtp` had community-reported broken tensors, swapped off entirely; `qwable-9b-fable5`'s safetensors base uses an unrecognized hybrid-attention architecture, swapped to its official GGUF). The footer box no longer disappears during local model loads — it was being erased, not scrolling away. New animated `Spinner` class runs for every chat turn now, not just quiet ones. `/settings thinking-display` (hidden/grayed/normal/red/theme) replaces the old hide-thinking toggle with real colored rendering.
- **0.71.4b8** — Fixed `nanbeige4.2-3b-gguf` (wrong filename case, and wired to the wrong backend for its custom architecture — added a `nanbeige` multilama backend). Robust Python interpreter resolution across the launcher, bash scripts, and TUI so the bridge/GUI always match whichever interpreter has `hypernix` installed. Thinking/reasoning output (`<think>` blocks) is now hidden from replies by default. New `/settings` (max input/output/thinking tokens) and `/tools` (real file create/edit/read/search, workspace-scoped, wired into an actual agentic tool-calling loop for cloud and GGUF models).
- **0.71.4b7** — New `hypernix.multilama` module: one interface over several llama.cpp variants (vanilla, `ik_llama.cpp`, PrismML's 1-bit-kernel fork, KoboldCpp) so a GGUF that won't load on one backend can fall back to another. `hyped-pro`'s GGUF dispatch now runs through it.
- **0.71.4b6** — `hyped+`/`hyped-pro` rewrite: dropped the OpenClaw-inspired branding in favor of its own look; every model is now backed by a real Python dispatch layer (`hypernix.hyped_pro_core`) shared with a new desktop GUI. Qwen and Kimi K3 are correctly classified as cloud (Alibaba Cloud DashScope and Moonshot AI respectively, with real API base URLs and auth env vars); local HuggingFace models auto-download on selection and via `/download`. New `/gui` command and `hyped-pro-gui` entry point launch a real Qt6 (X11 + Wayland) desktop app with a GTK4 fallback, both logging coded errors/debug info to the terminal. `/key` now persists real per-vendor API keys to `~/.hypernix/config.json`.
- **0.71.4b2** — `hyped+` (`hyped-pro`) Node.js TUI with locked multi-panel layout inspired by OpenClaw, Qwen Code CLI, Claude Desktop/CLI, and 2D pixel art coffee mascot startup animation. Expanded model catalog (Kimi K3, Claude Sonnet 4.6/5, Opus 4.8, Haiku 4.5, Fable 5, GPT-4o, GPT-5.6 Terra/Sol, GPT-5.5, DeepSeek R1/V4 Flash, Qwen 3.7 Plus, Gemma 4). Unified model directory (`~/.hypernix/models` / `HYPERNIX_MODELS_DIR`) and HuggingFace token support (`HF_TOKEN`). New Brewer 33.6429M parameter preset (`hypernix0x_v2_33m`). Fixed local model load error in normal `hyped`. Added `/system-prompt`, `/compact-prompt`, `/auto-compact`, `/price`, `/vision` slash commands and tab autocompletion. Prominent `hyper-Nix.2` undertrained warning banner.

- **0.70.4b11** — `qa` module (`QAProcessor`) for turning Q&A datasets into causal LM training strings with optional salt/pepper seasoning. `stml` module: `STML` context manager (untrained hard cap + sequence segment folding into batch dimension) and `calculate_vram_context` VRAM calculator CLI (`hypernix stml`). `TurboAbbicus` exponential curriculum regulator with configurable `hard_cap`, sine-wave oscillation adjusted by CPU load, and VRAM safeguard. Fixed `tvtop++` layout tree bug (border shifting), colors (CPU=green, RAM=magenta, GPU=red), dynamic console resizing, and dynamic graph/log widths. 59 new tests.
- **0.70.4b1** — Added `hnx` CLI shortcut alias. Created premium TUI dashboard `tvtop++` with moving spinner, process monitor, CPU/RAM/GPU block histories, and dampened loss decay curve estimations. Overhauled older `tv` module with resilient log parsing and block histories.
- **0.70.3b2** — Web UI rebuilt from static assets; Tailscale opt-in via `-T`. New `Tupperware` round splitter, `StovetopV3CookerPlus`, `HyperNixQuantizer` facade. Wiki pages for Pressure Cooker v3, Abbicus, Frameworks, Roadmap. `old_fridge.unwrap_model` for DDP/FSDP.
- **0.70.3** — `lazy_suzan` decentralized multi-GPU linking, `StovetopV3Cooker` / `CookerLite`, ComputeFramework crash fixes.
- **0.61.2** — `tvtop` btop-style multi-panel rewrite. Old single `hardware` panel replaced by **four** richer panels: `cpu` (TOTAL bar + per-core grid + 3-row history graph), `memory` (USED / CACHE / FREE / SWAP breakdown bars + 2-row history), `gpu` (UTIL / VRAM / TEMP / PWR gauges + 2-row history + name tag), `training` (unchanged). New probes: `_safe_psutil_per_core`, `_read_proc_stat_per_core` (Linux fallback), `_read_memory_breakdown` (used/free/cached/swap), `_query_nvidia_smi_full` (adds temperature + power.draw + power.limit + GPU name). Footer shows core count + GPU label. All original — no btop code copied. 9 new regression tests in `tests/test_v061_2.py`.
- **0.61.1** — chat CLI + 5 bug-fix / utility passes + **MAJOR `hyper-Nix.2` undertrained warning**. New `hyped` console script: high-quality TUI chat CLI with a configurator that lets you pick from a curated short-list (`hyper-Nix.2`, `hyper-nix.1`, `nix2.7a`, `nix2.6-mm`, `nix2.5`, `qwen3.5-{0.8b,2b,4b,9b}`, `nano-nano-v4`, `nano-mini-6.99-v2`, `nano-nano-927-v3`) or browse all `KNOWN_MODELS`, then pick a persona from `MENU` and tweak sampling. New `hypernix.utils` module — `healthcheck()` / `diagnostic_info()` / `list_models()` / `print_models()` / `session_dir()` / `is_module_available()` / `has_binary()`. New `hypernix.utils.warn_hyper_nix_2()` fires a MAJOR red-bordered warning whenever the under-trained `ray0rf1re/hyper-Nix.2` checkpoint is loaded (suppress with `HYPERNIX_SUPPRESS_HYPERNIX2_WARNING=1`). 5 bugs fixed: hyped chat-loop now routes through `Countertop.say` for proper history management; ASCII picker uses `*` not `★`; `UPS` instantiation no longer blocks on IP-geolocation; `plasma.calibrate_alarm` resets instead of compounding on re-call; `tv._sanitise` preserves `\r` so Windows CRLF logs render. Plus utilities: `Menu.find()` fuzzy persona lookup; `injection.thinking()` / `testing()` / `system_override()` shortcuts. 37 new tests in `tests/test_v061_1.py`.
- **0.61.0** — Python 3.14 support + 3 new modules + tvtop visual rewrite. New: `ups` (uninterruptible-power-supply mode — checks open-meteo for severe-weather codes + a pluggable scheduled-outage callback; on panic, fires `snapshot_fn` once and 3×'s the trainer's `save_every`), `injection` (token splicers — `ThinkingInjector` wraps in `<think>...</think>`, `TestingInjector`, `SystemOverrideInjector`, `CustomInjector`), `plasma` (quick GPU benchmark — runs a 6-step Llama-shape fwd/bwd/step loop and returns a `calibration_factor` you apply to a `smoke_alarm` to make ETAs reflect actual hardware). **`tvtop` rewrite**: btop++-style multi-panel layout with rounded panel frames + side-by-side hardware/training panels + a 5-row Unicode block-bar loss-curve graph + a full-width log panel. Auto-detect now skips logs that don't contain `step N/M loss=…` lines (no more accidentally tailing Konsole/browser logs); binary chars are sanitised before render; nvidia-smi cached for 3 s. 32 new tests in `tests/test_v061_b1.py` + 2 supplementary in `test_v060.py`.
- **0.60.0** — eight new modules. Headline four: `tv` (btop++-style training dashboard, run with the `tvtop` CLI; ANSI-colour, no hard deps; tails the latest log, shows step/ETA/throughput/loss-sparkline/CPU/RAM/GPU vitals/log tail), `compactor` (zip older checkpoints — `Compactor(root, keep_recent=3, fmt="zip")` or one-shot `compact()`), `ethanol` (bounded GPU overclock helper, run with `eth 0` to `eth 30`; refuses to apply without `--confirm` or `HYPERNIX_ETHANOL_CONFIRM=1`; nvidia-settings + nvidia-smi + rocm-smi + intel_gpu_frequency backends), `outage` (display blanker context manager — restores the screen when training finishes, errors, or you Ctrl-C; xset / wlopm / pmset / Windows backends). Plus four 4-tier modules: `timer` (KitchenTimer / EggTimer / IntervalTimer / PomodoroTimer), `thermometer` (InstantThermometer / ProbeThermometer / InfraredThermometer / DigitalThermometer for CPU+GPU temp sampling), `dishwasher` (HandWash / QuickWash / NormalWash / HeavyDuty cleanup of stale logs / checkpoints / build artefacts), `strainer` (Colander / FineMesh / NutMilkBag / Cheesecloth dataset filtering, including 8-gram Jaccard near-dup detection). 44 new tests in `tests/test_v060.py`.
- **0.52.6** — more forgiving `smoke_alarm` kwargs: `time_budget_seconds` now defaults to `600.0` (10 min) so `GasAlarm(cpu_preset="i7_7th_gen")` Just Works, and the base `Alarm` accepts `log_every` / `save_every` / `eval_every` so a downstream training-config dict can be `**`-spread into the constructor without `TypeError`. 20 regression tests in `tests/test_v052_6.py`.
- **0.52.5** — `smoke_alarm` is forgiving: every tier (`RadsAlarm` / `GasAlarm` / `ModernAlarm` / `AutoAlarm`) now accepts `cpu_preset=` / `gpu_preset=` / `max_steps=` directly. `cpu_preset=` accepts both a CPU SKU name (`"i7-7700hq"`) and a generation-family alias (`"i7_7th_gen"` → `i7-7700hq`, `"i9-12th-gen"` → `i9-12900k`, `"core-ultra"` → `core-ultra-7-155h`, etc.). `max_steps` caps `recommended_steps()` so a downstream training loop can hard-limit what the alarm hands back. 27 regression tests in `tests/test_v052_5.py`.
- **0.52.4** — bug fix: `CodeOven.chat` no longer crashes with `ValueError: too many dimensions 'str'` when the tokenizer's `apply_chat_template` returns an unexpected shape (a plain string, a 2-D batched tensor, a `BatchEncoding`, etc.). New `_coerce_token_ids` helper normalises every legal return shape into a flat `list[int]`; `_run` now also defensively coerces its argument and raises a clear `TypeError` if anything still slips through. 19 regression tests in `tests/test_v051_4.py`.
- **0.52.3** — auto version bump from CI (no code changes vs 0.51.3).
- **0.51.3** — `quantize` rewrite: the 6-type alias dict from 0.51.2 grew into a 30-entry `QUANT_CATALOG` of `QuantSpec` dataclasses (frozen: `name`, `bits_per_weight`, `category` ∈ `{float, legacy, k, iq}`, `size_factor`, `notes`, `recommended`). 49 aliases now map to the full llama.cpp ladder — floats (`F32` / `F16` / `BF16`), legacy quants (`Q4_0` / `Q4_1` / `Q5_0` / `Q5_1` / `Q8_0`), k-quants (`Q2_K` … `Q6_K`), and IQ-quants (`IQ1_S` … `IQ4_XS`). New helpers: `quant_recommended()`, `quant_by_category("k")`, `quant_for_size(target_bytes, fp16_bytes)`, `quant_estimate_size("q4km", fp16_bytes)`, `quant_resolve_spec("q4km")`. README + wiki refreshed; `hyper-nix.1` stays a fully-supported model alongside `hyper-Nix.2`. 37 new tests in `tests/test_v051_3.py` covering the catalog, helpers, alias resolution, and backward-compat paths.
- **0.51.2.1** — fix the PyPI logo broken-image that shipped in 0.51.2: README's `<img src=…/main/assets/logo.png>` returned 404 because `main` didn't have the file yet (it was on the working branch). Switched to a SHA-pinned `…/2d5eb37/assets/logo.png` URL that's guaranteed to resolve regardless of branch state, so PyPI renders the logo from this release onward.
- **0.51.2** — auto version bump from CI (no code changes vs 0.51.1.1).
- **0.51.1.1** — logo present and accounted for: `assets/logo.png` (1408×768 RGBA, 670 KB) + the smaller transparent variant `assets/logo1.png` are now in the repo. (PyPI render still broken in this release — see 0.51.1.2.)
- **0.51.1** — 5 bug-fix patches across 3 review passes (1 by-hand source-read + 2 hand-driven testing): `bell` no longer leaks the stop marker into the streamed output; `countertop._trim` always preserves the freshly-appended user turn; `cookbook` `_HYPER_NIX_2` no longer shares dict objects with `_CHATML` (mutation-aliasing fix); `flour.process` accepts torch tensors / generators as `produced_ids`; `pressure_cooker.UniversalCooker.select` now detects Pascal (sm_61, e.g. GTX 1080) and forces `fused=False` instead of silently picking a kernel that requires sm_70+. Project logo wired into README + `assets/logo.png` for the PyPI page.
- **0.51.0** — chat-first release: 5 new modules + first-class support for `ray0rf1re/hyper-Nix.2`. `cookbook` (chat-template registry: chatml / hyper-nix.2 / llama3 / llama2 / alpaca / vicuna / plain + `for_model(repo_id)` resolver), `countertop` (multi-turn session with persisted history, system prompt, auto-trim), `menu` (system-prompt presets: default / code-helper / judge / creative / chef / hyper-nix), `bell` (streaming-token callback + done notification), `flour` (chat-quality logits processor: repetition penalty + no-repeat n-gram + role-leak suppression + decoded-text stop-sequence detection — the bundle that makes hypernix's chat surface better than raw transformers for chatting). `DEFAULT_REPO_ID` now points at `ray0rf1re/hyper-Nix.2`; `CodeOven.repo_id` is plumbed through to `_format_chat` so the cookbook fallback fires automatically.
- **0.50.0** — 4 new modules + 3 bug-fix passes: `whisk` (SWA / EMA / geometric-mean checkpoint averaging) + `cutting_board` (train/val/test split, deterministic + stratified) + `apron` (RNG-state guard context manager) + `recipe_book` (named-config registry with `cook(name, **overrides)`). Bug fixes: `pressure_cooker` falls back to scalar AdamW when private `_functional` API is unavailable; `deep_fryer` uses per-parameter `torch.Generator` for reproducible noise; `food_processor.SliceBlade` validates `overlap_chars`; `industrial_range` pairwise parser detects "tie/tied/equal" anywhere in the head; `instant_pot.brew` fast-fails on missing dataset; `microwave._preheat` requires `config.json` before treating a string as a local snapshot path; `cake_pan` rolls back on `step_timeout` before raising; `apron` snapshots state **before** seeding so exit really restores the caller's pre-call RNG.
- **0.49.0** — `lunchbox` dataset packager (fixes HF-Hub `CastError: column names don't match`) + 31 new coverage tests across lunchbox / pressure_cooker / deep_fryer / cake_pan / freezer / shakers / smoke_alarm + end-to-end evaluator integration test
- **0.48.0** — `pressure_cooker` rewrite: 4 new tiers (`StovetopCooker`, `ElectricCooker`, `InductionCooker`, `ProCooker`) + `universal_cooker` selector + grad accumulation + `GradScaler` integration + fused/foreach AdamW kernels
- **0.47.1** — relaxed install pin to `torch>=1.13,<3` so `pip install hypernix` works on old Intel Macs
- **0.47.0** — `deep_fryer` (weight-noise) + `cake_pan` (CPU+GPU training guard) + 32 new CPU presets (i5, i9, Ultra 5/9) + 51 new GPU presets (full GTX 10 / RTX 20/30/40/50 lineups, Apple M-series, AMD Instinct + Radeon)
- **0.46.1** — `nix` short-name fallback chain: 2.7a → 2.6-mm → 2.5
- **0.46.0** — `salt_shaker` / `pepper_shaker` augmenters + `torch_compat` shim for old Intel Macs with torch 1.13
- **0.45.3** — `smoke_alarm` accepts `preset=` one-string kwarg
- **0.45.2** — pans accept `context_length` / `max_chars` (kw-only)
- **0.45.1** — pans init fix: positional args no longer bind to `name`
- **0.45.0** — espresso_maker, blender, toaster, food_processor, smoker; +3 microwave tiers; +2 coffee_maker tiers + cold_brew; CLI `brew`
- **0.44.0** — pans / microwave / table / sink / instant_pot / coffee_maker / pressure_cooker
- **0.43.0** — `smoke_alarm` (Rads / Gas / Modern / Auto) + 16 CPU + 20 GPU presets
- **0.42.0** — `new_range` / `old_range` / `industrial_range` labeling rubrics
- **0.41.0** — CUDA 6.1 / Pascal helpers, HyperNix 1.5 (92.1 M) training script
- **0.40.0** — `freezer` module (OldFreezer / NewFreezer / FlashFreezer)
- **0.36.0** — `old_fridge` / `mediocre_fridge` / `new_fridge` + evaluator example
- **0.35.0** — Gemma 4, Qwen 3.5 / 3.6, GLM 5.x, Nix family presets
- **0.34.0** — AutoModel fallback, Gemma/Phi/GLM/DeepSeek/GPT-OSS presets
- **0.33.0** — Windows + macOS support, Python 3.13, runtime auto-install
- **0.32.x** — CUDA 11.8 torch, slow-tokenizer fallback
- **0.31.x** — Chat REPL, Nano-nano / Nano-mini model family
- **0.30.x** — Code-generation oven (`old_oven.preheat`)

See [Changelog.md](Changelog.md) for per-release details including
patch versions, UX fixes, and bug reports.
