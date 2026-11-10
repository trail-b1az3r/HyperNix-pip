# Roadmap

> Planned release scopes summarize [`.prompts/.v1/main.md`](../.prompts/.v1/main.md), the detailed implementation specification. A roadmap item is not complete until its implementation, tests, documentation, and supporting evidence satisfy its acceptance criteria.

## 0.71.4b2 (Shipped)

- `hyped+` (`hyped-pro`) Node.js TUI based on OpenClaw, Qwen Code CLI, and Claude Desktop
- Updated Hyped Model Catalog (Kimi K3, Claude Sonnet 4.6/5, Opus 4.8, Haiku 4.5, Fable 5, GPT-4o, GPT-5.6 Terra/Sol, GPT-5.5, DeepSeek R1/V4 Flash, Qwen 3.7 Plus, Gemma 4)
- Unified model directory (`~/.hypernix/models`) & HuggingFace token support
- Brewer 33.6429M parameter architecture preset (`hypernix0x_v2_33m`) & Vision model support
- Slash command auto-completion, price estimator, prompt compaction, auto context compaction, and `hyper-Nix.2` warning banner

## 0.71.4b6 (Shipped)

- `hyped+`/`hyped-pro` real provider dispatch (cloud HTTP calls, local inference, T1 Gatekeeper) via `hypernix.hyped_pro_core` + `hyped_pro_bridge`, replacing the old mocked chat reply
- Qwen and Kimi K3 reclassified as `cloud` with real, documented provider info (DashScope / Moonshot AI)
- Automatic local-model downloads on `/model` selection and `/download`
- `/gui` desktop mode: Qt6 (X11 + Wayland) via PySide6, GTK4 fallback, coded terminal logging on both
- Real `/key` persistence to `~/.hypernix/config.json`
- Dropped OpenClaw-inspired branding

## 0.71.5b1 → b3 — HyperNix T1 API (Shipped)

A controlled HTTP gateway into HyperNix-pip (`hypernix.t1api`), its client
SDK (`hypernix.t1sdk`), and the `waiter` TUI/CLI. Delivered in three
betas; see [T1-API.md](T1-API.md) for the contract and
[Waiter-TUI.md](Waiter-TUI.md) for the client.

- **b1** — core FastAPI server, T1 auth + scoped tokens, the model
  registry, per-key/per-model usage tracking, SQLite, OpenAPI docs, the
  basic `waiter` CLI
- **b2** — module registry and upload/sync, server registry, async jobs,
  event streaming, the quota-cascade routing engine, billing and payment
  tokens, Tailscale/local deployment
- **b3** — production hardening: PostgreSQL, audit logging, mTLS,
  advanced rate limiting, IP allow/blocklists, real remote multi-server
  module transport, the key directory, cost/estimates/forecasts, the
  complete SDK, the full curses TUI, production configuration validation,
  deployment examples, and a security audit checklist

Open, deliberately: module blobs are checksummed but not encrypted at
rest — see [T1-API.md#known-limitation](T1-API.md#known-limitation).

## 0.71.5rc2 — Beta 4 (Shipped)

- `POST /usage/report`: the endpoint that lets a client report the tokens
  it actually spent, so the server's quota cascade advances for clients
  that run inference themselves
- `hyped-pro` T1-key support against a local or remote T1 API server
  (`t1api` vendor, `t1-routed` model, `/t1api` command)
- `hyped-pro` auto-displaying the current public release version
  (`hypernix.system.release`, `/version`)
- `qwen3.8-27b` registry entry
- `hypernix path`: automatic, reversible `PATH` setup for the console
  scripts (`hypernix.system.pathfix`)

## 0.72.0 — T1 v1.0.26.8.0.1 (Shipped)

- The `waiter lmstudio` LM Studio bridge (localhost, LAN with CORS, or
  Tailscale) and `hyped-pro` over it
- **HyperLink**, the iOS client: chat, images, file upload, code, and
  Hugging Face GGUF downloads from a model page merged with a direct
  file link, on and off the local network
- The six-part T1 version scheme (`api.major.year.month.feature.fix`)

## 0.72.1 — T1 v1.0.26.8.1.0 (Shipped)

- **T2 keys**: access levels, admin passwords, SSPKID, and T2S for
  HyperLink. T1-compatible in both directions
- **noodle** — multi-agent orchestration and subagent execution across
  nine providers, inside `hyped-pro`
- **scriptgen** — the training-script GUI
- **steamroller** — the llama.cpp quantiser, and the expanded quant
  format table
- Pascal sm_61 auto-tuning with a hard FP32 fallback
- `waiter -F`, the web TUI live stream, auth undo/redo, and backups

## 0.72.2 — the installer (Shipped)

- **`install-t1.sh`** — interactive setup and installer for the T1 API:
  bind address, key policy, T2 admin password, connection allowlist,
  rate limits, cost accounting, model source, HyperLink, and the
  `waiter` manager TUI, written out as a matching configuration
- `T1_ACCEPT_T1_KEYS`, the other half of the key-family policy — "T2
  only" is now enforced rather than merely recorded
- `T1_KEYMASTER_DIR`, so a deployment's key store lives with the rest of
  its configuration instead of always in `~/.hypernix/keymaster`

## 0.72.3 — T1 v1.0.2026.8.1.1 (Shipped)

- **Payment connections on a T2 key.** A **T2P** key carries a billing
  binding — a provider's customer and method references, a currency, and
  a spend cap — so a key can be issued to someone who pays for their own
  usage instead of drawing on the operator's budget. Access level and
  billing stay separate concerns: a level-9 key with no binding is still
  free to use, and a level-2 key with one is still level 2. A T2P key is
  never an administrator.

  How the design notes landed:

  - The binding references a provider's customer/method token. Card
    details never reach the server: the store refuses anything shaped
    like a 13–19 digit card number at the boundary, and the binding is
    not carried in the credential — keys land in shell history.
  - Spend caps are checked against the *estimated* cost before any model
    work happens, so an over-cap request is refused rather than billed
    and refunded.
  - Revoking a key releases its binding, and rotating one moves the
    binding — with its recorded spend — to the key that replaced it. Both
    run off Keymaster lifecycle hooks, because revocation happens in the
    security layer, which knows nothing about billing.
  - T2C keys stay reserved, as planned.

- **A server can refuse them.** `T1_BILLING_KEY_POLICY` is `allow`
  (default), `deny` — with a `T1_PAYMENT_URL` to point at, for an
  operator who sells access through their own site — or `separate`,
  which requires the payment key in `X-Payment-Key` so the credential
  that identifies a caller and the one that spends money have separate
  lifetimes. Enforced at authentication.

- **The bootstrap key.** A new server mints itself one admin key on first
  start: usable only from that machine, expired after three days, minted
  once. An empty key store plus admin-only key routes was a closed loop,
  and it is why `waiter hyperlink pair` could not run on a fresh install.

- **`hypernix-t1`** — one dependency-free executable for the whole server
  lifecycle: start, stop, kill, restart, status, logs, create, configure,
  test, key, autostart, remove.

- **HyperLink connects.** The advertised port is the one the request
  arrived on; `ts.net` is an ATS exception domain, since Tailscale's
  100.64.0.0/10 is shared address space and not one of the RFC 1918
  ranges `NSAllowsLocalNetworking` exempts; a missing tailnet now names
  its cause. The app takes a T2S key as well as a pairing code.

- **Consent before the agent runs anything** (`HYPERNIX_TOOL_POLICY`) —
  tool calls are parsed out of the model's own reply, so anything that
  can influence that reply could otherwise run shell commands.

- **CI and the public release gate on a live server.** Two jobs each mint
  their own T2 key, drive a real API and a booted iPhone simulator
  against a fake model, and delete every key they made. Nothing publishes
  until both are green.

## 0.72.4 (Shipped)

Cut across `dev1`–`dev13` and `post1`–`post5`; see the
[Changelog](Changelog.md) for the full list.

- **HyperLink says why it cannot reach a server**, and knows which
  machine it is talking to — a pinned identity fingerprint, readable
  from `hypernix-t1 status` so the comparison has two sides
- **`hypernix-t1 launch-script`** with a real job supervisor, trusted
  network mode, and `hypernix-t1 training` — what training is doing on
  this machine, and the controls
- **One way to ask about a GPU**, whoever made it: NVIDIA/CUDA,
  AMD/ROCm, CPU-only behind one interface
- **A llama.cpp that reads sub-bit models**, and the desktop app
  (Studio) that runs them — a GGUF catalogue with no Qt and no
  llama.cpp, a `LocalEngine` over llama.cpp, and `hnx runtime` so other
  applications can use the patched build
- **beta 1 pt 1 — `gather`**, and Neo oven learning the house
  architecture (`hyperNix0x-v2`)
- **beta 1 full — `fuse box`**, GPU thermal management, and a claim
  about it that did not survive being measured

## 0.72.5 — done

- **The release guard stops refusing the version the tree is prepared
  with.** *Done* — `.github/scripts/version_guard.py`. Preparing a
  release means writing the number into the three source files and
  adding the changelog heading under it; the guard refused exactly that
  and told people to invent a `.postN`, which is why 0.72.4 `post1` and
  `post3` shipped with no notes. It now asks the question that actually
  matters — does a `v<version>` tag already name *different* code — and
  only refuses then. Backwards is still refused.
- **`noodle` works inside `hyped-pro`.** *Done* — `/noodle` in the TUI,
  over five bridge verbs. It adopts vendor keys already in the HyperNix
  config, with the environment winning over the stored copy, so a
  session that works in `hyped` is not configured twice.
- **`hyprslug` builds Dflash2 drafts.** *Done*, and Dflash1 as well. All
  nine target precisions are there — `q8`, `int8`, `fp16`, `bf16`,
  `fp32`, `IQ0.5`, `Q6_K`, `Q4_M`, `int2` — and Dflash2 embeds its draft
  in the base GGUF, which is where the speed-up is: one file, one
  download, and a runtime that has never heard of Dflash2 reads the base
  model straight through. Dflash1 is the standalone `--model-draft`
  file; it renumbers its kept blocks from zero and rewrites
  `<arch>.block_count` to match, because a draft claiming 32 blocks and
  shipping 6 loads and then reads past the end of the tensor table.
  `hnx-bundle` puts several quantisations in one GGUF under
  `hnxq.<slug>.`, storing once the tensors every variant left untouched.
- **Three additions to `tvtoppro`.** *Done* — all three:
  1. the spinner module, and `tvtop-older`'s animated "decoding" startup
     text
  2. a module system, so a new stat is a file rather than a patch
  3. a stall detector — `train.log` untouched for over a week means look
     at the highest-usage running Python process instead, and read its
     logs and progress from there
- **`cctvtop`'s remote desktop is broken.** *Done* — it called a held
  port a live session. The probe completes an RFB handshake now, and a
  missing session says which of four reasons it is.
- **The README.** *Done* — a 0.72.5 section, the module reference
  brought up to date, and every console script this release adds listed
  in the CLI reference.
- **The `hypernix` CLI.** *Done* — an unknown subcommand exits 2 on
  stderr with the nearest real command suggested, aliases are accepted,
  and four commands that were reachable only by knowing they existed are
  in the menu.
- **T1 v1.0.26.9.2.3 — accounts and web auth without a T1 API key.**
  *Done*, in all four modes, with every rule below held. Local account creation and browser-based sign-in, served four
  ways: from localhost, over Tailscale, from the operator's own site, or
  from a prebuilt Cloudflare site hosted by the API host. Non-negotiable
  for this one, and each is an existing rule rather than a new one:
  - a public unauthenticated connection must never be able to escalate
    into administrator access
  - the human-readable server name is not an authenticator on its own
  - generated passwords must be securely random, with no predictable
    sequence
  - discovery stays separate from execution: a host-provided application
    is never run without explicit user confirmation and validation
- **Security pass.** *Done* — two real findings. The config file holding
  API keys was world-readable, and is now written `0600` into a `0700`
  directory through a temporary file and `os.replace`. Key
  authentication compared key strings in a loop whose position depended
  on the prefix, a timing oracle for the stored keys; keys are indexed
  by digest and compared with `hmac.compare_digest`, and the measured
  60x spread is flat.
- **Issue templates.** *Done* — one for a bug, one for a feature
  request.
- **A training guide.** *Done* — [Model-Training-Guide](Model-Training-Guide.md),
  written around "which of these do I use, and in what order". Every API
  in it was run against the package; six were wrong.
- **HyperLink: CarPlay, Siri, themes, attachments, renaming.** *Done*.
  CarPlay offers a keyboard only when the car reports it will allow one.
  Four App Intents, none of which open the app. Eight themes, each
  checked for WCAG AA against its own bubbles. The attachment menu
  offers all four ways in rather than two. And a chat can be renamed —
  the server has taken a title on `PATCH` since HyperLink shipped and
  nothing on the phone ever sent one.

## 0.72.6 (Shipped)

Released 2026-09-24, after `0.72.5.post14` to `post17` and three
release candidates; the [Changelog](Changelog.md) has each batch.

- **HyperNix Studio picks a shell.** *Done* (0.72.5.post14): `fish`
  interactive, `bash` for the scripts it writes, persisted and
  validated.
- **`neuron`** *Done* (0.72.5.post14). Trains small networks that act
  rather than predict (supervised, imitation and reinforcement
  learning) for game automation, fast image recognition and robotics.
  `hypernix neuron {demo,clone,dagger,rl,eval}`.
- **A code scanner on a schedule.** *Done* (0.72.5.post14): the
  `auto-scan` workflow, Monday and Thursday. It publishes a `.post`
  release only when four independent checks agree, among them that the
  last release is a plain stable version and that Claude reviewed the
  diff.
- **A flow chart that updates itself.** *Done* (0.72.5.post14): the
  `arch-map` workflow rewrites [Architecture](Architecture.md) in place
  on each public release. It has a second chart for the beta surface,
  with dotted lines for beta and red for declared but not built.
- **A real audio processor.** *Done* (0.72.5.post14):
  `hypernix.audio.processor` does resampling, filters, a gate and
  compressor, silence splitting, noise reduction and a `Pipeline`.
- **More example scripts.** *Done*: `examples/neuron/` and
  `examples/audio/`.
- **`hyped-pro`: git support, full local file editing.** *Done*
  (0.72.5.post16), with `hyped-pro` rebuilt on OpenTUI and the readline
  TUI kept as `hyped-plus`.
- **Basic `hyped`, rebuilt.** *Done*. Out of the box it is a plain
  local-AI TUI, and everything else is a Python *dot*
  (`hypernix.interfaces.dots`).
- **Added along the way:**
  - elements (`hydrogen`, `magnesium`, `carbon`), error codes, and
    tool calling for local models (post16);
  - HyperLink on-device models, web search, memories and the shell
    (post16);
  - v2.1 keys and Rotorvault, conceal mode with 36-hour retention, and
    `/server/info` (post17);
  - `waiter serv` letter groups with `-b -u -k -c -T -Y -S -e` (post17);
  - Siri on App Intents entities and `hnx-t1` (post17).

### 0.72.6.post1 – post3 (Shipped)

- **HyperLink on the web** (post1–post2): the T1 API serves the app's
  web version on port 37965, to this machine and the tailnet.
- **Brewer models run in llama.cpp** (post3): a real GGUF export, the
  `llama` architecture, checked against llama.cpp itself.
- **hyprslug**: `Q8_K`, `INT3`, `FP8`, the `hnx_Q6_H` hybrids, and
  hyperNix0x-v2 input (post3); `hnx convert -P -Q` (post3).
- **Model sync and symlinked models** (post3): `T1_MODEL_SYNC` /
  `hypernix-sync`, symlinked folders found everywhere a model is looked
  for, and linking a model from HyperLink.
- **Release plumbing** (post3): Codacy's crash fixed, nightly builds
  behind `PUBLIC_RELEASE_NIGHTLY`.

## 0.72.7 — planned: Python 3.12 → 3.15

**Status:** implemented on the 0.72.7 branch — what was done, what was
measured and the tested matrix are on [Python 3.12–3.15](Python-3.15.md).

A real migration, not a `python_requires` edit. Final target:

| Python | |
|---|---|
| ≤ 3.11 | unsupported |
| 3.12, 3.13, 3.14 | fully supported |
| 3.15 | fully supported and specifically optimised |

`Requires-Python >=3.12,<3.16`.

- **Audit the whole tree for 3.10-era assumptions** — `asyncio`,
  `typing`, `collections`, `importlib`, `inspect`, `pathlib`,
  `datetime`, `enum`, `dataclasses`, `subprocess`, `threading`,
  `multiprocessing`, exception handling, serialisation, networking,
  filesystem APIs, package discovery, entry points, CLI handling,
  environment variables, native extensions, the C/C++/Rust build
  systems, and CPython-specific behaviour. Drop shims that exist only
  for ≤3.10. No rewrites for their own sake.
- **PEP 810 — explicit lazy imports.** `lazy import` / `lazy from` for
  imports that are expensive or rarely needed: optional integrations,
  large dependencies, CLI-only modules, debugging, model and backend
  integrations, networking, cryptography, developer tooling. Imports
  stay **eager** for security initialisation, plugin registration,
  environment validation, configuration, API registration, import-time
  guarantees, and predictable error reporting. Not every import.
- **PEP 798 — unpacking in comprehensions.** Where it removes a nested
  comprehension, an `itertools.chain`, a temporary list, a manual
  flatten, or an intermediate dict. 3.12–3.14 cannot *parse* it, so it
  lives in 3.15-only modules behind a compatibility boundary, never in
  shared source.
- **PEP 799 — the `profiling` package.** `profiling.tracing` for
  deterministic work, `profiling.sampling` for statistical. A
  `hypernix/_profiling/` abstraction picks the backend by version;
  3.12–3.14 keep what they have and nothing is removed.
- **PEP 831 — frame pointers.** Every native component must inherit
  Python's compiler configuration from `sysconfig` rather than
  overwriting it, so `-fno-omit-frame-pointer` survives. Test stack
  traces, unwinding, profiling, debugging and crash reporting under
  3.15 — and do not force the flags onto a toolchain that does not
  support them.
- **A permanent benchmark suite** across 3.12–3.15: cold and warm
  startup, import time, CLI startup, memory, API and auth
  initialisation, config loading, networking, serialisation, optional
  module loading, native extension performance, profiling overhead. On
  3.15, additionally PEP 810 on vs off, and legacy vs PEP 798. Measure
  before claiming an optimisation helped.
- **Dependencies** — verify each against 3.12/3.13/3.14/3.15, upgrade
  the obsolete, replace the abandoned, check native wheels and source
  builds. No unnecessary pins.
- **Typing, packaging, and a mandatory CI matrix** over all four
  versions: install, wheel, sdist, import, unit, integration, API, CLI,
  auth, networking, serialisation, async, type checking, lint, format,
  and security checks — with the four PEP suites and the benchmarks
  added on 3.15.
- **`tests/python315/`** — `test_pep798.py`, `test_pep799.py`,
  `test_pep810.py`, `test_pep831.py`.
- **Backwards compatibility is not negotiable.** Public APIs, CLI
  commands, API endpoints, authentication, configuration, environment
  variables, serialisation formats, key formats, networking protocols
  and plugin behaviour do not change silently. Anything unavoidable gets
  isolated, a compatibility layer, migration docs and regression tests.
- **A final migration report**, ending in a per-version test matrix.
  Nothing is marked passing until it has actually been run.

## 0.72.8 — model tooling, HyperLink, and hardware support (Planned)

This release consolidates the 0.72.8 alpha and release-candidate requirements in the current prompt. Treat the subsections below as planned until the corresponding implementation and release gates are verified.

- **HyperLink sessions and model selection:** make chats private to the signed-in account, or to the calling API key when accounts are disabled. Sign and validate requests consistently across account/key identity, message timestamp, chat/session identifiers, and fresh request nonces; editing or resending a message must revalidate it. Improve timeout behavior, reject empty/invalid requests, prevent global chat leakage, and make multiple loaded models visible and selectable in both the Models view and chat.
- **T1 runner and API behavior:** add an interactive, refreshable model/backend picker without launching a separate full-screen TUI. Improve launch-script staged arguments and live log following. Introduce the versioned web-search API and configurable, permission-aware search storage; retain the documented compatibility window for the older endpoint before removal. Add bounded system-prompt compaction for models whose context cannot fit the full prompt. Harden the API, close outstanding security findings, publish a security policy, and document licensing changes before release.
- **Downloads, conversion, and quantization:** accept Hugging Face repository URLs and direct GGUF-file URLs, make repository IDs optional where a URL provides the source, and provide explicit output paths. Add `hnx auto` for download → conversion when needed → user-selected quantization. Develop the in-house `hnx-hf` login/client path with safe fallback to `huggingface_hub` when available; do not silently break existing workflows. Add and test the requested `hnx_q6h1` and `hnx_q6h1m` quantization profiles, and validate the non-stock llama.cpp quantization path. Move less-common commands behind `hnx extra`, update help and aliases, and audit the commands for stubs, regressions, or misleading success messages.
- **Model analysis and training:** develop the C4 model-analysis workflow for inspecting weights/layers, measuring candidate components, and restoring or assembling derived models. Keep destructive pruning or synthesis separate from the source model, track provenance, and validate outputs. Integrate its proposed `reap`/`sow` workflow and its specialized Pressure Cooker V5S/AdamW hybrid only behind real tests and reproducible comparisons.
- **HyperLink, integrations, and observability:** support full CommonMark/GFM-style Markdown, dataset context imports, richer code/file access, and compatible MCP integrations. Add the requested browser extension for explicitly authorized web search and browser interaction. Expand `tvtopmax` sorting, section navigation, off-screen panning, and hotkeys; correct RAM and per-core temperature reporting and use measured audio levels for any decibel warning rather than estimates. Expose real power/efficiency telemetry when hardware provides it. Add safer monitor backlight control to `prot`, including a reliable restore/off switch. Any memory creation or tool execution must follow the user's configured consent and privacy rules.
- **Supported hardware and model architecture RC gate:** compare the existing catalogues against verifiable upstream sources and add at least **251 net-new CPU SKU/model entries**, **151 net-new GPU entries**, and **36 net-new model-architecture families** beyond the baseline. Each entry must describe which operations are genuinely supported; recognition alone must not imply successful training or inference. Generate versioned CPU, GPU, and architecture JSON from canonical registries, publish a searchable GitHub Pages matrix, retain sources/revisions, and test generated data and page filters.

**Completion gate:** measure the net-new counts against a saved baseline; test alias and unknown-device handling, CPU-only and mixed-device systems, supported architecture operations, installation/package artifacts, offline catalogue loading, API/session security, and the production Pages build. Do not mark a hardware or architecture combination supported without evidence for the specific operation.

## 0.73.0 — Strife runtime and Optimizer Framework V2 (Planned)

### Alpha 1 — Strife custom GGUF inference runtime

Build Strife as a genuine local inference backend, not merely a launcher around the existing modified llama.cpp binary. Own model lifecycle, scheduling, inference execution, token streaming, cache management, cancellation, and hardware-aware placement through a stable Python API and CLI. Reuse compatible and license-appropriate HyperNix readers and fixtures where sound. Local inference must work without a T1 key, cloud access, or a running T1 server. Publish an honest architecture/quantization/platform/backend support matrix; file recognition or successful loading alone is not proof of working inference.

### Alpha 2 — Optimizer Framework V2 and Pressure Cooker V7

Unify the Pressure Cooker V4, V5, V5+, V5S, V6, and V6V implementations behind a shared framework without erasing their distinct update rules or breaking existing APIs/checkpoints. Deliver the V7 optimizer family (including Mini, Q, S, and M variants), measured auto-selection, optimizer-specific controls and diagnostics, and repeatable benchmarks for throughput, memory, stability, and quality. Include safe precision/state-mode selection, hardware compatibility and fallback behavior, checkpoint compatibility, failure handling, and a test gate. Studio should consume this shared optimizer API rather than reimplementing the optimizer core.

## 0.73.1–0.73.5 — stability and compatibility series (Planned)

Five release steps focused on bug fixes, code review, security, performance, documentation cleanup, cross-platform build failures, and expanded tests rather than a broad feature expansion. The explicitly scoped additions are working T1 API calls from models running in HyperLink, improved AMD compatibility, and a GitHub Actions workflow that uses an available model to refresh the README “What’s new” section after a major release. Keep generated content reviewable and never publish invented release notes or unverified claims.

## 0.73.6 — HGPS and Strife early beta (Planned)

This release is the early-beta integration milestone described by the 0.73.6 prompt sections. It is not complete merely because the scheduler or runner can be launched.

- **HyperNix GPU Process Scheduler (HGPS):** detect actual device architecture, free/total VRAM, memory bandwidth where measurable, backend/driver capabilities, temperature, utilization, power, and host load. Use those observations to recommend and apply bounded batch/micro-batch sizes, gradient accumulation, supported kernels, transfer behavior, allocation, and CPU/GPU placement. Prioritize Pascal/Turing compatibility without assuming newer CUDA features exist on sm_61. Make every selected profile and fallback reason visible.
- **One scheduler, not competing schedulers:** Strife must consume HGPS resource decisions and share its device capability model, tuning cache, and memory planner. Cache entries must be bounded and invalidated by model/workload, device, driver/runtime, and relevant configuration changes. Missing telemetry or unsupported optimizations must fall back to a portable path.
- **Strife early beta:** harden GGUF inspection/loading, tokenizer and tensor compatibility, CPU/GPU partial placement, KV/context cache bounds, batching, streaming, stop conditions, cancellation, timeouts, concurrent session isolation, checkpoint-free inference lifecycle, diagnostics, and packaging. Preserve a local CPU path without a T1 key or cloud dependency.
- **Quantization:** implement and validate `hnx_q7h` and `hnx_q7E` with explicit format identifiers, target bit-rate/component definitions, tensor layout and metadata interpretation, supported runtime kernels, and round-trip/compatibility tests. Unsupported formats must fail clearly rather than be misread as another type.
- **Thermal/resource safety:** extend `prot` for multi-monitor power/backlight control with a guaranteed restore path on normal exit, error, and interruption. Do not leave secondary displays disabled.
- **Release gate:** publish the architecture/quant/backend/platform support matrix; test malformed/truncated GGUFs, unsupported types, memory pressure, OOM/fallback, cancellation, CPU fallback, hardware-specific dispatch, and cache invalidation. Performance goals must be reported as measured, skipped with reasons, or unmet—not presented as results without evidence.

## 0.74.0 — HyperNix0xV3 architecture (Planned)

Design and implement HyperNix0xV3 as a genuinely new architecture rather than a renamed Transformer or a collection of standard layers. Explore two new custom layers for efficient context and adaptive processing, a scalable block architecture, context-aware synchronous optimization, adaptive and thermal-aware control, and hardware-aware execution. Support multiple model sizes using shared configuration and implementation paths. Treat any claimed gains as research hypotheses until baselines, numerical checks, training stability, hardware compatibility and repeatable experiments support them; record failed experiments as well as successful ones.

## 0.74.1 — HyperLink local runtime and context telemetry (Planned)

Add the legacy **Origins** theme and fix hardware reporting so supported telemetry can identify CPU, RAM, GPU, and power use rather than showing partial or guessed readings. Report hourly power consumption and efficiency only where the system exposes enough information, clearly labeling estimates. Integrate a usable local model runner based on the modified llama.cpp runtime. Add a three-ring chat indicator for context and remaining usage, with clear color thresholds and automatic context compaction at the specified 96% full point. Add an actionable example that trains a small Brewer model using the latest supported Pressure Cooker, Neo Oven, and synchronous-training components.

## 0.74.2 — experiment ledger and reproducible workbench (Planned)

Make training, evaluation, inference benchmarking, and quantization runs reproducible and recoverable. Add versioned experiment manifests and records, a CLI workflow, safe checkpoint lifecycle/recovery, comparable metrics, and artifact/storage safeguards. Integrate monitoring without coupling experiment records to a single UI. Validate privacy, cleanup, interrupted-run recovery, deterministic metadata, and honest before/after comparisons.

## 0.74.3 — HyperNix Studio, local runtime, and agent tools (Planned)

Deliver a coherent Studio experience that can discover, load, inspect, and run local models without a T1 key or hosted server. Integrate Strife and the shared optimizer framework instead of duplicating them. Add observable local tool calling, parallel calls where safe, capability detection, built-in tools, agent profiles/tool packs, environment snapshots, project diagnostics, patch/apply and test-and-fix loops, shell support for Bash/Fish/PowerShell, and safe streaming/cancellation. Preserve permissions and user approval at tool boundaries; isolate execution, constrain file/process/network access, and make failures recoverable. Include model/context/tool-result management, caching, training dashboard integration, configuration and CLI support, accessible high-quality UI, backward compatibility, performance tests and complete documentation.

## 0.75.0 — HyperNix0xV4 cognitive architecture (Planned)

Research and build a new model architecture rather than simply modifying V3 or renaming conventional Transformer blocks. The proposed design explores perception, curiosity, parallel internal processing, hypothesis formation, synchronization, memory, learning and perspective formation. Keep internal computation distinct from user-facing explanations; do not expose or require a traditional chain-of-thought transcript. Provide configurable model sizes and validate the architecture through reproducible training/evaluation, memory and compute measurements, and comparisons with documented baselines before claiming advantages.

## 0.76.0 — scope not yet specified

The current prompt reserves this version number but does not define deliverables.

## 0.76.1 — hyperNix.25 and safer shutdown UX (Planned)

Add support for `hyperNix.25`. For the wall-clock/shutdown script, show a clear five-to-one countdown and a final shutdown notice, with a documented **Shift+C** cancellation path. Make cancellation reliable during the entire countdown and test that cancellation prevents shutdown.

## 0.80.0 — scope not yet specified

The current prompt reserves this version number but does not define deliverables; no additional scope is inferred here.

## 0.81.0 — unified in-house runtime (Planned)

Consolidate the runtime into the single `hypernix` Python package; do not split the design into separately installed `hypernix-torch`, `hypernix-poor`, `hypernix-hardware`, or `hypernix-cuda` packages. Build the proposed HyperNix Torch interface around the specified PyTorch 2.14 codebase where required, with internal hardware, runtime, CUDA, CPU, logging, terminal, Unicode and compatibility modules. Add automatic backend/runtime selection, explicit compatibility handling, native build and packaging support, and a versioned migration path. Validate the supported CPU/GPU combinations and package once-install behavior with real CI, build, install, runtime and performance tests; do not promise universal CUDA or hardware compatibility by catalog entry alone.

## 0.81.1 — HyperLink Android, navigation, providers, and HyperNixUP (Planned)

Deliver the complete HyperLink and developer-experience scope in `.prompts/.v1/main.md`, with implementation and release evidence rather than placeholder screens or documentation-only claims.

- **HyperNixUP:** define a versioned Markdown extension with a formal grammar, precedence, escaping, nesting, parser errors, examples, and safe fallback rendering. Use one parser/renderer across supported HyperLink views and system-prompt workflows; test malformed input, accessibility, reveal/redaction behavior, and security boundaries.
- **HyperLink Android and navigation:** build the native Android client; improve swipe/gesture navigation, accessible alternatives, Code tab workflows, model browsing and Hugging Face search, and provider/API setup. Fix effort-level mapping across prompts and providers so displayed settings match the provider request actually sent.
- **Providers and local/cloud workflows:** add the specified Anthropic, Gemini, GPT, and Qwen adapters with mocked contract/streaming tests, secret redaction, timeouts, retries, cancellation, and capability checks. Keep local inference independent of a T1 key.
- **Authorization and T1:** enforce administrator, delegated-admin, approved-device, and ordinary-user permissions server-side. Tailscale is an additional validated trust signal, never a substitute for authentication or role checks. Reproduce and fix the T1 API version auto-update failure with a regression test and a safe, user-approved update path.
- **Repository wiki redesign:** preserve useful existing pages, historical notes, and inbound links while adding consistent navigation, landing/index pages, cross-links, version labels, and robust link validation. Correct stale commands, broken links, duplicated guidance, and contradictions with source behavior. The new [Examples module encyclopedia](Examples.md) must have a linked table of contents and explain every Python module in the source tree—not only the names in `MODULE_CATEGORIES`—with its path, role, public/internal status, and links to detailed references. Keep its inventory synchronized with the actual source tree; never infer behavior from a filename or mark planned code as implemented.
- **Runnable examples and cookbook:** add organized, tested examples for text generation/streaming, provider configuration, T1 SDK/key scopes, HyperLink pairing, model discovery/download, CPU-only training, datasets, checkpoints, evaluation, quantization, benchmarking, image/file inputs, cancellation/retry/timeouts, approved tool workflows, HyperNixUP, permissions, Tailscale deployment, diagnostics, and recovery. Each example needs prerequisites, expected output, cleanup, safe defaults, and a mocked/tiny-fixture path that does not require paid APIs, secrets, or a GPU.
- **In-app Docs:** build a separate PyTorch-style Docs section—not the wiki menu—with search, category navigation, breadcrumbs, page TOCs, version labels, code snippets, tutorials, API/CLI references, offline behavior where practical, and accessible reading layouts. Do not load arbitrary remote HTML into privileged views.
- **Release integration:** keep docs, examples, generated data, package metadata, and release notes aligned with verified implementation. Test navigation/deep links, offline behavior, link checks, role boundaries, parser behavior, provider contracts, Android builds, and all existing workflows.

## 0.82.0 — major performance and GPU-efficiency release (Planned)

This is a measurement-led implementation release. Profiling reports and benchmark scripts are deliverables, but they do not count as performance improvements by themselves.

- **Baselines and profiling:** establish repeatable cold/warm baselines for package and CLI startup/imports, model loading, prompt prefill, token decode, training steps, optimizer-only steps, data preprocessing, conversion, quantization, and local serving. Profile real hot paths and preserve raw machine-readable results plus readable reports.
- **CPU/runtime:** reduce avoidable imports, Python overhead, repeated discovery/parsing, serialization and copies; improve bounded data loading, preprocessing, memory use, and thread behavior without oversubscribing CPU/BLAS/PyTorch workers or silently changing OS settings.
- **GPU:** target measured synchronization, allocation, transfer, launch, batching, placement, and kernel bottlenecks. Add fused/foreach operations, non-blocking transfers, reusable buffers, compile/graph paths, attention/KV-cache improvements, and backend-specific kernels only where supported and benchmarked. Keep portable fallbacks and Pascal-safe behavior where the installed stack allows it.
- **Adaptive resource planning:** provide a single planner that accounts for weights, KV cache, activations, optimizer state, runtime overhead, free memory, context, batch shape, and backend capability. Show estimates before expensive work, compare them with measured peaks, expose safe overrides, and make OOM recovery actionable without silently truncating context or changing precision/objectives.
- **Training and optimizers:** benchmark AdamW and each supported Pressure Cooker generation (V5/V5+, V5S, V6/V6V, V7 family) in isolation and end-to-end. Preserve update semantics, checkpoint formats, numerical behavior, gradient/scaler handling, reproducibility, and convergence; optimizer-only speed is not sufficient if full training regresses.
- **Quantization and inference:** reduce intermediate allocations and conversion buffers; improve streaming/memory-mapped paths and supported quantization kernels. Measure load time, time-to-first-token, prefill/decode throughput, concurrency, KV-cache memory, and placement. Validate quality/logit/loss behavior against trusted baselines and never reuse compiled/cache artifacts across incompatible model, ABI, quant, or device combinations.
- **Safe dispatch:** select paths using runtime feature detection (device, driver/runtime, backend, memory, kernel availability), report the selected path and fallback reason, and keep tuning/compilation caches bounded and correctly invalidated.
- **Release gates:** target at least **10% median improvement in two or more representative end-to-end workloads**, or at least **10% lower peak RAM/VRAM without more than 3% throughput regression** for a representative workload. If the target is not met safely, publish profiler evidence and the limitation instead of inventing a result.
- **Evidence and documentation:** record commit/version, model revision/hash, quantization, input/output/context settings, backend/driver, device, peak RAM/VRAM, latency distributions, throughput, warmups/repeats, and measured-vs-estimated telemetry. Include uncertainty/spread, correctness checks, regression tests, CPU fallback, hardware skips with reasons, raw artifacts, and a final report listing each change and its measured effect.

## 1.xx.0 — the T2 API

The T2 API itself does not release until 1.xx.0. At that point it supports T2 and T2S keys only, plus T2C once its key derivation is resolved. Until then, the existing T1 API recognizes and validates T2 keys; the standalone T2 API is not exposed.

---

See [Changelog](Changelog.md) for shipped history.
