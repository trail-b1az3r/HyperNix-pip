# Roadmap

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

- The `waiter bridge` LM Studio bridge (localhost, LAN with CORS, or
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

## 0.72.4 — beta 1 (Shipped)

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

## 0.72.5 — planned

- **The release guard stops refusing the version the tree is prepared
  with.** *Done* — `.github/scripts/version_guard.py`. Preparing a
  release means writing the number into the three source files and
  adding the changelog heading under it; the guard refused exactly that
  and told people to invent a `.postN`, which is why 0.72.4 `post1` and
  `post3` shipped with no notes. It now asks the question that actually
  matters — does a `v<version>` tag already name *different* code — and
  only refuses then. Backwards is still refused.
- **`noodle` works inside `hyped-pro`.** The autonomous multi-agent
  executor has its own entry point and does not yet run as a mode of the
  TUI it was written for.
- **`hyprslug` builds Dflash2 drafts.** `hypernix.quant.dflash2` can
  already `attach` a draft to a base GGUF; hyprslug cannot yet *produce*
  the draft. Target precisions: `q8`, `int8`, `fp16`, `bf16`, `fp32`,
  `IQ0.5`, `Q6_K`, `Q4_M` — then embedding the result in a single GGUF,
  which is where the speed-up is: one file, one download, and a runtime
  that has never heard of Dflash2 still reads the base model straight
  through.
- **Three additions to `tvtoppro`:**
  1. the spinner module, and `tvtop-older`'s animated "decoding" startup
     text
  2. a module system, so a new stat is a file rather than a patch
  3. a stall detector — `train.log` untouched for over a week means look
     at the highest-usage running Python process instead, and read its
     logs and progress from there
- **`cctvtop`'s remote desktop is broken.** Fix it.
- **The README.** Update or replace; several sections describe a version
  that is several releases behind.
- **The `hypernix` CLI.** General improvement pass.
- **T1 v1.0.26.9.2.2 (or .2.3) — accounts and web auth without a T1 API
  key.** Local account creation and browser-based sign-in, served four
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
- **Security pass.** Fix what the audit finds.

## 0.72.6 — planned

- **HyperNix Studio picks a shell.** `fish` by default, `bash` for
  coding.
- **`neuron`** — a module for training small neural networks for
  concrete jobs rather than language: game automation, fast image
  analysis and recognition, robotics, and similar.
- **A code scanner on a schedule.** Twice a week: Python bug hunt, the
  API exercised across two jobs, a security pass, and basic maintenance
  applied automatically where it is safe to. Where the run finds enough
  to be worth shipping, it cuts a public `.postN` release — using the
  Claude API on the owner's account (Sonnet 5+ at high or extra effort,
  the newest Opus at medium or high as an advisor). It releases only
  when the most recent release is not a prerelease, a `.dev` commit, or
  a beta.
- **A flow chart that updates itself.** A GitHub Action on public
  release that *updates* the existing chart rather than adding another:
  major dependencies, features, and which module links to what. Beta
  features go in a second, smaller chart below it — solid lines for
  shipped, dotted for added in the beta, red for not yet built.
- **A real audio processor.** The current one is a stub.
- **More example scripts.**
- **`hyped-pro`: git support, full local file editing**, and more.
- **Basic `hyped`, rebuilt.** No custom configuration in the box — out
  of the box it is a plain local-AI TUI with T1 support and a Hugging
  Face key for searching and downloading GGUF models. Underneath,
  everything is configurable, and every config is a **dot**: a file
  written in Python (or Lua — pick one and commit to it).

## 0.72.7 — planned: Python 3.12 → 3.15

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

## 0.73.0 — planned

HyperNix Studio:

- full local model support with **no T1 key and no hosted server**
- both `bash` and `fish`
- substantially more tool-calling
- and more

## 0.73.1 – 0.73.5 — planned: five releases that add nothing

Stability, bug fixes, code review and revision, safety, and performance.
No new features, with one exception:

- **a CLI for clearing out old training checkpoints and datasets**, to
  get storage back
- improved AMD compatibility
- a better README — outdated sections updated or removed

## 0.73.6 — planned: the HyperNix GPU Process Scheduler (HGPS)

Aimed squarely at older NVIDIA hardware: GTX 1080, 1080 Ti, RTX 2080 and
the rest of the Pascal/Turing generation.

- **Detect, then decide.** Architecture, VRAM capacity and free VRAM,
  CUDA capability, memory bandwidth, thermal state, utilisation and
  system load — feeding batch size, micro-batch size, gradient
  accumulation, kernel configuration, CPU↔GPU transfer behaviour and
  allocation strategy. Optimise for *sustained* throughput, not a
  benchmark burst.
- **Legacy execution mode.** Pascal gets FP16/FP32 CUDA, fused kernels,
  memory reuse, gradient checkpointing, pinned host memory and async
  transfers. Turing gets Tensor Cores *when they measurably help*, with
  a CUDA fallback. Benchmark the available paths at startup and pick the
  fastest stable one rather than assuming. Profiles: `Pascal-Legacy`,
  `Turing-Legacy`, `Auto` (default).
- **VRAM management with a margin.** Watch usage and cut micro-batch
  size or turn on checkpointing *before* an OOM. Keep weights, optimizer
  state and hot tensors resident; avoid fragmentation. For models that
  do not fit, mixed CPU/GPU with async prefetch and layer/tensor
  offload — and never move the same tensor back and forth repeatedly.
- **A real-time controller** over utilisation, temperature, power,
  memory, kernel time, dataloader latency, PCIe latency and
  tokens or samples per second, with thermal-throttle protection and
  telemetry that names the bottleneck: compute-, memory-, CPU- or
  PCIe-bound, plus the estimated optimal batch size.
- **Cheap, and cached.** Minimal overhead, graceful degradation when a
  feature is missing, never slower merely because a newer optimisation
  is unavailable, and per-GPU tuning profiles so a configuration
  discovered once is reused on the next run.

## 1.xx.0 — the T2 API

The T2 API itself does not release until 1.xx.0. At that point the T2
API supports T2 and T2S keys only, plus T2C once its key derivation is
resolved. Until then T2 keys are recognised and validated by the T1
API — which is what 0.72.1 shipped — and the T2 API is not exposed.

---

See [Changelog](Changelog.md) for shipped history.
