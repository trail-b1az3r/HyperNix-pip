# CLI reference

`hypernix` ships a console script, `hypernix` (also aliased to `hnx` for brevity), which dispatches
to **34 subcommands** plus the `all` pipeline as the default (see `_SUBCOMMANDS`
in `src/hypernix/cli.py`). Most subcommands wrap one library function, so
they're easy to script; a few (`brew`, `cli`, `net`, `gkey`) are themselves
small sub-CLIs with their own sub-subcommands.

```
usage: hypernix <subcommand> [options]  (or: hnx <subcommand> [options])

Core pipeline:
  all                    download -> convert -> [quantize]  (default)
  download               fetch a HuggingFace snapshot
  convert                produce fp32 / fp16 GGUF from a snapshot
  quantize               run llama-quantize on an fp16 / fp32 GGUF
  verify                 read-validate a GGUF and print headers
  info                   package + optional GGUF header summary
  upload                 push files to a HuggingFace repo
  doctor                 environment diagnostic (pass --fix to install deps)
  fetch-llama-quantize   pre-seed the llama-quantize cache

Training & inference:
  train                  init / expand / run training utilities
  generate               sample text from a local snapshot
  oven                   code-generation wrapper
  chat                   interactive REPL against any supported model
  brew                   one-shot recipe pipeline, or the brewer architecture-builder sub-CLI
  camo / camouflage      RLHF/RLAF alignment scaffolding
  fizzle / fiz           architecture fuse/merge module
  stml                   VRAM -> trainable-context-length calculator

Assistants & dashboards:
  cli                    interactive TUI/CLI menu over all of the above
  pipeline               ASR -> LLM -> TTS pipeline (see note below)
  assistant              interactive assistant REPL (see note below)
  vera                   Vera assistant CLI
  tvtop                  classic live training dashboard
  cctvtop                Python training dashboard w/ hardware metrics + optional VNC
  map                    steampunk schematic TUI for model/training state

Data & infra:
  scavenger              search + pull HF datasets under storage/quality budgets
  gather                 crawl a site into a corpus (see Gather.md)
  fusebox                GPU thermal governor for a run (see FuseBox.md)
  runtime                use the HyperNix llama.cpp from LM Studio etc.
  websearch              non-API web search utility
  net                    Tailscale mesh connect / export / log tailing
  gkey                   API key issuance, scoping, revocation (Gatekeeper + Keymaster)
  config                 HyperNix configuration management
  prot / protect         hardware health monitoring and protection
  wiki                   this documentation, read straight from installed source

Shortcuts:
  --auto-oven            download default snapshot + run code completion
                         (equivalent to `hypernix oven --auto ...`)

Run `hypernix <subcommand> --help` for per-command flags.
```

> **Known limitations — read before relying on these:** `hypernix pipeline`
> and `hypernix assistant` both accept an LLM-selection flag (`--llm`,
> `--model`) but currently **ignore it**. Internally they call a hardcoded
> stub responder that returns a small set of canned/simulated replies —
> not real inference from any model you point them at. The ASR and TTS
> stages, and every other subcommand on this page, are real. See the
> `pipeline` and `assistant` sections below for specifics.

## Additional Companion Scripts

Apart from the main `hypernix` / `hnx` entry points, the package installs these
companion scripts (see `[project.scripts]` in `pyproject.toml` for the
authoritative list):

* `hnx` — Short alias for `hypernix`.
* `hypernix-quantize` — Alias for `hypernix` (historical name, predates the `quantize` subcommand naming).
* `tvtop` — Classic TUI training dashboard.
* `tvtop-old` / `tvtop-plus-plus` / `tvtoppp` — Older TUI dashboard generations, kept for compatibility.
* `tvtop-older` — Original single-panel dashboard.
* `cctvtop` — Python training dashboard with hardware metrics and optional VNC (same as `hypernix cctvtop`).
* `hyped` — Configurable high-quality chat TUI with a model/persona configurator.
* `hyped+` / `hyped-pro` — Standalone Node.js TUI backed by a real Python dispatch layer (cloud APIs, auto-downloaded local models, Gatekeeper routing), slash autocompletion, price estimator, system prompt compactor, and a `/gui` desktop mode (Qt6 X11/Wayland, GTK4 fallback).
* `hyped-pro-gui` — Launch the hyped-pro desktop GUI directly, without the TUI.
* `multilama` — Unified interface over multiple llama.cpp variants (vanilla, ik_llama.cpp, PrismML fork, KoboldCpp).
* `eth` — Ethanol GPU overclock and VRAM helper. Refuses to apply changes without `--confirm`.
* `gkey` — Same as `hypernix gkey`, as its own executable.
* `hnx-map` — Same as `hypernix map`, as its own executable.
* `waiter` — The official [T1 API client](Waiter-TUI.md) CLI/TUI. Client-side only: stdlib `urllib`, no `[t1api]` extra needed.
* `hypernix-t1` — Start, stop, configure, test and autostart a local T1 API server (see below). A shell program, so it ships via `script-files` rather than `[project.scripts]`.

## `all` — the classic pipeline

```bash
hypernix --repo-id ray0rf1re/hyper-nix.1 --output-dir ./out \
    --quants fp32 fp16 q8_0 q6_k q4_k_m

hypernix --model-dir ./local-snapshot --output-dir ./out   # skip download
```

Full flag set:

| Flag | Default | What |
|---|---|---|
| `--repo-id REPO` | `ray0rf1re/hyper-nix.1` | HF repo id |
| `--revision REF` | latest | git ref / tag |
| `--model-dir PATH` | — | reuse a local snapshot |
| `--output-dir PATH` | `./hypernix-gguf` | where GGUFs land |
| `--name NAME` | `HyperNix` | header display name |
| `--arch NAME` | `hypernix` | GGUF `general.architecture` |
| `--quants [Q ...]` | `fp32 fp16` | any mix from the [quant aliases](Quantization.md#quantize_gguf) |
| `--n-head N` | from config | override head count |
| `--context-length N` | from config | override context length |
| `--threads N` | `cpu_count//2` | llama-quantize threads |
| `--llama-quantize PATH` | auto | explicit binary path |
| `--no-auto-fetch` | false | disable the GitHub-release fallback |
| `--auto` | false | walk back releases + PyPI fallback |
| `--keep-intermediate` | false | keep the fp16 GGUF |
| `--token TOKEN` | `$HF_TOKEN` | for gated repos / uploads |
| `--upload-to REPO` | — | push produced GGUFs |
| `--upload-private` | false | mark the target repo private |

## `download`

```bash
hypernix download --repo-id nix2.5                 # short name
hypernix download --repo-id Qwen/Qwen3.5-4B --token $HF_TOKEN
```

| Flag | Default |
|---|---|
| `--repo-id` | `ray0rf1re/hyper-nix.1` |
| `--revision` | latest |
| `--local-dir PATH` | HF cache |
| `--cache-dir PATH` | `~/.cache/huggingface/hub` |
| `--token` | `$HF_TOKEN` |
| `--quiet` | false |
| `--no-verify` | false |

Prints the local snapshot path to stdout.

## `convert`

```bash
hypernix convert --model-dir ./snapshot --output ./out-fp16.gguf --dtype fp16
```

| Flag | Default |
|---|---|
| `--model-dir PATH` | required |
| `--output PATH` | required |
| `--dtype` | `fp16` (`fp16` / `f16` / `fp32` / `f32`) |
| `--arch NAME` | `hypernix` |
| `--name NAME` | `HyperNix` |
| `--n-head N` | from config |
| `--context-length N` | from config |

## `quantize`

```bash
hypernix quantize --source ./out-fp16.gguf --output ./out-q4.gguf --type q4_k_m
```

| Flag | Default |
|---|---|
| `--source PATH` | required |
| `--output PATH` | required |
| `--type Q` | required (see [quant aliases](Quantization.md#quantize_gguf)) |
| `--threads N` | `cpu_count//2` |
| `--llama-quantize PATH` | auto |
| `--no-auto-fetch` | false |
| `--auto` | false (walks back releases + PyPI fallback) |
| `-hnx` | false — quantise with hyprslug, never touch llama.cpp |
| `--imatrix PATH` | none (hyprslug tiers) |
| `--quantize-embeddings` / `--no-…` | tier's default |
| `--quantize-output` / `--no-…` | tier's default |

`-hnx` is what the sub-bit tiers need: `llama-quantize` has never heard
of `IQ0.9_L`, `IQ0.75_M` or `IQ0.5_XXXL`, so nothing is looked for,
downloaded or built.

```bash
hypernix quantize --source model.f32.gguf --output model.iq05.gguf \
    --type IQ0.5_XXXL -hnx --quantize-embeddings --quantize-output
```

Those last two matter more than they look. A sub-bit tier leaves
`token_embd` and the output head in float by default, because at half a
bit the embedding table *is* the model — but on a 7B an untouched F32
table and head are then most of the resulting file, and a tier called
`IQ0.5_XXXL` lands nearer 1.7 bits per weight than 0.5. Pass both to get
the number the tier is named for; leave them off to keep the quality the
default is protecting. [HnxRun](HnxRun.md) runs either.

## `verify`

```bash
hypernix verify ./out-q4_k_m.gguf            # header summary
hypernix verify ./out-q4_k_m.gguf --tensors  # + tensor list
```

Exit code 0 on successful parse, non-zero otherwise.

## `info`

```bash
hypernix info                   # version + python + torch
hypernix info --gguf a.gguf     # + full verify output
```

## `upload`

```bash
hypernix upload --repo-id ray0rf1re/HyperNix.1-gguf a.gguf b.gguf c.gguf
```

| Flag | Default |
|---|---|
| `--repo-id` | `ray0rf1re/HyperNix.1-gguf` |
| `--token` | `$HF_TOKEN` |
| `--private` | false |
| `--commit-message MSG` | `"Add HyperNix GGUF quantizations"` |
| positional: file list | required |

## `doctor`

```bash
hypernix doctor              # report
hypernix doctor --fix        # install missing runtime deps
```

Reports Python / torch / numpy / safetensors / huggingface-hub / gguf /
tqdm / sentencepiece versions, OS + distro, and the resolved
`llama-quantize` path. `--fix` routes through `hypernix.deps.ensure`
to install any runtime deps that aren't pinned by the wheel
(`torch` is never touched — users control their CUDA flavor).

## `fetch-llama-quantize`

Pre-seed the `~/.cache/hypernix/bin/` cache so the first `quantize`
call is fast:

```bash
hypernix fetch-llama-quantize
hypernix fetch-llama-quantize --force          # redownload
hypernix fetch-llama-quantize --auto           # include PyPI fallback
hypernix fetch-llama-quantize --search-releases 20
```

## `train`

Sub-subcommands: `init`, `expand`, `run`. See [Training.md](Training.md)
for the mental model. Full flag reference:

### `train init`

```bash
hypernix train init \
    --out-dir ./fresh --tokenizer-source ./hyper-nix-v1 \
    --vocab-size 32000 --hidden-size 1024 --intermediate-size 4096 \
    --num-hidden-layers 16 --num-attention-heads 16 \
    --max-position-embeddings 2048 --rope-theta 10000.0 \
    --seed 0
```

### `train expand`

```bash
hypernix train expand \
    --src-dir ./hyper-nix-v1 --dst-dir ./hyper-nix-v2 \
    --hidden-size 1536 --intermediate-size 6144 \
    --num-hidden-layers 24 --num-attention-heads 24 \
    --init-std 0.02 --seed 0
```

### `train run`

```bash
hypernix train run \
    --model-dir ./hyper-nix-v2 --dataset ./corpus.txt \
    --out-dir ./trained \
    --steps 1000 --batch-size 2 --context-length 512 \
    --lr 3e-4 --weight-decay 0.1 --grad-clip 1.0 \
    --dtype float32 --log-every 10 --save-every 500 --seed 0
```

### `train run` — VRAM flags

```bash
hypernix train run --model-dir ./m --dataset ./d.txt --out-dir ./out \
    --gradient-checkpointing --checkpoint-every 2 --tune-allocator
```

| Flag | |
|---|---|
| `--gradient-checkpointing` | Recompute activations in backward instead of storing them: ~30% more compute for most of the activation memory, which is what a long-context run actually runs out of. |
| `--checkpoint-every N` | Checkpoint every Nth block. `1` for all, `2` for half the saving at half the extra compute. |
| `--fuse-optimizer` | Step and free each gradient as it is produced, instead of holding a full copy of them between `backward` and `step`. Needs `--grad-clip 0` — a global norm cannot be computed one gradient at a time, and the combination is refused before the checkpoint loads rather than after. |
| `--tune-allocator` | `expandable_segments` on the CUDA allocator, so a long run with varying sequence lengths fragments less. |

Full detail, plus the parts with no CLI surface (optimizer-state offload,
peak measurement): [VRAM](VRAM.md).

## `generate`

```bash
hypernix generate --model-dir ./snapshot --prompt "def fib(n):" \
    --max-new-tokens 128 --temperature 0.2 --top-k 40 --top-p 0.95 \
    --seed 0 --device cuda --dtype float16
```

Small sampler, no chat template, no stop-sequence trimming. For
code-oriented generation use `oven`; for conversations use `chat`.

`--model-dir` also takes a `.gguf` file, not just a snapshot directory —
GGUF is the format this package spends most of its time producing.
Upstream quant types go to llama.cpp; the sub-bit tiers go to
[HnxRun](HnxRun.md), because no llama.cpp can read them.

```bash
hypernix generate --model-dir model.iq05.gguf --prompt "hello" \
    --cache-bytes 2G
```

`--hnx-device` chooses where a sub-bit model runs: `auto`, `cpu`, `cuda`,
`cuda:1`, `mps`, `xpu`. On an accelerator the packed bytes are uploaded
once and decoded there. `auto` falls back to the CPU; a named device that
cannot run says why and what to install.

`--cache-bytes` is the memory-for-speed dial for those tiers only. They
hold their weights packed and unpack inside every matmul, which costs
about 4× float32 in time; this spends memory to buy some of it back,
pinning the largest tensors first. Sizes are human (`512M`, `2G`, or a
plain byte count); a size it cannot read is refused rather than quietly
becoming zero. It has no effect on a model llama.cpp runs, which has its
own answer to the same question.

## `devices`

```bash
hypernix devices
hypernix devices --json
```

What can and cannot run a sub-bit model here, with the reason for each
refusal. The interesting answer is usually a refusal: a GTX 1080 is
`sm_61`, recent torch wheels build for `sm_75` and up,
`torch.cuda.is_available()` returns True anyway, and the failure arrives
at the first kernel launch worded as though the driver were broken. See
[Devices](Devices.md).

## `oven`

Code-generation wrapper — preheat + `complete` or `fill` in one call.

```bash
# Prompt completion.
hypernix oven --repo-id nano-mini --prompt "def fib(n):" \
    --max-new-tokens 128 --temperature 0.2 --top-k 40 --top-p 0.95

# Fill-in-the-middle.
hypernix oven --model-dir ./snapshot \
    --fill-prefix "def add(a, b):\n    return " \
    --fill-suffix "\n\nprint(add(1, 2))" \
    --max-new-tokens 32

# Just download + save the self-contained .pt bundle.
hypernix oven --repo-id nix2.5 --save-pt ./nix.pt
```

Shortcut: `hypernix --auto-oven --prompt "..."` == `hypernix oven --auto --prompt "..."`.

## `chat`

```bash
# Single-turn (scripting).
hypernix chat --repo-id nix2.5 --message "Capital of France?"

# Interactive REPL.
hypernix chat --repo-id gemma-4-e4b --system "You are terse."
```

Same flags as `oven` minus the FIM options, plus `--system` and
`--message`. `--model-dir` takes a `.gguf` here too, `--cache-bytes`
included, and the model is loaded once and held for the whole session
rather than per turn.

```bash
hypernix chat --model-dir model.iq05.gguf --cache-bytes 2G
```

## `brew`

```bash
# One-shot pipeline from a recipe file:
hypernix brew recipe.json --set output_dir=./out

# Architecture-builder sub-CLI (brewer):
hypernix brew new --preset small --out-dir ./arch
hypernix brew list
```

A path ending in `.json` is treated as an `instant_pot` recipe and run
end-to-end (download/convert/quantize/etc. as described in the recipe).
Anything else dispatches into the `brewer` architecture-builder sub-CLI,
with presets from `33m`/`micro`/`small`/`medium`/`large` (GPU) through
`cpu-nano`/`cpu-tiny`/`cpu-small` (CPU-only).

## `pipeline` and `assistant` — current limitations

Both of these are real, runnable REPLs — but the LLM stage in each is
currently a **hardcoded stub**, not a live model:

```bash
hypernix pipeline --audio recording.wav --asr whisper --llm nix2.5 --tts piper
hypernix assistant --model nix2.5
```

- `pipeline`'s `--llm` value is accepted but never read; the "LLM" stage
  always calls an internal `SimpleLLM` class that returns a short
  simulated response string.
- `assistant`'s `--model` value is accepted but never read; replies come
  from a small hardcoded demo responder ("I'm a demo assistant — integrate
  a real LLM for full responses!").
- The ASR and TTS stages of `pipeline`, and everything else in this
  document, call real code paths.

If you need real generation today, use `chat`, `generate`, or `oven`
directly against a downloaded model — those are fully wired up.

## `cli`

```bash
hypernix cli            # rich TUI menu
hypernix cli --simple   # plain-text menu, no `rich` dependency
```

Interactive menu covering downloads, conversion, quantization, training,
and evaluation without needing to remember individual subcommand flags.

## `tvtop` / `cctvtop`

```bash
hypernix tvtop      # classic single-panel dashboard, tails a training log
hypernix cctvtop     # richer dashboard with hardware telemetry, optional VNC
```

Run `hypernix tvtop --help` / `hypernix cctvtop --help` for the full flag
set — both are primarily driven by pointing them at a training log file.

## `camo` / `camouflage`

```bash
hypernix camo -Lmodel ./snapshot -Ai -M gpt-4o-mini -Sp "Be terse." -s 200
```

RLHF/RLAF alignment scaffolding. `-Ai` enables AI-assisted evaluation
(scoring generations with a separate evaluator model via `-M`) instead of
manual scoring; `-Sp` sets the system prompt used during rollout, `-s` the
number of steps.

## `fizzle` / `fiz`

```bash
hypernix fizzle --help
```

The Fuzed Architecture module — fuses/merges model weights and LoRA
adapters. See `hypernix fizzle --help` for the current flag set.

## `stml`

```bash
hypernix stml --vram 8 --params 3 --precision fp16 --batch-size 1
```

Calculates a trainable context length given available VRAM (GB), model
size (billions of params), batch size, and precision. Useful before
`train run` to avoid an OOM crash mid-run.

## `scavenger`

```bash
hypernix scavenger --keywords "code,python" --max-storage 20 \
    --min-likes 5 --data-type parquet --download some-org/some-dataset
```

Searches HuggingFace datasets and pulls them under a storage/quality
budget (`--max-storage` in GB, `--min-likes`, `--max-age`, etc.) instead
of downloading everything that matches a keyword.

## `gather`

```bash
hnx gather -W https://example.org -Q 2 -T 4 -p 1.5 -f jsonl -o ./corpus
hnx gather -L "a.org,b.org" -f parquet -C --xz -O corpus-2026-09
hnx gather probe -W https://example.org -u 8      # measure a host's limit
hnx gather formats --json                         # what this build supports
```

Crawls a site and writes it as training data. `-W` a site or `-L` a
comma-separated list, `-T` threads, `-Q` depth, `-p` the pause between
requests to a host, `-f` the format (`html`, `html-full`,
`html-full-wimages`, `text`, `jsonl`, `parquet`, `js`), `-o` where,
`-O` the file header or archive name, `-C` with `--xz` / `--7z` /
`--zip` / `--gz` to compress, and `-U` to upload to a GitHub or Hugging
Face repo (which needs `--yes`).

robots.txt is honoured and there is a delay between requests by default,
and a host's own `Crawl-delay` overrides `-p` when it asks for longer.
`-f js` **saves** JavaScript; nothing in the module runs any.

Script-shaped: `--json` on stdout with progress on stderr, and exit
codes `0` crawled / `1` nothing fetched / `2` bad arguments / `3`
interrupted. Full documentation in [Gather](Gather.md).

## `fusebox`

```bash
hnx fusebox status                    # what the cards are doing now
hnx fusebox watch --target 78         # hold 78 °C by pacing the run
hnx fusebox watch --target 78 --underclock --yes
hnx fusebox plan                      # what underclocking would do
hnx fusebox restore                   # undo what a crashed run left
```

Holds a GPU at a temperature you chose, and trips like a fuse if it goes
past `--trip` anyway. Two levers: pausing between steps (free, needs no
privileges, on by default) and lowering a power limit (`--underclock`,
needs root and `--yes`, outlives the process).

**It is not a speedup and does not claim to be.** Holding a temperature
costs throughput — a power limit 2–7 %, pausing 12–24 % — and the module
prints what it cost when the run ends. What you buy is the temperature.
The measurement is in `tests/test_fusebox.py`, and the reasoning is in
[FuseBox](FuseBox.md).

It never raises a power limit above the card's default, never escalates
privileges, and records every change to `~/.hypernix/fusebox-state.json`
so `hnx fusebox restore` can undo it after a crash.

`hnx train run --thermal-target 78` paces a training run the same way,
without touching any card setting.

## `runtime`

```bash
hnx runtime status                # what is built, detected, installed
hnx runtime serve model.gguf      # start the patched server
hnx runtime path                  # the build's bin directory
hnx runtime install --yes         # put it inside LM Studio
hnx runtime restore --yes         # put LM Studio back
```

Makes the llama.cpp from `native/ggml-hnx` — the one that reads the
sub-bit types — usable from other applications. `serve` starts an
OpenAI-compatible server anything can point at and changes nothing on
the machine; `install` replaces the libraries LM Studio bundles, which
needs `--yes`, is backed up, and is undone by `restore`.

Full documentation in [Runtime](Runtime.md).

## `websearch`

```bash
hypernix websearch "llama.cpp quantization formats" -n 5 --json
```

A non-API web search utility (`-e` to pick an engine, `--json` for
machine-readable output) — useful in scripts/agents that need search
results without a paid search-provider API key.

## `net`

```bash
hypernix net connect 100.64.0.5        # connect to a Tailscale mesh peer
hypernix net mport 8080                # expose the mesh port locally
hypernix net export 8080 --apply       # export a local port to the mesh
hypernix net tail acheck ./train.log   # tail a remote log; add -r to resume
```

Distributed network manager built on Tailscale — mainly for driving
`tvtop`/`cctvtop` against a training run happening on another machine.

## `gkey`

```bash
gkey create --type service --scopes download,upload --expires 2027-01-01
gkey create -v v2 --level 5            # a T2 key:  T2_…-5
gkey create -v v2short                 # a T2S key: T2S_…   (HyperLink)
gkey list                              # `gkey list id <key-id>` for detail
gkey revoke <key_id>
gkey version                           # what this build can issue
```

Unified CLI over the Gatekeeper + Keymaster modules for issuing, scoping,
and revoking API keys — for `hyped-pro`'s cloud routing and for the
[T1 API](T1-API.md#authentication).

**`-v` / `--level` — which format to mint.** A T2 key is a *spelling* of
a T1 key, not a separate credential: it is converted back and looked up in
the same store, which is why one minted anywhere else authenticates as
nothing.

| `-v` | Mints | For |
|---|---|---|
| `v1` (default) | `T1_…` | servers, admin, everything |
| `v2` | `T2_…-N` | a client key with an access level `--level N` |
| `v2short` | `T2S_…` | HyperLink and the iOS app; 26-character body; **never** an administrator |

`v2.1` is recognised and refused with the reason — it is reserved, not
unknown. `--password` / `--word` set the human-carried half where the
format has one; without them one is generated, and it is never a
predictable sequence.

**`gkey version`** prints the four numbers that move independently — the
`hypernix` package, the T1 API contract (short and long spellings), the
latest key format this build can issue, and the formats it accepts —
because "my key is refused" is usually one of them disagreeing.

`gkey` honours **`T1_KEYMASTER_DIR`**, which is what the server reads. On
an install using `--config-dir`, a `gkey` that ignored it would write keys
the server could not see, and both halves would appear to work.

## `hypernix-t1`

```bash
hypernix-t1 create                       # set up a server (hands off to install-t1.sh)
hypernix-t1 create --non-interactive     # …or unattended, accepting every default
hypernix-t1 start                        # start / stop / kill / restart / status
hypernix-t1 logs -f                      # follow the log (or `logs 200`)
hypernix-t1 test                         # health, status, and a real end-to-end probe
hypernix-t1 key create -v v2 --level 5   # gkey, against this server's own store
hypernix-t1 configure                    # open the config in $EDITOR
hypernix-t1 autostart on                 # install a systemd user service
hypernix-t1 remove                       # tear it back down
```

A single dependency-free shell program covering the whole lifecycle of a
[T1 API](T1-API.md) server, so running one does not mean remembering a
uvicorn invocation or hunting for a pid.

| Command | |
|---|---|
| `start` | background the server (`setsid`, so closing the terminal does not take it with you), then wait for `/health` — and say *why* if it exits during startup instead of timing out |
| `stop` | `SIGTERM`, then wait 15s. Still there? It says so and points at `kill` rather than escalating on its own. |
| `kill` | `SIGKILL`, immediately. In-flight requests are lost, and it says so. |
| `restart` | `stop`, escalating to `kill` if needed, then `start` |
| `status` | pid, address, version, whether `/health` actually answers |
| `logs` | tail; `-f` to follow |
| `create` | hands off to `install-t1.sh` when it is available — the guided setup, and every flag passes through (`--non-interactive`, `--yes`, …). Installed from a wheel there is no checkout and no installer, so it writes a **minimal** local-only config instead (`--host`, `--port`, `--force`) and says plainly what that does not cover: no allowlist, no rate limits, no pricing, no model registry. |
| `configure` | open the config in `$EDITOR`; with none set it prints the path rather than picking an editor for you |
| `test` | not a health ping — `/health`, then `/status`, then (in a checkout) the same end-to-end probe CI runs, reported per stage |
| `key` | pass straight through to `gkey`, against **this server's** key store — `hypernix-t1 key create -v v2 --level 5` |
| `autostart` | `on` / `off` / `status` — a systemd **user** service, with an absolute `ExecStart` because systemd rejects a relative one at load |
| `remove` | stop, disable, delete the config — but **keeps the key store**, which is not recoverable and may still be in use elsewhere. Confirmed by typing the word, not by `y`. |

The pid file is checked against the process actually running under it, so
a recycled pid is never mistaken for a live server and a stale file is
cleaned up rather than reported as running.

## `config`

```bash
hypernix config --help
```

Reads/writes HyperNix's persistent configuration (defaults for
`--output-dir`, `--token`, etc). See `hypernix config --help` for the
current subcommands.

## `map`

```bash
hypernix map --help
```

A steampunk-themed schematic TUI — dials, pipes, and steam-gauge style
visualization of model/training state. Also installed standalone as
`hnx-map`.

## `vera` / `prot` (`protect`)

```bash
hypernix vera --help
hypernix prot --help
```

`vera` is a separate assistant CLI; `prot`/`protect` is a hardware health
monitoring and protection module (thermal/power guardrails during long
training runs). Both are early/minimal — check `--help` for what's
currently implemented rather than assuming full parity with `chat` or
`cli`.

## Environment variables

| Var | What |
|---|---|
| `HF_TOKEN` | HuggingFace token for gated repos / upload |
| `HYPERNIX_AUTO_INSTALL=0` | Disable the runtime pip-install shim |
| `HYPERNIX_CACHE_DIR` | Override `~/.cache/hypernix/` |
| `HYPERNIX_TOOL_POLICY` | `ask` (default) / `deny` / `allow` — consent before `hyped-pro`'s agent runs a side-effecting tool. Tool calls are parsed out of the *model's own reply*, so anything that can influence that reply could otherwise run shell commands. `ask` with no terminal degrades to **deny**, so a CI job or daemon is not a shell for whoever can reach the model. |
| `T1_KEYMASTER_DIR` | The key store `gkey` and the [T1 API](T1-API.md) server share. Set it on both, or neither. |

## Exit codes

- `0` — success
- `1` — runtime error (download failed, quantize crashed, etc.)
- `2` — bad usage (missing required flag, non-existent model-dir, …)
