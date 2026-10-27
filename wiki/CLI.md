# CLI reference

`hypernix` ships a console script, `hypernix` (also aliased to `hnx` for brevity), which dispatches
to its subcommands plus the `all` pipeline as the default (see `_SUBCOMMANDS`
in `src/hypernix/interfaces/cli.py`; `hnx --help` lists the main ones and
`hnx extra` the rest). Most subcommands wrap one library function, so
they're easy to script; a few (`brew`, `cli`, `net`, `gkey`) are themselves
small sub-CLIs with their own sub-subcommands.

```
usage: hypernix <subcommand> [options]  (or: hnx <subcommand> [options])

Core pipeline:
  auto                   download a repo, convert it if it is not a GGUF, quantise it
  all                    download -> convert -> [quantize]
  download               fetch a HuggingFace snapshot, one file, or owner/repo:QUANT
  convert                produce fp32 / fp16 GGUF; -P -Q TARGET also quantises
  quantize               run llama-quantize on an fp16 / fp32 GGUF
  verify                 read-validate a GGUF and print headers
  info                   package + optional GGUF header summary
  upload                 push files to a HuggingFace repo
  doctor                 environment diagnostic (pass --fix to install deps)
  fetch-llama-quantize   pre-seed the llama-quantize cache (shortcut: fllq)
  hyprslug-headers       self-describing GGUF headers + the sub-bit server

Training & inference:
  train                  init / expand / run training utilities
  chat                   interactive REPL against any supported model
  claude                 Claude Code on a model this machine's T1 server serves
  brew                   one-shot recipe pipeline, or the brewer architecture-builder sub-CLI
  image                  hnx0xv2image: image-to-text fused onto LFM2.5, Qwen3.5 or HyperNix
  fizzle / fiz           architecture fuse/merge module
  stml                   VRAM -> trainable-context-length calculator
  dilute                 best-of-n sampling: many answers, keep the good ones
  c4                     explode a model; measure, reap and sow its layers and weights (see C4.md)

Assistants & dashboards:
  cli                    interactive TUI/CLI menu over all of the above
  tvtopmax               tvtop-max: the run's dashboard on OpenTUI (also `tvtop-max`)
  hypedpro               hyped-pro: the OpenTUI chat client with tools (also `hyped-pro`)
  vera                   check a script: syntax, smoke tests, errors explained
  wakeup                 train a wake word, then listen for it

Data & infra:
  gather                 crawl a site into a corpus (see Gather.md)
  fusebox                GPU thermal governor for a run (see FuseBox.md)
  runtime                use the HyperNix llama.cpp from LM Studio etc.
  net                    DEPRECATED: Tailscale and peers are now the T1 API's /net
  gkey                   API key issuance, scoping, revocation (Gatekeeper + Keymaster)
  config                 HyperNix configuration management
  devices                GPUs/accelerators here, and what each supports
  errors                 look up an error code (L#-NNNNN.kS)
  elements               addons: hydrogen, carbon, magnesium, your own
  extra                  the commands below, kept out of the main menu
  list-all               every command: hnx, hnx extra, hnx-t1, and every other one

`hnx extra` (0.72.8a3; each works exactly as before):
  neuron                 small networks that act: RL, imitation, supervised
  pipeline               speech in, a model's answer, speech out (ASR -> LLM -> TTS)
  assistant              interactive assistant REPL over a real model, optional voice
  generate               sample text from a local snapshot
  wiki                   this documentation, read straight from installed source
  prot / protect         blank the screen until a wake word, without locking it
  oven                   code-generation wrapper
  path                   put hypernix's console scripts on your PATH
  map                    steampunk schematic TUI for model/training state
  tvtop                  the classic training dashboard
  cctvtop                training dashboard w/ hardware metrics + optional VNC
  camo / camouflage      RLHF/RLAF alignment
  websearch              web search without an API key, real result links
  scavenger              search + size HF datasets under storage/quality budgets

Shortcuts:
  --auto-oven            download default snapshot + run code completion
                         (equivalent to `hypernix oven --auto ...`)

Also accepted: fllq (fetch-llama-quantize) · fiz (fizzle) · camouflage (camo) ·
               protect (prot) · fuse-box (fusebox)

Run `hypernix <subcommand> --help` for per-command flags.
```

## `auto`

```bash
hnx auto HuggingFaceTB/SmolLM2-135M-Instruct              # Q4_K_M
hnx auto Qwen/Qwen3-4B -q Q4_K_M -q Q8_0 -o ./qwen3        # several quants
hnx auto bartowski/SmolLM2-135M-Instruct-GGUF -q Q5_K_M    # a repo of GGUFs
hnx auto unsloth/Qwen3-4B-GGUF:IQ4_XS                      # llama.cpp's repo:QUANT form
hnx auto ./my-hf-folder -q q6h4                             # a local folder, a HyperNix hybrid
```

One command from a Hugging Face repo to the quantised GGUF you want
(0.72.8a3). What it does depends on the repo:

| The repo has | `auto` |
|---|---|
| a GGUF of that quant | downloads it (every part of a split one) and stops |
| GGUFs, not that quant | downloads the highest-precision one (F32, BF16, F16, else Q8_0) and quantises it; says so when that is Q8_0 |
| safetensors / `pytorch_model.bin` | downloads, converts to F16 GGUF, quantises to each `-q` |
| a hyperNix0x-v2 (Brewer) model | the same, with HyperNix's exporter (written as `llama`) |

Conversion uses **llama.cpp's own `convert_hf_to_gguf.py`** for every
architecture llama.cpp knows, so the result runs in llama.cpp and LM
Studio; it is fetched from the official release tag (or, where GitHub
cannot be reached, from the `llama-cpp-python` source release on PyPI,
checked against PyPI's SHA-256) and cached in
`~/.cache/hypernix/llama-convert/`. A model it cannot convert falls back
to HyperNix's converter, which writes `general.architecture=hypernix` —
`auto` says so, because only HnxRun runs that. `--converter llama.cpp`
or `hypernix` forces one.

`-q` takes hyprslug's targets (the k-quants, Q8_0, the HyperNix tiers
and hybrids); llama.cpp's other types (IQ*, TQ*, MXFP4) go to
`llama-quantize`. A bad `-q` is refused before anything is downloaded.
`-o` is the folder (default `~/.hypernix/models/NAME-GGUF`);
`--keep-f16`, `--keep-download`, `--revision`, `--token`. Prints the GGUF
paths.

## `extra`

`hnx extra` lists the commands kept out of `hnx --help` since 0.72.8a3:
`neuron`, `pipeline`, `assistant`, `generate`, `wiki`, `prot`, `oven`,
`path`, `map`, `tvtop`, `cctvtop`, `camo`, `websearch` and `scavenger`.
They dispatch exactly as before. Each was run when it moved; what was
broken was fixed (see their sections below and the changelog).

## `list-all`

```bash
hnx list-all                     # every command, grouped
hnx list-all quant               # only those whose name or description says "quant"
hnx list-all --section t1        # one group: hnx, extra, aliases, t1, other
hnx list-all --json              # for a script
```

Every command in one list (0.72.8a4): `hnx`'s main menu, `hnx extra`, the
other spellings `hnx` runs (`fllq`, `fiz`, …), every `hnx-t1` /
`hypernix-t1` command, and every other console script the package
installs (`waiter`, `gkey`, `hyprslug` and its other names, `tvtop-max`,
…). The last group is read from the installed package, so a new script
appears without an edit. The suggestions `hnx` prints after a typo
(`dl`, `quantise`) do not run, so they are not listed.

## `claude`

```bash
hnx claude                                   # this machine's T1 server and its loaded model
hnx claude --model qwen3-4b                  # a registered model
hnx claude --server http://100.64.0.7:8000 --key T1-...
hnx claude --print-env                       # export the variables yourself
hnx claude -- -p "explain this repo"         # after --: claude's own arguments
```

Runs [Claude Code](https://docs.anthropic.com/en/docs/claude-code) on a
model HyperNix is serving, through the T1 server's Anthropic-compatible
`/v1/messages` ([T1 API](T1-API.md#the-anthropic-messages-api)). It sets
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (the T1 key; with none,
`t1-keyless`, which works only where the server's trusted-network mode
lets keyless requests in), `ANTHROPIC_MODEL` and the Opus/Sonnet/Haiku/
subagent defaults to the one model, a ten-minute `API_TIMEOUT_MS` and
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, and unsets
`ANTHROPIC_API_KEY`. The server and key also come from
`HYPERNIX_T1_URL`/`T1_URL` and `HYPERNIX_T1_KEY`/`T1_API_KEY`. Install
Claude Code with `npm install -g @anthropic-ai/claude-code`.

A local model needs tool calling to be useful here: one served by
llama-server with `--jinja` (the built-in runner does this) answers
Claude Code's tool calls.

## `image`

```bash
hnx image new -o ./eye --preset small
hnx image fuse ./eye LiquidAI/LFM2.5-1.2B-Instruct -o ./lfm-eye
hnx image train ./lfm-eye --data captions.jsonl --stage projector --steps 2000
hnx image caption ./lfm-eye photo.jpg -p "What is in this picture?"
hnx image split ./lfm-eye -o ./eye-for-lfm
hnx image quant ./lfm-eye Q4_K_M -o ./lfm-eye.q4km
```

The [hnx0xv2image](HNX0xV2Image.md) image-to-text model, fused onto an
LFM2.5 (not -VL), a Qwen3.5 or a HyperNix model. `--stage projector |
image | text | all` trains one part with the rest frozen; `--optimizer
c4` uses the c4 hybrid. `--link` refers to the text model instead of
copying it. `quant` (0.72.8a5) packs the whole fused model into one GGUF
with hyprslug, at any tier, and every other `hnx image` command reads
the result; `hyprslug FUSED TIER -o OUT` does the same.

## `tvtopmax` / `hypedpro`

`hnx tvtopmax` is `tvtop-max` and `hnx hypedpro` is `hyped-pro` (both
OpenTUI on Bun), listed in the main menu since 0.72.8a3; the arguments
are theirs. See [TvTopMax](TvTopMax.md) and the `hyped-pro` section
below.

## `download`

```bash
hypernix download nix2.5                                     # short name
hnx download https://huggingface.co/Qwen/Qwen3.5-4B -o ./qwen  # a link to a repo
hnx download https://huggingface.co/owner/repo/blob/main/m-Q4_K_M.gguf -o m.gguf
hnx download owner/repo-GGUF -f m-Q8_0.gguf                  # one file from a repo
hypernix download --repo-id Qwen/Qwen3.5-4B --token $HF_TOKEN  # the old form
```

The model is a repo id, a short name, a `huggingface.co` (or `hf.co`)
link to a repo — `/tree/<rev>` picks a revision — or a link to one file
in it (`/blob/`, `/resolve/`, `hf://owner/repo/file`, `owner/repo/file`)
(0.72.8). A repo with no safetensors and one GGUF downloads that GGUF;
with several it takes the `Q4_K_M` one, or lists them and asks for `-f`.
Dataset and Space links are refused. `owner/repo:Q4_K_M` (llama.cpp's
`-hf` form) takes that quant, every part of a split file (0.72.8a3).

A gated repo (the default `ray0rf1re/hyper-nix.1` is one) says so, with
the page to ask for access and how to log in, instead of three chained
tracebacks; a repo that does not exist says that. A short name's folder
is named after the repo it resolved to (`nix` used to become `nix/`).

| Flag | Default |
|---|---|
| `MODEL` (or `--repo-id`) | `ray0rf1re/hyper-nix.1`; optional |
| `-f`, `--file NAME` | the whole model; one file from the repo instead |
| `-o`, `--output PATH` | the models folder; a folder, or for one file a file name |
| `--revision` | latest, or a [pin](../SECURITY.md#running-it-safely) |
| `--local-dir PATH` | the same as `-o` for a whole repo |
| `--cache-dir PATH` | `~/.cache/huggingface/hub` |
| `--token` | `$HF_TOKEN` |
| `--quiet` | false |
| `--no-verify` | false |

Prints the local snapshot path to stdout.

## `convert`

```bash
hypernix convert --model-dir ./snapshot --output ./out-fp16.gguf --dtype fp16

# -P: convert and quantise in one step (hnx is the same command)
hypernix convert ./snapshot -P -Q Q4_K_M
hnx convert ./snapshot/model.safetensors -P -Q q6h4 -o model.q6h4.gguf
hnx convert ./brewer_models/mine -P -Q FP8
hnx convert ./snapshot -q -p                 # lowercase works; -q alone is Q4_K_M
```

`-P` always converts first and quantises second, as two steps it
reports (`[1/2] converting …`, `[2/2] quantising to …`): only the
finished F16 GGUF reaches the quantiser (0.72.8).

| Flag | Default |
|---|---|
| `MODEL` (or `--model-dir PATH`) | required |
| `-o`, `--output PATH` | required without `-P`; with it, `<model>.<target>.gguf` beside the model |
| `-P`, `-p`, `--pipeline` | off. Converts safetensors to a GGUF, then quantises it with [hyprslug](HyprSlug.md) to the `-Q` target |
| `-Q`, `-q`, `--quant [TARGET]` | none. Any hyprslug target: `Q4_K_M`, `Q8_0`, `INT3`, `FP8`, `IQ0.5`, the hybrids `q6h`/`q6h4`/`q6h2`/`hnx_q6h1`/`hnx_q6h1m`. Implies `-P`; with no target, `Q4_K_M` |
| `--keep-intermediate` | off. With `-P`, keep the F16 GGUF beside the output |
| `--imatrix PATH` | none. With `-P`, an importance matrix for hyprslug |
| `--dtype` | `fp16` (`fp16` / `f16` / `fp32` / `f32`) |
| `--arch NAME` | `hypernix` |
| `--name NAME` | `HyperNix` |
| `--n-head N` | from config |
| `--context-length N` | from config |

What `-P` does with each kind of model:

| Model | What happens |
|---|---|
| Hugging Face folder, or a `.safetensors` in one | converted to an F16 GGUF, quantised, and the F16 copy deleted |
| hyperNix0x-v2 (Brewer) folder or `.pt` | exported to a llama.cpp GGUF, then quantised |
| `.gguf` | quantised as it is |

The target is checked before anything is converted, so a typo in `-Q`
costs nothing. The F16 copy is staged beside the output, not in `/tmp`,
because it is the biggest file in the pipeline.

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

`-hnx` is what the HyperNix targets need: `llama-quantize` has never
heard of the sub-bit tiers (`IQ0.9_L`, `IQ0.75_M`, `IQ0.5_XXXL`), the
codebooks (`INT3`, `FP8`…), `Q8_K` as a weight, or the hybrids (`q6h`,
`q6h4`, `q6h2`), so nothing is looked for, downloaded or built. See
[HyprSlug](HyprSlug.md).

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

Reports the OS and distro, Python (3.12–3.15), torch, and every other
dependency in `pyproject.toml` (numpy, safetensors, huggingface-hub,
gguf, tqdm, rich, sentencepiece) -- those decide the exit code. Then,
without failing the run: the GPU and the brewer preset and optimizer it
suits, `llama-quantize` (fetched on first use if missing), the patched
llama.cpp and whether it reads every HyperNix type, the models folder
(through symlinks, with any broken link and its target), and which
extras are installed.

`--fix` installs every dependency but torch, plus the `train` extra,
through `hypernix.deps.ensure`, and puts the console scripts on PATH
(`torch` is never touched — users control their CUDA flavor). Full
table: [PipelineMechanics](PipelineMechanics.md#hypernixdoctor--environment-diagnostic).

## `fetch-llama-quantize`

Pre-seed the `~/.cache/hypernix/bin/` cache so the first `quantize`
call is fast:

```bash
hypernix fetch-llama-quantize
hypernix fetch-llama-quantize --force          # redownload
hypernix fetch-llama-quantize --auto           # include PyPI fallback
hypernix fetch-llama-quantize --search-releases 20
hnx fllq                                       # the same, shorter
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

| Flag | |
|---|---|
| `--repo-id ID` / `--model-dir DIR` | the model: a short name or Hugging Face id, or a local snapshot |
| `--revision REV`, `--token TOKEN` | Hugging Face revision and token, for a gated or pinned download |
| `--device DEV`, `--dtype float32\|float16\|bfloat16` | where and in what precision to load it |
| `--prompt TEXT` | text to complete; not with `--fill-prefix` |
| `--fill-prefix`, `--fill-suffix` | fill in the middle |
| `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p`, `--seed` | sampling |
| `--save-pt PATH` | also write a self-contained bundle `torch.load` reads |
| `--quiet` | no progress output |

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

| Flag | |
|---|---|
| `--system TEXT` | a system prompt prepended to every turn |
| `--message TEXT` | run one turn and exit |
| `--cache-bytes N` | memory to spend keeping sub-bit weights decoded (`512M`, `2G`), largest tensors first; ignored for models llama.cpp runs |
| `--hnx-device auto\|cpu\|cuda\|cuda:1\|mps\|xpu` | where a sub-bit model runs; a named device that cannot run says why instead of quietly using the CPU |

The rest (`--repo-id`, `--model-dir`, `--revision`, `--token`,
`--device`, `--dtype`, sampling, `--seed`, `--quiet`) are `oven`'s.

## `brew`

```bash
# One-shot pipeline from a recipe file:
hypernix brew recipe.json --set output_dir=./out

# Architecture-builder sub-CLI (brewer). Each model is a folder,
# ./brewer_models/<name>, so these work across separate runs:
hypernix brew new --preset cpu-nano --name my-model
hypernix brew train --name my-model --data corpus.txt --steps 2000   # saves the folder
hypernix brew export --name my-model --outtype f16                   # a GGUF llama.cpp runs
hypernix brew list

# Any Brewer model folder or .pt, HyperNix.3-mini included, to GGUF:
hypernix brew gguf ~/.hypernix/models/HyperNix.3-mini --outtype q8_0
```

A path ending in `.json` is treated as an `instant_pot` recipe and run
end-to-end (download/convert/quantize/etc. as described in the recipe).
Anything else dispatches into the `brewer` architecture-builder sub-CLI,
with presets from `33m`/`micro`/`small`/`medium`/`large` (GPU) through
`cpu-nano`/`cpu-tiny`/`cpu-small` (CPU-only).

| Brewer command | |
|---|---|
| `new --preset P --name N [--save-dir D]` | write `D/N/config.json` (default `./brewer_models/N`) |
| `train --name N \| --dir D --data F` | train, continuing from saved weights if there are any, and save `model.safetensors` and `char_vocab.json` into the folder. `--steps`, `--lr`, `--batch-size`, `--device` all apply |
| `export --name N \| --dir D` | `--format gguf` (default; runs in llama.cpp), `pt`, or `hnxg` (the old binary, not GGUF). `--outtype f32\|f16\|bf16\|q8_0`, `--tokenizer`, `--context`. Refuses a folder with no trained weights |
| `gguf SOURCE` | any Brewer folder (config.json + weights + tokenizer) or `.pt` to a GGUF; `--pad-ffn 256` (default) keeps every tensor k-quantisable |
| `list` | the folders in `./brewer_models`, whether each is trained, and its GGUFs |

A Brewer GGUF is written as llama.cpp's `llama` architecture, which is
what a Brewer block is, so it runs in any llama.cpp build, LM Studio, and
the `native/ggml-hnx` patched build alike; see
[Model-Training-Guide § Running a brewed model in llama.cpp](Model-Training-Guide.md#running-a-brewed-model-in-llamacpp).

## `pipeline` and `assistant`

```bash
hypernix assistant --model nix2.5            # chat with a real model
hypernix assistant -m nix2.5 --voice         # and speak the replies
hypernix pipeline --audio recording.wav --asr whisper --tts piper -o reply.wav
hypernix pipeline --active                   # record from the microphone
```

**`assistant`** is a real chat REPL. Turns go through the same provider
registry as `hyped` (a local snapshot, a local GGUF, or a cloud API), so
`--model`/`-m` picks the model; without it the default set by
`hypernix config dmodel` is used, and with neither it says how to pick
one. `--voice`/`-v` speaks replies (also `/voice` in the session). Inside
it: `/model`, `/persona`, `/models`, `/personas`, `/system`, `/voice`,
`/listen`, `/reset`, `/save`, `/quit`.

**`pipeline`** runs speech in, speech out (`--audio`/`-a` a file, or
`--active` to record until Ctrl+C; `--prompt`/`-p` for the system prompt,
`--output`/`-o`). Since 0.72.8a3 all three stages are real: `--llm` is
the chat model (the same names and dispatch as `hyped` and `assistant`;
default the one `hypernix config` set), `--asr` a Whisper
(`nano-whisper`, `whisper-base`, `whisper-small` or a Hub id) and `--tts`
a text-to-speech model (`mms` or a Hub id). Until then the reply was a
canned "this is a simulated response" and `--llm`, `--asr` and `--tts`
were not read; the TTS stage also failed on token ids cast to float and
on a `generate()` VITS does not have, and the WAV was written at
22,050 Hz whatever the voice spoke at. A WAV is read without torchaudio.

## `cli`

```bash
hypernix cli            # rich TUI menu
hypernix cli --simple   # plain-text menu, no `rich` dependency
```

Interactive menu covering downloads, conversion, quantization, training,
and evaluation without needing to remember individual subcommand flags.

## `tvtop` / `cctvtop`

Two different dashboards, and the names are easy to mix up:

| You type | You get |
|---|---|
| `hypernix tvtop` | the classic dashboard (`hypernix.monitoring.tv`) |
| `hypernix cctvtop`, or the `cctvtop` or `tvtop` command | the pure-Python dashboard with hardware metrics and a remote-desktop panel (`hypernix.monitoring.cctvtop`) |

```bash
hypernix tvtop --log train.log --small     # classic
hypernix cctvtop                           # ~/checkpoints/train.log, then ./checkpoints/, then the newest training log here
hypernix cctvtop --log run.log --remote-desktop
```

`hypernix tvtop` takes `--log PATH`, `--refresh SECONDS`, `-s`/`--small`,
`--ascii` and `--no-color`, and finds the newest `*.log` under the
current folder with `step N/M loss=X` lines when `--log` is not given.

`cctvtop` takes `--log PATH` and the remote-desktop flags:
`--remote-desktop` starts a VNC server with a password, bound to
localhost (`--remote-desktop-listen localhost|tailscale|lan|all`,
`--remote-desktop-view-only`, `--remote-desktop-insecure` for no
password, which it says on screen), and `--remote-desktop-status` prints
what is running and exits. Without `--remote-desktop` nothing is
started. See [Dashboards](Dashboards.md).

The other generations are their own commands: `tvtop-older`,
`tvtop-old` / `tvtop-plus-plus`, [`tvtop-pro`](TvTopPro.md) and
[`tvtop-max`](TvTopMax.md).

## `camo` / `camouflage`

```bash
hypernix camo -Lmodel ./snapshot -Ai -M gpt-4o-mini -Sp "Be terse." -s 200
```

REINFORCE alignment of a local model. Each step samples an answer to a
prompt, scores it, and moves the model toward answers that scored above
the running baseline. `-Ai -M MODEL` scores with an evaluator model (a
folder, a Hub id or link, or a `.gguf`) told `-Sp`; without `-Ai` a
length-and-refusal heuristic scores, and says so. `-P FILE` the prompts
(one a line, or JSONL with `prompt`), `-s` steps, `-o` the output folder
(default `camo_aligned_model`), `--lr`, `--max-new-tokens`.

Fixed in 0.72.8a3: the evaluator's score was read with a pattern that
never matched a number, so every score was 5.0 and RLAF trained on
nothing; and the loss had the wrong sign, making well-scored answers
*less* likely. It now trains on the tokens actually sampled.

## `fizzle` / `fiz`

```bash
hypernix fiz model1 openai/whisper-tiny model2 Qwen/Qwen1.5-0.5B \
    loRA1 ./my_qwen_lora --out ./my_fuzed_model
```

The Fuzed Architecture module: fuses model weights and LoRA adapters.
Models are given as `model1 <id or link>`, `model2 …`, adapters as
`loRA1 <path>`, …, and `--out DIR` is where the result goes.

## `stml`

```bash
hypernix stml --vram 8 --params 3 --precision fp16 --batch-size 1
```

Calculates a trainable context length given available VRAM (GB), model
size (billions of params), batch size, and precision (`fp32`, `fp16`,
`int8`, `int4`). `--num-layers`, `--num-heads` and `--head-dim` replace
the shape it would otherwise estimate from the parameter count. Useful before
`train run` to avoid an OOM crash mid-run.

## `scavenger`

```bash
hypernix scavenger --keywords "code,python" --max-storage 20 \
    --min-likes 5 --data-type parquet --download some-org/some-dataset
```

Searches HuggingFace datasets and pulls them under a storage/quality
budget instead of downloading everything that matches a keyword:
`--max-storage` (GB), `--max-total` (datasets), `--min-entries`,
`--min-likes`, `--max-age` (days), `--data-type`, and `--limit` (results
per search query). Rows and size come from the Hub's dataset viewer
(0.72.8a3): sizing used to need the `datasets` package, and without it
every dataset read as 0 rows and the default `--min-entries 1000`
rejected all of them. A size the Hub does not know is shown as `?` and
is not a rejection; `--max-age` no longer crashes.

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

## `c4`

```bash
hnx c4 explode MODEL --save ./pieces          # what the model is made of
hnx c4 stem MODEL --test --calib calib.txt    # which layers are needed
hnx c4 dial MODEL --calib calib.txt --score taylor --unneeded 0.2 --priority pdial
hnx c4 reap MODEL --calib calib.txt --drop 2 --unneeded 0.2 --ffn-keep 0.75 -o harvest
hnx c4 reap --best A B C --calib calib.txt -o harvest
hnx c4 sow harvest -o ./new-model --heal 100 --calib calib.txt
hnx c4 restore MODEL -o ./model-pt --format pt
```

Takes a model apart on the GPU (CUDA, then MPS, then the CPU; `--device`
to choose) and finds out what each piece is worth. `MODEL` is a
safetensors folder or file, or a `.pt` state dict — both with a
`config.json` beside them — or a GGUF. `stem` ranks the layers
(`unneeded`, `candidate to drop`, `needed`, `keep`); `dial` scores every
weight (`magnitude`, `taylor`, `fisher`, `wanda`), picks the unneeded
share and ranks what to train first; `reap` writes a harvest (the layers,
dials and feed-forward channels to keep, as JSON plus a masks file) and
`sow` builds a model from it, optionally healing it with the c4
optimizer (`--optimizer hybrid|v5s|adamw`, `--lr`).

`--calib` is a text file (blank lines separate passages) read with the
model's tokenizer; without it the tests run on random tokens and say so.
`--json` prints JSON. Exit code `1` on a model c4 cannot use. `hnx-c4` is
the same command. Full documentation in [C4](C4.md).

## `fusebox`

`fuse-box` is the same command.

```bash
hnx fusebox status                    # what the cards are doing now
hnx fusebox watch --target 78         # hold 78 °C by pacing the run
hnx fusebox watch --target 78 --underclock --yes
hnx fusebox plan                      # what underclocking would do
hnx fusebox restore                   # undo what a crashed run left
```

Holds a GPU at a temperature you chose (`--target`), and trips like a
fuse if it goes past `--trip` anyway; a tripped run waits until the card
is back below `--reset`. Two levers: pausing between steps (free, needs no
privileges, on by default) and lowering a power limit (`--underclock`,
needs root and `--yes`, outlives the process).

**It is not a speedup and does not claim to be.** Holding a temperature
costs throughput — a power limit 2–7 %, pausing 12–24 % — and the module
prints what it cost when the run ends. What you buy is the temperature.
The measurement is in `tests/training/test_fusebox.py`, and the reasoning is in
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

`--build DIR` points at a llama.cpp build other than the one `build.sh`
makes (or `$HNX_LLAMA_BUILD`), and `--json` is machine-readable. Full
documentation in [Runtime](Runtime.md).

## `websearch`

```bash
hypernix websearch "llama.cpp quantization formats" -n 5 --json
```

A non-API web search utility (`-n`/`--max-results`, `-e`/`--engine
auto|duckduckgo|bing`, `--json` for machine-readable output) — useful in scripts/agents that need search
results without a paid search-provider API key. Results link to the page itself: Bing's
`bing.com/ck/a?...` and DuckDuckGo's `/l/?uddg=` redirects are decoded
(0.72.8a3; they used to come back as tracking links with `&amp;` in
them). DuckDuckGo refuses some networks with a challenge page; Bing is
then used.

## `net`

> **Deprecated (0.72.8a3).** Every command below is now a route of the T1
> API's [`/net`](T1-API.md#net), answering JSON on the machine the server
> runs on; `hnx net` still works, says which route replaces the command
> each time, and will be removed in 0.73.

```bash
hypernix net status                    # tailscale status
hypernix net m-ip                      # this machine's tailnet IP
hypernix net connect 100.64.0.5        # SSH to a peer
hypernix net ex-port 8080              # print the `tailscale serve` command; --apply runs it
hypernix net tail acheck train.py      # is train.py on each peer? -r runs it there
hypernix net tail stop train.py        # stop it on the peers
```

A distributed network manager on Tailscale and SSH. The subcommands:

| | |
|---|---|
| `config` | view or edit `~/.hypernix/net.json` |
| `auto-setup`, `m-setup` | configure Tailscale and SSH keys, or print how to by hand |
| `status`, `m-ip` | `tailscale status`, and this machine's tailnet IPv4 |
| `connect IP` | SSH to a peer |
| `a-il` | find the online peers and add their host keys |
| `mutli-a-port PORT` | record an active port in the config (the name is spelled that way) |
| `ex-port PORT [--apply]` | expose a port across the tailnet with `tailscale serve` |
| `s-storage` | rsync the storage folder to every peer |
| `onef-all` | mark this node as the central storage node and print its IP |
| `tail acheck FILE [-r]`, `tail stop FILE` | check each peer for a Python file and optionally run it; stop it again |

## `gkey`

```bash
gkey create --type service --scopes download,upload --expires 2027-01-01
gkey create -v v2 --level 5            # a T2 key:  T2_…-5
gkey create -v v2short                 # a T2S key: T2S_…   (HyperLink)
gkey create -v v2.1 --level 4          # a v2.1 key: T2C_…-4, and the T2CK_ kit that makes it
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
| `v2.1` | `T2C_…-N` and a `T2CK_…` kit | a v2 key sealed with Rotorvault: the client keeps the kit, and the key it sends changes every day |

**v2.1 (0.72.6).** `gkey create -v v2.1` mints the key, registers a device
for it and prints two things below the panel (each is too long to box):
today's `T2C_` key and the `T2CK_` **kit**. Give the kit to the client —
`waiter serv -A -I <server> -K <kit>` — and it makes each day's key from
it; the key itself is never written down in the clear. A day's key is
accepted on its UTC day and one day either side. Revoke a device with
`DELETE /auth/t2c/devices/{id}`. How the sealing works, and what it does
and does not protect, is in [T1 API → v2.1 keys](T1-API.md#v21-t2c-keys-and-rotorvault).

`--password` / `--word` set the human-carried half where the format has
one; without them one is generated, and it is never a predictable
sequence.

**`gkey version`** prints the four numbers that move independently — the
`hypernix` package, the T1 API contract (short and long spellings), the
latest key format this build can issue, and the formats it accepts —
because "my key is refused" is usually one of them disagreeing.

`gkey` honours **`T1_KEYMASTER_DIR`**, which is what the server reads. On
an install using `--config-dir`, a `gkey` that ignored it would write keys
the server could not see, and both halves would appear to work.

## `hypernix-t1`

```bash
hnx-t1 status                            # hnx-t1 is the same program, shorter
hnx-t1 setup                             # the interactive installer (b1); works without a checkout
hypernix-t1 create                       # set up a server (hands off to install-t1.sh)
hypernix-t1 create --non-interactive     # …or unattended, accepting every default
hypernix-t1 start                        # start / stop / kill / restart / status
hypernix-t1 version                      # HyperNix + T1 versions, and the Python it runs as
hypernix-t1 logs -f                      # follow the log (or `logs 200`)
hypernix-t1 upgrade                      # upgrade HyperNix in the server's own Python, and restart
hnx-t1 upgrade -S env                    # bring the server's .env up to this version
hnx-t1 upgrade -S all                    # both: the venv, then the env
hypernix-t1 sync                         # mirror ~/.hypernix/models into ~/.hypernix/t1api/models
hypernix-t1 test                         # health, status, and a real end-to-end probe
hypernix-t1 key create -v v2 --level 5   # gkey, against this server's own store
hypernix-t1 configure                    # open the config in $EDITOR
hypernix-t1 autostart on                 # install a systemd user service
hypernix-t1 remove                       # tear it back down
```

`hnx-t1` (0.72.6) is an alias, not a copy: it runs the `hypernix-t1` next
to it, so every subcommand, flag and exit code is the same.

When the server has a private venv (`~/.hypernix/t1api/venv`, made by
`install-t1.sh`), whichever `hypernix-t1` you run hands over to the copy
installed in that venv if it is a different script. That copy came with
the code the server actually runs, which the one first on your `PATH` may
not have: `pip install -U` into the venv upgrades only the venv's copy,
and two builds of one version cannot be told apart by their version. The
one exception is a copy newer than the server's HyperNix, which keeps
running and warns that the server is behind.

A copy from before 0.72.6.post3 cannot hand over, and `status` (run from
the server's copy) says so when one is first on `PATH`, with the fix: a
symlink to the server's copy, which then follows every upgrade.

```bash
~/.hypernix/t1api/venv/bin/hypernix-t1 status     # the server's own copy
ln -sf ~/.hypernix/t1api/venv/bin/hypernix-t1 "$(command -v hypernix-t1)"
```

A single dependency-free shell program covering the whole lifecycle of a
[T1 API](T1-API.md) server, so running one does not mean remembering a
uvicorn invocation or hunting for a pid.

| Command | |
|---|---|
| `start` | background the server in a session of its own, so closing the terminal or dropping the ssh connection does not take it with you, then wait for `/health` — and say *why* if it exits during startup instead of timing out. On a machine whose logind has `KillUserProcesses=yes` it also warns that logging out will kill it anyway, and names the two ways out. |
| `stop` | `SIGTERM`, then wait 15s. Still there? It says so and points at `kill` rather than escalating on its own. |
| `kill` | `SIGKILL`, immediately. In-flight requests are lost, and it says so. |
| `restart` | `stop`, escalating to `kill` if needed, then `start` |
| `upgrade` | `pip install -U 'hypernix[t1api]'` in the Python the server runs on (its private venv, when it has one), then restart; `--main` installs from GitHub. `status` and `start` warn when that Python's HyperNix is older than `hypernix-t1`'s own. `-S` picks what (0.72.8a4): `venv` is that, and the default; `env` upgrades the server's `.env` — every setting this version has that the file lacks is added **commented out at its default**, with a line on what it does, so nothing changes until you set one; a missing token secret is added, a file others can read is closed to 0600, and a `T1_*` name nothing reads is reported with the setting it probably meant; the old file is kept as `.env.bak-<time>`. `all` is both, the venv first so the new code does the env. `--dry-run` with `env` shows the result and writes nothing |
| `setup` | the interactive installer, b1 (0.72.8a4; also `install`). Unlike `install-t1.sh` it needs no checkout. It asks about the server (name, who can reach it, port, a public URL), access (keyless trusted networks, request limits), models (the runner, LM Studio, the default model) and what the server offers (HyperLink and HyperLink on the web, Claude Code, MCP and Hugging Face's MCP server, code sandboxes, remote runs, the public counter), shows the result, writes `.env` (0600) and starts or restarts the server. Run again, it starts from the server's own answers, keeps its token secret — a new one would invalidate every key — keeps settings it does not ask about, and keeps the old file as `.env.bak-<time>`. `--yes` takes every default, `--answers FILE` takes a JSON object of answers by name, `--dry-run` prints the file, `--no-start` stops after writing it |
| `status` | pid, address, config/key/log paths, this server's identity fingerprint, the HyperNix version and the Python it runs on, where HyperLink on the web answers (or that this HyperNix has none), and the T1 version from `/status`. When the server is not running, the stale pid, anything else on its port, and the last lines of the log. |
| `version` | the HyperNix and T1 versions, and the Python the server runs as — the line to read first when an upgrade seems to have done nothing |
| `logs` | tail; `-f` to follow |
| `create` | hands off to `install-t1.sh` when it is available — the guided setup, and every flag passes through (`--non-interactive`, `--yes`, …). Installed from a wheel there is no checkout and no installer, so it writes a **minimal** local-only config instead (`--host`, `--port`, `--force`) and says plainly what that does not cover: no allowlist, no rate limits, no pricing, no model registry. |
| `configure` | open the config in `$EDITOR`; with none set it prints the path rather than picking an editor for you |
| `test` | not a health ping — `/health`, then `/status`, then (in a checkout) the same end-to-end probe CI runs, reported per stage |
| `key` | pass straight through to `gkey`, against **this server's** key store — `hypernix-t1 key create -v v2 --level 5` |
| `autostart` | `on` / `off` / `status` — a systemd **user** service, with an absolute `ExecStart` because systemd rejects a relative one at load |
| `sync` | mirror `~/.hypernix/models` into this server's own `~/.hypernix/t1api/models`, one symlink per file, removing links whose file has gone (`--dry-run`, `--index`, `--watch S`). Also installed as `hypernix-sync` and `t1-sync`; `T1_MODEL_SYNC=1` has the server do it itself — see [T1-API § Model sync](T1-API.md#model-sync) |
| `index` | read every `.gguf` in a folder and write the model registry from what the files say — see [ModelIndex](ModelIndex.md) |
| `built-in-runner` | also `runner`: serve a model from this server's own llama.cpp instead of LM Studio — see [below](#hypernix-t1-built-in-runner) |
| `chat` | send a message to the served model and print the reply — see [below](#hypernix-t1-chat) |
| `training` | what training is doing on this machine, with pause / resume / stop; runs started by `launch-script` show up on their own |
| `launch-script` | run a script, or a command with `-1 'CMD'`, so it survives an ssh disconnect: supervised, with its logs and exit status kept, and `--status` / `--logs` / `--stop` from any later session. `-2` … `-9` answer the command's questions in order (`-5 ""` presses Enter; a gap is Enter too), and `--logs JOB -F` follows the log until Ctrl-C (0.72.8) |
| `config web` | the five [web search](T1-API.md#web-search-v2) settings: `--set NAME=VALUE`, `--json`, `--reset` (0.72.8) |
| `override lms move-dir` | point LM Studio's models folder at `~/.hypernix/models` (or a folder you name); `override lms revert` undoes it |
| `start-foreground` | run in this terminal instead of the background; what the systemd unit uses |
| `remove` | stop, disable, delete the config — but **keeps the key store**, which is not recoverable and may still be in use elsewhere. Confirmed by typing the word, not by `y`. |

The pid file is checked against the process actually running under it, so
a recycled pid is never mistaken for a live server and a stale file is
cleaned up rather than reported as running.

### If the server does not stay running

`start` detaches through the interpreter it is already launching —
`Popen(start_new_session=True)`, which is setsid(2) in the child — rather
than through a `setsid` binary. macOS does not ship one, and where it
does exist it forks when it is already a process-group leader, so the pid
recorded could belong to a process that had already exited. The pid in
`server.pid` is the uvicorn process itself, and `os.getsid(pid) == pid`
is the fact that makes a SIGHUP from the closing terminal irrelevant.

That does not cover **logout** on a machine whose systemd-logind sets
`KillUserProcesses=yes`: it kills everything the user owns, a separate
session included, and writes nothing to the log. `start` warns when it
sees that setting with lingering off. Either way out works:

```bash
hypernix-t1 autostart on              # a user service, started at login
sudo loginctl enable-linger "$USER"   # …and kept running after logout
```

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

`vera FILE` checks a HyperNix script: it verifies the syntax, runs smoke
tests, and explains a failure with a local model.

| Flag | |
|---|---|
| `-dr`, `--dry-run` | run the file with its dry-run flag |
| `-C`, `--full-run` | run the file for real |
| `-FT`, `--function-test`, `-q`/`--depth 1-10` | test each argument on its own, to that depth |
| `-t`/`--timeout N`, `-T ml\|s\|M\|h` | per-stage limit, and its unit |
| `-tt`/`--total-timeout SECONDS` | a limit on the whole run |
| `-Na`, `--no-ai` | print the raw error, load no model |
| `-m`, `--pick-model`, `--models-dir DIR` | choose the GGUF that explains, from `~/.hypernix/models` or DIR |

`prot`/`protect` blanks the monitor,
puts the terminal into raw mode, and waits for a wake word typed blind —
type it and the screen comes back. It can also log CPU and memory to
`~/.hypernix/protect_health.log` while it waits, since the screen is off
and printing would be both invisible and destructive to raw mode.

```bash
hypernix prot                     # blank, wait for the wake word
hypernix prot bind set <word>     # change the wake word (default: bon)
hypernix prot bind reset
hypernix prot --force             # wait even if the screen cannot be blanked
hypernix prot restore             # put back lock settings a crash left off
```

**How it blanks.** The method comes from the session, not the platform,
and lives in `hypernix.system.blanking` shared with `outage`:

| Session | Method |
| --- | --- |
| X11 | `xset +dpms` then `xset dpms force off` |
| Hyprland | `hyprctl dispatch dpms off` |
| sway | `swaymsg output * power off` |
| wlroots | `wlopm --off '*'` |
| GNOME | Mutter's `PowerSaveMode` over D-Bus (`gdbus`) |
| KDE Plasma | `kscreen-doctor --dpms off` |
| macOS | `pmset displaysleepnow` |

The `+dpms` is the fix for "prot doesn't make the monitors black":
`xset dpms force off` is a *request to the DPMS extension*, and when
DPMS is disabled — which it is on a lot of desktops, because the desktop
environment handles power management itself — the X server accepts it,
does nothing, and exits 0. A session that had DPMS off deliberately gets
it back on the way out.

**It does not lock the session** (0.72.8a3). The GNOME/KDE path used to
blank by *activating the screensaver*, and an active screensaver locks:
the wake word woke a lock screen. That path is gone. And because a dark
screen is an idle session, which locks after the desktop's idle delay,
`prot` switches lock-on-idle off for as long as it runs — GNOME,
Cinnamon and MATE (`lock-enabled`), KDE (`Autolock`), Xfce's screensaver
and the X screensaver — and puts each back on the way out, including on
SIGTERM or a closed terminal. Settings already off are not touched.
What it changed is also written to `~/.hypernix/protect-lock-state.json`,
so if it is killed outright `hnx prot restore` puts them back. A Mac set
to lock the moment the display sleeps is refused unless `--allow-lock`.

**Hyprland, with hyprlock or Quickshell** (0.72.8a5). Hyprland never
locks by itself; hypridle does, after its timeouts (running hyprlock,
`loginctl lock-session`, or a Quickshell lock), and Quickshell can watch
for idle itself with an `IdleMonitor`. While the screen is dark `prot`:

- holds a logind idle inhibitor (`systemd-inhibit --what=idle`), which
  hypridle honours unless `hypridle.conf` says
  `ignore_systemd_inhibit = true` — then hypridle is stopped and started
  again on the way out (`systemctl --user` when it runs as a unit,
  `hyprctl dispatch exec hypridle` when it does not);
- holds a Wayland idle inhibitor with `wlinhibit` when the Quickshell
  config (`~/.config/quickshell`) uses an `IdleMonitor`, which ignores
  logind's — and says to install `wlinhibit` when it is missing;
- turns `misc:key_press_enables_dpms` (and `mouse_move_enables_dpms`)
  off if they are on, since the first letter of the wake word would light
  the screen, and back on afterwards.

The inhibitors are processes, so they end with `prot` however it ends;
the option and hypridle changes are in the state file for
`hnx prot restore`.

**It will not lock a screen it did not blank.** If there is no method
for this session, `prot` says which of the four cases it is — no
graphical session, a Wayland compositor with nothing installed, a
missing `xset`, a wedged compositor — and **refuses to enter raw mode**.
Doing it anyway is the worst of both: the machine looks unlocked and the
keyboard looks dead. `--force` is there for someone who wants the
wake-word prompt regardless.

## `hypernix-t1 built-in-runner`

Load, unload and inspect the model the server is serving — without LM
Studio, and without walking over to the machine. Also spelled
`hypernix-t1 runner`; the two are the same command.

```bash
hypernix-t1 built-in-runner start                 # pick from a list (a terminal), else the default model
hypernix-t1 built-in-runner start qwen3-8b --gpu-layers 24 --backend cuda
hypernix-t1 built-in-runner backend               # pick the backend from a list
hypernix-t1 built-in-runner start m.gguf --backend multilama   # through a llama.cpp fork
hypernix-t1 built-in-runner status
hypernix-t1 built-in-runner plan qwen3-8b
hypernix-t1 built-in-runner stop
```

**Choosing from a list (0.72.8).** In a terminal, `start` with no
model, or with a name that matches nothing exactly, opens a picker like
`lms load`: type to filter, ↑/↓ to move, Enter to load, Esc to cancel.
The list refreshes while it is open and never takes over the screen.
`backend` picks the runner's backend the same way. `--no-pick` keeps the
old behaviour for scripts.

**Multi-llama (0.72.8).** `--backend multilama` serves a GGUF through
the llama.cpp fork it needs — ik_llama.cpp for its IQ*_K, IQ*_KS,
IQ*_KT and repacked types, Nanbeige for its architecture — and
`multilama-ik`, `-prismml`, `-kobold` or `-nanbeige` names one. `auto`
does it by itself when a file needs a fork. `hyprslug --check FILE` says where a
file runs.

**The default model.** Since 0.72.8, `start` with no model named (and
no terminal to pick in) loads Liquid AI's
[LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF), and the
server downloads `LFM2.5-2.6B-Q4_K_M.gguf` (about 1.7 GB) the first
time; `lfm2.5-2.7b` names it too. Before 0.72.8 it was
[HyperNix.3-mini](https://huggingface.co/ray0rf1re/HyperNix.3-mini),
which `start hypernix.3-mini` still loads, and
the server downloads it into `~/.hypernix/models/HyperNix.3-mini` the
first time (about 195 MB: the safetensors, config and tokenizer, never
the pickled `model.pt`). It is HyperNix's own model: 48.7M parameters,
trained from scratch on one GTX 1080, with a 512-token context. It is a
**base model**. It completes text, it does not follow instructions or
call tools, and at this size its text is often not coherent. For
chatting with tools, name a GGUF instead. `T1_DEFAULT_MODEL` names a
different default, and `T1_DEFAULT_MODEL=` (empty) turns the default
off, so `start` loads the only model on the machine as it did before.

**Native HyperNix models.** A folder with a brewer `config.json` and
weights, like HyperNix.3-mini or anything `hnx brew` trained, is served
through llama.cpp when there is a llama.cpp build: it is converted to a
GGUF once, cached in `<T1_CONFIG_DIR>/cache/brewed-gguf` and converted
again only when its weights change. Otherwise, or when it cannot be
converted (no tokenizer), `hypernix.hyperlink.brewed_server` serves it in
PyTorch. Either way it is the same OpenAI API, so HyperLink cannot tell
the difference. `HYPERNIX_BREWED_BACKEND=torch` always uses PyTorch;
`=llama` requires llama.cpp and says why when it cannot. The
chat is written out as a `User:` / `Assistant:` transcript, which is the
only format a base model can continue.

**Where the server is.** `runner` and `chat` use `--url`, then `T1_URL`,
then `T1_HOST`/`T1_PORT` from the environment or the server's `.env`,
then port 8000. It used to try only 8000, so a server on 8001 was "not
running" to it. If nothing answers there, it asks the nearby ports on
this machine whether they are a T1 API, and says where it found one.

**`start` does not make you name a model when there is only one.**
`load` always did, which is right on a machine with forty of them and
wrong on the commonest machine there is: one GGUF on disk and nothing
serving it, where the name is not a decision anybody has to make and
asking for it is a lookup they have to do first.

With more than one it refuses to guess, and lists them — guessing wrong
is not a typo, it is a minute of loading and the VRAM of a model nobody
asked for. With none it says which sources answered, because "no
models" and "LM Studio is not running" produce the same empty list and
want completely different things doing about them.

Running it twice leaves the model that is already serving alone and
says so, rather than charging a reload — and an evicted conversation —
for a command that was meant to be a no-op. `--restart` reloads it
anyway. `stop` is `unload` under a name somebody who just typed `start`
will reach for.

`plan` says where a model's layers would go and changes nothing.
Loading evicts whatever people are currently talking to, so seeing the
consequence first is not a nicety:

```
qwen3-8b
  file: /home/you/.hypernix/models/qwen3-8b-q4_k_m.gguf
  33 of 33 on the GPU
  24 GB of VRAM, 4.9 GB of weights plus a 32k cache

Nothing has changed — this was a plan.
```

**It is a client, not a second runner.** It talks HTTP to the server on
the same machine, exactly as HyperLink does. Loading the model in this
process would start a *second* llama.cpp, which takes the VRAM the
server's copy is using — and the failure then lands on the one that was
working rather than the one being started.

**The key** comes from `--key`, then `T1_ADMIN_KEY`, then the admin key
in the server's own `.env` — which is the common case, since whoever is
typing this is usually at the keyboard of the machine running it. A
server in trusted-network mode with partial admin accepts the request
with no key at all.

A refusal is printed rather than raised, because a refusal is
information: a 403 names the three ways to be allowed, and a 404 lists
what this server *can* load.

## `hypernix-t1 chat`

```bash
hypernix-t1 chat "What can you do here?"
hypernix-t1 chat -s "You are terse. Answer in one line." "Which model are you?"
hypernix-t1 chat "and on the GPU?"                # carries on the same chat
hypernix-t1 chat --new "a fresh conversation"
echo "summarise this" | hypernix-t1 chat
```

A message to the model this server serves, with the reply streamed to
the terminal. It uses the same HyperLink route as the phone, so the
answer comes from the built-in runner when a model is loaded there, with
HyperLink's default system prompt and tools, and from LM Studio
otherwise. `-s` sets this chat's own system prompt, which is added after
the default one. The chat is remembered, so the next `chat` carries it
on until `--new`. Tools the model runs are printed as `[tool ✓]` on
stderr. `-q` prints only the reply.

It replaces a curl recipe that needed `VAR=$(...)`, which fish refuses
("Unsupported use of '='"), and works the same in any shell.

## `wakeup`

```bash
hypernix wakeup record "hey nix" -o ./takes               # capture takes from the microphone
hypernix wakeup train "hey nix" --positives ./takes --negatives ./ambient
hypernix wakeup check clip.wav --model hey-nix.pt         # score one clip
hypernix wakeup listen --model hey-nix.pt                 # listen on the microphone
```

Train a wake word and listen for it. The full guide — where examples come
from, how the classifier works, how to tune it — is [Wake-up](WakeUp.md).

## `dilute`

```bash
hypernix dilute run  --prompts prompts.txt -o traces.jsonl --traces 200
hypernix dilute jit  --prompts prompts.txt -o traces.jsonl --judge-model qwen3
hypernix dilute inspect traces.jsonl
```

Best-of-n sampling: several answers per prompt, keep the best. `run`
writes at the end, `jit` streams traces as they are made, `inspect` says
whether a trace file is worth training on. The evaluator decides what you
get: `--judge` uses a model and is the honest default; `--length` is a
smoke test and will teach a model to ramble if you train on what it picks.

## `neuron`

```bash
hypernix neuron demo   --env grid --episodes 50 -o demos.npz
hypernix neuron clone  --env grid --demos demos.npz -o policy.pt
hypernix neuron dagger --env grid --rounds 5 -o policy.pt
hypernix neuron rl     --env beam --algorithm reinforce -o policy.pt
hypernix neuron eval   --env grid --policy policy.pt
```

Train small networks that act: imitation (`clone`, `dagger`),
reinforcement learning (`rl`), and evaluation. Reach for `clone` before
`rl` — if you can demonstrate the task, cloning gets a working policy in
minutes.

## `errors`

```bash
hypernix errors explain S1-00060.b3     # what a code means and what to do
hypernix errors explain HPC-CFG-001     # an older name: says which code it is now
hypernix errors list                    # every code, or one domain's
hypernix errors aliases                 # every older name and its code
```

HyperNix error codes have the shape `L#-NNNNN.kS`: domain, tier, number,
kind (a-f) and severity (1-5). The T1 API returns one beside every error
as `hx_code`. Every code is listed on [Error codes](Error-Codes.md).

Since 0.72.8a4 hyped-pro raises registered codes too: its own
`HPC-…`, `HPB-…`, `HPG-…`, `HPO-…`, `HPT-…` and `TOOL-…` names are aliases
of them, and its bridge sends the old name beside the new one as
`legacy_code`.

## `elements`

```bash
hypernix elements list                  # every element, built-in and yours
hypernix elements info magnesium        # one element's spec
hypernix elements plan magnesium        # what it would change, changing nothing
hypernix elements run magnesium         # activate until Ctrl-C
hypernix elements new Na                # scaffold a user element (a real symbol)
```

Addons named after the periodic table (hydrogen, carbon, magnesium, and
your own). Periods 6 and 7 — element 55 onward — need `--experimental`.
Magnesium needs `psutil` (`pip install "hypernix[elements]"`).

## `hyprslug-headers`

```bash
hypernix hyprslug-headers status
hypernix hyprslug-headers scan ~/.hypernix/models
hypernix hyprslug-headers show model.gguf
hypernix hyprslug-headers serve model.gguf          # OpenAI-compatible endpoint
```

Self-describing headers for GGUFs that carry HyperNix extension types, and
the runtime that executes them: `install`, `status`, `uninstall`, `scan`,
`show`, `stamp`, `wrap` (re-encode to a type stock llama.cpp reads),
`install-model` and `serve`. `--json` on every subcommand. See
[HyprSlug headers](HyprSlug-Headers.md).

## `path`

```bash
hypernix path                 # --check: report what would happen (the default)
hypernix path --apply         # write the PATH block into your shell's startup file
hypernix path --undo          # remove it
hypernix path --print         # just the snippet
```

Put HyperNix's console scripts (`hypernix`, `hnx`, `hyped`, `waiter`, …)
on your PATH. `--shell` and `--profile` override detection. Inside a
virtualenv `--apply` is refused unless `--force` is given.

## `wiki`

```bash
hnx wiki                      # table of contents
hnx wiki pressure_cooker_v5   # one module's documentation
hnx wiki -q freezer           # stream it section by section
hnx wiki workshop -b          # open it in a browser
hnx wiki --search QAT         # search every module
```

A documentation browser that reads docstrings from the installed source,
so it describes the version you have.

## Environment variables

| Var | What |
|---|---|
| `HF_TOKEN` | HuggingFace token for gated repos / upload |
| `HYPERNIX_AUTO_INSTALL=0` | Disable the runtime pip-install shim |
| `HYPERNIX_CACHE_DIR` | Override `~/.cache/hypernix/` |
| `HYPERNIX_DEPRECATION_WARNINGS=0` | Silence HyperNix's own one-line deprecation notice on stderr. It does **not** touch the `DeprecationWarning` itself — that would let a stray export disarm `-W error::DeprecationWarning` for a whole CI run. Silence that with `PYTHONWARNINGS=ignore::DeprecationWarning`; set both to hear nothing. See [deprecated modules](#deprecated-modules). |
| `HYPERNIX_TOOL_POLICY` | `ask` (default) / `deny` / `allow` — consent before `hyped-pro`'s agent runs a side-effecting tool. Tool calls are parsed out of the *model's own reply*, so anything that can influence that reply could otherwise run shell commands. `ask` with no terminal degrades to **deny**, so a CI job or daemon is not a shell for whoever can reach the model. |
| `T1_KEYMASTER_DIR` | The key store `gkey` and the [T1 API](T1-API.md) server share. Set it on both, or neither. |

## Deprecated modules

Five modules are deprecated and each says so, on stderr, the moment it
is imported:

| Module | Use instead | Since |
|---|---|---|
| `hypernix.models.old_oven` | `hypernix.models.neo_oven` | 0.71.5a2 |
| `hypernix.system.old_fridge` | `hypernix.models.neo_oven` | 0.71.5a2 |
| `hypernix.data.mediocre_fridge` | `hypernix.models.neo_oven` | 0.71.5a2 |
| `hypernix.evaluation.new_fridge` | `hypernix.models.neo_oven` | 0.71.5a2 |
| `hypernix.monitoring.tvtop` | `hypernix.monitoring.tv` | 0.70.0 |

None of them is scheduled for removal; the files are kept intact.

Each import emits a real `DeprecationWarning` — so `-W error`,
`pytest.warns` and any audit tooling can see it — **and** a one-line
notice on stderr, because outside `__main__` Python hides
`DeprecationWarning` by default and would show nothing at all.
`hypernix.system.deprecation` prints the stderr line only when the
warning was actually suppressed, so no one sees it twice.

Nothing goes to **stdout**. That matters: until 0.72.4.post8 the notice
was printed with `rich.Console()`, which defaults to stdout, so
`hnx … > out.json` got a line of English in its JSON.

`hypernix.preheat` and `hypernix.new_oven` resolve to `neo_oven`, so
the top-level API does **not** raise a deprecation notice. They pointed
at `old_oven` until 0.72.4.post9 even though the 0.71.5a2 notes said
they returned a `NeoOven` — which meant the top-level shortcuts told
callers to stop using a module they had never imported.

## Exit codes

- `0` — success
- `1` — runtime error (download failed, quantize crashed, etc.)
- `2` — bad usage (missing required flag, non-existent model-dir, …)
