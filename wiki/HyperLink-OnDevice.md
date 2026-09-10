# HyperLink: models on the phone

New in **0.72.4.post10**. Search Hugging Face, download a GGUF, and run
it on the phone with no server involved.

The hard part is not the running. It is answering *"will this one
work?"* before a four-gigabyte download, and being right — because
being wrong means the phone downloads for twenty minutes on cellular and
is then killed by the OS partway through the first reply.

## Three things that make the obvious answer wrong

### Total RAM is not the budget

This is the big one, and it is the mistake almost every naive
implementation makes.

`ProcessInfo.processInfo.physicalMemory` returns 8 GB on an iPhone 15
Pro. **An app may not use it.** iOS gives each process a jetsam limit
well below total RAM — commonly 2–3 GB for a normal app on an 8 GB
device — and exceeding it is not a swap, not a slowdown, and not an
exception you can catch. The process is killed, with no warning first.

So a fit check written against `physicalMemory` tells the user a 5 GB
model fits, and it does not. The number that matters is
`os_proc_available_memory()`, which reports what remains of *this
process's* limit right now.

```
8 GB iPhone, Llama-3.1-8B Q4_K_M at 4k context  ->  NO
  needs 5.5 GiB, this app has 3.0 GiB
```

The `com.apple.developer.kernel.increased-memory-limit` entitlement
raises the ceiling and Apple grants it selectively. HyperLink reads it
from the embedded provisioning profile and **reports** it; it never uses
it to inflate an estimate, because claiming headroom the process may not
have been granted is the same bug in a new place.

The budget also *shrinks* while the app is open, as other apps run. A
model that fit when the list was drawn may not fit when the user taps
it, so the check runs again immediately before every load.

### The KV cache is not small

It scales with context length, and at long contexts it exceeds the
weights:

| context | KV cache (8B, fp16) | weights (Q4_K_M) |
|---|---|---|
| 2,048 | 0.25 GiB | 4.8 GiB |
| 8,192 | 1.00 GiB | 4.8 GiB |
| 32,768 | 4.00 GiB | 4.8 GiB |
| 131,072 | 16.00 GiB | 4.8 GiB |

A planner that sizes only the weights is badly wrong exactly when
someone tries to use the long context they chose the model for. So the
useful question is usually not *"does it fit"* but *"how much context
can I have"*, and `largestContext` answers that instead of refusing.

The cache is sized by the **key/value** head count, not the attention
head count. A grouped-query model shares each KV head across several
attention heads, so using the larger number overestimates by the GQA
ratio — 4× on Llama 3 — and refuses models that run comfortably.

An 8-bit KV cache halves it, and is frequently what makes a long context
possible on a phone at all.

### GPU offload does not reduce memory

On a desktop with a discrete card, moving layers to the GPU moves them
out of system RAM. **Apple silicon has unified memory**: a Metal buffer
and a `malloc` come from the same pool and count against the same limit.

Offloading to Metal is worth doing for speed and does nothing whatever
for fit. A planner that subtracts offloaded layers from the memory
estimate approves models that cannot run.

## The Neural Engine

It cannot run a GGUF model, and HyperLink does not offer a switch that
claims otherwise.

The ANE is reachable only through Core ML, and llama.cpp has no Core ML
backend for LLM inference — its Apple backend is Metal, with Accelerate
on the CPU path. Running on the ANE would mean converting the model to
Core ML: a different file, in a different format, produced by a
different toolchain. It is not a setting.

Metal is the acceleration that exists here, and on Apple silicon it is
fast. The settings screen says this rather than hiding the question.

## Sizing a model before downloading it

A GGUF's size is not bits × parameters. llama.cpp keeps the embedding
and output tensors at a higher precision than the file's name suggests,
and for a small model those are a large fraction of it — Llama-3.2-1B
has a 128k vocabulary over 2,048 dimensions, which is **21% of its
parameters**.

Sizing that model flat under-counts by 8%, and under-counting is the
direction that gets the process killed. So the exception is priced
separately when the vocabulary and dimension are known, and a
conservative factor stands in when they are not:

| model | estimate | real file | error |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 0.82 GB | 0.81 GB | +1.1% |
| Llama-3.2-3B Q4_K_M | 2.07 GB | 2.02 GB | +2.2% |
| Llama-3.1-8B Q4_K_M | 5.18 GB | 4.92 GB | +5.2% |
| Llama-3.1-8B Q6_K | 6.72 GB | 6.60 GB | +1.8% |
| Llama-3.1-8B Q8_0 | 8.70 GB | 8.54 GB | +1.9% |
| Qwen2.5-7B Q4_K_M | 4.93 GB | 4.68 GB | +5.3% |

Every one lands **at or above** the real file. That asymmetry is
deliberate: over-estimating hides a model that would have run, which is
annoying; under-estimating approves one that cannot, which ends the app.

## Two implementations, kept in step

The decision has to be made *on the phone* — before a download, and
again before a load, when there may be no network at all. So the
arithmetic exists twice:

- `hypernix/hyperlink/ondevice.py` — the reference, with tests against
  published Hugging Face file sizes
- `ios/HyperLink/Sources/OnDevice/ModelFit.swift` — the mirror

Duplicated arithmetic drifts, and here the symptom of drift is the Swift
side approving a model the Python side would refuse. So
`tests/test_hyperlink_ondevice_mirror.py` parses the Swift and compares
every constant and every quantisation bit width against the Python.
Change one and the build says so.

There is no Swift toolchain in CI, so that test cannot compile or run
the Swift. It checks the numbers, which are the part that decides
whether someone's phone survives.

## Downloading

A GGUF is between half a gigabyte and eight, and that drives everything:

- **Resumable**, through a background `URLSession`. A download that
  restarts from zero when the user walks past a lift is not a download.
- **Excluded from iCloud backup.** Apple rejects apps that back up
  re-downloadable data, and a user's backup is not the place for
  something Hugging Face already has.
- **Disk checked first**, with
  `volumeAvailableCapacityForImportantUsage` rather than
  `systemFreeSize` — the second counts space iOS will not actually
  hand over, so the download dies at 90%.
- **Size verified.** A truncated GGUF fails to load with an error that
  reads like a broken model rather than a broken download, so a short
  file is deleted rather than kept.
- **Multi-part files are excluded** from listings. They need joining
  before use, and offering one produces a file that cannot load.

## The Hugging Face token

Lives in the **Keychain**, with `ThisDeviceOnly` so it does not travel
to a restored backup on another phone. Not in `UserDefaults`, whose
plist is readable from a file-system backup.

It is sent to `huggingface.co` and nowhere else, never logged, and never
included in an error message.

Gated repositories are flagged in search results rather than discovered
as a 403 forty minutes into a download.

## Settings

| setting | default | why it is exposed |
|---|---|---|
| Backend | Metal | CPU is more predictable under thermal load |
| Threads | performance cores | More is faster until it is not: on a phone the ceiling is thermal, and every core throttles within a minute or two |
| Context length | 4,096 | Drives the KV cache, which is usually the binding constraint |
| KV cache bits | 16 | 8 halves the cache for a small quality cost |
| System prompt | empty | |
| Pause in background | on | A backgrounded app holding gigabytes is the most likely thing to be jetsammed |
| Enforce the memory check | on | Overridable, because the estimate is an estimate |

## The engine

```bash
cd ios
./scripts/build_llama_xcframework.sh          # 15-25 minutes
python3 scripts/prepare_project.py            # picks it up
xcodegen generate --spec project.generated.yml
```

That clones llama.cpp at the ref `native/ggml-hnx/build.sh` pins,
applies the HyperNix tensor-type patch so the phone can read sub-bit
models, and runs **upstream's own `build-xcframework.sh`** to produce
`ios/vendor/llama.xcframework`.

Upstream's script rather than a hand-listed Xcode target on purpose:
llama.cpp restructures its build between releases — `ggml-metal.m`
became `ggml-metal.cpp`, the Metal backend moved directory — and a
hand-maintained file list breaks on every bump in a way that reads as a
compiler error rather than as *"the list is stale"*. It also removed its
`Package.swift`, so the SPM route is gone.

Requires macOS with Xcode: an xcframework is produced by `xcodebuild`,
which does not cross-compile. On Linux, build the desktop engine with
`native/ggml-hnx/build.sh` instead.

### Builds without it

A checkout that has never run the script still generates and builds —
but only because `prepare_project.py` leaves the dependency out.

XcodeGen cannot decide this itself. Its `optional: true` sets weak
*linking*, a dynamic-linker property; the framework still has to exist
at build time. A spec that always names it fails with `There is no
XCFramework found at ...` on any checkout that has not built the
engine, which is what happened the first time this shipped. So the
decision is made in Python, where it can be tested both ways, and
`project.generated.yml` is what XcodeGen actually reads.

With the engine absent, `HNX_LOCAL_LLAMA` is empty,
`LlamaRunner.swift` compiles out entirely, and `LocalInference` falls
back to `EchoRunner`, which tells the user this build has no local
engine. Someone changing a view should not need a twenty-minute
llama.cpp compile.

The framework is **linked, not embedded**: upstream builds it with
`BUILD_SHARED_LIBS=OFF`, so it is static and its code goes into the app
binary. Embedding a static framework copies an archive into the bundle
for nothing, and App Store validation rejects it.

CI is the same: `local_engine` is a workflow input, off by default,
because two slices on a hosted macOS runner is 15–25 minutes nobody
should pay on a PR that touched a view.

### The API churns, so the symbols are pinned

`ios/vendor/llama-api-b10883.json` records every function, struct and
constant `include/llama.h` declared at the pinned ref — 236 functions —
and `tests/test_ios_llama_link.py` checks that every `llama_*` symbol
`LlamaRunner.swift` calls is in it.

That check exists because the C API moves a lot, and each of these was
the right name recently:

| was | is |
|---|---|
| `llama_load_model_from_file` | `llama_model_load_from_file` |
| `llama_new_context_with_model` | `llama_init_from_model` |
| `llama_free_model` | `llama_model_free` |
| `llama_kv_cache_clear(ctx)` | `llama_memory_clear(llama_get_memory(ctx), _)` |
| `params.use_mmap` / `use_mlock` | `params.load_mode` |

Tokenizer calls take a `const llama_vocab *` from
`llama_model_get_vocab(model)`, not the model.

### "JIT" loading is real, and it is `lazy_mode`

`llama_model_params.lazy_mode` reads the rows of tensors the
architecture marks **on demand** rather than pulling whole tensors up
front. HyperLink uses `LLAMA_LAZY_MODE_AUTO`, which applies it only to
tensors over 4 GiB — full `ON` is a per-model decision and not one to
make on someone's behalf.

Paired with `LLAMA_LOAD_MODE_MMAP`, so the weights are file-backed and
evictable rather than dirty anonymous pages. On iOS that is the
difference between pages the kernel can reclaim under pressure and pages
that count fully against the jetsam limit. Emphatically **not**
`MLOCK`: pinning gigabytes on a phone is the fastest way to be killed.

## What is still not verified

**None of the Swift has been compiled, and the engine has not been
built.** There is no Xcode, no Swift toolchain and no macOS in the
environment this was written in, so the build script has never run.

What *has* been checked: every llama.cpp symbol against the real header
at the pinned ref, the iOS and desktop refs agreeing, the framework
dependency being optional, the flag shipping off, and brace balance.
The first real macOS build is where compile errors would surface.

See also: [HyperLink sync](HyperLink-Sync.md) · [CLI](CLI.md) · [Home](Home.md)
