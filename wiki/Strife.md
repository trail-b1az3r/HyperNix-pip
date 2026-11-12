# Strife

Strife is HyperNix's own GGUF inference runtime. It reads a `.gguf`
itself (header, tokenizer, weights) and runs the transformer itself.
There is no llama.cpp process, no T1 key and no network.

> **Status: alpha 1.** Strife loads and generates with real GGUF files of
> the architectures below, matches a `transformers` reference to within
> 1e-4 in tests, and does everything the engine section describes.
> - Its native CPU kernels are built on your machine with `hnx strife
>   kernels build`. Until you run that, it computes with numpy.
> - `--gpu-layers` puts whole blocks on a CUDA device through PyTorch. That
>   path is tested on PyTorch's CPU device only: **it has not been run on a
>   real GPU by its author**.
> - The performance targets below are goals. **None has been measured**, so
>   none is claimed. `hnx strife benchmark` on your hardware is the
>   evidence. For speed today, the llama.cpp runner (`hnx-t1 runner start`)
>   is the safer choice.

## Commands

```bash
hnx strife inspect model.gguf            # what it is, and whether Strife runs it (loads nothing)
hnx strife inspect model.gguf --tensors  # every tensor, its type and shape
hnx strife run model.gguf -p "Once upon a time" -n 64
hnx strife run model.gguf -p "Hi" --chat --system "Be brief"   # through the chat template
hnx strife chat model.gguf               # a conversation; one session, prefix reused each turn
hnx strife diag model.gguf --gpu-layers 20   # hardware, backend, placement, fallback reasons
hnx strife benchmark model.gguf --out-dir results/   # measure it: JSON, CSV and a report
hnx strife types                         # every tensor type Strife reads
hnx strife selftest                      # build tiny llama/qwen2/qwen3 models and run them
hnx strife kernels build                 # compile the native kernels for this CPU and verify them
hnx strife kernels status                # what is built, and for which tensor types
```

Every command takes `--json`. `run --json` prints one JSON object per
streamed token, then a final `done` object with the stats. Errors carry
a stable code (`strife.<area>.<reason>`) and a remedy.

| Flag | Meaning |
|---|---|
| `-c`, `--ctx` | Context length. Default: the model's trained length, at most 4096. |
| `-t`, `--threads` | CPU threads for BLAS. Applied before numpy loads; `diag` and `status()` report whether it took. |
| `--cache-mb` | RAM for decoded weights. Default: 60% of free RAM. Weights beyond it are decoded a slab at a time on each use, so RAM stays bounded and speed drops. |
| `--kernels` | `auto` (default) uses the native kernels when they are built; `off` computes with numpy. |
| `--gpu-layers` | Blocks whose weight matrices go on the GPU, through PyTorch. `-1` is as many as fit. What was placed, and why not more, is reported. See [GPU offload](#gpu-offload). |
| `--gpu-device` | `cuda:N`. Default: the first CUDA device. |
| `-n`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed` | Sampling. `--temp 0` is greedy. A seed makes sampling reproducible. |
| `--stop` | A stop string, repeatable. The text is cut before it. |
| `--timeout` | Seconds before the request stops with `timeout`. |

## Native kernels

Strife's kernels are one C file (`strife/native/strife_kernels.c`) that
decodes quantised blocks and multiplies by a weight while it is still
quantised. They are compiled on the machine that runs them, for that CPU:

```bash
hnx strife kernels build      # needs cc, gcc or clang; about a second
hnx strife kernels status
hnx strife kernels verify     # check the built library again
hnx strife kernels remove
```

Nothing is compiled unless you ask. Until `build` has been run, Strife
computes with numpy exactly as before. `HNX_STRIFE_KERNELS=build` builds on
first use, and `off` ignores a built library.

What `build` does:

- It compiles with `-O3 -march=native -fopenmp`, and falls back to fewer
  flags when the compiler lacks one. The dot products and the K-quant
  unpacking are written for AVX-512 and AVX2, with plain C for everything
  else.
- It **verifies every tensor type** before recording the build. Generated
  blocks covering every bit pattern are decoded by the library and by the
  Python decoders, and the quantised multiply is compared with the decoded
  one. A type that does not agree is not used; the others are.
- It writes the library under `~/.hypernix/strife/kernels/`, owner-only.
  Strife refuses to load it if anyone else can write it. The build is keyed
  by the source, the compiler and the CPU, so a new CPU or a new HyperNix
  gets a new build.

Kernels exist for F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
Q4_K, Q5_K and Q6_K. The HyperNix sub-bit and low-bit types have none yet:
a model that is mostly those types keeps the numpy path.

When the kernels can carry a model (at least 90% of its matrix bytes are in
types they have), Strife:

- multiplies by each weight in place, so nothing is decoded whole and the
  decode cache holds only the small tensors;
- holds BLAS to one thread while the engine is open, and gives the threads
  back when it closes. Two thread pools on the same cores are slower than
  either alone: a native multiply that takes 0.7 ms alone took 8 ms right
  after a BLAS call. Without `threadpoolctl`, Strife does this itself for
  the OpenBLAS that numpy's wheels ship, on Linux. Where it cannot limit
  BLAS, the kernels are used for decoding only, and `diag` says so.

`status()`, `diag` and the server's start line report which path is in use
and why (`kernels.note`).

Measured on one 4-vCPU x86-64 machine with AVX-512, on LFM3 1.2B in
Q4_K_M (1.1 GB), with `benchmarks/strife_bench.py -p 256 -n 64 -r 3`:

| | numpy | native kernels |
|---|---|---|
| Decode | 6.2 tok/s | 7.8 tok/s |
| Prefill, warm | 139 tok/s | 76 tok/s |
| First answer after load (a 20-token prompt) | 41 s | 9 s |
| RAM for decoded weights | 6.6 GB | 1 MB |

These are one machine's numbers, not a claim about yours. Note the prefill
row: with many tokens at once, four-thread BLAS on fully decoded weights is
faster than the kernels, once the 6.6 GB of decoding has been paid for. The
kernels win on decode speed, on memory and on the time to the first answer.
The llama.cpp comparison has not been run: there is no llama.cpp build on
that machine.

## GPU offload

`--gpu-layers N` puts the weight matrices of the first N blocks on a CUDA
device through PyTorch. Each is decoded once to half precision and its
multiplies run there. Attention arithmetic, norms and the KV cache stay in
host memory, so this is partial offload of the heavy tensors, not of the
whole computation.

```bash
hnx strife diag model.gguf --gpu-layers -1      # plan the split; loads nothing
hnx strife run model.gguf -p "Hi" --gpu-layers 12
```

- **It needs PyTorch with CUDA.** Without one, the answer is
  `gpu_layers_used: 0` and the reason.
- **Free memory decides.** Layers are placed in order while they fit in the
  card's free memory less a 768 MiB reserve. A model that does not fit is
  split, and the placement states each side: layers on the GPU, layers on
  the CPU, device bytes, and why.
- **A card PyTorch was not built for is refused by name.** A GTX 1080 has
  compute capability 6.1, and current PyTorch wheels do not include its
  kernels. Strife says so and runs on the CPU. A PyTorch build that includes
  the card would be used.
- **Not run on real hardware by its author.** The device path is tested on
  PyTorch's CPU device, where it gives the same tokens as the CPU path on
  every architecture. No speed on any GPU has been measured.

## Your own build as a T1 runner

A Strife you built yourself can serve for T1. It might be a checkout with
your changes, or HyperNix installed in another interpreter. Register it
once:

```bash
hnx strife build mine --source ~/src/HyperNix-pip          # a checkout (its src/ is used)
hnx strife build fast --python ~/venvs/strife/bin/python   # an interpreter with HyperNix
hnx strife build mine --source ~/src/HyperNix-pip --default   # what strife loads use by default
hnx strife build --list
hnx strife build --remove fast
```

Registering runs the build's own selftest in its own interpreter: a
generated model through load, prefill and decode. Only a build that passes
is recorded, in `~/.hypernix/strife/builds.json` (`HYPERNIX_HOME` moves it).
The file is owner-only, and Strife refuses it if anyone else can write it.

On T1, `GET /runner/status` lists `strife_builds`, and a load picks one by
name:

```json
POST /runner/load
{"model_id": "qwen3.5-2b", "backend": "strife", "strife_build": "mine"}
```

The API takes registered names only, never a path. `"builtin"` is the
Strife in the installed package. Without `strife_build`, the default build
is used (`--default`, or `HNX_STRIFE_BUILD=NAME` on the server). The
placement reason says which build is serving.

## Python

```python
from hypernix.strife import Engine, SamplerConfig, inspect_model

report = inspect_model("model.gguf")          # problems listed, nothing allocated
with Engine("model.gguf", n_ctx=4096, max_concurrent=2, max_sessions=8) as engine:
    for event in engine.stream(messages=[{"role": "user", "content": "Hi"}], max_tokens=64,
                               sampling=SamplerConfig(temperature=0.7, seed=1)):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
    print(event["stop_reason"], event["stats"])

    sid = engine.new_session()                  # a conversation that reuses its prefix
    engine.generate("…", session=sid, max_tokens=32, stop=["\n\n"], timeout=30)
    engine.status()                             # memory, placement and settings
```

`stop_reason` is one of `stop_token`, `stop_string`, `length`,
`context_full`, `cancelled` or `timeout`. The stats separate prefill
from decode, giving tokens and seconds for each and the time to first
token. They also report how much of the prompt a session reused.

## Capability matrix

### Architectures

| Architecture | Status | Differences handled |
|---|---|---|
| `llama` | supported | adjacent-pair rope (the GGUF converter's q/k permutation) |
| `mistral` | supported | as llama |
| `qwen2` | supported | NeoX rope, q/k/v biases |
| `qwen3` | supported | NeoX rope, per-head q/k RMSNorm |
| anything else | refused at inspection (`strife.arch.unsupported`) | the llama.cpp runner covers more |

Tied output (no `output.weight`), grouped-query attention and linear
rope scaling are handled for all four. Sliding-window attention, MoE,
YaRN/NTK rope scaling and multimodal models are not implemented. Models
that need them are refused at inspection or listed here as gaps.

### Tensor types

A type is supported only when it has a decoder tested against its own
encoder. An unknown type is refused by name and never read as a
neighbouring type.

| Types | Source | Status |
|---|---|---|
| F32, F16, BF16 | read directly | supported |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | `hypernix.quant.llamaquants` (bit-compatible with llama.cpp) | supported |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | the same | supported; this covers Q4_K_M and Q3_K_* files |
| HyperNix sub-bit: IQ0.9_L, IQ0.75_M, IQ0.5_XXXL, IQ0.25_UXL, INT1, HNX_1375BIT, HNX_15125BIT, HNX_IQ1S | `hypernix.quant.subbit` via `hyprslug.TIER_TYPES` | supported |
| HyperNix low-bit: INT2, INT3, INT4, INT8, FP2, FP8 | `hypernix.quant.lowbit` | supported |
| IQ1_S, IQ1_M, IQ2_*, IQ3_*, IQ4_NL, IQ4_XS, Q8_1, I8–I64, F64 | none | refused (`strife.quant.unsupported`) |

Q4_K_M and Q3_K_M are mixes of these types, so a file in either runs.

### Backends and platforms

| Backend | Status |
|---|---|
| CPU (numpy + the BLAS numpy ships with) | alpha 1: what Strife runs on until the kernels are built |
| CPU native quantized kernels (AVX-512, AVX2, plain C; threads through OpenMP) | alpha 1, after `hnx strife kernels build`: the llama.cpp block types and the float types. The weights are decoded a tile at a time and multiplied in float32; llama.cpp's integer dot products are not implemented |
| NVIDIA CUDA through PyTorch, partial layer offload (`--gpu-layers`) | alpha 1, tested on PyTorch's CPU device only; not run on a real GPU. Which cards work depends on the installed PyTorch: Pascal (GTX 1080) is not in current wheels and is refused by name |
| AMD | only where the repository's backend can be exercised and tested |

| Platform | Tier |
|---|---|
| Linux x86_64 | alpha target |
| Windows | alpha target |
| macOS, Apple M5 or newer | experimental |
| macOS, older Apple silicon or Intel | not supported (`diag` says so) |

## Performance targets

These are the targets the alpha is measured against. They are goals, not
results. A benchmark that does not meet one reports it as unmet.

| Hardware | Workload | Target |
|---|---|---|
| GTX 1080, 8 GB | 27B Q4_K_M (Qwen family), partial offload | ≥ 8 tok/s, aiming for 8–14 |
| GTX 1080, 8 GB | 32B Q3_K (QwQ family) | 5 tok/s |
| RTX 3090 Ti / 4080 / 5080 | comparable supported workloads | 12+ tok/s |

A 27B or 32B quantised model does **not** fit entirely in 8 GB of VRAM.
On that card Strife will place what fits and keep the rest in host
memory, and it will say so. Today, on the CPU path, a model of that size
decodes at well under one token per second.

Measure it with the benchmark:

```bash
hnx strife benchmark model.gguf -p 512 -n 128 -r 5 --out-dir results/
hnx strife benchmark model.gguf --target 8          # exits 1 when unmet
hnx strife benchmark model.gguf --numpy             # also without the native kernels
hnx strife benchmark model.gguf --gpu-layers -1
```

`python benchmarks/strife_bench.py` is the same program. It takes a model
that is already on the machine and never downloads one.

| Measured | How |
|---|---|
| Cold run | The first run, reported on its own and kept out of the medians. |
| Prefill and decode tok/s | The median of `-r` runs in fresh sessions, with the spread. `--warmup` runs are discarded first. |
| First token | Seconds from the request to the first generated token. |
| Token latency | The time between tokens: p50, p90, p99 and max. |
| Memory | Peak resident memory, the decode cache, mapped bytes, and device memory when layers are offloaded. |
| Quality | Greedy output from a fixed prompt must repeat exactly. With `--numpy`, the native kernels and numpy must give the same tokens. A failed check fails the run (exit 3). |
| llama.cpp | `llama-bench` on the same file with the same lengths, threads and GPU layers, when a build is found. Otherwise the row says it was skipped and why. |

`--out-dir` writes `strife-bench.json` (the file's SHA-256, the HyperNix
version, the machine, every setting and result), `strife-bench.csv` (one
row per runtime) and `strife-bench.md` (a report). Compare records from one
machine only.

The `strife-benchmark` workflow runs this on a self-hosted runner with the
hardware and the model, on request, and uploads the record.

## How it is built

The implementation plan, layer by layer, and where each lives:

| Layer | File | Notes |
|---|---|---|
| GGUF parsing | `hypernix/quant/gguf.py` (shared) | every length is checked against the bytes left in the file before it is read, so a hostile header cannot force a huge allocation |
| Inspection and validation | `strife/inspect.py` | offsets, alignment, truncation, overlap, overflow, architecture shapes, tokenizer sanity, all before any weight is touched |
| Tensor types | `strife/quants.py` | the verified decoders only |
| Weights | `strife/weights.py` | one read-only mmap; an LRU decode cache with a RAM budget; slab-wise matmul past it; row-only embedding lookup; the native multiply when the kernels are built |
| Native kernels | `strife/native/strife_kernels.c`, `strife/kernels.py`, `strife/blasctl.py` | the C source, its build/verify/load, and the BLAS thread control that keeps the two thread pools apart |
| Tokenizer | `strife/tokenizer.py` | gpt2 byte-level BPE with llama.cpp's pre-tokenizer patterns; SentencePiece with byte fallback; specials; sandboxed Jinja chat templates; a streaming decoder that never splits a character |
| Model | `strife/model.py` | RMSNorm, GQA with a causal mask, rope (norm/neox, linear scaling), SwiGLU, tied output; a KV cache bounded by `n_ctx` |
| Sampling | `strife/sampler.py` | repetition penalty, top-k, temperature, top-p, min-p, seeded |
| Engine | `strife/engine.py` | sessions (LRU-bounded, prefix reuse, per-session lock), fresh caches for sessionless requests, bounded concurrency and queue, stop strings with hold-back, cancellation, timeouts, stats |
| Placement | `strife/placement.py` | RAM, GPUs (via `hypernix.system.gpus`, every vendor), platform tier, the layer split and fallback reasons |
| CLI | `strife/cli.py` | `hnx strife …` |
| Fixtures | `strife/fixtures.py` | tiny real GGUFs of every architecture and type, with their weights in Hugging Face layout |

Next steps, in order:
1. Run the benchmark on the target cards and record the results against
   the targets above, with the llama.cpp runner beside them.
2. Native kernels for the HyperNix sub-bit and low-bit types, ARM NEON
   paths, and integer dot products for the K-quants.
3. GPU offload without PyTorch in the path, and the KV cache and attention
   on the device.

## Checked on a real model

Qwen2.5-0.5B-Instruct, Q4_K_M (468 MiB, a mix of Q4_K, Q5_0, Q6_K, Q8_0 and
F32), on a 4-thread x86_64 CPU:
- **Tokenizer:** identical token ids to Hugging Face's, accented and CJK
  text included.
- **Logits:** within 1.5e-4 of `transformers` loading the same GGUF, on
  logits of about ±25. The most likely next token is the same at every
  position.
- **Generation:** "What is the capital of France?" through the chat
  template gives "The capital of France is Paris." and stops on the
  end-of-turn token. Decode ran at 15.4 tok/s. The first token took
  15 s, because the first pass decodes every weight into the cache.

Run it yourself with `STRIFE_TEST_MODEL=/path/to/model.gguf pytest
tests/strife -k real`. It is opt-in, and nothing is downloaded.

## Tests

`tests/strife/test_strife.py` builds its own tiny models, so nothing is
downloaded:
- **Logits:** llama, qwen2, qwen3 and tied-output models match
  `transformers` at every position within 1e-4. Cached one-token decoding
  matches a full recompute. A zero decode-cache budget changes nothing.
- **Quant types:** every K-quant and legacy type, and every advertised
  HNX type, runs exactly what the file stores (`transformers` over the
  decoded weights).
- **Tokenizers:** the BPE tokenizer matches the `tokenizers` library.
  Text round-trips, special tokens stay single tokens, and SentencePiece
  byte fallback works. A malicious chat template is refused by the
  sandbox.
- **Inspection:** truncated files, bad magic, a hostile array length, an
  unsupported type and an unsupported architecture are each refused with
  their code.
- **Engine:** greedy and seeded determinism, streaming equals the result,
  stop strings, cancellation before and during a stream, timeouts, a full
  context, session prefix reuse, the session bound, concurrent requests
  matching sequential ones, and a full queue refused.
- **Placement and CLI:** a GPU request answered honestly, Macs scoped to
  M5 and newer, and the CLI commands.

## Troubleshooting

| Message | Meaning |
|---|---|
| `strife.arch.unsupported` | The architecture is not one of the four. Use the llama.cpp runner. |
| `strife.quant.unsupported` | A tensor type has no verified decoder (IQ-family types, for example). Re-quantise, or use llama.cpp. |
| `strife.model.truncated` | The file is shorter than its tensor table says. Download it again. |
| `strife.runtime.context_full` | The prompt and output don't fit `--ctx`. Raise it or shorten the prompt. |
| `strife.engine.busy` | Every slot is busy and the queue is full. Retry, or raise `max_queue`. |
| Very slow decode | Run `hnx strife kernels build`. Without the kernels, check `diag`: if the weights exceed `--cache-mb`, they are decoded on every use. |
| `--gpu-layers` placed nothing | `diag MODEL --gpu-layers N` gives the reason: no PyTorch, no CUDA device, a card this PyTorch was not built for, or not enough free memory. |
| `strife.kernels.no_compiler` | Install gcc or clang, or set `CC`. Strife runs on numpy without the kernels. |
