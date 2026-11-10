# Strife

Strife is HyperNix's own GGUF inference runtime. It reads a `.gguf`
itself (header, tokenizer, weights) and runs the transformer itself.
There is no llama.cpp process, no T1 key and no network.

> **Status: alpha 1, CPU only, in progress.** Strife loads and generates
> with real GGUF files of the architectures below, matches a
> `transformers` reference to within 1e-4 in tests, and does everything
> the engine section describes. It does **not** yet run on a GPU, and its
> CPU kernels are numpy, not native SIMD. For speed today, use the
> llama.cpp runner (`hnx-t1 runner start`). The performance targets below
> are goals; none are met yet, and nothing here claims otherwise.

## Commands

```bash
hnx strife inspect model.gguf            # what it is, and whether Strife runs it (loads nothing)
hnx strife inspect model.gguf --tensors  # every tensor, its type and shape
hnx strife run model.gguf -p "Once upon a time" -n 64
hnx strife run model.gguf -p "Hi" --chat --system "Be brief"   # through the chat template
hnx strife chat model.gguf               # a conversation; one session, prefix reused each turn
hnx strife diag model.gguf --gpu-layers 20   # hardware, backend, placement, fallback reasons
hnx strife types                         # every tensor type Strife reads
hnx strife selftest                      # build tiny llama/qwen2/qwen3 models and run them
```

Every command takes `--json`. `run --json` prints one JSON object per
streamed token, then a final `done` object with the stats. Errors carry
a stable code (`strife.<area>.<reason>`) and a remedy.

| Flag | Meaning |
|---|---|
| `-c`, `--ctx` | Context length. Default: the model's trained length, at most 4096. |
| `-t`, `--threads` | CPU threads for BLAS. Applied before numpy loads; `diag` and `status()` report whether it took. |
| `--cache-mb` | RAM for decoded weights. Default: 60% of free RAM. Weights beyond it are decoded a slab at a time on each use, so RAM stays bounded and speed drops. |
| `--gpu-layers` | Accepted and reported. Alpha 1 runs every layer on the CPU and says why. |
| `-n`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed` | Sampling. `--temp 0` is greedy. A seed makes sampling reproducible. |
| `--stop` | A stop string, repeatable. The text is cut before it. |
| `--timeout` | Seconds before the request stops with `timeout`. |

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
| CPU (numpy + the BLAS numpy ships with) | alpha 1: what Strife runs on |
| CPU native quantized kernels (SIMD, no decode-to-float) | next step |
| NVIDIA CUDA, Pascal (GTX 1080) through Blackwell (RTX 5080), partial layer offload | next step; `--gpu-layers` is reported, not used |
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
decodes at well under one token per second. The benchmark command
(`benchmarks/strife_bench.py`) comes with the native kernels, and
compares against the llama.cpp runner on the same machine and files.

## How it is built

The implementation plan, layer by layer, and where each lives:

| Layer | File | Notes |
|---|---|---|
| GGUF parsing | `hypernix/quant/gguf.py` (shared) | every length is checked against the bytes left in the file before it is read, so a hostile header cannot force a huge allocation |
| Inspection and validation | `strife/inspect.py` | offsets, alignment, truncation, overlap, overflow, architecture shapes, tokenizer sanity, all before any weight is touched |
| Tensor types | `strife/quants.py` | the verified decoders only |
| Weights | `strife/weights.py` | one read-only mmap; an LRU decode cache with a RAM budget; slab-wise matmul past it; row-only embedding lookup |
| Tokenizer | `strife/tokenizer.py` | gpt2 byte-level BPE with llama.cpp's pre-tokenizer patterns; SentencePiece with byte fallback; specials; sandboxed Jinja chat templates; a streaming decoder that never splits a character |
| Model | `strife/model.py` | RMSNorm, GQA with a causal mask, rope (norm/neox, linear scaling), SwiGLU, tied output; a KV cache bounded by `n_ctx` |
| Sampling | `strife/sampler.py` | repetition penalty, top-k, temperature, top-p, min-p, seeded |
| Engine | `strife/engine.py` | sessions (LRU-bounded, prefix reuse, per-session lock), fresh caches for sessionless requests, bounded concurrency and queue, stop strings with hold-back, cancellation, timeouts, stats |
| Placement | `strife/placement.py` | RAM, GPUs (via `hypernix.system.gpus`, every vendor), platform tier, the layer split and fallback reasons |
| CLI | `strife/cli.py` | `hnx strife …` |
| Fixtures | `strife/fixtures.py` | tiny real GGUFs of every architecture and type, with their weights in Hugging Face layout |

Next steps, in order:
1. Native CPU kernels that multiply quantised blocks directly, with the
   numpy path kept as the portable fallback.
2. CUDA offload of whole layers, checked against device capability
   (Pascal has no tensor cores and no bf16).
3. The benchmark command and its JSON/CSV/report output, compared with
   the llama.cpp runner.
4. An OpenAI-compatible local endpoint, tested against its contract.
5. Linux and Windows CI smoke jobs.

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
| Very slow decode | Expected in alpha 1 on large models. Check `diag`: if the weights exceed `--cache-mb`, they are decoded on every use. |
