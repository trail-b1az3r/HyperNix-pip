# Quantization — the GGUF pipeline

The original reason `hypernix` exists: take a PyTorch snapshot of any
HyperNix-family model — the chat-tuned `ray0rf1re/hyper-Nix.2` (current
default) or the original `ray0rf1re/hyper-nix.1` (still fully supported)
— and emit GGUF files that `llama.cpp`, `ollama`, LM Studio, and friends
can load directly.

> **v0.51.3** rewrote the catalog: 30 distinct `QuantSpec` types
> (49 aliases) covering floats (`F32` / `F16` / `BF16`), legacy
> RTN quants (`Q4_0` … `Q8_0`), the full k-quant ladder
> (`Q2_K` … `Q6_K`), and the newer IQ-quants
> (`IQ1_S` … `IQ4_XS`). Helpers: `quant_recommended()`,
> `quant_by_category("k")`, `quant_for_size(target_bytes,
> fp16_bytes)`, `quant_estimate_size("q4km", fp16_bytes)`,
> `quant_resolve_spec("q4km")`. See **Catalog** below.

## The pipeline, in three stages

```
    HF snapshot           fp16 GGUF             q-quant GGUFs
  (config.json +       (llama.cpp native)      (Q8_0 / Q6_K /
   model.safetensors)                           Q4_K_M / Q5_K_M)
         │                      │                     │
         │  convert_to_gguf     │  quantize_gguf     │
         └──────────────────────┴─────────────────────┘
```

Each stage is one library call and one CLI flag:

```bash
# all stages in one go (default: just fp32 + fp16)
hypernix --repo-id ray0rf1re/hyper-nix.1 --output-dir ./out \
    --quants fp32 fp16 q8_0 q6_k q4_k_m
```

```python
from hypernix import download_model, convert_to_gguf, quantize_gguf

snap = download_model("ray0rf1re/hyper-nix.1")
fp16 = convert_to_gguf(snap, "hn-fp16.gguf", dtype="fp16")
q4   = quantize_gguf(fp16, "hn-q4_k_m.gguf", "q4_k_m")
q6   = quantize_gguf(fp16, "hn-q6_k.gguf",   "q6_k")
q8   = quantize_gguf(fp16, "hn-q8_0.gguf",   "q8_0")
```

## `convert_to_gguf`

Architecture-agnostic. It reads the state dict directly, infers all
dimensions from tensor shapes, and maps HF tensor names onto
llama.cpp's canonical GGUF layout when a recognizable pattern
matches (Llama, GPT-NeoX, GPT-2, nanoGPT). Unknown names round-trip
verbatim so the output is always loadable somewhere downstream.

```python
convert_to_gguf(
    model_dir="./snapshot",
    output="./out-fp16.gguf",
    dtype="fp16",                 # or "fp32"
    arch_name="hypernix",         # sets "general.architecture"
    name="HyperNix",              # display name in the header
    n_head_hint=None,             # override head count if config is wrong
    context_length=None,          # override context length
)
```

No llama.cpp installation needed for this stage — it uses the
pure-Python `gguf` library. Produces fp32 or fp16 tensors.

## One step: `hypernix convert -P`

```bash
hnx convert ./snapshot -P -Q Q4_K_M                 # safetensors -> F16 -> Q4_K_M
hnx convert ./snapshot/model.safetensors -P -Q q6h4 -o model.q6h4.gguf
hnx convert ./brewer_models/mine -P -Q FP8          # a hyperNix0x-v2 model
```

Converts a safetensors model and quantises the result with
[hyprslug](HyprSlug.md), deleting the F16 in between. A `.gguf` is
quantised as it is. Any hyprslug target works, including the HyperNix
tiers and hybrids. See [CLI](CLI.md#convert).

## Without llama.cpp: `-hnx` and hyprslug

`hypernix quantize ... -hnx`, `steamroller ... -hnx` and `hyprslug`
itself quantise in Python, with no `llama-quantize` looked for,
downloaded or built. They write every upstream type below (`Q2_K` to
`Q8_0`, and the `_S`/`_M`/`_L` mixes) plus what `llama-quantize` cannot:

| Targets | What | Runs in |
|---|---|---|
| `IQ0.9_L` … `IQ0.25_UXL`, `INT1`, `HNX_1375BIT` | sub-bit and one-bit | HnxRun, patched llama.cpp |
| `INT8`, `INT4`, `INT3`, `INT2`, `FP2`, `FP8` | fixed codebooks | HnxRun, patched llama.cpp |
| `Q8_K` | llama.cpp's activation block, as a weight | HnxRun, patched llama.cpp |
| `hnx_Q6_H_k` (`q6h`), `hnx_Q6_H_4` (`q6h4`), `hnx_Q6_H_2` (`q6h2`) | hybrids chosen per tensor by role and depth | patched llama.cpp |
| `FP32`, `FP16`, `BF16` | width conversions | any llama.cpp |

"Patched llama.cpp" is a build with
[`native/ggml-hnx`](../native/ggml-hnx/README.md). See
[HyprSlug](HyprSlug.md) for the recipes and [LowBit](LowBit.md) for the
types.

## `quantize_gguf`

Drives the `llama-quantize` binary. The full alias table lives in
`hypernix.QUANT_TYPES` (49 entries → 30 distinct quant types in
`hypernix.QUANT_CATALOG`). Recommended starting points:

| Alias | llama.cpp enum | bpw | Typical use |
|---|---|---|---|
| `fp32` / `f32` | F32 | 32.0 | Reference / debugging |
| `fp16` / `f16` | F16 | 16.0 | Default intermediate |
| `bf16` | BF16 | 16.0 | Better range than F16 |
| `q8` / `q8_0` | Q8_0 | 8.5 | Small model, near-lossless |
| `q6` / `q6_k` | Q6_K | 6.56 | Balanced accuracy/size |
| `q5km` / `q5_k_m` | Q5_K_M | 5.83 | A notch above Q4_K_M |
| `q4km` / `q4_k_m` | Q4_K_M | 4.83 | Most common small-GPU choice |
| `iq4_xs` | IQ4_XS | 4.25 | New IQ family, sub-Q4_K_M tier |
| `q3_k_m` | Q3_K_M | 3.75 | Aggressive size reduction |
| `iq3_s` | IQ3_S | 3.44 | Beats Q3_K_M at similar size |
| `q2_k` | Q2_K | 2.625 | Smallest k-quant, significant loss |
| `iq2_m` | IQ2_M | 2.7 | 2-bit IQ, beats Q2_K |
| `iq1_s` | IQ1_S | 1.5625 | Extreme size reduction |

## Catalog helpers (v0.51.3)

```python
from hypernix import (
    QUANT_CATALOG, quant_recommended,
    quant_by_category, quant_for_size,
    quant_estimate_size, quant_resolve_spec,
)

# 30 specs total
print(len(QUANT_CATALOG))                       # 30

# Curated short-list (F16, Q8_0, Q4_K_M, Q5_K_M, Q6_K)
for s in quant_recommended():
    print(s.name, s.bits_per_weight, s.notes)

# All k-quants, sorted ascending by bpw
for s in quant_by_category("k"):
    print(s.name, s.bits_per_weight)

# Pick the largest quant that fits in a target byte budget
fp16_bytes = 2_000_000_000  # 2 GB
spec = quant_for_size(target_size_bytes=900_000_000, fp16_size_bytes=fp16_bytes)
print("recommend:", spec.name)                   # "Q6_K" for this budget

# Estimate output size without running llama-quantize
print(quant_estimate_size("q4km", fp16_bytes))   # ≈ 603 MB

# Look up a single spec from any alias
print(quant_resolve_spec("q4km"))                # QuantSpec(name='Q4_K_M', ...)
```

`QuantSpec` is a frozen dataclass with `name`, `bits_per_weight`,
`category` (`"float" / "legacy" / "k" / "iq"`), `size_factor`
(`bpw / 16`), `notes`, and `recommended`.

### `llama-quantize` binary resolution

In priority order:

1. Explicit `--llama-quantize /path/to/bin`
2. `llama-quantize` on `PATH`
3. `llama.cpp` bundled with `pip install "hypernix[llama-cpp]"`
4. System package manager locations:
   - Arch: `/usr/bin/llama-quantize`
   - Fedora: `/usr/bin/llama-quantize`
   - openSUSE: `/usr/bin/llama-quantize`
   - Homebrew (macOS): `/opt/homebrew/bin/llama-quantize`
   - Windows: `scoop install llama.cpp` → `~/scoop/apps/llama.cpp/current/bin/`
5. Auto-download from the upstream
   [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp/releases)
   release to `~/.cache/hypernix/bin/` (CPU-only binary, CUDA variants
   available under other asset names).

Disable auto-download with `--no-auto-fetch`, pre-seed the cache with
`hypernix fetch-llama-quantize`, or provide `--llama-quantize` explicitly.

### The `--auto` flag

```bash
hypernix quantize --source hn-fp16.gguf --output hn-q4.gguf \
                  --type q4_k_m --auto
```

Adds two fallbacks on top of the normal resolution:

- Walks back through the latest N `llama.cpp` releases if the newest
  release has no asset for your OS / CPU variant.
- Falls back to `pip install llama-cpp-python` (which ships a bundled
  `llama-quantize`) if every GitHub fetch fails.

## `verify` and `info`

```bash
hypernix verify hn-q4_k_m.gguf
hypernix verify hn-q4_k_m.gguf --tensors     # also list tensors

hypernix info --gguf hn-q4_k_m.gguf          # package + GGUF headers
```

`verify` parses the GGUF header with the `gguf` library and prints
fields + tensor metadata. Use it as a CI guard to catch corrupted
downloads; the exit code is non-zero if parsing fails.

## `upload`

```bash
HF_TOKEN=hf_xxx hypernix upload \
    --repo-id ray0rf1re/HyperNix.1-gguf \
    ./hn-q4_k_m.gguf ./hn-q6_k.gguf ./hn-q8_0.gguf
```

Or as part of `hypernix all`:

```bash
hypernix --repo-id ray0rf1re/hyper-nix.1 --output-dir ./out \
    --quants fp32 fp16 q8_0 q6_k q4_k_m \
    --upload-to ray0rf1re/HyperNix.1-gguf
```

Uses `huggingface_hub.HfApi.upload_folder` under the hood.

## Common gotchas

**"huggingface-hub not found"** — the lazy import failed. `pip install
huggingface-hub>=0.24`; it's a core dep of hypernix so this shouldn't
happen on a clean install.

**GGUF won't load in llama.cpp** — check the
`general.architecture` string with `hypernix verify`. llama.cpp only
knows certain architecture strings; if you built a custom shape with
`new_oven` and called it `"my-arch"`, the GGUF will still write but
llama.cpp will reject it. Use `--arch llama` or `--arch hypernix` when
converting.

**Out-of-memory during quantization** — `llama-quantize` loads the
whole fp16 tensor into RAM for each quantization pass. On a 32 GB
box the largest model you can quantize in one shot is roughly 13 B
params (the tensor in fp16 alone is 26 GB). Use a box with more RAM
or a split-file GGUF.

**Laptop-grade quantization** — `scripts/quantize_i7_7660u.sh` caps
BLAS/OpenMP to 4 threads, runs at reduced CPU + I/O priority when
`nice` / `ionice` are available, and only keeps one fp16 intermediate
on disk. Good reference for tuning your own small-box pipeline.
