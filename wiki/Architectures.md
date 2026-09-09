# Architectures — `ARCH_PRESETS` & `KNOWN_MODELS`

`hypernix` ships two registries. They solve different problems:

| Registry | Keys are | Values hold | Used by |
|---|---|---|---|
| `KNOWN_MODELS` | short names (`"nix2.5"`, `"gemma-4-e4b"`) | `ModelInfo(repo_id, arch, notes)` | `download_model`, `old_oven.preheat`, `hypernix chat` |
| `ARCH_PRESETS` | arch names (`"qwen2.5"`, `"gemma4"`, `"nix"`) | dict of config knobs | `new_oven(arch=...)` |

**`KNOWN_MODELS` resolves short names for *loading* pretrained
checkpoints.** You never need an ARCH_PRESET to load a pretrained
model — non-HyperNix `model_type` values route through
`transformers.AutoModelForCausalLM`.

**`ARCH_PRESETS` is for *building* a fresh parametric model** in the
shape of a known family. `new_oven(arch="qwen3.5", hidden_size=..., ...)`
stamps out an untrained model with the Qwen 3.5 config knobs set
correctly (rope_theta=1e7, tied embeddings, no qkv bias).

## `ARCH_PRESETS`

```python
from hypernix import ARCH_PRESETS

ARCH_PRESETS["gemma4"]
# {"attention_bias": False,
#  "model_type": "llama",       # runtime class we use for this shape
#  "rope_theta": 1_000_000.0,
#  "rms_norm_eps": 1e-6,
#  "tie_word_embeddings": True}
```

Full list at v0.41:

**HyperNix**
- `hypernix`

**Llama**
- `llama` — Llama 2 defaults (rope_theta=10k, untied)
- `llama3` / `llama3.1` / `llama3.3` / `llama4` — rope_theta=500k, untied
- `llama3.2` — rope_theta=500k, **tied** (small 1B/3B variants)

**Qwen**
- `qwen2` / `qwen2.5` — Qwen2 shape, attention_bias=True, rope_theta=1M, tied
- `qwen3` — Qwen3 shape, attention_bias=False, rope_theta=1M, tied
- `qwen3.5` — rope_theta=**10M**, tied
- `qwen3.6` — rope_theta=10M, **untied** (MoE)

**Mistral / Nemotron**
- `mistral` — attention_bias=False, rope_theta=1M
- `nemotron` — Llama-shape, rope_theta=500k

**Gemma**
- `gemma` / `gemma2` — Llama-backed, rope_theta=10k, tied
- `gemma3` — rope_theta=1M, tied
- `gemma4` — rope_theta=1M, tied (matches Gemma4ForConditionalGeneration)

**Phi**
- `phi3` — rope_theta=10k, untied
- `phi4` — rope_theta=**250k**, untied

**GLM**
- `glm4` — Qwen2-backed (has qkv bias), untied
- `glm5` / `glm5.1` — Llama-backed, rope_theta=1M, untied

**DeepSeek**
- `deepseek` / `deepseek-r1` — Llama-backed R1-distill shape

**OpenAI / Nix**
- `gpt-oss` / `gptoss` — Llama-backed, rope_theta=500k
- `nix` / `nix2` — Qwen2-backed, **no qkv bias** (unlike stock Qwen2), tied

### `model_type` inside presets

Because HyperNix implements Llama and Qwen2 natively, every preset
`model_type` is one of `"hypernix"`, `"llama"`, `"qwen2"`, or
`"mistral"` — the four arches the `HyperNixModel` class can realize
directly. Presets for other families (Gemma, Phi, GLM, Nemotron, etc.)
pick whichever of those four backs their shape most closely:

- bf16-friendly, no qkv bias → `"llama"`
- needs qkv bias (like GLM4) → `"qwen2"`
- exactly Mistral's conventions → `"mistral"`

This is why, for example, `ARCH_PRESETS["gemma4"]["model_type"]` is
`"llama"`. It's the runtime-class proxy, not the HF-side `model_type`.

## `KNOWN_MODELS`

```python
from hypernix import KNOWN_MODELS, resolve_repo_id, resolve_model_info

resolve_repo_id("nix2.5")          # -> "ray0rf1re/Nix2.5"
resolve_model_info("gemma-4-e4b")  # -> ModelInfo(repo_id="google/gemma-4-E4B-it",
                                   #             arch="auto",
                                   #             notes="Gemma 4 E4B it …")
```

Key sections (see the source for the full list — 60+ entries at v0.41):

### HyperNix native
`hyper-nix.1`, `hyper-nix`, `hypernix`, `nano-nano-v4`, `nano-nano`,
`nano-mini-6.99-v2`, `nano-mini`, `nano-nano-927-v3`, `nano-nano-927`

### Nix (ray0rf1re/nix collection, Qwen2-shape)
`nix`, `nix2.5`, `nix2.6`, `nix2.6-m`, `nix2.6-mm`, `nix-2.7a`, `nix2.7`

### Llama 3.x (gated)
`llama-3.1-8b`, `llama-3.1-8b-instruct`, `llama-3.2-1b`, `llama-3.2-3b`,
`llama-3.3-70b-instruct`

### Qwen 2.5 / 3 / 3.5 / 3.6
`qwen2.5-0.5b`, `qwen2.5-7b`, `qwen2.5-7b-instruct`, `qwen2.5-coder-7b`,
`qwen3-0.6b`, `qwen3-8b`,
`qwen3.5-0.8b`, `qwen3.5-2b`, `qwen3.5-4b`, `qwen3.5-9b`,
`qwen3.5-27b`, `qwen3.5-35b-a3b`, `qwen3.5-122b-a10b`, `qwen3.5-397b-a17b`,
`qwen3.6-35b-a3b`

### Gemma 2 / 3 / 4
`gemma-2-2b`, `gemma-2-9b`, `gemma-2-27b`,
`gemma-3-1b`, `gemma-3-4b`,
`gemma-4-e2b`, `gemma-4-e4b`, `gemma-4-26b-a4b`, `gemma-4-31b`

### Phi
`phi-3-mini`, `phi-3.5-mini`, `phi-4`

### DeepSeek
`deepseek-r1-distill-llama-8b`, `deepseek-r1-distill-qwen-7b`,
`deepseek-v2-lite`, `deepseek-v3`

### GLM
`glm-4-9b-chat`, `glm-4.1v`, `glm-5`, `glm-5.1`, `glm-5.1-fp8`

### Mistral / NVIDIA / gpt-oss
`mistral-7b-instruct`, `mixtral-8x7b-instruct`,
`nemotron-4-15b`, `llama-3.1-nemotron-70b-instruct`, `mistral-nemo-12b`,
`gpt-oss-20b`, `gpt-oss-120b`

### New in v0.71.4b2
`kimi-k3`, `claude-sonnet-4.6` (API), `claude-sonnet-5` (API), `claude-opus-4.8` (API),
`claude-haiku-4.5` (API), `fable-5`, `gpt-4o` (API), `gpt-5.6-terra` (API),
`gpt-5.6-sol` (API), `gpt-5.5` (API), `deepseek-r1`, `deekseek-v4flash`,
`qwen3.7-plus`, `gemma-4-27b`

## Brewer Presets (`hnx brew`)

The `hypernix.brewer` module defines the **hyperNix0x-v2** family of from-scratch
architecture presets. These are used for building new models, not loading pretrained checkpoints.

| Preset | CLI alias | Layers | d_model | Params | ctx |
|---|---|---|---|---|---|
| `hypernix0x_v2_33m` | `33m`, `micro` | 6 | 512 | **~33.6429M** | 4096 |
| `hypernix0x_v2_small` | `small` | 9 | 1024 | ~458M | 20482 |
| `hypernix0x_v2_medium` | `medium` | 18 | 1280 | ~918M | 40964 |
| `hypernix0x_v2_large` | `large` | 36 | 2048 | ~3.5B | 103724 |
| `hypernix0x_v2_cpu_nano` | `cpu-nano` | 4 | 128 | — | 512 |
| `hypernix0x_v2_cpu_tiny` | `cpu-tiny` | 6 | 256 | — | 1024 |
| `hypernix0x_v2_cpu_small` | `cpu-small` | 8 | 384 | — | 2048 |

The three `cpu-*` presets are sized to train on a CPU in a sitting
rather than to be good — they exist so the pipeline can be exercised
end to end without a GPU.

The **33m** preset (`hypernix0x_v2_33m`) was added in v0.71.4b2. It targets lightweight edge
devices, fast inference, and image text-to-text models (Vision support), with 33,642,900 parameters
exactly (`n_layers=6`, `d_model=512`, `d_ff=1444`, `n_heads=16`, `n_kv_heads=4`, GQA, sliding window `1024`).

### hyperNix0x-v2 in Neo oven

Since 0.72.4.dev10, Neo oven loads and trains the Brewer family
directly. `hypernix.models.brewer_adapter` is the bridge:

```python
from hypernix.models import neo_oven

model, config = neo_oven.preheat_brewed("./checkpoints/run-3.pt")
fresh        = neo_oven.new_brewed("small")
```

Plain `neo_oven.preheat(path)` also works: it recognises a Brewer
checkpoint and routes itself, so a path that used to fail now loads.

Recognition looks inside the file rather than trusting the extension. A
directory is identified by `d_model` and `n_layers` in its
`config.json` — together those are Brewer's and nothing else's, since
`HyperNixConfig` and every HF config use `hidden_size` and
`num_hidden_layers`. A `.pt` is a zip whose `data.pkl` member holds the
pickled object graph, so the top-level keys are read out of the first
64 KB of that member as literal bytes. Nothing is unpickled to answer
the question: `torch.load` on an untrusted file executes code, and
*"is this one of ours"* must never be the reason to run it.

The adapter is an `nn.Module` subclass rather than a proxy, so `.to()`,
`.parameters()`, `state_dict()`, gradient checkpointing and the
optimizer all keep working without knowing it exists. What it exists for
is one argument position:

```python
BrewerModel.forward(input_ids, attn_mask)   -> logits
neo_oven  calls  model(ids, labels=labels)  -> {"loss": ...}
```

Passed straight through, `labels` binds to `attn_mask`, and a tensor of
token ids gets used as an additive attention mask. That runs. It stays
finite. It trains into noise and never raises, which is the worst shape
a bug can have — so the binding is the adapter's whole job.

`wrap()` is idempotent and a no-op for anything that already speaks the
convention, so passing an HF model or an already-wrapped one back
through returns it unchanged.

### `arch="auto"` vs native

`ModelInfo.arch == "auto"` means the loader will use
`transformers.AutoModelForCausalLM`. This is correct for model types
without a native HyperNixModel implementation (`gemma4`, `qwen3_5`,
`qwen3_5_moe`, `glm_moe_dsa`, `phi3`, …). The other values
(`"hypernix"`, `"llama"`, `"qwen2"`, `"mistral"`) pick the matching
native path.

## Adding new entries

`KNOWN_MODELS` lives in
[`src/hypernix/download.py`](../src/hypernix/download.py).
`ARCH_PRESETS` lives in
[`src/hypernix/old_oven.py`](../src/hypernix/old_oven.py). Both are plain
Python dicts — add an entry, run the tests, open a PR.

For a KNOWN_MODELS entry, confirm the HF repo with a quick
`huggingface_hub.snapshot_download(..., allow_patterns=["config.json"])`
and make sure the `arch` string matches what `load_snapshot` will do
with it (`"auto"` for unfamiliar model_types).

For an ARCH_PRESET, verify against the official config.json on the
Hub: `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`,
`attention_bias` are the four knobs that matter.
