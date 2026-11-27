# Training — scratch, expand, and fine-tune

> **Looking for "which of these should I use?"** — that is
> [the Model Training Guide](Model-Training-Guide.md). This page is the
> `hypernix.train` API reference.

Three flows live under `hypernix.train`:

1. **`init_from_scratch`** — build a fresh, randomly-initialized
   HyperNix snapshot at any shape.
2. **`expand_checkpoint`** — warm-start a bigger model from a smaller
   one; overlapping rows / columns copy over, extra slots get
   `N(0, std)`.
3. **`train`** — minimal causal-LM training loop; consumes a
   snapshot directory + a raw-text file and writes a new snapshot.

All three output the same thing: a standard HuggingFace-style directory
(`config.json` + `model.safetensors` + optional tokenizer files) that
round-trips through `load_snapshot`, `old_oven.preheat`, or
`hypernix convert`.

Plus a fourth path for non-native models:

4. **AutoModel fallback** — any config with a `model_type` outside
   `{hypernix, llama, qwen2, mistral}` is loaded via
   `transformers.AutoModelForCausalLM`, wrapped in a thin adapter, and
   trained or sampled through the same `load_snapshot` surface.

## `HyperNixConfig`

The parametric knob set:

```python
from hypernix import HyperNixConfig

cfg = HyperNixConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=None,     # default = num_attention_heads (no GQA)
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
    attention_bias=False,         # True for Qwen2-shape
    model_type="hypernix",
    rope_style="interleaved",     # "interleaved" for HyperNix native,
                                  # "half-rotate" for HF Llama/Qwen2
)
```

Validation runs in `__post_init__` — divisibility checks
(`hidden_size % num_attention_heads == 0`), head-dim parity,
`num_attention_heads % num_key_value_heads == 0`, and the
`rope_style in {"interleaved", "half-rotate"}` gate.

## `init_from_scratch`

```python
from hypernix import HyperNixConfig, init_from_scratch

out = init_from_scratch(
    "./fresh",
    cfg,
    tokenizer_source="./hyper-nix-v1",   # copies tokenizer files over
    seed=0,                              # deterministic init
)
# -> ./fresh/config.json, ./fresh/model.safetensors,
#    ./fresh/tokenizer.json, etc.
```

If `tokenizer_source` is `None`, no tokenizer files are copied; the
oven falls back to a byte tokenizer. Clamp `vocab_size=256` in that
case so the embedding matrix matches the byte range 0..255.

## `expand_checkpoint`

Warm-start a bigger model from a smaller one. Overlapping rows /
columns copy verbatim; new slots init from `N(0, init_std)`; extra
blocks duplicate the last old block.

```python
from hypernix import expand_checkpoint

expand_checkpoint(
    src_dir="./hyper-nix-v1",
    dst_dir="./hyper-nix-v2",
    hidden_size=1536,            # up from 1024
    intermediate_size=6144,      # up from 4096
    num_hidden_layers=24,        # up from 16
    num_attention_heads=24,      # up from 16
    vocab_size=None,             # keep original
    init_std=0.02,
    seed=0,
)
```

Width axes (hidden / intermediate / heads / vocab) and depth (layers)
can both grow. Each tensor that differs in shape gets a fresh
`torch.empty()` initialized with `N(0, init_std)`, then the source
tensor's values are copied into the top-left block.

## `train`

```python
from hypernix import train

out = train(
    model_dir="./hyper-nix-v2",
    dataset_path="./corpus.txt",
    out_dir="./hyper-nix-v2-trained",
    steps=1000,
    batch_size=2,
    context_length=512,
    lr=3e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    device=None,                 # "cuda" if available else "cpu"
    dtype="float32",             # "float32", "float16", "bfloat16"
    log_every=10,
    save_every=500,
    seed=0,
)
```

Training details:

- **Optimizer** — `AdamW(lr, weight_decay, betas=(0.9, 0.95))`.
- **Scheduler** — `CosineAnnealingLR(T_max=steps)`.
- **Loss** — `CrossEntropyLoss` on next-token prediction (token at
  position `i+1` predicted from positions `0..i`).
- **Chunking** — the dataset is concatenated, tokenized, and split
  into non-overlapping `context_length`-sized chunks that are stacked
  into batches.
- **Dtype** — passed through to the model's `.to(dtype=...)`; does not
  wrap in `torch.autocast`. For mixed-precision Pascal training, wrap
  the call yourself:

  ```python
  with torch.amp.autocast("cuda", dtype=torch.float16), \
       torch.cuda.amp.GradScaler():
      train(...)
  ```

- **Checkpointing** — every `save_every` steps a full snapshot is
  written to `out_dir`. Set `save_every=0` to only save at the end.

This is not a replacement for a production trainer. Good enough for
smoke-testing a model shape, short continue-pretrain runs, and
fine-tunes on hundreds of thousands of tokens. Anything serious should
go through 🤗 `accelerate`, `torchtune`, or `nanotron`.

## Advanced Distributed Training

HyperNix provides massive abstractions for hardware orchestration and distributed training. You can invoke these directly or through an `instant_pot.brew()` configuration.

### `ComputeFramework`
A single abstraction layer spanning CPU, CUDA, MPS, and TPU. It automatically wraps your model in `DistributedDataParallel` (DDP) or `FullyShardedDataParallel` (FSDP) depending on the topology.
```python
from hypernix.compute_framework import ComputeFramework
cf = ComputeFramework(local_rank="cuda", use_ddp=True, use_fsdp=False)
model = cf.prepare_model(model)
cf.backward(loss)
cf.step(optimizer)
```

### `lazy_suzan` (Decentralized Multi-GPU)
An extremely efficient multi-GPU training topology specifically designed to link disparate GPUs (e.g., 3x H200s or 12x 5090s) together *without* a physical NVLink bridge.
- **Gradient Compression**: Compresses backward gradients utilizing FP8, INT8, or Top-K sparsity.
- **P2P Ring-AllReduce**: Bypasses heavy NCCL bottlenecks via custom PyTorch peer-to-peer ring topologies.
- **Overlapped Communication**: Hooked directly into PyTorch's backward pass, overlapping synchronization with the actual backward cycle.

`lazy_suzan` is activated in an `instant_pot` recipe by simply passing `"use_lazy_suzan": True`.

### `abbicus` (Dynamic Token Regulation)
Automatic regulation of context length, curriculum tuning, and token padding/truncation during training. `Abbicus` dynamically calculates these parameters based on the model's total parameter size, dataset type, and current global step, ensuring maximum throughput.

```python
from hypernix.abbicus import Abbicus, AbbicusConfig

cfg = AbbicusConfig(model_size="8B", dataset_type="instruct")
regulator = Abbicus(cfg)
regulator.step(global_step=5000)
context_len = regulator.current_max_length
```

See [Abbicus](Abbicus.md) for the full config options and the
`TurboAbbicus` variant (exponential growth + VRAM safeguard).

## AutoModel fallback

```python
from hypernix import load_snapshot

# model_type="gemma4" (Gemma 4) has no native HyperNix implementation.
# load_snapshot lazy-installs transformers, wraps the HF model in a
# forward(input_ids, labels=None) -> {"logits", "loss"} adapter, and
# returns it alongside a config shim.
model, cfg = load_snapshot("./gemma-4-4b-snapshot")
# train() can work against this model directly.
```

Supported model_types (pattern: HF model_type → backing path):

| HF `model_type` | Loader path |
|---|---|
| `hypernix`, `llama`, `qwen2`, `mistral` | Native `HyperNixModel` |
| `gemma4`, `qwen3_5`, `qwen3_5_moe`, `glm_moe_dsa`, `phi3`, `gemma2`, `gemma3`, `deepseek_v2`, `deepseek_v3`, `gpt_oss`, … | `AutoModelForCausalLM` |

The wrapper stays small — it just matches the
`forward(input_ids, labels=None) -> {"logits", "loss"}` contract that
`train` expects. Real generation / sampling goes through the HF model
directly.

## CLI

The same flows are available on the command line:

```bash
hypernix train init \
  --out-dir ./hyper-nix-v2 \
  --tokenizer-source ./hyper-nix-v1 \
  --hidden-size 1536 --intermediate-size 6144 \
  --num-hidden-layers 24 --num-attention-heads 24

hypernix train expand \
  --src-dir ./hyper-nix-v1 \
  --dst-dir ./hyper-nix-v2 \
  --hidden-size 1536 --intermediate-size 6144 --num-hidden-layers 24

hypernix train run \
  --model-dir ./hyper-nix-v2 \
  --dataset ./corpus.txt \
  --out-dir ./hyper-nix-v2-trained \
  --steps 1000 --batch-size 2 --context-length 512
```

See [CLI.md](CLI.md) for full flag reference.

## Tips

- **Use `seed=` on every call for reproducibility.** Init / expansion
  / training all accept it.
- **Size the model first, train second.** `expand_checkpoint` is the
  cheapest way to grow a model that already works; you keep the
  pretraining signal and only learn the new parameters.
- **Freeze to fine-tune cheaply.** `old_fridge.freeze(model,
  patterns=("embed_tokens", "layers.0", "layers.1", ...))` cuts the
  backward pass and optimizer state proportionally.
- **Validate the config before training** — `HyperNixConfig.__post_init__`
  catches head-divisibility and rope errors at construction time,
  not at the first forward pass.
