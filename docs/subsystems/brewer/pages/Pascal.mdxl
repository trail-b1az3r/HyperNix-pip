# Pascal — GTX 1080 / CUDA 6.1 / sm_61 playbook

Pascal is the compute-capability 6.x architecture found in the
consumer 10-series (GTX 1050 through GTX 1080 Ti), the Titan Xp / X
Pascal, and the Tesla P40 / P100. It predates tensor cores, has no
native bf16 or TF32, and has no FP16 accumulators — but it runs fp16
matmul at 2× the fp32 rate via its regular CUDA cores, which is
still an enormous win for training small-to-medium models.

`hypernix` has first-class support for Pascal throughout.

## One-minute checklist

```bash
# 1. Install torch for CUDA 11.8 (the last line with sm_61 in the
#    default kernel cache):
pip install --index-url https://download.pytorch.org/whl/cu118 torch

# 2. Then install hypernix (it will reuse the already-installed torch):
pip install hypernix
```

```python
# 3. Let the freezer pick settings:
from hypernix import freezer

hints = freezer.pascal_mode_hints()
# {"dtype": torch.float16, "use_sdpa": False, "use_compile": False,
#  "tf32": False, "matmul_precision": "highest",
#  "install_hint": "pip install --index-url https://download.pytorch.org/whl/cu118 torch"}

fz = freezer.flash_freezer(base=freezer.auto_freezer(), slow=True)
# -> OldFreezer wrapped for OOM safety.  Uses fp16 natively on sm_61.
```

## Detection helpers

```python
from hypernix.freezer import compute_capability, is_pascal, pascal_safe_dtype

compute_capability()        # (6, 1) on a GTX 1080; None on CPU
is_pascal()                 # True on sm_6x
pascal_safe_dtype()         # torch.float16 on Pascal/Volta/Turing,
                            # torch.bfloat16 on Ampere+, fp32 on CPU
```

`is_pascal()` is the cheap version of "should I disable all the
Ampere-only fast paths?". Use it to branch:

```python
if is_pascal():
    torch.backends.cuda.matmul.allow_tf32 = False   # TF32 doesn't exist here
    torch.backends.cudnn.allow_tf32 = False
    # Avoid torch.compile and SDPA's flash / mem-efficient kernels;
    # they assume Ampere tensor cores and break on sm_61.
```

## What NOT to use on Pascal

| Feature | Why it's broken on sm_61 |
|---|---|
| `torch.compile` | Triton kernels assume Ampere tensor cores; compile succeeds, generated code crashes or produces garbage. |
| `torch.nn.functional.scaled_dot_product_attention` with `enable_flash_sdp(True)` | FlashAttention requires sm_80. The fallback math path works but is slow. |
| `torch.bfloat16` | Not supported natively; `is_bf16_supported()` returns False; forcing it falls back to a bitwise-emulated path that's 10× slower than fp16. |
| TF32 (`allow_tf32=True`) | No-op — Pascal has no TF32 units. |
| `torch.cuda.amp.autocast("bfloat16")` | Same as bf16. Use `autocast("float16")` or don't autocast. |

## What TO use

| Feature | Why it works well |
|---|---|
| `torch.float16` | Native 2× throughput over fp32 on sm_61 (no tensor cores needed). |
| Gradient scaling (`torch.cuda.amp.GradScaler`) | Essential for fp16 stability. |
| AdamW with fp32 master weights | Standard mixed-precision recipe. |
| Small batch sizes (1 – 4) on 8 GB | Use `OldFreezer` — that's exactly its target envelope. |
| `FlashFreezer(slow=True)` | A single bad batch (too-long sequence, attention mask edge case) gets caught and the run halves down instead of dying. |

## Memory budget for a GTX 1080 (8 GB)

Training a ~100 M parameter model at fp16 with AdamW:

| Component | Approx. size |
|---|---|
| Model weights (fp16) | 200 MB |
| Gradients (fp16) | 200 MB |
| AdamW state (fp32 m + v + master) | 1.2 GB |
| Activations (batch=1, ctx=1024, 12 layers) | 600 MB – 1.5 GB |
| CUDA context / cuDNN workspace | 400 – 800 MB |
| **Total** | **~3 – 4 GB** |

Leaves 4 GB of headroom for the allocator's fragmentation and for a
second copy during checkpoint save. On a GTX 1080 you can push
batch_size to 2–4 if context_length stays at 1024.

## End-to-end recipe: HyperNix 1.5 on a GTX 1080

The `examples/train_hypernix_1_5_gtx1080.py` script does exactly this.
92.1 M params, fp16, batch=1, ctx=1024, FlashFreezer-wrapped:

```bash
python examples/train_hypernix_1_5_gtx1080.py \
    --dataset corpus.txt \
    --tokenizer-source ./hyper-nix-v1 \
    --out-dir ./hypernix-1.5 \
    --steps 2000 --batch-size 1 --context-length 1024 \
    --freeze-embed        # optional; cheapens the gradient pass
```

Startup prints:

```
[pascal] detected sm_61 (Pascal). Forcing fp16, disabling SDPA/compile/TF32.
[freezer] base=OldFreezer dtype=torch.float16
[freezer] wrapped with FlashFreezer (slow=True, max_retries=5)
[vram]    VRAMBudget(device='cuda:0', total=8.0GB free=7.8GB ...)
[init] fresh HyperNix 1.5 snapshot at ./hypernix-1.5/scratch
[params]  total=92,130,048  trainable=92,130,048  size(fp32 equiv)=175.8 MB
```

## Troubleshooting

**"CUDA error: no kernel image is available for execution on the
device"** — you're on a CUDA-12 torch wheel that dropped sm_61.
Reinstall torch from the cu118 index.

**Training loss spikes to NaN** — fp16 overflow. Use a `GradScaler`,
lower the learning rate, or narrow gradient clipping. If you used
`OldFreezer().preferred_dtype` on a CPU box you'd get fp32 instead;
this NaN path is a GPU-side fp16 edge case, not a library bug.

**Unexpectedly slow forward pass** — you're hitting the SDPA math
fallback. Run with `PYTORCH_ENABLE_MPS_FALLBACK=0` and verify with
`torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
enable_mem_efficient=False)`. For raw matmul throughput, check
`torch.cuda.get_device_name()` and confirm you're actually on the
Pascal card (many laptops have hybrid GPUs).
