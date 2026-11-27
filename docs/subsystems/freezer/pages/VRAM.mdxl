# VRAM optimizations

[`hypernix.system.freezer`](Freezer.md) answers *"how big a batch
fits?"*. `hypernix.system.vram` answers the next question: **"how do I
make more of it fit without changing what the model learns?"**

Five techniques. Each is opt-in, each is reversible, and each refuses
rather than silently degrading — which matters more here than usual,
because every one of them is invisible when it quietly fails. You get the
same loss curve and the same OOM, with no way to tell which one did not
take effect.

| Technique | Frees | Costs |
|---|---|---|
| [`configure_allocator`](#the-allocator) | fragmentation — often 5–20% of reserved-but-unusable VRAM | nothing |
| [`checkpoint_blocks`](#activation-checkpointing) | most activation memory | ~30% more compute |
| [`fuse_optimizer_into_backward`](#optimizer-in-backward) | one full copy of the gradients | no clipping, no accumulation |
| [`offload_optimizer_state`](#optimizer-state-offload) | optimizer state, for the duration of a block | a host round-trip each way |
| [`measure_peak`](#measuring) | nothing — it tells you whether the others worked | nothing |

## From the CLI

```bash
hypernix train run --model-dir ./m --dataset ./d.txt --out-dir ./out \
    --gradient-checkpointing \
    --checkpoint-every 2 \
    --tune-allocator

# The fused optimizer needs clipping off, and says so if you forget:
hypernix train run ... --fuse-optimizer --grad-clip 0
```

## The allocator

```python
from hypernix.system import vram

report = vram.configure_allocator()
print(report.report())      # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The CUDA caching allocator carves VRAM into fixed-size segments and
cannot satisfy a large request from several small free ones. On a long
run with varying sequence lengths that shows up as an OOM while
`nvidia-smi` still reports gigabytes free: the memory is *reserved and
unusable*, not in use. `expandable_segments:True` lets a segment grow
instead.

**This has to run before the first CUDA allocation.** The allocator reads
the variable once, at init, and never again — so importing
`hypernix.system.vram` deliberately does **not** import `torch`, and the
function is callable from a launcher that has not touched CUDA yet.

Call it too late and it tells you:

```python
>>> vram.configure_allocator().report()
'PYTORCH_CUDA_ALLOC_CONF unchanged (CUDA is already initialized — the
allocator read PYTORCH_CUDA_ALLOC_CONF at init and will not read it
again. Call this before the first CUDA allocation.)'
```

Other keys: `garbage_collection_threshold=0.8` reclaims cached blocks
above 80% of capacity (useful on a card that also drives a display), and
`max_split_size_mb` refuses to split blocks above a size — which helps a
workload with a few large stable allocations and hurts one with many
small ones, so it has no default.

An existing `PYTORCH_CUDA_ALLOC_CONF` is **kept**, not overwritten: an
explicit setting is a decision somebody made. Pass
`override_existing=True` to replace it.

## Activation checkpointing

```python
handle = vram.checkpoint_blocks(model)          # every block
handle = vram.checkpoint_blocks(model, every=2) # every other one
...
handle.disable()                                 # back to where you started
```

Activations, not parameters, are what a long-context run actually runs
out of: they scale with batch × sequence × layers, and the parameters do
not scale with any of those. Checkpointing keeps only each block's input
and recomputes the rest during backward.

`every=2` is the useful middle when you are close to fitting rather than
far from it — roughly half the saving for roughly half the extra compute.

**Finding the blocks.** The longest `nn.ModuleList` of structurally
identical children. That is what a transformer's layer stack is in every
architecture this package supports, without hard-coding an attribute name
per architecture: `.layers`, `.h`, `.blocks` and `.decoder.layers` are
the same shape underneath. A model that is *not* a stack of identical
layers is reported (`wrapped == 0`) rather than raised on — there is
simply nothing there to checkpoint.

**`use_reentrant=False` is not a preference.** The reentrant
implementation silently produces no gradients at all when none of the
checkpointed region's *inputs* require grad — which is the normal case
for the first block, whose input is the embedding output. That failure
mode is a model that trains on a subset of its own layers and never says
so.

Two more things it gets right, both of which are silent bugs otherwise:

- Under `torch.no_grad()` the wrapper passes straight through. There is
  no backward to recompute for, so checkpointing there would run every
  block twice and save nothing.
- `prefer_native=True` (the default) uses the model's own
  `gradient_checkpointing_enable` when it has one, because a
  transformers implementation knows about that architecture's cache and
  attention-mask handling and a generic wrapper does not.

While the generic path is active each block carries a closure as its
`forward`, so pickling the *whole model object* (`torch.save(model)`)
fails. Saving a `state_dict` — what `save_snapshot` and every other saver
in this package do — is unaffected, and `handle.disable()` restores
picklability.

## Optimizer-in-backward

```python
handle = vram.fuse_optimizer_into_backward(
    model, lambda params: torch.optim.AdamW(params, lr=3e-4)
)
loss.backward()      # every parameter is stepped and freed in here
handle.step()        # a no-op, so an existing loop keeps its shape
```

An ordinary loop holds every gradient at once between `backward` and
`step` — a second full copy of the model, in gradient dtype, at exactly
the moment activations peak. Attaching the step to each parameter's
post-accumulation hook means a gradient exists only between the instant
it is finished and the instant it is applied.

`step()` and `zero_grad()` are no-ops rather than errors, so a training
loop does not have to change shape to use this.

**Three things it refuses, loudly:**

| Passed | Why it cannot work |
|---|---|
| `grad_clip` | A global norm cannot be computed from one gradient at a time. |
| `accumulation_steps > 1` | Accumulation needs gradients to survive across micro-batches; this frees them at the end of each backward. |
| `scaler` | A `GradScaler` must unscale every gradient before any step. bf16 autocast needs no scaler and works here. |

Each of those would otherwise produce a plausible-looking loss curve for
a model that trained differently than you asked for, which is why they
are `ValueError` and not a warning. `hypernix train run --fuse-optimizer`
refuses the same combination *before* it loads the checkpoint, so you are
not told after a model load that the combination was never going to work.

Note that accumulation is an *alternative* to this, not a companion:
both buy memory, and accumulation is the one that keeps clipping.

## Optimizer-state offload

```python
with vram.offload_optimizer_state(opt) as moved:
    metrics = evaluate(model, val_loader)
print(f"freed {moved} bytes for the eval pass")
```

Adam-family state is two tensors per parameter — for an fp32 model that
is twice the parameter memory, sitting idle through any pass that is not
a training step. Around a mid-run eval or a generation sample, moving it
to the host is most of a model's worth of VRAM back, for two transfers.

It is a **context manager and not a mode** on purpose: state that lived
on the host during the step would cross the bus every step, which is a
different and much worse trade than doing it twice. The restore is in a
`finally`, so an exception inside the block cannot leave the optimizer
split across two devices — which would fail on the next step with an
error naming neither this call nor that exception.

`release_cache()` is the smaller relative: it returns cached-but-unused
blocks to the driver and reports how many bytes that was. Worth doing
after a phase change; not worth doing every step, where it only makes the
allocator re-acquire what it just gave back.

## Measuring

```python
with vram.measure_peak() as peak:
    train_one_epoch(...)
print(peak.report())
# peak allocated 7.41 GiB, reserved 8.90 GiB (+1.49 GiB allocator overhead)
```

`allocated` is what the tensors needed; `reserved` is what the allocator
held from the driver. **The gap between them is fragmentation** — which
is the number `configure_allocator` is trying to move, so a run where
`reserved` falls and `allocated` does not is that change working, not
noise.

The counters are reset on entry, so a measurement is of the block and not
of everything since process start. The `PeakMemory` is filled in on
*exit*; reading it inside the block gives zeros, because the peak is not
known yet.

## Which one first

```python
for rec in vram.recommend(
    parameters=7_000_000_000, layers=32,
    batch_size=4, context_length=4096, hidden_size=4096,
    grad_clip=False,
):
    print(rec.report())
```

Ordered by estimated saving, largest first. Every number is arithmetic on
the arguments — parameter count, dtype width, the shape of the batch —
**not a measured result and not a benchmark**; `measure_peak` is how you
find out what actually happened. A technique your loop cannot use is left
out rather than listed with a caveat nobody reads: clipping removes the
fused optimizer from the list entirely.

## What this is not

- Not sharding. Single-device only; `lazy_suzan` and `ComputeFramework`
  are the multi-GPU story.
- Not quantization. [`Pressure Cooker v5`](Pressure-Cooker-V5.md) already
  quantizes optimizer momentum, and [`Quantization`](Quantization.md)
  covers weights.
- Not automatic. Nothing here is applied for you, and nothing here
  changes a default. Every function is an explicit call, and every one of
  them is reversible.

## See also

- [Freezer](Freezer.md) — batch/context sizing and OOM-retry
- [STML](STML.md) — the VRAM → trainable-context-length calculator
- [Pressure Cooker v5](Pressure-Cooker-V5.md) / [v6](Pressure-Cooker-V6.md) — memory-first and speed-first optimizers
- [CakePan](CakePan.md) — memory-pressure rollback during training
