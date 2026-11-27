# Bell — `hypernix.bell`

A bell rings when an order's done. `hypernix.bell` does two things:

1. Wraps any `CodeOven`-compatible oven so generation **streams** one
   token at a time through callbacks, instead of blocking until the full
   reply is ready.
2. Provides a notification primitive (stdout / file / custom callback)
   that fires once when a long chat or batch job finishes.

You don't need it to chat — [`Countertop`](Kitchen.md) uses a `Bell`
internally when you pass `stream=True` — but it works standalone with a
raw `CodeOven` too.


> **Canonical source:** `src/hypernix/timing/bell.py`. The flat `hypernix.bell` import spelling used in examples is a compatibility-facing API; the path above is the implementation to check when verifying behavior. This is the callback/iterator streaming helper, not the model runtime itself. It expects an oven-like object exposing the methods documented below.

---

## Push-based streaming (callbacks)

```python
from hypernix.bell import Bell
from hypernix.old_oven import preheat

oven = preheat("hyper-nix.2")

bell = Bell()
bell.on_token(lambda tok, idx: print(tok, end="", flush=True))
bell.on_done(lambda full: print(f"\n[done, {len(full)} chars]"))

reply = bell.stream_chat(
    oven,
    [{"role": "user", "content": "hello"}],
    max_new_tokens=128,
)
```

`stream_complete(oven, prompt, **gen_kwargs)` is the plain-completion
equivalent of `stream_chat`.

## Pull-based streaming (generators)

```python
for tok in bell.iter_chat(oven, messages):
    print(tok, end="", flush=True)
```

`iter_complete(oven, prompt, **gen_kwargs)` is the plain-completion
equivalent of `iter_chat`.

## `Bell` (dataclass)

| Field | Type | Notes |
|---|---|---|
| `token_callbacks` | `list[Callable[[str, int], None]]` | Each `(token_str, index)` forwarded as produced. |
| `done_callbacks` | `list[Callable[[str], None]]` | Fired once with the full decoded reply. |
| `flour` | `hypernix.flour.Flour \| None` | Optional chat-quality processor — see [Flour](Flour.md). When set, every step's logits are run through `flour.process(...)` before sampling, and the running decoded text is checked against `flour.matched_stop()` for early termination *before* the matching token is yielded (so a stop marker never leaks into the stream). |

### Methods

| Method | Signature | Notes |
|---|---|---|
| `.on_token(fn)` | `(Callable[[str, int], None]) -> Bell` | Registers and returns `self` (chainable). |
| `.on_done(fn)` | `(Callable[[str], None]) -> Bell` | Registers and returns `self` (chainable). |
| `.ring(full_reply)` | `(str) -> None` | Manually fires the done-callbacks — useful for wiring a non-streamed `oven.complete()` call into a flow that already has a bell set up for notifications. |
| `.iter_complete(oven, prompt, *, max_new_tokens=128, temperature=0.7, top_k=40, top_p=0.95, seed=None)` | `-> Iterator[str]` | Compatible with any oven exposing `_encode`, `_decode`, and `model(...).logits`. |
| `.iter_chat(oven, messages, *, max_new_tokens=256, temperature=0.7, top_k=40, top_p=0.95, seed=None)` | `-> Iterator[str]` | Routes through `oven._format_chat`, so any chat template the oven already uses (HF `apply_chat_template` or a hypernix [Cookbook](Cookbook.md) template) is preserved. |
| `.stream_complete(oven, prompt, **gen_kwargs)` | `-> str` | Drains `iter_complete`, fires `done_callbacks`, returns the full text. |
| `.stream_chat(oven, messages, **gen_kwargs)` | `-> str` | Drains `iter_chat`, fires `done_callbacks`, returns the full text. |

### Sampling

Token sampling is self-contained inside `bell.py` (`_sample_one`) rather
than reusing `old_oven`'s private sampler, specifically so `Bell` keeps
working with custom ovens that don't expose the same internals. Supports:
- `temperature <= 0` → greedy (`argmax`)
- `top_k` truncation
- `top_p` (nucleus) truncation, always keeping at least the top-1 token
- EOS detection via `_eos_ids(oven)`, which only fires for HF-tokenizer
  ovens (`oven.tokenizer_kind == "hf"`)

### Ready-made bell variants

| Function | Returns |
|---|---|
| `stdout_bell()` | Writes each token to `sys.stdout`, flushes per token, prints `\n[bell: N chars]\n` on done. |
| `file_bell(path)` | Truncates `path`, appends each token (flushed per token), writes `\n--- end of reply ---\n` and closes the handle on done. Creates parent dirs. |
| `silent_bell()` | Plain `Bell()` with no callbacks — get the streaming iterator with zero side effects. |
| `bell(*, stdout=False)` | `stdout_bell()` if `stdout=True`, else a plain `Bell()`. |

### Required modules

- `torch` (hard dependency — sampling and the forward pass both run through it)
- Standard library: `sys`, `dataclasses`, `pathlib`, `collections.abc`
- Optional: `hypernix.flour.Flour` (only if you set `flour=`)

---

## See also

- [Flour](Flour.md) — the chat-quality logits processor `Bell` can wrap around sampling
- [Cookbook](Cookbook.md) — chat templates consumed via `oven._format_chat`
- `hypernix.countertop.Countertop` — higher-level multi-turn session that wires a `Bell` for you when `stream=True`
