# Flour — `hypernix.flour`

Flour is what binds a dough together — without it the rest of the
ingredients fall apart on the countertop. `Flour` is the chat-quality
logits processor that binds together every heuristic you'd otherwise wire
by hand on top of raw `transformers`: repetition penalty, no-repeat
n-gram blocking, bad-word suppression, role-leak suppression, and
decoded-text stop-sequence detection — one dataclass, one `.process()`
call.

This is the piece that makes hypernix's chat surface noticeably better
than raw `transformers` generation, for two concrete reasons:

1. **Stop sequences match on decoded text, not raw token ids** — so
   `"<|im_end|>"` works even when the tokenizer splits it into 3 pieces.
2. **Role-leak suppression** catches the failure mode where a
   half-trained chat model starts echoing `user:` and writing its own
   follow-up question. Vanilla `transformers` has no built-in primitive
   for this.

All the knobs live on one dataclass, so a config can be saved/reloaded as
JSON without serializing `LogitsProcessorList` instances.


> **Canonical source:** `src/hypernix/chat/flour.py`. The `hypernix.flour` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. Flour only processes logits if it is wired into a generation/sampling loop (for example through Bell); it is not a server-side policy or a standalone safety boundary.

---

## Quick use

```python
from hypernix.flour import Flour
from hypernix.bell import Bell

f = Flour.smart_default(template="hyper-nix.2")

bell = Bell(flour=f)   # works with hypernix.bell.Bell
# chat.say("hello") — all heuristics now applied automatically per token
```

`Flour` does **not** require `Bell` — call `f.process(logits,
produced_ids, tokenizer=...)` directly inside any sampling loop instead.

## Constructors / recipes

| Constructor | Settings |
|---|---|
| `Flour.smart_default(*, template=None)` | `repetition_penalty=1.1`, `no_repeat_ngram=4`, `frequency_penalty=0.0`, `presence_penalty=0.0`, `suppress_role_leaks=(template is not None)`. The recommended starting point for chat. |
| `Flour.aggressive(*, template=None)` | `repetition_penalty=1.3`, `frequency_penalty=0.5`, `presence_penalty=0.3`, `no_repeat_ngram=3`. For models that loop a lot. |
| `Flour.off()` | Every field at its default/no-op value — passes logits through unchanged. |
| `flour(*, repetition_penalty=1.1, no_repeat_ngram=4, template="hyper-nix.2", aggressive=False)` | Module-level quick constructor; `aggressive=True` just delegates to `Flour.aggressive(template=template)`. |

## `Flour` dataclass fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `repetition_penalty` | `float` | `1.0` (off) | Multiplicative: divides logits of already-seen *positive* tokens, multiplies *negative* ones. `1.1` is the OpenAI chat default. |
| `frequency_penalty` | `float` | `0.0` | Linear penalty proportional to how many times each token has appeared. |
| `presence_penalty` | `float` | `0.0` | Linear penalty applied once per unique token already produced (regardless of count). |
| `no_repeat_ngram` | `int` | `0` (off) | Blocks any token that would close an n-gram already seen in the produced sequence. |
| `bad_words` | `list[str]` | `[]` | Decoded-token blocklist. Only enforced when a `tokenizer` is passed to `.process()`. |
| `stop_sequences` | `list[str]` | `[]` | Decoded-text suffixes that end generation when matched. |
| `suppress_role_leaks` | `bool` | `False` | Auto-adds the chat template's role markers (see table below) to both `bad_words` and `stop_sequences`. |
| `template_name` | `str \| None` | `None` | Which entry of `ROLE_LEAK_MARKERS` to use when `suppress_role_leaks=True`. |

### `ROLE_LEAK_MARKERS` (built-in templates)

| Template | Markers |
|---|---|
| `"chatml"` / `"hyper-nix.2"` | `<\|im_start\|>`, `<\|im_end\|>`, `<\|im_start\|>user`, `<\|im_start\|>system` |
| `"llama3"` | `<\|start_header_id\|>`, `<\|end_header_id\|>`, `<\|eot_id\|>`, `<\|start_header_id\|>user<\|end_header_id\|>` |
| `"llama2"` | `[INST]`, `[/INST]`, `<<SYS>>`, `<</SYS>>` |
| `"alpaca"` | `### Instruction:`, `### Response:` |
| `"vicuna"` | `USER:`, `ASSISTANT:` |
| `"plain"` | `user:`, `system:`, `assistant:` |

Only the **close-of-turn** markers (containing `im_end`, `eot_id`,
`</s>`, or `[/INST]`) are promoted to `stop_sequences` automatically; the
rest are added only to `bad_words` via `effective_bad_words()`.

## Methods

| Method | Signature | Notes |
|---|---|---|
| `.effective_stop_sequences()` | `() -> list[str]` | `stop_sequences` plus auto role-leak close markers, deduped. |
| `.effective_bad_words()` | `() -> list[str]` | `bad_words` plus auto role-leak markers, deduped. |
| `.matched_stop(decoded_so_far)` | `(str) -> str \| None` | First stop sequence `decoded_so_far` ends with, else `None`. Compares on decoded text, so token boundaries don't matter. |
| `.strip_stop(decoded)` | `(str) -> str` | Strips a trailing matched stop sequence (and preceding whitespace), if present. |
| `.process(logits, produced_ids, *, tokenizer=None)` | `(Tensor, Sequence[int]) -> Tensor` | Applies every active heuristic in order: repetition → frequency → presence → no-repeat-ngram → bad-words. Accepts `(vocab,)` or `(1, vocab)` shaped logits and returns the same shape. `produced_ids` may be a list, tensor, or any iterable of ints. `tokenizer` is required only if `bad_words` or `suppress_role_leaks` is active — every other heuristic is tokenizer-free. |
| `.clean_reply(reply)` | `(str) -> str` | Post-hoc cleanup of a finished reply: strips a trailing stop sequence, cuts at the first role-leak marker, and cuts at a generic `\nuser:` / `\nsystem:` leak via regex (case-insensitive). |

### Required modules

- `torch` (hard dependency for `.process()`'s tensor ops)
- Standard library: `re`, `dataclasses`, `collections.abc`
- Tokenizer object (any HF-compatible `.encode(...)`) only needed for
  `bad_words` resolution — `_resolve_bad_word_ids` caches resolved ids
  per-tokenizer (`id(tokenizer)`) and silently skips words that don't
  resolve to a single token.

---

## See also

- [Bell](Bell.md) — streaming wrapper that applies `Flour` per-token automatically
- `hypernix.cookbook` — chat templates whose names line up with `template_name` / `ROLE_LEAK_MARKERS`
- `hypernix.countertop.Countertop` — multi-turn session; pass `flour=` to apply this automatically
