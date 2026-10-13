# Countertop — `hypernix.countertop`

A countertop is where you keep the dishes you're actively working with.
`Countertop` is the same idea for chat: a persistent multi-turn session
bound to an oven, tracking history, chat template, streaming, and
chat-quality cleanup in one object. Together with [Cookbook](Cookbook.md)
and [Menu](Menu.md), this is the headline chat surface of the package.


> **Canonical source:** `src/hypernix/chat/countertop.py`. The `hypernix.countertop` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. The history limit is measured in rendered characters as a token-budget proxy. Saved session JSON does not include T1 credentials or the Gatekeeper/Keymaster objects.

---

## Quick start

```python
from hypernix.neo_oven import preheat
from hypernix.countertop import Countertop

oven = preheat("hyper-nix.2")
chat = Countertop(oven, system="You are a helpful chef.")

print(chat.say("How do I dice an onion?"))
print(chat.say("And how about a shallot?"))

chat.save("session.json")
```

## Streaming

```python
from hypernix.bell import stdout_bell

chat = Countertop(oven, system="…", bell=stdout_bell())
chat.say("explain transformers in 3 sentences")  # tokens stream live
```

## `Countertop` (dataclass)

| Field | Type | Default | Notes |
|---|---|---|---|
| `oven` | `Any` | required | Anything exposing `.chat(messages, **kwargs) -> str`, e.g. `hypernix.old_oven.CodeOven`. |
| `system` | `str \| None` | `None` | System prompt prepended on every turn. |
| `template` | `str \| ChatTemplate \| None` | `None` | When `None`, auto-resolved from `oven.repo_id` via `hypernix.cookbook.for_model()` on first use. |
| `max_history_tokens` | `int \| None` | `None` | When the rendered transcript exceeds this many **characters** (a cheap token proxy), the oldest user/assistant pair is dropped. `None` disables trimming. |
| `bell` | `hypernix.bell.Bell \| None` | `None` | If set, `.say()` streams the reply through it. |
| `flour` | `hypernix.flour.Flour \| None` | `None` | If set, every reply is cleaned via `Flour.clean_reply()`. If `bell` is also set and its `.flour` is unset, this same `Flour` is attached to the bell automatically (`__post_init__`), so logits get processed *during* streamed generation too. |
| `sampling` | `dict[str, Any]` | `{}` | Default sampling kwargs forwarded to `oven.chat`. |
| `history` | `list[dict[str, str]]` | `[]` | The running transcript, *not* including `system`. |

### Methods

| Method | Signature | Notes |
|---|---|---|
| `.say(user, *, max_new_tokens=None, temperature=None, top_k=None, top_p=None, seed=None)` | `(str) -> str` | Appends `user`, trims history if needed, generates (streamed via `bell` if set, else `oven.chat(...)` directly), strips one trailing newline, cleans via `flour` if set, appends the reply, returns it. Per-call kwargs override `self.sampling` only where explicitly passed (non-`None`). |
| `.reset()` | `() -> None` | Clears `history`; keeps `system` and all config. |
| `.messages()` | `() -> list[dict[str, str]]` | Full message list including the system prompt (if any) — handy for handing off to a different runner. |
| `.render(*, add_generation_prompt=True)` | `() -> str` | Renders the current transcript through the resolved chat template **without generating** — useful for debugging the exact prompt about to be sent. |
| `.save(path)` | `(Path \| str) -> Path` | Writes `{system, history, template (name), max_history_tokens, sampling}` as indented JSON. Creates parent dirs. |
| `Countertop.load(path, oven)` | `classmethod (Path \| str, Any) -> Countertop` | Rebuilds a session from saved JSON, bound to the `oven` you pass (the oven itself is never serialized). T1 key/authentication objects are not saved; authenticate again if needed. |

### Optional T1 key accounting

`authenticate_t1(t1_key)` creates or uses a local `Keymaster` and `Gatekeeper`, authenticates the key, and binds the resulting metadata to the session. When a key is bound, `say()` checks the Gatekeeper quota before generation and records approximate usage afterwards; these estimates are character-based token approximations, not exact tokenizer counts. `t1_meta` exposes the bound key metadata and `t1_stats()` returns its Gatekeeper usage statistics. This local accounting layer does not by itself establish that a model request is being sent to a remote T1 service.

### History trimming (`_trim`)

While `max_history_tokens` is set and `len(history) > 1`: renders the
full transcript, and if it exceeds the character budget, drops the
oldest pair (2 messages, capped so at least 1 always remains). This
guarantees the most-recently-appended user turn always survives, even
when the budget is smaller than a single rendered turn.

## Module-level convenience constructor

```python
from hypernix.countertop import countertop

chat = countertop(oven, persona="code-helper", max_history_tokens=4000)
```

`countertop(oven, *, system=None, persona=None, template=None,
max_history_tokens=None, bell=None, flour=None, **sampling)` — `persona`
is a shortcut for picking a system prompt by name from
[`hypernix.menu.MENU`](Menu.md), equivalent to `system=MENU.get(persona)`
but avoids importing `menu` yourself at the call site. Passing both
`system` and `persona` raises `ValueError`.

## Interactive CLI menu (also lives in this file)

`countertop.py` also hosts the `hypernix`/`hnx` interactive TUI menu
(the 0.61.4 "Interactive TUI/CLI" feature) — a plain function, unrelated
to the `Countertop` class itself:

| Function | Signature | Notes |
|---|---|---|
| `interactive_cli(use_rich=True)` | `(bool) -> int` | Rich-based interactive menu (Download / Convert / Quantize / Train / Chat / ASR-TTS Pipeline / Assistant / Web UI / System Info / Quit). Falls back to `_simple_cli()` if `rich` isn't installed or `use_rich=False`. Returns a process exit code. |
| `_simple_cli()` | `() -> int` | Plain-`input()` fallback with the same menu options, no `rich` dependency. |

Both variants just print the equivalent `hypernix <subcommand> --help`
invocation for each menu choice rather than running it inline — they're
a discovery aid pointing you at the real subcommand, not a wrapper that
executes it for you.

### Required modules

- `hypernix.bell.Bell`, `hypernix.cookbook` (`ChatTemplate`, `for_model`,
  `COOKBOOK`), `hypernix.flour.Flour` — all internal, imported at module
  load.
- `hypernix.menu.MENU` — imported lazily inside `countertop()`, only
  when `persona=` is used.
- `rich` — optional, only for `interactive_cli(use_rich=True)`; the
  function catches `ImportError` and falls back to `_simple_cli()`.
- Standard library: `copy`, `json`, `dataclasses`, `pathlib`.

---

## See also

- [Cookbook](Cookbook.md) — chat templates `Countertop` resolves via `for_model`
- [Menu](Menu.md) — persona system prompts, wired via `persona=`
- [Bell](Bell.md) — streaming; auto-linked with `flour` when both are set
- [Flour](Flour.md) — reply cleanup via `clean_reply()`
