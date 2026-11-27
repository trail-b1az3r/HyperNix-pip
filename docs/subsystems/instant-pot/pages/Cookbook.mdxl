# Cookbook — `hypernix.cookbook`

A cookbook holds the *recipes* that turn a list of plain `{"role",
"content"}` messages into the exact prompt string a given model expects.
Different model families use different formats (ChatML, Llama 3 turn
tags, Alpaca instructions, plain `role: content` transcripts), and
getting one wrong silently makes a chat model behave like a base model —
`cookbook` is the registry that prevents that.


> **Canonical source:** `src/hypernix/chat/cookbook.py`. The `hypernix.cookbook` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. `for_model()` uses a fixed, ordered substring-pattern table; it does not enumerate `KNOWN_MODELS`. A registry entry therefore does not automatically imply that the right chat template will be selected.

---

## Built-in templates

| Name | Format |
|---|---|
| `"chatml"` | OpenAI ChatML / Qwen / hyper-Nix.2 native (`<\|im_start\|>role\n...<\|im_end\|>`). |
| `"hyper-nix.2"` | Alias for ChatML, plus a HyperNix-flavoured `default_system`. Recommended for `ray0rf1re/hyper-Nix.2`. Also reachable via aliases `"hypernix.2"`, `"hyper-nix2"`, `"hypernix2"`, `"hyper-nix"`, `"hypernix"`. |
| `"llama3"` | `<\|start_header_id\|>role<\|end_header_id\|>` form. Alias `"llama-3"`. |
| `"llama2"` | `[INST] ... [/INST]` with system in `<<SYS>>`. Alias `"llama-2"`. |
| `"alpaca"` | `### Instruction:` / `### Response:`. |
| `"vicuna"` | `USER:` / `ASSISTANT:` lines. |
| `"plain"` | `role: content` transcript fallback — string formatting is tokenizer-agnostic, but it may not match the model's training template. |

## Quick use

```python
from hypernix.cookbook import COOKBOOK

prompt = COOKBOOK.get("hyper-nix.2").apply(
    [
        {"role": "system", "content": "You are a helpful chef."},
        {"role": "user", "content": "How do I dice an onion?"},
    ],
    add_generation_prompt=True,
)
```

`COOKBOOK` is a module-level `Cookbook` preloaded with every built-in —
use it directly rather than constructing your own unless you want an
empty registry.

## `ChatTemplate` (dataclass)

Each turn renders as `{prefix}{content}{suffix}`, with `prefix`/`suffix`
looked up per role (falling back to `"{role}: "` / no suffix if not
found in the role tables).

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | |
| `role_prefixes` | `Mapping[str, str]` | `{}` | Per-role prefix, e.g. `<\|im_start\|>user\n`. |
| `role_suffixes` | `Mapping[str, str]` | `{}` | Per-role suffix, e.g. `<\|im_end\|>\n`. |
| `bos` | `str` | `""` | Prepended once at the very start, if set. |
| `eos` | `str` | `""` | Used by `.stop_tokens()`. |
| `assistant_prefix` | `str` | `"assistant: "` | Appended when `add_generation_prompt=True`, telling the model it's now the assistant's turn. |
| `default_system` | `str \| None` | `None` | Auto-prepended as a system message if the first message in `.apply()` isn't already one. |
| `notes` | `str` | `""` | Human-readable description, surfaced by `list_templates()`. |

### Methods

| Method | Signature | Notes |
|---|---|---|
| `.apply(messages, *, add_generation_prompt=True)` | `(list[dict]) -> str` | Renders `messages` into a single prompt string. Injects `default_system` first if applicable. |
| `.stop_tokens()` | `() -> list[str]` | Best-effort list of strings marking the end of an assistant turn (the assistant role's suffix, plus `eos` if set) — useful for early-stopping a sampler. |

## `Cookbook` (registry)

| Method | Signature | Notes |
|---|---|---|
| `Cookbook.from_builtins()` | `classmethod -> Cookbook` | Preloaded with every built-in — same construction as `COOKBOOK`. |
| `.add(name, template)` | `(str, ChatTemplate) -> None` | Raises `TypeError` if `template` isn't a `ChatTemplate`. Names are lower-cased on insert. |
| `.get(name)` | `(str) -> ChatTemplate` | Case-insensitive lookup. Raises `KeyError` (listing known names) if not found. |
| `name in cookbook` | `__contains__` | Case-insensitive. |
| `.names()` | `() -> list[str]` | Sorted list of registered names (including aliases). |

## Module-level helpers

| Function | Signature | Notes |
|---|---|---|
| `for_model(name_or_repo, *, default="plain")` | `(str) -> ChatTemplate` | Picks the best template by matching substrings of the (lower-cased) short name or full HF repo id against an ordered pattern table (more specific patterns first — e.g. `"hyper-nix.2"` is checked before the bare `"hyper-nix"` fallback, which maps to `"plain"` since v1 had no chat template). Falls back to `default` if nothing matches. |
| `cookbook(*, builtins=True)` | `() -> Cookbook` | Constructs a `Cookbook`, optionally preloaded. |
| `apply_template(messages, *, template="hyper-nix.2", add_generation_prompt=True)` | `(list[dict]) -> str` | One-shot helper — resolves `template` (a name or an already-built `ChatTemplate`) via `COOKBOOK` and calls `.apply()`. |
| `list_templates()` | `() -> dict[str, str]` | Returns `{template.name: template.notes}` for every *distinct* registered template (dedupes aliases pointing at the same object via `id()`). |

### `for_model()` pattern table (order matters)

`hyper-nix.2` / `hypernix.2` / `hyper-nix2` / `hypernix2` → `"hyper-nix.2"`,
then `hyper-nix` → `"plain"`, then `nix-2.7`/`nix2.7`/`nix2.6`/`nix2.5` →
`"chatml"`, `qwen3`/`qwen2` → `"chatml"`, `llama-3`/`llama3` →
`"llama3"`, `llama-2`/`llama2` → `"llama2"`, `vicuna` → `"vicuna"`,
`alpaca` → `"alpaca"`, `nano-nano`/`nano-mini` → `"plain"`.

### Required modules

Standard library only — `dataclasses`, `collections.abc`. No HyperNix
internal dependencies; `cookbook.py` is safe to import standalone.

---

## See also

- [Kitchen](Kitchen.md) — general inference/chat subsystem overview
- `hypernix.old_oven.CodeOven._format_chat` — wired to `cookbook` so a hyper-Nix.2 snapshot's chat formatting Just Works
- `hypernix.flour` — `template_name`/`ROLE_LEAK_MARKERS` line up with these same template names for role-leak suppression
- `hypernix.menu` — persona system prompts, often layered on top of `default_system`
