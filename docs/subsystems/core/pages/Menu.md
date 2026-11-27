# Menu — `hypernix.menu`

A menu lists the dishes a kitchen serves; `hypernix.menu` lists the
**personas** (system prompts) you point a chat model at. It's a small,
zero-dependency named-string registry — `dict[str, str]` with lookup,
fuzzy matching, and JSON persistence.


> **Canonical source:** `src/hypernix/chat/menu.py`. The `hypernix.menu` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. Persona text is an instruction to the model, not an authorization mechanism. The `code-helper` prompt's wording about destructive commands does not enforce shell or filesystem permissions.

---

## Built-in personas

| Name | Persona |
|---|---|
| `"default"` | Terse helpful assistant. |
| `"concise"` | Same, hard-capped to three sentences, no preamble. |
| `"code-helper"` | Python-leaning code assistant; refuses destructive shell suggestions (`rm -rf`, `drop table`) without an explicit warning. |
| `"judge"` | Pair-rating judge — replies with exactly `A`, `B`, or `T`, no other text. |
| `"creative"` | Story / brainstorm partner; gives five distinct directions on request. |
| `"chef"` | Flavour-text persona matching the package's kitchen idiom. |
| `"none"` | Empty prompt string; built in directly. `Menu.add()` rejects empty prompts, so this entry cannot be recreated with `.add()`. |
| `"hyper-nix"` | The same default HyperNix system-prompt text used by `hypernix.cookbook.HYPER_NIX_2`. |

> **Security note:** A persona is not a security boundary. Validate and authorize shell, file, network, and other tool actions in application code; do not rely on the `code-helper` prompt to prevent destructive behavior.

## Quick use

```python
from hypernix.menu import MENU

sys_prompt = MENU.get("code-helper")

from hypernix.countertop import Countertop
ct = Countertop(oven, system=sys_prompt)
ct.say("write me a fizzbuzz")
```

`MENU` is a module-level `Menu` instance preloaded with every built-in —
import and use it directly rather than constructing your own unless you
want an empty registry.

## `Menu` class

```python
@dataclass
class Menu:
    prompts: dict[str, str] = field(default_factory=dict)
```

| Method | Signature | Notes |
|---|---|---|
| `Menu.from_builtins()` | `classmethod -> Menu` | Construct preloaded with every built-in persona. Same as `MENU`'s construction. |
| `.add(name, prompt)` | `(str, str) -> None` | Raises `TypeError` if `prompt` isn't a `str`, `ValueError` if empty/whitespace-only. |
| `.get(name)` | `(str) -> str` | Raises `KeyError` (listing known names) if `name` isn't registered. |
| `.remove(name)` | `(str) -> None` | |
| `name in menu` | `__contains__` | |
| `len(menu)` | `__len__` | |
| `.names()` | `() -> list[str]` | Sorted list of registered names. |
| `.default()` | `() -> str` | Returns the `"default"` entry, or the first registered prompt if `"default"` isn't present. Raises `KeyError` if the menu is empty. |
| `.find(query)` | `(str) -> str \| None` | Fuzzy lookup — see below. |
| `.save(path)` | `(Path \| str) -> Path` | Writes the registry as indented JSON (`{name: prompt}`), creating parent dirs as needed. |
| `Menu.load(path)` | `classmethod (Path \| str) -> Menu` | Loads a previously-saved JSON file. |

### `find()` matching order

`find(query)` tries, in order, and returns the matched **key** (not the
prompt text):

1. Exact match
2. Case-insensitive exact match
3. Substring match — only if **exactly one** entry contains the query
4. Prefix match — only if **exactly one** entry starts with the query

If any step finds more than one ambiguous candidate, it falls through to
the next step rather than guessing; if nothing resolves to a single match,
`find()` returns `None` so the caller can force disambiguation rather than
silently picking the wrong persona.

### Extending and persisting

```python
from hypernix.menu import menu

m = menu()  # builtins=True by default; pass builtins=False for an empty registry
m.add("my-bot", "You are a terse pirate. Answer every question in nautical slang.")
m.save("./prompts.json")

# ...later, anywhere:
from hypernix.menu import Menu
m2 = Menu.load("./prompts.json")
m2.get("my-bot")
```

### Required modules

Standard library only — `json`, `dataclasses`, `pathlib`. No HyperNix
internal dependencies; `menu.py` is safe to import standalone.

---

## See also

- [Kitchen](Kitchen.md) — general inference/chat subsystem overview
- `hypernix.countertop` — multi-turn chat session that consumes a `system=` prompt, typically sourced from `Menu`
- `hypernix.cookbook` — chat-template registry (`HYPER_NIX_2` and friends) referenced by the `"hyper-nix"` persona
