# Recipe Book — `hypernix.recipe_book`

A recipe book is the registry of named configurations. Instead of typing
the same 12-key dict every time you brew, save it once with a name and
refer back by name forever — across processes, via JSON.

```python
from hypernix.recipe_book import RecipeBook

book = RecipeBook()
book.add("evaluator-quick", {
    "repo_id": "nix2.5", "dataset": "judge.txt", "out_dir": "./trained",
    "steps": 500, "batch_size": 1, "context_length": 1024,
    "device": "cuda", "dtype": "float16",
})
book.save("recipes.json")

# later, in a different process:
book = RecipeBook.load("recipes.json")
book.cook("evaluator-quick")   # routes through instant_pot.brew
```

## Built-in recipes (`HYPERNIX_RECIPES`)

| Name | `kind` | What it does |
|---|---|---|
| `"evaluator-quick"` | `"instant_pot"` | 200-step smoke test on `nix2.5`, auto device, fp32. |
| `"ftune-pascal"` | `"instant_pot"` | 2000-step fine-tune of `ray0rf1re/hyper-nix.1`, fp16, `freeze_embed=True` — tuned for Pascal (GTX 1080-class) hardware. |
| `"nightly-coldbrew"` | `"cold_brew"` | 7-phase cold-brew job, one phase per day (`phase_interval_seconds=86400`). |
| `"espresso-eval"` | `"espresso"` | Eval-only preset — `"double-shot"` tier, 64 tokens, temp 0.2. |

Get a book preloaded with these via `RecipeBook.from_builtins()` or
`recipe_book(builtins=True)`.

## `RecipeBook` (dataclass)

`recipes: dict[str, dict[str, Any]]` — every add/get deep-copies the
dict, so mutating a recipe you got back from `.get()` never affects the
book's stored copy.

| Method | Signature | Notes |
|---|---|---|
| `.add(name, recipe)` | `(str, dict) -> None` | Raises `TypeError` if `recipe` isn't a dict. |
| `.get(name)` | `(str) -> dict` | Deep copy. Raises `KeyError` (listing known names) if not found. |
| `.remove(name)` | `(str) -> None` | |
| `name in book` | `__contains__` | |
| `len(book)` | `__len__` | |
| `.names()` | `() -> list[str]` | Sorted list of recipe names. |
| `.save(path)` | `(Path\|str) -> Path` | Writes all recipes as indented JSON. |
| `RecipeBook.load(path)` | `classmethod (Path\|str) -> RecipeBook` | |
| `RecipeBook.from_builtins()` | `classmethod -> RecipeBook` | Preloaded with `HYPERNIX_RECIPES`. |

## `.cook(name, **overrides)` — dispatch by `kind`

```python
book.cook("evaluator-quick", steps=1000)          # instant_pot, with steps overridden
book.cook("nightly-coldbrew", brew_fn=my_phase_fn) # cold_brew needs brew_fn=
book.cook("espresso-eval", oven=my_oven, prompts=["Hi", "2+2?"])  # espresso needs oven= and prompts=
```

Looks up `name`, applies `**overrides` on top of the stored recipe
(`recipe.update(overrides)`), pops the `kind` key (defaults to
`"instant_pot"` if absent), and dispatches:

| `kind` | Routes to | Required override(s) |
|---|---|---|
| `"instant_pot"` | `hypernix.instant_pot.brew(recipe)` | none — the recipe dict itself is the config |
| `"cold_brew"` | `hypernix.coffee_maker.cold_brew(brew_fn, **recipe).brew()` | `brew_fn=` (a `Callable[[dict, int], dict]`) — raises `ValueError` if omitted |
| `"espresso"` | `hypernix.espresso_maker.espresso_maker(tier, oven=oven, **recipe).pull(prompts)` | `oven=` and `prompts=` — raises `ValueError` if either is omitted |

Any other `kind` raises `ValueError`. All three internal modules are
imported lazily inside `.cook()` (not at module load), so `recipe_book`
itself has no hard dependency on `instant_pot`/`coffee_maker`/
`espresso_maker` unless you actually cook a recipe of that kind.

## Module-level constructor

```python
from hypernix.recipe_book import recipe_book
book = recipe_book(builtins=True)
```

`recipe_book(*, builtins=False) -> RecipeBook`.

### Required modules

Standard library only at import time — `copy`, `json`, `dataclasses`,
`pathlib`, `typing`. `hypernix.instant_pot` / `hypernix.coffee_maker` /
`hypernix.espresso_maker` are internal, lazily-imported dependencies
used only by `.cook()`.


> **Canonical source:** `src/hypernix/training/recipe_book.py`. The `hypernix.recipe_book` import spelling used in examples is the compatibility-facing API; the source path above is the implementation to check when verifying behavior. The book stores configuration dictionaries as JSON; runtime callables such as `brew_fn` must be supplied at dispatch time and are not serializable recipe values.

---

## See also

- `hypernix.instant_pot` — `brew()`, the default `kind`
- [Coffee Maker](CoffeeMaker.md) — `cold_brew()`
- [Espresso Maker](EspressoMaker.md) — `espresso_maker()` / `.pull()`
