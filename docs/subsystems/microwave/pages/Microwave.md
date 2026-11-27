# Microwave — `hypernix.microwave`

When you don't want to preheat an oven and keep it around, zap the
prompt through a microwave: one function call takes a repo id or local
path, produces a completion, and tears down. Ideal for scripting
("make a filename from this heading"), CI smoke tests, and 5-line
Jupyter-cell snippets.


> **Canonical source:** `src/hypernix/models/microwave.py`. The `hypernix.microwave` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. A string is treated as a local directory only when it exists, is a directory, and contains `config.json`; a `Path` object is treated as a local path directly. The helper currently wraps the legacy oven API.

---

## Five power levels

| Tier | Function | Profile | Use case |
|---|---|---|---|
| 1 | `defrost(repo_id_or_dir, *, device=None, dtype="float32", quiet=True)` | No generation — just preheats and returns the oven. | Warm up once, then issue many `oven.complete()`/`.chat()` calls without rebuilding state each time. |
| 2 | `low_zap(repo_id_or_dir, prompt, *, max_new_tokens=16, temperature=0.0, top_k=1, top_p=1.0, stop=("\n",), seed=0, ...)` | Short, deterministic, single-line. | Filename / slug / one-word-answer generation. |
| 3 | `zap(repo_id_or_dir, prompt, *, max_new_tokens=64, temperature=0.2, top_k=40, top_p=0.95, stop=(), seed=None, ...)` | Standard. | The default if you don't know which tier you want. |
| 4 | `high_zap(repo_id_or_dir, prompt, *, max_new_tokens=512, temperature=0.7, top_k=50, top_p=0.95, stop=(), seed=None, ...)` | Long, hot. | Draft-a-paragraph / synth-a-story mode. |
| 5 | `chat_zap(repo_id_or_dir, message, *, system=None, max_new_tokens=128, temperature=0.7, top_k=40, top_p=0.95, seed=None, ...)` | Single-turn chat via the tokenizer's chat template (falls back to raw completion if none present). | One-off chat calls without a `Countertop` session. |

All five accept the same `repo_id_or_dir`, `device`, `dtype`, `quiet`
keyword args — only the sampling profile changes between tiers.

## Usage

```python
from hypernix.microwave import zap, low_zap, chat_zap, defrost, reheat

filename = low_zap("hyper-nix.2", "Title: Weekly Standup Notes ->")

reply = chat_zap("hyper-nix.2", "Explain BPE tokenization in one paragraph.")

# Warm once, issue several completions without reloading:
oven = defrost("hyper-nix.2")
out1 = oven.complete("def fib(n):")
out2 = oven.complete("def is_prime(n):")
```

## `reheat()` — continue a prior output without reloading

```python
def reheat(
    oven,
    prior_output: str,
    continuation_prompt: str = "",
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    stop: tuple[str, ...] = (),
    seed: int | None = None,
) -> str
```

`oven` is typically the return value of `defrost()`. `prior_output` is
the text to extend; `continuation_prompt` is an optional bridge string
inserted between the prior output and the new generation window.
Concatenates `prior_output + continuation_prompt` and calls
`oven.complete()` on the result — no template re-formatting.

## `TIERS` lookup table

```python
TIERS: dict[str, callable] = {
    "defrost": defrost,
    "low": low_zap,
    "standard": zap,
    "high": high_zap,
    "chat": chat_zap,
}
```

Useful for dispatch-by-name (e.g. a CLI `--tier` flag) without a chain
of `if`/`elif`.

## Repo-id / local-path resolution (`_preheat`)

Internally, every tier routes through `_preheat()`, which decides
whether `repo_id_or_dir` is a HF repo id or a local snapshot directory:

- A `Path` object is always treated as a local snapshot dir (repo id
  defaults to `"ray0rf1re/hyper-nix.1"` purely as a label).
- A `str` is only treated as a local dir if it exists, is a directory,
  **and contains a `config.json`** — otherwise it's passed through as a
  repo id / short name for `old_oven.preheat` to resolve. This
  `config.json` check (v0.50) prevents a string that happens to match an
  existing same-named directory in the cwd from silently shadowing a
  short-name lookup.

### Required modules

- `hypernix.old_oven` (internal — `preheat`, and the returned `CodeOven`'s `.complete()`/`.chat()`)
- Standard library: `pathlib`

---

## See also

- [Kitchen](Kitchen.md) — general inference subsystem overview
- [Ovens](Ovens.md) — `CodeOven` / `preheat`, what `microwave` wraps for one-shot use
- `hypernix.countertop.Countertop` — for multi-turn sessions instead of one-off zaps
