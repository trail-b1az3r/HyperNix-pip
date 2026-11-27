# Coffee Maker — `hypernix.coffee_maker`

A coffee maker brews on a schedule. `coffee_maker` runs a training/eval
function repeatedly — on an interval, as a one-shot batch, cyclically
with feedback, or as a long single run with disk checkpoints — and
keeps a history of results. It's intentionally lightweight: a cron
replacement, not a full scheduler. For a real scheduler use
systemd-timers / APScheduler / Airflow; for "run this N times and tell
me how it went", use `coffee_maker`.

All four tiers record results as `Brew` entries:

```python
@dataclass
class Brew:
    cycle: int
    started_at: float     # time.time() at start
    duration_s: float     # wall time of this brew
    ok: bool
    result: Any = None    # brew_fn's return value, if ok
    error: str | None = None  # "{ExceptionType}: {message}", if not ok
```

Exceptions inside `brew_fn` are always caught and recorded as a failed
`Brew` (except in `ColdBrewMaker`, which re-raises — see below).


> **Canonical source:** `src/hypernix/timing/coffee_maker.py`. The `hypernix.coffee_maker` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. `ColdBrewMaker` checkpoints are JSON and therefore the returned state must be JSON-serializable. Its phase function is re-raised on failure after the failed run is recorded; the other tiers have different failure behavior.

---

## Tier 1 — `CoffeeMaker` (drip, scheduled/repeated)

```python
from hypernix.coffee_maker import CoffeeMaker

maker = CoffeeMaker(brew_fn=run_one_training_cycle, interval_seconds=3600)
results = maker.run(cycles=5)          # exactly 5 brews, sleeping between
# or:
maker.serve(install_sigint=True)       # runs until maker.stop() / Ctrl-C
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `brew_fn` | `Callable[[], Any]` | required | Zero-arg callable. |
| `interval_seconds` | `float` | `60.0` | Sleep between calls is `interval_seconds - last_brew_duration`, so cadence stays regular regardless of how long a brew takes. |
| `history` | `list[Brew]` | `[]` | |

| Method | Notes |
|---|---|
| `.brew_once(cycle=0)` | Runs `brew_fn()` once, appends and returns the `Brew`. |
| `.run(cycles)` | Runs exactly `cycles` brews. No sleep after the final cycle or after `.stop()` is called mid-run. |
| `.serve(*, install_sigint=True)` | Runs until `.stop()` is called. If `install_sigint=True`, installs a `SIGINT` handler so Ctrl-C sets the stop flag cleanly (finishing the current brew) instead of raising `KeyboardInterrupt`; the original handler is restored in a `finally` block. |
| `.stop()` | Cooperative cancel — finishes the current brew, then exits. |
| `.summary()` | `{"cycles", "failed", "mean_duration_s", "last_ok"}`. |

Module-level shortcut: `coffee_maker(brew_fn, *, interval_seconds=60.0) -> CoffeeMaker`.

## Tier 2 — `FrenchPressMaker` (batch brew)

```python
from hypernix.coffee_maker import french_press
results = french_press([eval_run_a, eval_run_b, eval_run_c]).plunge()
```

Doesn't loop on a schedule — takes a list of zero-arg callables, runs
them all once in order, returns results. Useful for running an ensemble
of evaluators, training a handful of LoRAs in sequence, or any "N
independent jobs" pattern.

`FrenchPressMaker(brews: list[Callable[[], Any]])` → `.plunge() -> list[Brew]`.
Shortcut: `french_press(brews) -> FrenchPressMaker`.

## Tier 3 — `PercolatorMaker` (cyclic / refinement)

```python
from hypernix.coffee_maker import percolator

result = percolator(
    lambda draft: judge_and_revise(draft),
    seed_input=initial_draft,
    max_cycles=5,
    convergence=lambda prev, new: prev == new,
).percolate()
```

Fixed-point iteration: the output of cycle N feeds cycle N+1. Great for
"draft, critique, revise, …" loops with a judge model. `brew_fn` takes
the previous result and returns the next; the first call gets
`seed_input`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `brew_fn` | `Callable[[Any], Any]` | required | |
| `seed_input` | `Any` | `None` | Input to the first cycle. |
| `max_cycles` | `int` | `5` | Hard cap even without convergence. |
| `convergence` | `Callable[[Any, Any], bool] \| None` | `None` | Called as `convergence(prev, new)` after each successful cycle; if it returns `True`, `.percolate()` stops early and returns `new`. |

`.percolate() -> Any` returns the final `current` value. On an exception,
the failure is recorded and the loop breaks (current, pre-exception
value is returned — the exception itself is **not** re-raised, unlike
`ColdBrewMaker`). Shortcut: `percolator(brew_fn, seed_input=None, *, max_cycles=5, convergence=None)`.

## Tier 4 (new type) — `ColdBrewMaker` (long, checkpointed)

```python
from hypernix.coffee_maker import cold_brew

def phase_fn(state: dict, phase: int) -> dict:
    state["step"] = phase
    # ... do real work, mutate/extend state ...
    return state

cb = cold_brew(phase_fn, phases=24, checkpoint_path="run.json")
final_state = cb.brew()   # resumable — rerunning picks up from the last checkpoint
```

"Set it and forget it" — one `brew_fn` that may take hours/days across
many phases, with a mandatory JSON checkpoint written after every
successful phase so a crashed/killed run can resume. `brew_fn(state,
phase)` receives the last-persisted state dict (`{}` on the first ever
call) and the 0-based phase index, and must return the next state dict.

| Field | Type | Default | Notes |
|---|---|---|---|
| `brew_fn` | `Callable[[dict, int], dict]` | required | |
| `phases` | `int` | `24` | Total phase count. |
| `checkpoint_path` | `Path \| str` | `"cold_brew.json"` | JSON file: `{"state": ..., "next_phase": ...}`. |
| `phase_interval_seconds` | `float` | `0.0` | Optional sleep between phases (not after the last). |

**Important:** unlike the other three tiers, `.brew()` **re-raises** any
exception from `brew_fn` after recording the failed `Brew` — a cold-brew
run does not silently swallow errors, since resuming from a bad
checkpoint state would be worse than stopping. Shortcut: `cold_brew(brew_fn, *, phases=24, checkpoint_path="cold_brew.json", phase_interval_seconds=0.0)`.

## `TIERS` lookup table

```python
TIERS: dict[str, type] = {
    "drip": CoffeeMaker,
    "french-press": FrenchPressMaker,
    "percolator": PercolatorMaker,
    "cold-brew": ColdBrewMaker,
}
```

Note there's no factory *function* here (unlike `pans`/`toaster`/
`blender`/`food_processor`) — construct the class from `TIERS` directly,
or use each tier's dedicated shortcut function (`coffee_maker`,
`french_press`, `percolator`, `cold_brew`).

### Required modules

Standard library only — `json`, `signal`, `time`, `dataclasses`,
`pathlib`, `collections.abc`. No HyperNix internal dependencies.

---

## See also

- `hypernix.espresso_maker` — evaluation-focused tiers, a natural pairing with `CoffeeMaker`'s scheduled retraining
- `hypernix.instant_pot` — one-shot end-to-end training pipeline (`brew()`), a different `brew`-named API worth not confusing with this module
