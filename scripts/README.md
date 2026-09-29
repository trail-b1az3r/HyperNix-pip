# autofix

Three repair scripts, one router, and a shared scope module. Each script
owns exactly one failure class and refuses the others.

| Script | Owns | What it does |
|---|---|---|
| `autofix-B` | ruff diagnostics | `ruff check --fix`, then `--unsafe-fixes`; commits `[autofix-B]`. |
| `autofix-E` | imports, syntax, collection | Wraps optional imports, drops duplicates, adds `from __future__ import annotations`, replaces bare `except:`, then verifies every file compiles. |
| `autofix-F` | failing tests in a module category (timing by default) | Widens wall-clock margins in the individual tests that failed. |
| `autofix` | — | Reads a CI log (or reproduces the failure), then runs whichever of the three owns it. |
| `autofix_scope.py` | — | Shared: category → tests, category → time-valued kwargs, log → failure class. Not run directly in normal use. |

## Routing

```
scripts/autofix                       # reproduce locally, classify, fix
scripts/autofix --log ci-output.txt   # classify a CI log and fix
scripts/autofix --log -               # ... from stdin
scripts/autofix --dry-run             # say what would run, change nothing
```

The router checks in the order that failures block each other:

1. **ruff** — cheap, and a lint report on a file that doesn't parse is noise.
2. **`pytest --collect-only`** — does the tree import at all?
3. **the category's tests** — for `timing`, about ninety tests, a few seconds.

It never runs the full suite. Classification puts imports first for the same
reason: a `ModuleNotFoundError` in a timing test is `autofix-E`'s problem, not
`autofix-F`'s, even though a timing test is the thing that went red.

## autofix-F

Scope comes from `hypernix.MODULE_CATEGORIES`, not a hardcoded list.
`autofix_scope.py` resolves each test file's imports and finds the individual
test *functions* that use those modules, so a file like `tests/system/test_v060.py`
— which covers eight modules — contributes only its timer tests.

**It engages only when some but not all of the category's tests fail.** That
is the signature of the one thing it can fix: a wall-clock assertion that
lost a race on a loaded runner. All of them failing means something upstream
broke, and patching individual tests would bury it, so the script hands the
log to `autofix-E` or `autofix-B` and changes nothing itself.

The fix is to scale every wall-clock constant in a failing test — the sleeps
and the `duration=` / `interval_seconds=` / `work_seconds=` keywords alike —
by the same factor:

```python
t = timer.KitchenTimer(duration=0.05).start()   # -> duration=0.1
assert not t.expired()
time.sleep(0.25)                                # -> time.sleep(0.5)
assert t.expired()
```

Uniform scaling keeps every relationship in the test intact (the sleep stays
five times the duration) while making it tolerate an absolute stall twice as
long. The time-valued keyword names are read off the modules' own dataclass
fields, so a renamed field can't leave a stale rule behind.

Everything else — an `AttributeError` from a renamed symbol, a `TypeError`
from a changed signature, a real logic regression — is reported with its
actual message and left alone. There is no fix here that makes a test pass
without making it correct. If the widened tests still fail after
`--max-rounds`, nothing is committed and the working tree is left for
inspection.

```
scripts/autofix-F                     # run, fix, commit
scripts/autofix-F --dry-run           # show the edits
scripts/autofix-F --no-commit
scripts/autofix-F --log ci.txt        # classify a CI log instead of running
scripts/autofix-F --scale 4 --max-rounds 2 --max-sleep 3
scripts/autofix-F --category data     # a different module category
```

## Not re-testing everything

Two layers:

**In the script.** After widening, `autofix-F` re-runs only the tests it
changed. The edits are inside those test bodies and cannot reach any other
test, so a full run would tell you nothing new.

**In CI.** The commit carries trailers:

```
Autofix-Script: autofix-F
Autofix-Scope: timing
Autofix-Tests: tests/system/test_v060.py::TestTimer::test_interval_timer_only_fires_after_interval
```

`.github/workflows/ci.yml` has a `triage` job that reads them. When
`Autofix-Scope` is present the 4-OS × 4-Python `test` matrix and the sdist
build are skipped, and `autofix-verify` runs the named tests on one
interpreter instead. `lint` still runs either way.

Only `autofix-F` writes these trailers, because only `autofix-F` can promise
that narrow a blast radius. `autofix-B` and `autofix-E` edit `src/`, so their
commits carry no trailer and get the full matrix like any other change.

The trailers come from a commit message, which on a pull request is
attacker-controlled. CI passes them through the environment rather than
interpolating them into a shell script, and every node id is checked against
the category's own discovery before pytest sees it:

```
python scripts/autofix_scope.py --category timing --validate "$AUTOFIX_TESTS"
```

A node id outside the category, or anything that isn't a node id, fails the
job rather than running.

## Other utilities here

`autofix_scope.py` is also useful on its own:

```
python scripts/autofix_scope.py --list                  # timing test node ids
python scripts/autofix_scope.py --list --category data
python scripts/autofix_scope.py --time-kwargs
python scripts/autofix_scope.py --classify ci-log.txt
```

Tests for all of this live in `tests/repo/test_autofix_scripts.py`.
