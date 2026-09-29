# scripts/

Everything here is run by hand or by a workflow; none of it is imported
by the `hypernix` package. Repository automation that only CI runs
lives in [`.github/scripts/`](#githubscripts) instead.

| Script | What it is for | Run by |
|---|---|---|
| [`autofix`, `autofix-B`, `autofix-E`, `autofix-F`, `autofix_scope.py`](#autofix) | Classify a CI failure and repair the one class each script owns | you; `ci.yml`'s `triage` reads their commit trailers |
| `generate_docs_data.py` | Writes `docs/public/v1/{api-deep,t1-api,code-stats,changelog}.json` for the docs site, from the source tree and git history | `update-docs-data.yml` |
| `t1api_examples.py` | Drives a real T1 API and records `examples/t1api/API-EXAMPLES.md` and `openapi.json`. Rerun after changing a route | you |
| `update_json_stats.py` | PyPI download statistics into `docs/public/v1/json` | `update-json-stats.yml` (daily) |
| `ci/fake_model_server.py` | An OpenAI-shaped model that answers with canned text, for the integration jobs | `ci.yml`, `public-release.yml` |
| `ci/integration_probe.py` | Mints its own key, chats through a running T1 API, and deletes every key it made | same |
| `ci/wait_for_http.py` | Waits for a server to answer, and says why when it never does | same |
| `benchmark_v5.py`, `benchmark_v5s.py`, `benchmark_v6.py` | Pressure Cooker V5 / V5S / V6 against AdamW, each on its own identically seeded model. CUDA when there is one | you |
| `measure_optimizer_memory.py` | Exact optimizer-state bytes per parameter for AdamW and Pressure Cooker V5 / V5S / V6 | you |
| `install_deps.sh` | Cross-distro bootstrap into `./.venv`, preferring Python 3.12 | you |
| `install_macos_legacy.sh` | Intel Macs on Catalina / Big Sur that cannot run PyTorch 2.x: torch 1.13 and `hypernix[legacy-torch]` | you |
| `quantize_i7_7660u.sh` | Quantise `ray0rf1re/hyper-nix.1` to GGUF on a 2-core laptop CPU, one quant at a time | you |
| `apply_hypernix_fixes.py` | **Obsolete.** A one-off CI repair from before the package was split into subpackages; every file it edits has since moved, so it changes nothing. Kept for history | nobody |

## `.github/scripts/`

| Script | What it does | Run by |
|---|---|---|
| `version_guard.py` | Refuses a release version that goes backwards, or that names code a tag already names, and prints the PEP 440 spelling | `public-release.yml` |
| `release_plan.py` | Decides whether a run is a release, a nightly, or nothing: the nightly setting (`PUBLIC_RELEASE_NIGHTLY`), and skipping nights with no change but the stat bots' | `public-release.yml` |
| `codacy_sarif_filter.py` | Cuts Codacy's SARIF to correctness and security findings, under GitHub's 25,000-result limit | `codacy.yml` |
| `security_autofix.py` | Turns open code-scanning and Dependabot alerts into fixes on an autofix branch | `security-autofix.yml` |
| `update_readme.py` | Rewrites the README header between its markers with the current PyPI version | `update-readme.yml` |

`version_guard.py`, `release_plan.py` and `codacy_sarif_filter.py` have
tests under `tests/repo/`; `security_autofix.py` and `update_readme.py`
do not.

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
