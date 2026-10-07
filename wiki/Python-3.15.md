# Python 3.12–3.15

**hyperNix-pip requires Python 3.12 or newer and officially supports
Python 3.12, 3.13, 3.14, and 3.15.** `Requires-Python` is
`>=3.12,<3.16`; 3.10 and 3.11 stop at 0.72.6.post4, which pip installs
there on its own. Its source is the `lts/0.72.6` branch, which points at
exactly the commit that release was built from.

This page is what changed in 0.72.7 to get there, how hyperNix-pip uses
Python 3.15's new features without breaking 3.12–3.14, what the
dependencies allow, what was measured, and what was tested — with
nothing marked as passing that was not run.

## Python 3.15 modernization

Four 3.15 features are used. Each sits behind a boundary, so 3.12–3.14
run exactly the code they ran before.

### PEP 810 — explicit lazy imports

3.15 adds `lazy import x` / `lazy from x import y`: the module is loaded
at first use rather than at the import statement. The keyword is a
syntax error before 3.15, so shared source uses the PEP's compatibility
form instead: a module-level `__lazy_modules__` list. On 3.15 the imports
it names are deferred; on 3.12–3.14 it is an ordinary variable.

It is used where an import was measured to be paid for and not needed:

| Module | Deferred | Who pays otherwise |
|---|---|---|
| `hypernix.quant.llamaquants` | `numpy` | every quant CLI, including `--help` |
| `hypernix.models.download` | `huggingface_hub` | `hyped`, which reads only `KNOWN_MODELS` |

Imports stay eager for security, configuration, key handling, API
registration and plugin loading: those must fail when the process
starts, not at the first request. `tests/python315/test_pep810.py`
enforces that, checks every name in a `__lazy_modules__` list is really
imported there, and on 3.15 checks the deferral happens and that a
deferred module still works on first use. The rest of the package was
already lazy at its top (`hypernix/__init__.py` resolves names on first
attribute access, PEP 562), and the T1 API's import cost is route
registration, which has to stay eager.

### PEP 798 — unpacking in comprehensions

`[*row for row in rows]`, `{*s for s in sets}` and `{**d for d in dicts}`
replace `itertools.chain`, a nested comprehension or an update loop.
3.12–3.14 cannot *parse* a file that contains them, so they live only in
`hypernix/_compat/*_py315.py`:

- `hypernix._compat` exports `flatten`, `union` and `merge`, imported
  from `_iter_py315` on 3.15 and `_iter_legacy` before it — same API,
  same results (the tests compare them case by case on 3.15).
- Used where hyperNix-pip flattens: optimizer parameter groups, the LM
  Studio header scan across roots.
- `hypernix._compat.is_py315_only(path)` tells every tool that parses
  the whole tree (autoscan, archmap, the docs generator, the deprecation
  tests) to skip those files on an older interpreter. Ruff and mypy
  exclude them; 3.15 compiles and runs them in `tests/python315`.

### PEP 799 — the `profiling` package

3.15 moves the deterministic profiler to `profiling.tracing` (`cProfile`
stays as an alias) and adds `profiling.sampling`, a statistical profiler
that can attach to a running process. `hypernix._profiling` is one API
over both:

```python
from hypernix._profiling import trace, profile_call, sampling_argv, sample

with trace() as run:                 # profiling.tracing on 3.15, cProfile before
    do_work()
print(run.report(limit=15))
run.dump("run.pstats")

result, stats = profile_call(do_work, arg)

sample(pid=1234, duration=10, output="run.pstats")   # 3.15: profiling.sampling
```

Before 3.15, `sampling_argv` / `sample` raise `SamplingUnavailable`
naming the alternatives; nothing that worked before is removed.

### PEP 831 — frame pointers

CPython 3.15 is built with `-fno-omit-frame-pointer
-mno-omit-leaf-frame-pointer`, so perf, py-spy, bpftrace and
`profiling.sampling --native` can walk a stack through C. Native code
built here keeps that:

- `hypernix._native_flags` reads the interpreter's `sysconfig` `CFLAGS`
  and *inherits* the frame-pointer flags — it never invents them for an
  interpreter built without them, never passes them to MSVC, and drops
  any later flag that would switch them back off.
- `setup.py`'s optional C++ extension (`BUILD_CCTVTOP=1`) builds through
  it.
- `native/ggml-hnx` keeps frame pointers by default
  (`GGML_HNX_FRAME_POINTERS=ON`), adding only the flags the compiler
  accepts.

`tests/python315/test_pep831.py` checks the flag logic in every case,
compiles real code at `-O2` and reads the prologue to confirm `%rbp` is
set up, and on 3.15 builds a setuptools extension end to end and reads
its compile line. The CI `python315` job also builds ggml-hnx and checks
its disassembly.

## What else changed

- **Metadata.** `requires-python`, `python_requires` and the classifiers
  (`3 :: Only`, 3.12–3.15) in `pyproject.toml` and `setup.cfg`.
- **Tools.** `install-t1.sh` searches `python3.15` … `python3.12` and
  refuses anything outside 3.12–3.15 by name; `hypernix doctor`'s range
  is 3.12–3.15; the `hnx` launcher tries 3.12, 3.13, 3.14, 3.15.
- **Shims.** An audit of `src/` for `sys.version_info` gates, `tomli`,
  `typing_extensions` and `importlib_metadata` fallbacks found none left
  to remove: the code already targeted 3.12 syntax (ruff's
  `target-version` was `py312`). New generic code uses PEP 695 type
  parameters.
- **Typing.** mypy runs in CI on 3.12 and 3.15 over the modules this
  release adds (strict) and reworks (`check_untyped_defs`). It found two
  real issues, both fixed: a `limits` dict typed for ints that holds
  `None` and a list, and a method named `list` shadowing the builtin in
  later annotations.
- **CI.** Every OS on 3.12, 3.13, 3.14 and 3.15; a 3.15 job for the four
  PEPs with a native frame-pointer check; a benchmark job per version.
- **Performance.** The benchmarks found image uploads spending 1.2 s
  (screenshot) to 3.5 s (photo) in the WebP encoder's highest effort
  setting for byte-identical output; effort 4 is used now.

## Dependency audit

Every declared dependency, checked against PyPI on 2026-10-06 (latest
release, and the wheels it ships for CPython 3.12–3.15):

| Dependency | Latest | 3.12 | 3.13 | 3.14 | 3.15 | Notes |
|---|---|---|---|---|---|---|
| torch | 2.14.1 | ✓ | ✓ | ✓ | **✗** | no cp315 wheel yet — see Known issues |
| numpy | 2.5.3 | ✓ | ✓ | ✓ | ✓ | |
| safetensors | 0.8.0 | ✓ | ✓ | ✓ | ✓ | abi3 |
| huggingface-hub | 2.1.1 | ✓ | ✓ | ✓ | ✓ | pure Python |
| gguf | 0.19.0 | ✓ | ✓ | ✓ | ✓ | pure Python |
| tqdm | 4.70.1 | ✓ | ✓ | ✓ | ✓ | pure Python |
| rich | 15.0.0 | ✓ | ✓ | ✓ | ✓ | pure Python |
| sentencepiece | 0.2.2 | ✓ | ✓ | ✓ | source | no cp315 wheel; builds from the sdist (C++ compiler, CMake) — verified |
| cryptography | 50.0.2 | ✓ | ✓ | ✓ | ✓ | abi3 |
| fastapi / uvicorn / pydantic | 0.142.2 / 0.54.0 / 2.13.5 | ✓ | ✓ | ✓ | ✓ | pydantic-core has cp315 wheels |
| python-dotenv / python-multipart | 1.2.4 / 0.0.32 | ✓ | ✓ | ✓ | ✓ | pure Python |
| Pillow | 12.3.0 | ✓ | ✓ | ✓ | ✓ | |
| pillow-heif | 1.8.0 | ✓ | ✓ | ✓ | ✓ | `images` extra |
| pillow-jxl-plugin | 1.3.8 | ✓ | ✓ | ✓ | ✗ | `images` extra; no cp315 wheel |
| rawpy | 0.27.1 | ✓ | ✓ | ✓ | ✗ | `images` extra; no cp315 wheel (DNG falls back to its embedded preview) |
| psutil | 7.2.2 | ✓ | ✓ | ✓ | ✓ | abi3 |
| transformers / accelerate | 5.19.0 / 1.15.0 | ✓ | ✓ | ✓ | ✓ | pure Python (`train` extra; need torch) |
| llama-cpp-python | 0.3.36 | src | src | src | src | sdist only, builds anywhere |
| psycopg[binary] | 3.3.6 | ✓ | ✓ | ✓ | ✓ | `t1api-pg` extra |
| PySide6 | 6.11.2 | ✓ | ✓ | ✓ | ✗ | `gui-qt` extra; declares `Requires-Python <3.15` |
| pytest / mypy / build / twine / ruff | | ✓ | ✓ | ✓ | ✓ | `dev` extra |

No pin was raised or lowered in this release. The resolver already picks
torch ≥ 2.2 and numpy ≥ 1.26 on 3.12, the first releases with 3.12
wheels, so the existing floors cost nothing; `torch>=1.13` stays because
0.72.6.post4's legacy-Mac path is documented against it.

## Benchmarks

`benchmarks/bench.py` measures startup (bare interpreter, `import
hypernix`, the CLIs' `--help`, cold and warm), imports of the expensive
entry modules, T1 API construction and requests, serialisation, image
conversion, the sub-bit codec, profiler overhead, and on 3.15 PEP 810 on
vs filtered off and PEP 798 vs the legacy code. Medians of repeated runs,
in fresh processes where startup is the thing measured.

    python benchmarks/bench.py --repeat 7 --json bench.json

### Results

From CI run 37549870364 (the `benchmarks` job, `ubuntu-latest`, x86-64,
`--repeat 7`, medians in ms). **Each Python ran on a different hosted
runner**, so treat differences between columns of less than about 1.5×
as machine noise: the bare interpreter alone spans 8.6–13.5 ms. The
same-process comparisons in the last table are the reliable ones.

| Benchmark | 3.12.14 | 3.13.15 | 3.14.7 | 3.15.0rc3 |
| --- | ---: | ---: | ---: | ---: |
| `python -c pass` | 12.33 | 8.60 | 13.47 | 8.88 |
| `import hypernix` | 27.11 | 13.20 | 22.26 | 13.87 |
| `hnx --help` | 111.76 | 61.64 | 102.15 | 58.11 |
| `hyprslug --help` | 142.89 | 78.65 | 151.19 | 55.57 |
| `gkey --help` | 89.71 | 51.52 | 81.92 | 46.53 |
| runner CLI `--help` | 83.89 | 48.22 | 89.49 | 62.80 |
| `import hypernix.t1api.app` | 997 | 512 | 958 | 607 |
| `import hypernix.hyperlink.managed` | 100.33 | 58.29 | 101.10 | 64.78 |
| `import hypernix.interfaces.hyped` | 291.03 | 162.85 | 307.65 | 56.12¹ |
| `import hypernix.quant.hyprslug` | 134.14 | 72.74 | 138.99 | 37.74¹ |
| `import hypernix.quant.convert` | 1543 | 865 | 1562 | — (needs torch) |
| `create_app()` | 68.37 | 54.16 | 70.62 | 47.01 |
| `GET /health` | 2.89 | 1.37 | 2.66 | 1.43 |
| authenticated `GET /hyperlink/models` | 4.83 | 2.62 | 4.63 | 2.71 |
| JSON dumps, 400-message chat | 0.39 | 0.25 | 0.35 | 0.13 |
| JSON loads, 400-message chat | 0.25 | 0.16 | 0.26 | 0.14 |
| image PNG → WebP (640×480) | 21.72 | 14.51 | 21.65 | 14.21 |
| sub-bit quantise (64 blocks) | 2.53 | 1.46 | 2.16 | 1.49 |
| deterministic profiler overhead | 1.73× | 1.96× | 1.68× | 2.82×² |

¹ PEP 810 at work, not a faster machine: `hyped` and `hyprslug` reach
`numpy` only through `hypernix.quant.llamaquants`, whose `__lazy_modules__`
defers it. On the 3.15 test box, importing both loads 182 modules with
lazy imports on and 553 with them filtered off (numpy among them).
² `profiling.tracing` on 3.15 vs `cProfile` before it. One run, with a
±1.5 ms spread on a 7.8 ms measurement; not yet a reliable regression
signal.

**Python 3.15 only — same process, same machine:**

| Benchmark | With the 3.15 feature | Without | Gain |
| --- | ---: | ---: | ---: |
| `import hypernix.quant.llamaquants` (PEP 810 vs filter off) | 23.59 | 92.34 | 3.9× |
| `import hypernix.models.download` (PEP 810 vs filter off) | 22.73 | 203.90 | 9.0× |
| flatten 5000 rows (PEP 798 vs `itertools.chain`) | 0.07 | 0.18 | 2.6× |
| merge 5000 dicts (PEP 798 vs an update loop) | 0.12 | 0.25 | 2.1× |

The PEP 810 gain is the import being deferred, not removed: the first
attribute access on `numpy` or `huggingface_hub` still pays for it. It
is a real saving for the commands that never touch them (`--help`, the
T1 API's request path, the header tools).

## What was tested

Only what actually ran is listed. "Pass" means the whole job was green:
pytest, the CLI smoke tests and the `setup.py` check after it.

**CI, full test suite** (`ci.yml`, run 37585342208, the 0.72.7 head):

| | 3.12 | 3.13 | 3.14 | 3.15.0rc3 |
| --- | --- | --- | --- | --- |
| ubuntu-latest | pass | pass | pass | pass¹ |
| ubuntu-22.04 | pass | pass | pass | pass¹ |
| windows-latest | pass | pass | pass | pass¹ |
| macos-latest (arm64) | pass² | pass² | pass² | pass¹ ² |

¹ Without torch, which has no 3.15 wheel: the tests that need it skip,
by name, with that reason; everything else runs.
² From run 37575260834, one commit series earlier: macOS runners were
still queued when this was written. The two changes since (the `gather`
queue accounting and the timer clock) are platform-neutral Python and
were tested on Linux and Windows in run 37585342208.

Also green in that run: ruff, mypy on 3.12 and 3.15, the Python 3.15
feature job (PEP 798/799/810/831 tests verbosely, plus a CMake build of
ggml-hnx whose object code is checked for frame-pointer prologues), the
benchmarks on all four versions, the native and CUDA kernel builds, and
the sdist + wheel build.

**Locally** (Linux x86-64; CPython 3.12.15, 3.13.12, 3.14.8, 3.15.0rc3;
torch 2.14.0 CPU on 3.12–3.14): the full suite on each, 9,344, 9,348,
9,348 and 7,802 tests passing. Fixes made after those runs were re-tested
locally on 3.12 and 3.15 by the suites they touch, and in full by CI.

Not tested: Python 3.15 with torch (none exists to install), PySide6 on
3.15 (it declares `<3.15`), and a free-threaded (`t`) build of any
version.


## Known issues

- **torch on 3.15.** PyTorch publishes no CPython 3.15 wheel yet (2.14.1
  stops at cp314), so `pip install hypernix` on 3.15 cannot resolve its
  torch requirement from PyPI. Until it does, install the rest with
  `python scripts/ci/requirements_without_torch.py dev t1api > reqs.txt
  && pip install -r reqs.txt && pip install --no-deps hypernix` (what CI
  does), or a torch built from source. Everything that does not touch
  torch — the T1 API, HyperLink, the CLIs, GGUF and header tools — runs;
  training, conversion and PyTorch inference need torch.
- **sentencepiece on 3.15** builds from source: install a C++ compiler
  and CMake first.
- **Extras without 3.15 wheels:** `gui-qt` (PySide6 declares `<3.15`),
  and rawpy and pillow-jxl-plugin in `images`.
- **3.15 is a release candidate** (3.15.0rc3 was tested). The final
  release is not expected to change these APIs.
