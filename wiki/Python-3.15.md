# Python 3.12–3.15

**hyperNix-pip requires Python 3.12 or newer and officially supports
Python 3.12, 3.13, 3.14, and 3.15.** `Requires-Python` is
`>=3.12,<3.16`; 3.10 and 3.11 stop at 0.72.6.post4, which pip installs
there on its own.

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

<!-- RESULTS -->

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
