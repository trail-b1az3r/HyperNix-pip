"""PEP 799 -- the ``profiling`` package -- behind hypernix._profiling.

3.15 has ``profiling.tracing`` (cProfile, renamed) and
``profiling.sampling`` (new). 3.12-3.14 have only cProfile. These tests
check that the abstraction picks the right backend on each, that the
deterministic path gives a real profile everywhere, and that sampling
runs for real on 3.15 and refuses clearly before it.
"""
from __future__ import annotations

import pstats
import subprocess
import sys
import textwrap

import pytest

from hypernix import _profiling as hp

needs_315 = pytest.mark.skipif(sys.version_info < (3, 15), reason="profiling package is 3.15")
before_315 = pytest.mark.skipif(sys.version_info >= (3, 15), reason="checks the pre-3.15 path")


def _busy(n: int) -> int:
    return sum(i * i for i in range(n))


def test_the_backend_matches_the_interpreter():
    if sys.version_info >= (3, 15):
        assert hp.BACKEND == "profiling.tracing" and hp.SAMPLING_AVAILABLE
        assert type(hp.new_tracer()).__module__ == "profiling.tracing"
    else:
        assert hp.BACKEND == "cProfile" and not hp.SAMPLING_AVAILABLE
        assert type(hp.new_tracer()).__module__ == "cProfile"


def test_profile_call_returns_the_result_and_a_real_profile():
    result, stats = hp.profile_call(_busy, 5000)
    assert result == _busy(5000)
    assert isinstance(stats, pstats.Stats)
    assert any(func[2] == "_busy" for func in stats.stats), "the profiled function is not in it"


def test_trace_reports_and_dumps(tmp_path):
    with hp.trace() as run:
        _busy(2000)
    report = run.report(limit=5)
    assert "_busy" in report and "function calls" in report
    out = tmp_path / "run.pstats"
    run.dump(out)
    assert any(func[2] == "_busy" for func in pstats.Stats(str(out)).stats)


def test_a_failing_block_still_stops_the_profiler():
    with pytest.raises(ZeroDivisionError), hp.trace() as run:
        _busy(10)
        1 / 0  # noqa: B018
    # Disabled: a second profiler can start, which fails while one is active.
    with hp.trace():
        pass
    assert run.stats is not None


@before_315
def test_sampling_says_it_needs_3_15():
    with pytest.raises(hp.SamplingUnavailable, match="3.15"):
        hp.sampling_argv(pid=1)


@needs_315
def test_sampling_argv_shapes():
    attach = hp.sampling_argv(pid=4321, duration=2, rate="10khz", output="x.pstats")
    assert attach[1:4] == ["-m", "profiling.sampling", "attach"]
    assert attach[-1] == "4321" and "--pstats" in attach and "--all-threads" in attach
    run = hp.sampling_argv(script="t.py", script_args=["--n", "3"], fmt="collapsed")
    assert run[3] == "run" and "--collapsed" in run and run[-3:] == ["t.py", "--n", "3"]
    with pytest.raises(ValueError):
        hp.sampling_argv(pid=1, script="t.py")
    with pytest.raises(ValueError):
        hp.sampling_argv(pid=1, fmt="svg")


@needs_315
def test_sampling_runs_a_script_for_real(tmp_path):
    script = tmp_path / "spin.py"
    script.write_text(textwrap.dedent("""
        import time
        def spin():
            end = time.perf_counter() + 0.6
            n = 0
            while time.perf_counter() < end:
                n += 1
            return n
        spin()
    """), encoding="utf-8")
    out = tmp_path / "spin.pstats"
    done = hp.sample(script=script, output=out, fmt="pstats", timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert out.exists() and out.stat().st_size > 0
    names = {func[2] for func in pstats.Stats(str(out)).stats}
    assert "spin" in names, sorted(names)[:20]


@needs_315
def test_cprofile_is_still_there_on_3_15():
    """Nothing is removed: code that imports cProfile keeps working."""
    done = subprocess.run([sys.executable, "-c", "import cProfile; cProfile.Profile()"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
