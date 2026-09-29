"""The autofix tooling: scope discovery, routing, and the timing repair.

``scripts/autofix_scope.py`` decides which tests belong to a module
category; ``scripts/autofix`` routes a failure to the script that owns it;
``scripts/autofix-F`` repairs timing margins and refuses to touch anything
else. CI depends on all three agreeing, so they are tested together.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"


def _load(name: str, filename: str):
    """Import a script by path — ``autofix-F`` isn't a valid module name."""
    spec = importlib.util.spec_from_loader(
        name,
        importlib.machinery.SourceFileLoader(name, str(SCRIPTS / filename)),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scope = _load("autofix_scope_under_test", "autofix_scope.py")
autofix_f = _load("autofix_f_under_test", "autofix-F")


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["HYPERNIX_AUTO_INSTALL"] = "0"
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600, env=env,
    )


# ---------------------------------------------------------------------------
# Scope discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_finds_the_timer_tests(self) -> None:
        found = {ref.node_id for ref in scope.discover_tests("timing")}
        assert "tests/system/test_v060.py::TestTimer::test_kitchen_timer_expires_after_duration" in found
        assert "tests/system/test_v060.py::TestTimer::test_interval_timer_only_fires_after_interval" in found

    def test_excludes_tests_for_other_categories(self) -> None:
        found = {ref.node_id for ref in scope.discover_tests("timing")}
        assert not any("test_t1api" in node_id for node_id in found)
        assert "tests/system/test_v060.py::TestCompactor::test_unknown_fmt_raises" not in found

    def test_records_which_modules_a_test_touches(self) -> None:
        refs = {ref.node_id: ref for ref in scope.discover_tests("timing")}
        ref = refs["tests/system/test_v060.py::TestTimer::test_factory"]
        assert ref.modules == frozenset({"timer"})

    def test_reads_the_category_table_from_the_package(self) -> None:
        import hypernix

        assert scope.category_modules("timing") == list(
            hypernix.MODULE_CATEGORIES["timing"],
        )

    def test_unknown_category_is_a_clean_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            scope.category_modules("not-a-category")
        assert "unknown category" in str(excinfo.value)

    @pytest.mark.parametrize(
        "source",
        [
            "from hypernix import timer\ndef test_a():\n    timer.KitchenTimer()\n",
            "from hypernix.timing import timer\ndef test_a():\n    timer.KitchenTimer()\n",
            "from hypernix.timer import KitchenTimer\ndef test_a():\n    KitchenTimer()\n",
            "from hypernix import timer as clock\ndef test_a():\n    clock.KitchenTimer()\n",
            "import hypernix.timing.timer as clock\ndef test_a():\n    clock.KitchenTimer()\n",
            "def test_a():\n    from hypernix import timer\n    timer.KitchenTimer()\n",
        ],
    )
    def test_every_import_spelling_is_recognised(self, tmp_path: Path, source: str) -> None:
        (tmp_path / "test_spelling.py").write_text(source, encoding="utf-8")
        found = scope.discover_tests("timing", tmp_path)
        assert [ref.node_id.split("::")[-1] for ref in found] == ["test_a"]

    def test_a_test_that_only_mentions_other_modules_is_not_matched(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "test_other.py").write_text(
            "from hypernix import timer, compactor\n"
            "def test_uses_timer():\n    timer.KitchenTimer()\n"
            "def test_uses_compactor():\n    compactor.list_checkpoints('.')\n",
            encoding="utf-8",
        )
        found = [ref.node_id.split("::")[-1] for ref in scope.discover_tests("timing", tmp_path)]
        assert found == ["test_uses_timer"]

    def test_node_ids_are_runnable(self) -> None:
        """A discovered node id must actually select something in pytest."""
        node_id = "tests/system/test_v060.py::TestTimer::test_factory"
        assert node_id in {ref.node_id for ref in scope.discover_tests("timing")}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--collect-only", node_id],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300,
            env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


class TestTimeKwargs:
    def test_discovers_the_timer_knobs_from_the_dataclasses(self) -> None:
        knobs = scope.time_kwargs("timing")
        assert {"duration", "interval_seconds", "work_seconds", "rest_seconds"} <= knobs

    def test_does_not_invent_knobs(self) -> None:
        knobs = scope.time_kwargs("timing")
        assert "on_ring" not in knobs
        assert "rang" not in knobs


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

class TestClassify:
    def test_import_errors_win_over_everything(self) -> None:
        log = (
            "I001 [*] Import block is un-sorted\n"
            "E   ModuleNotFoundError: No module named 'hypernix.timing'\n"
            "FAILED tests/system/test_v060.py::TestTimer::test_factory - AssertionError\n"
        )
        assert scope.classify(log) == "imports"

    def test_lint_diagnostics(self) -> None:
        log = "Run ruff check src tests\nI001 [*] Import block is un-sorted\nFound 1 error.\n"
        assert scope.classify(log) == "lint"

    def test_a_failing_timing_test(self) -> None:
        log = (
            "FAILED tests/system/test_v060.py::TestTimer::test_interval_timer_only_fires_after_interval"
            " - AssertionError: assert True is False\n1 failed, 1899 passed\n"
        )
        assert scope.classify(log) == "timing"

    def test_a_failure_outside_the_category(self) -> None:
        log = "FAILED tests/t1api/test_t1api_http.py::TestModels::test_availability - assert 404 == 200\n"
        assert scope.classify(log) == "tests"

    def test_a_clean_log(self) -> None:
        assert scope.classify("1900 passed, 4 skipped in 77s\n") == "clean"

    def test_parametrised_node_ids_still_match(self) -> None:
        log = (
            "FAILED tests/system/test_v060.py::TestTimer::test_factory[case-1] - AssertionError\n"
        )
        assert scope.classify(log) == "timing"

    def test_failing_node_ids_are_extracted_in_order(self) -> None:
        log = (
            "FAILED tests/a.py::test_one - E\n"
            "FAILED tests/b.py::test_two\n"
            "FAILED tests/a.py::test_one - E\n"
        )
        assert scope.failing_node_ids(log) == ["tests/a.py::test_one", "tests/b.py::test_two"]


class TestValidateCli:
    def test_accepts_node_ids_from_the_category(self) -> None:
        proc = _run_script(
            "autofix_scope.py", "--validate",
            "tests/system/test_v060.py::TestTimer::test_factory",
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == "tests/system/test_v060.py::TestTimer::test_factory"

    def test_rejects_a_test_outside_the_category(self) -> None:
        proc = _run_script(
            "autofix_scope.py", "--validate",
            "tests/t1api/test_t1api_http.py::TestModelsEndpoints::test_availability",
        )
        assert proc.returncode == 1
        assert "not timing tests" in proc.stderr

    def test_rejects_shell_metacharacters(self) -> None:
        """CI feeds a commit trailer through here before pytest sees it."""
        proc = _run_script("autofix_scope.py", "--validate", "; rm -rf / #")
        assert proc.returncode == 1


# ---------------------------------------------------------------------------
# The repair itself
# ---------------------------------------------------------------------------

def _plan(source: str, node_id: str, factor: float = 2.0, max_sleep: float = 5.0):
    tree = ast.parse(source)
    func = autofix_f._target_function(tree, node_id)
    assert func is not None, f"could not find {node_id}"
    return autofix_f.plan_edits(
        func, source.splitlines(keepends=True),
        scope.time_kwargs("timing"), factor, max_sleep,
    )


class TestPlanEdits:
    SOURCE = textwrap.dedent("""\
        import time
        from hypernix import timer


        class TestTimer:
            def test_expiry(self):
                t = timer.KitchenTimer(duration=0.05).start()
                assert not t.expired()
                time.sleep(0.25)
                assert t.expired()
    """)

    def test_scales_both_the_duration_and_the_sleep(self) -> None:
        edits, refusals = _plan(self.SOURCE, "f.py::TestTimer::test_expiry")
        assert refusals == []
        assert {(e.old, e.new) for e in edits} == {("0.05", "0.1"), ("0.25", "0.5")}

    def test_scaling_preserves_the_ratio_between_them(self) -> None:
        edits, _ = _plan(self.SOURCE, "f.py::TestTimer::test_expiry", factor=7.0)
        by_old = {e.old: float(e.new) for e in edits}
        assert by_old["0.25"] / by_old["0.05"] == pytest.approx(0.25 / 0.05)

    def test_leaves_non_time_arguments_alone(self) -> None:
        source = textwrap.dedent("""\
            from hypernix import timer
            def test_a():
                timer.PomodoroTimer(work_seconds=0.05, rest_seconds=0.5)
                timer.timer("interval", interval_seconds=1)
                helper(retries=3, tolerance=0.01)
        """)
        edits, _ = _plan(source, "f.py::test_a")
        assert {e.old for e in edits} == {"0.05", "0.5", "1"}

    def test_never_grows_a_sleep_past_the_cap(self) -> None:
        source = "import time\ndef test_a():\n    time.sleep(4.0)\n"
        edits, refusals = _plan(source, "f.py::test_a", factor=10.0, max_sleep=5.0)
        assert refusals == []
        assert [e.new for e in edits] == ["5.0"]

    def test_refuses_when_the_sleep_is_already_at_the_cap(self) -> None:
        source = "import time\ndef test_a():\n    time.sleep(6.0)\n"
        edits, refusals = _plan(source, "f.py::test_a", factor=2.0, max_sleep=5.0)
        assert edits == []
        assert refusals and "not the fix here" in refusals[0]

    def test_ignores_zero_and_negative_constants(self) -> None:
        source = "import time\ndef test_a():\n    time.sleep(0)\n    time.sleep(0.0)\n"
        edits, _ = _plan(source, "f.py::test_a")
        assert edits == []

    def test_a_test_with_no_time_constants_yields_nothing(self) -> None:
        source = "from hypernix import timer\ndef test_a():\n    assert timer.TIERS\n"
        edits, _ = _plan(source, "f.py::test_a")
        assert edits == []

    def test_applying_edits_keeps_the_file_parseable(self, tmp_path: Path) -> None:
        path = tmp_path / "f.py"
        path.write_text(self.SOURCE, encoding="utf-8")
        edits, _ = _plan(self.SOURCE, "f.py::TestTimer::test_expiry")
        autofix_f.apply_edits(path, edits)
        updated = path.read_text(encoding="utf-8")
        ast.parse(updated)
        assert "duration=0.1" in updated
        assert "time.sleep(0.5)" in updated


class TestFailureTypeDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "assert True is False",                 # pytest's rewritten assert
            "AssertionError",
            "AssertionError: assert 0.0 >= 0.05",
            "assert not True",
        ],
    )
    def test_assertion_failures_are_margin_candidates(self, message: str) -> None:
        assert autofix_f._non_assertion_type(message) is None

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("AttributeError: module has no attribute 'NoSuchTimer'", "AttributeError"),
            ("TypeError: __init__() got an unexpected keyword", "TypeError"),
            ("ModuleNotFoundError: No module named 'x'", "ModuleNotFoundError"),
            ("ValueError: duration must be >= 0", "ValueError"),
            ("Failed: DID NOT RAISE", None),
        ],
    )
    def test_other_failures_are_named_and_left_alone(
        self, message: str, expected: str | None,
    ) -> None:
        assert autofix_f._non_assertion_type(message) == expected


class TestTargetLookup:
    SOURCE = (
        "def test_free():\n    pass\n\n\n"
        "class TestThing:\n    def test_method(self):\n        pass\n"
    )

    def test_finds_a_module_level_test(self) -> None:
        func = autofix_f._target_function(ast.parse(self.SOURCE), "f.py::test_free")
        assert func.name == "test_free"

    def test_finds_a_method(self) -> None:
        func = autofix_f._target_function(
            ast.parse(self.SOURCE), "f.py::TestThing::test_method",
        )
        assert func.name == "test_method"

    def test_strips_parametrize_ids(self) -> None:
        func = autofix_f._target_function(
            ast.parse(self.SOURCE), "f.py::TestThing::test_method[case-1]",
        )
        assert func.name == "test_method"

    def test_returns_none_for_something_that_is_not_there(self) -> None:
        assert autofix_f._target_function(ast.parse(self.SOURCE), "f.py::test_absent") is None


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

# The synthetic test autofix-F is supposed to repair. It has to fail
# *deterministically* before the fix and pass deterministically after it,
# or the end-to-end tests below become the very thing this suite exists to
# catch. The original relied on "at least 1µs will have elapsed by the time
# should_fire() runs", which is a race the test can win: on a fast runner
# the interval had not elapsed, should_fire() returned False, the test
# passed, and autofix-F correctly stood down — failing the tests that
# expected it to act. `_elapse` closes that race by busy-waiting a fixed
# 100µs, two orders of magnitude past the 1µs interval.
#
# It busy-waits rather than calling time.sleep() on purpose: autofix-F
# scales sleep arguments, and this delay must stay fixed across the repair
# so the test fails for one reason beforehand and passes for one reason
# after. plan_edits() only rewrites time.sleep() arguments and time-valued
# keyword arguments, so a positional call to a local helper is left alone.
FAILING_TIMER_TEST = textwrap.dedent("""\
    from __future__ import annotations

    import time

    from hypernix import timer


    def _elapse(seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            pass


    def test_interval_does_not_fire_immediately() -> None:
        t = timer.IntervalTimer(interval_seconds=1e-06).start()
        _elapse(1e-04)
        assert t.should_fire() is False
        time.sleep(2e-06)
        assert t.should_fire() is True
""")

PASSING_TIMER_TESTS = textwrap.dedent("""\
    from __future__ import annotations

    from hypernix import timer


    def test_factory() -> None:
        assert isinstance(timer.timer("kitchen", duration=1.0), timer.KitchenTimer)


    def test_egg_timer_starts_unrung() -> None:
        assert timer.EggTimer(duration=1.0).rang is False
""")


@pytest.fixture
def timing_tests(tmp_path: Path) -> Path:
    """A tests directory with two passing timer tests and one that fails."""
    (tmp_path / "test_ok.py").write_text(PASSING_TIMER_TESTS, encoding="utf-8")
    (tmp_path / "test_flaky.py").write_text(FAILING_TIMER_TEST, encoding="utf-8")
    return tmp_path


class TestEndToEnd:
    def test_stands_down_when_nothing_fails(self, tmp_path: Path) -> None:
        (tmp_path / "test_ok.py").write_text(PASSING_TIMER_TESTS, encoding="utf-8")
        proc = _run_script("autofix-F", "--tests-dir", str(tmp_path), "--no-commit")
        assert proc.returncode == 0
        assert "stands down" in proc.stdout

    def test_refuses_when_every_test_fails(self, tmp_path: Path) -> None:
        """All-failing is an upstream break, not a margin — do not patch."""
        (tmp_path / "test_flaky.py").write_text(FAILING_TIMER_TEST, encoding="utf-8")
        before = (tmp_path / "test_flaky.py").read_text(encoding="utf-8")

        proc = _run_script("autofix-F", "--tests-dir", str(tmp_path), "--no-commit")

        assert proc.returncode == 1
        assert "will not patch tests to hide it" in proc.stdout
        assert (tmp_path / "test_flaky.py").read_text(encoding="utf-8") == before

    def test_widens_the_failing_test_only(self, timing_tests: Path) -> None:
        untouched = (timing_tests / "test_ok.py").read_text(encoding="utf-8")

        proc = _run_script(
            "autofix-F", "--tests-dir", str(timing_tests),
            "--scale", "100000", "--max-rounds", "1", "--no-commit",
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        fixed = (timing_tests / "test_flaky.py").read_text(encoding="utf-8")
        assert "interval_seconds=0.1" in fixed
        assert "time.sleep(0.2)" in fixed
        assert (timing_tests / "test_ok.py").read_text(encoding="utf-8") == untouched

    def test_reruns_only_the_tests_it_changed(self, timing_tests: Path) -> None:
        proc = _run_script(
            "autofix-F", "--tests-dir", str(timing_tests),
            "--scale", "100000", "--max-rounds", "1", "--no-commit",
        )
        assert "Re-running the 1 changed test(s)" in proc.stdout
        assert "The rest of the suite was not re-run" in proc.stdout

    def test_dry_run_changes_nothing(self, timing_tests: Path) -> None:
        before = (timing_tests / "test_flaky.py").read_text(encoding="utf-8")
        proc = _run_script(
            "autofix-F", "--tests-dir", str(timing_tests),
            "--scale", "100000", "--dry-run",
        )
        assert proc.returncode == 0
        assert "nothing written" in proc.stdout
        assert (timing_tests / "test_flaky.py").read_text(encoding="utf-8") == before

    def test_leaves_a_non_timing_failure_alone(self, tmp_path: Path) -> None:
        (tmp_path / "test_ok.py").write_text(PASSING_TIMER_TESTS, encoding="utf-8")
        (tmp_path / "test_broken.py").write_text(
            "from hypernix import timer\n"
            "def test_renamed_symbol():\n"
            "    timer.NoSuchTimer(duration=0.05)\n",
            encoding="utf-8",
        )
        before = (tmp_path / "test_broken.py").read_text(encoding="utf-8")

        proc = _run_script("autofix-F", "--tests-dir", str(tmp_path), "--no-commit")

        assert proc.returncode == 1
        assert "left alone" in proc.stdout
        assert "AttributeError" in proc.stdout
        assert (tmp_path / "test_broken.py").read_text(encoding="utf-8") == before


class TestRouting:
    @pytest.mark.parametrize(
        ("log", "expected_script"),
        [
            ("E   ModuleNotFoundError: No module named 'x'\n", "autofix-E"),
            ("I001 [*] Import block is un-sorted\nFound 1 error.\n", "autofix-B"),
            (
                "FAILED tests/system/test_v060.py::TestTimer::test_factory - AssertionError\n",
                "autofix-F",
            ),
        ],
    )
    def test_router_picks_the_owning_script(
        self, tmp_path: Path, log: str, expected_script: str,
    ) -> None:
        log_file = tmp_path / "ci.log"
        log_file.write_text(log, encoding="utf-8")
        proc = _run_script("autofix", "--log", str(log_file), "--dry-run")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert f"would run {SCRIPTS / expected_script}" in proc.stdout

    def test_router_refuses_failures_no_script_owns(self, tmp_path: Path) -> None:
        log_file = tmp_path / "ci.log"
        log_file.write_text(
            "FAILED tests/t1api/test_t1api_http.py::TestModels::test_availability - assert 404\n",
            encoding="utf-8",
        )
        proc = _run_script("autofix", "--log", str(log_file), "--dry-run")
        assert proc.returncode == 1
        assert "need a human" in proc.stdout

    def test_autofix_f_hands_import_failures_to_autofix_e(self, tmp_path: Path) -> None:
        log_file = tmp_path / "ci.log"
        log_file.write_text(
            "ERROR tests/system/test_v060.py\nE   ModuleNotFoundError: No module named 'x'\n"
            "FAILED tests/system/test_v060.py::TestTimer::test_factory - ModuleNotFoundError\n",
            encoding="utf-8",
        )
        proc = _run_script("autofix-F", "--log", str(log_file), "--dry-run")
        assert proc.returncode == 0
        assert "autofix-E's job, not autofix-F's" in proc.stdout


# ---------------------------------------------------------------------------
# CI wiring
# ---------------------------------------------------------------------------

class TestCiWorkflow:
    WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    def test_the_matrix_is_skipped_for_an_autofix_commit(self) -> None:
        assert "if: needs.triage.outputs.narrow != 'true'" in self.WORKFLOW

    def test_there_is_a_narrow_verification_job(self) -> None:
        assert "autofix-verify:" in self.WORKFLOW
        assert "if: needs.triage.outputs.narrow == 'true'" in self.WORKFLOW

    def test_the_trailers_are_passed_through_the_environment(self) -> None:
        """Never interpolated into a shell script — they come from a commit."""
        assert "AUTOFIX_TESTS: ${{ needs.triage.outputs.tests }}" in self.WORKFLOW
        assert 'pytest -q $tests' not in self.WORKFLOW

    def test_the_node_ids_are_validated_before_pytest_sees_them(self) -> None:
        assert "--validate" in self.WORKFLOW

    def test_autofix_f_writes_the_trailers_ci_reads(self) -> None:
        source = (SCRIPTS / "autofix-F").read_text(encoding="utf-8")
        assert "Autofix-Scope: " in source
        assert "Autofix-Tests: " in source
