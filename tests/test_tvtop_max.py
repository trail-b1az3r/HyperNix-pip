"""tvtop-max (0.72.6.rc2): the script reader, the bridge, the launcher.

The screen itself is TypeScript and has its own tests (bun test in
src/hypernix/monitoring/tvtop_max_app). These cover everything it is
told: what a training script is made of, what its log says, which
process is the run, and that the app ships and launches.
"""
from __future__ import annotations

import io
import json
import re
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

import hypernix
from hypernix.interfaces import hyped_pro_otui as otui
from hypernix.monitoring import run_inspect, tvtop_max, tvtop_max_bridge
from hypernix.monitoring.run_inspect import (
    analyze_script,
    arch_from_folder,
    cooker_catalogue,
    findings_from_log,
    script_from_command,
)

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "hypernix" / "monitoring" / "tvtop_max_app"


def _script(tmp_path: Path, body: str, name: str = "train.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


SAMPLE = """
    import os, json
    import torch
    from hypernix.models.neo_oven import new_oven
    from hypernix.models import old_oven
    from hypernix.optimizers.pressure_cooker_v5 import PressureCookerV5
    from hypernix.optimizers.pressure_cooker import PressureCooker
    import helpers
    import surely_not_installed_pkg

    oven = new_oven("./out", arch="qwen3", hidden_size=512, num_hidden_layers=8)
    opt = PressureCookerV5(oven.model.parameters(), lr=3e-4, weight_decay=0.1)
    state = torch.load("x.pt")
    oven.train(dataset="d.txt", steps=100, optimizer_class=PressureCooker)
"""


# ---------------------------------------------------------------------------
# Which script
# ---------------------------------------------------------------------------


class TestScriptFromCommand:
    @pytest.mark.parametrize("command", [
        "python train.py",
        "python -u train.py --lr 3e-4",
        "torchrun --nproc_per_node 2 train.py",
        "accelerate launch train.py --config c.yaml",
        "/usr/bin/python3.11 ./train.py",
    ])
    def test_the_first_py_argument(self, tmp_path, command):
        _script(tmp_path, "print(1)\n")
        assert script_from_command(command, cwd=tmp_path) == tmp_path / "train.py"

    def test_arguments_after_the_script_are_the_scripts(self, tmp_path):
        """tvtop-max's own launcher mentions the script it was pointed at."""
        _script(tmp_path, "print(1)\n")
        assert script_from_command("python /usr/bin/tvtop-max -S train.py", cwd=tmp_path) is None
        assert script_from_command("python -W ignore train.py --out x.py", cwd=tmp_path) == tmp_path / "train.py"

    def test_distributed_launchers(self, tmp_path):
        _script(tmp_path, "print(1)\n")
        for command in ("python -m torch.distributed.run --nproc_per_node 2 train.py",
                        "deepspeed --num_gpus 2 train.py"):
            assert script_from_command(command, cwd=tmp_path) == tmp_path / "train.py"

    def test_an_executable_script(self, tmp_path):
        _script(tmp_path, "print(1)\n")
        assert script_from_command("./train.py --lr 1", cwd=tmp_path) == tmp_path / "train.py"

    def test_a_missing_file_is_not_a_script(self, tmp_path):
        assert script_from_command("python gone.py", cwd=tmp_path) is None

    def test_dash_m_resolves_without_importing(self):
        found = script_from_command("python -m hypernix.monitoring.run_inspect")
        assert found is not None and found.name == "run_inspect.py"

    def test_no_script(self):
        assert script_from_command("python -c 'print(1)'") is None


# ---------------------------------------------------------------------------
# What the script is made of
# ---------------------------------------------------------------------------


class TestAnalyzeScript:
    @pytest.fixture
    def report(self, tmp_path):
        (tmp_path / "helpers.py").write_text("x = 1\n", encoding="utf-8")
        return analyze_script(_script(tmp_path, SAMPLE))

    def _item(self, report, module):
        return next(i for i in report.imports if i.module == module)

    def test_imports_are_sorted_into_kinds(self, report):
        kinds = {i.module: i.kind for i in report.imports}
        assert kinds["hypernix.models.neo_oven"] == "hypernix"
        assert kinds["os"] == "stdlib"
        assert kinds["helpers"] == "local"
        assert kinds["surely_not_installed_pkg"] == "missing"

    def test_hypernix_modules_carry_their_summary_and_the_running_version(self, report):
        oven = self._item(report, "hypernix.models.neo_oven")
        assert oven.names == ["new_oven"]
        assert oven.summary
        assert oven.version == hypernix.__version__

    def test_from_hypernix_import_a_module_is_that_module(self, report):
        assert "hypernix.models.old_oven" in {i.module for i in report.imports}

    def test_a_deprecated_module_says_what_to_use(self, report):
        assert self._item(report, "hypernix.models.old_oven").deprecated == "hypernix.models.neo_oven"

    def test_libraries_carry_their_installed_version(self, report):
        pytest.importorskip("torch")
        torch = self._item(report, "torch")
        assert torch.kind == "library" and torch.version

    def test_the_architecture_is_read_from_new_oven(self, report):
        arch = report.arch
        assert (arch.source, arch.arch, arch.line) == ("script", "qwen3", 11)
        assert arch.fields == {"hidden_size": 512, "num_hidden_layers": 8}
        assert arch.params and arch.params > 0

    def test_both_cookers_are_found_called_or_named(self, report):
        found = {c.name: c for c in report.cookers}
        assert found["PressureCookerV5"].generation == "v5"
        assert found["PressureCookerV5"].kwargs == {"lr": 3e-4, "weight_decay": 0.1}
        assert found["PressureCookerV5"].deprecated == ""
        # Passed as optimizer_class=, never called: still the optimizer.
        assert found["PressureCooker"].generation == "v1"
        assert "PressureCookerV4" in found["PressureCooker"].deprecated

    def test_findings(self, report):
        messages = [(f.level, f.message) for f in report.findings]
        text = "\n".join(m for _, m in messages)
        assert ("error", "imports surely_not_installed_pkg, which is not installed in this environment") in messages
        assert "hypernix.models.old_oven is deprecated" in text
        assert "Pressure Cooker V1, which is deprecated" in text
        assert "torch.load without weights_only=True" in text
        assert "no seed is set" in text
        assert "nothing in the script saves a checkpoint" in text

    def test_a_careful_script_has_fewer_findings(self, tmp_path):
        report = analyze_script(_script(tmp_path, """
            import torch
            from hypernix.models.neo_oven import new_oven
            torch.manual_seed(0)
            oven = new_oven("./out", arch="gemma3")
            state = torch.load("x.pt", weights_only=True)
            oven.train(dataset="d.txt")
            oven.save_pt("out.pt")
        """))
        assert report.findings == []
        assert report.arch.arch == "gemma3"

    def test_preheat_names_the_snapshot(self, tmp_path):
        report = analyze_script(_script(tmp_path, 'from hypernix import preheat\noven = preheat("me/my-model")\n'))
        assert (report.arch.source, report.arch.repo) == ("script", "me/my-model")

    def test_a_script_that_does_not_parse(self, tmp_path):
        report = analyze_script(_script(tmp_path, "def broken(:\n"))
        assert report.error.startswith("syntax error on line 1")
        assert report.findings[0].level == "error"

    def test_an_unreadable_script(self, tmp_path):
        assert analyze_script(tmp_path / "nope.py").error.startswith("could not read")

    def test_the_script_is_never_run(self, tmp_path):
        marker = tmp_path / "ran"
        analyze_script(_script(tmp_path, f"open({str(marker)!r}, 'w').write('yes')\nimport sys\n"))
        assert not marker.exists()


class TestCatalogue:
    def test_every_generation_is_there(self):
        generations = {gen for gen, _ in cooker_catalogue().values()}
        assert {"v1", "v3", "v4", "v5", "v5s", "v6", "v6v"} <= generations

    def test_read_without_importing_torch(self, monkeypatch):
        run_inspect._COOKERS = None
        monkeypatch.setitem(sys.modules, "torch", None)  # any torch import now fails
        assert cooker_catalogue()["PressureCookerV4"][0] == "v4"


class TestArchFromFolder:
    def test_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "llama", "hidden_size": 256, "num_hidden_layers": 4, "vocab_size": 1000,
        }), encoding="utf-8")
        arch = arch_from_folder(tmp_path)
        assert (arch.source, arch.arch) == ("config.json", "llama")
        assert arch.fields["hidden_size"] == 256

    def test_nothing_there(self, tmp_path):
        assert arch_from_folder(tmp_path).source == "none"


# ---------------------------------------------------------------------------
# What the log says
# ---------------------------------------------------------------------------


class TestLogFindings:
    def test_errors_first_repeats_counted_once(self):
        lines = [
            "step 1/10 loss=2.3",
            "/x.py:3: UserWarning: TypedStorage is deprecated",
            "/x.py:3: UserWarning: TypedStorage is deprecated",
            "step 2/10 loss=nan",
            "Traceback (most recent call last):",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB",
            "WARNING: dataloader is slow",
        ]
        found = findings_from_log(lines)
        assert [f.level for f in found][:3] == ["error", "error", "error"]
        messages = {f.message: f for f in found}
        assert messages["UserWarning: TypedStorage is deprecated"].count == 2
        assert "the loss is nan" in messages
        assert "out of GPU memory" in messages
        assert "the run raised an exception" in messages
        assert "dataloader is slow" in messages

    def test_a_clean_log(self):
        assert findings_from_log([f"step {i}/10 loss=1.0" for i in range(10)]) == []


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


@pytest.fixture
def run(tmp_path):
    script = _script(tmp_path, SAMPLE)
    log = tmp_path / "train.log"
    log.write_text("step 1/100 loss=2.31 lr=3e-4\nUserWarning: x is deprecated\nstep 2/100 loss=2.10 lr=3e-4\n",
                   encoding="utf-8")
    return script, log


class TestBridge:
    def test_frame_carries_the_numbers_log_and_findings(self, run):
        script, log = run
        monitor = tvtop_max_bridge.Monitor(log=str(log), script=str(script), discover=False)
        frame = monitor.frame()
        assert (frame["step"], frame["total_steps"], frame["loss"]) == (2, 100, 2.1)
        assert frame["log_lines"][-1] == "step 2/100 loss=2.10 lr=3e-4"
        assert frame["log_findings"][0]["message"] == "UserWarning: x is deprecated"
        assert "cpu_per_core" in frame and "processes" in frame
        json.dumps(frame, default=str)

    def test_info_carries_the_report(self, run):
        script, log = run
        monitor = tvtop_max_bridge.Monitor(log=str(log), script=str(script), discover=False)
        info = monitor.info()
        assert info["script"] == str(script) and info["log"] == str(log)
        assert info["arch"]["arch"] == "qwen3"
        assert {c["name"] for c in info["report"]["cookers"]} == {"PressureCookerV5", "PressureCooker"}
        assert info["hypernix_version"] == hypernix.__version__

    def test_no_script_is_said(self, tmp_path):
        monitor = tvtop_max_bridge.Monitor(discover=False)
        assert monitor.info()["report"] is None

    def test_protocol(self, run):
        script, log = run
        monitor = tvtop_max_bridge.Monitor(log=str(log), script=str(script), discover=False)
        out = io.StringIO()
        requests = io.StringIO('{"id": 1, "cmd": "ping"}\nnot json\n{"id": 2, "cmd": "nope"}\n')
        assert tvtop_max_bridge.serve(monitor, stdin=requests, stdout=out) == 0
        replies = {r["id"]: r for r in map(json.loads, out.getvalue().splitlines())}
        assert replies[1] == {"id": 1, "ok": True, "data": {"pong": True}}
        assert replies[2]["ok"] is False and replies[2]["code"] == "TVM-BRIDGE-001"

    def test_a_failing_command_is_a_reply_not_a_crash(self, run, monkeypatch):
        script, log = run
        monitor = tvtop_max_bridge.Monitor(log=str(log), script=str(script), discover=False)
        monkeypatch.setattr(monitor, "frame", lambda: 1 / 0)
        reply = monitor.handle({"id": 9, "cmd": "frame"})
        assert reply["ok"] is False and "ZeroDivisionError" in reply["error"]


class _Candidate:
    def __init__(self, pid, command):
        self.pid, self.command = pid, command

    def to_dict(self):
        return {"pid": self.pid, "command": self.command}


class TestWhichProcess:
    def _monitor(self, monkeypatch, candidates, family, **kwargs):
        from hypernix.monitoring import stale_log

        monkeypatch.setattr(stale_log, "rank_python_processes", lambda limit=10: candidates)
        monkeypatch.setattr(stale_log, "_process_cwd", lambda pid: None)
        monkeypatch.setattr(tvtop_max_bridge, "_own_family", lambda: set(family))
        monkeypatch.setattr(tvtop_max_bridge.Monitor, "_find_log", lambda self, cwd: None)
        return tvtop_max_bridge.Monitor(**kwargs)

    def test_the_busiest_that_is_not_tvtop_max_itself(self, monkeypatch):
        monitor = self._monitor(monkeypatch, [_Candidate(1, "python launcher"), _Candidate(2, "python run.py")],
                                family={1})
        assert monitor.pid == 2

    def test_a_named_script_only_matches_a_process_running_it(self, monkeypatch, tmp_path):
        script = _script(tmp_path, "print(1)\n")
        other = _script(tmp_path, "print(2)\n", name="other.py")
        candidates = [
            # tvtop-max's own launcher mentions the script but runs something else
            _Candidate(10, f"python /usr/bin/tvtop-max -S {script}"),
            _Candidate(11, f"python {other}"),
            _Candidate(12, f"python -u {script} --lr 1"),
        ]
        monitor = self._monitor(monkeypatch, candidates, family=set(), script=str(script))
        assert monitor.pid == 12

    def test_nothing_running_it_is_no_process_not_a_guess(self, monkeypatch, tmp_path):
        script = _script(tmp_path, "print(1)\n")
        other = _script(tmp_path, "print(2)\n", name="other.py")
        monitor = self._monitor(monkeypatch, [_Candidate(11, f"python {other}")], family=set(), script=str(script))
        assert monitor.pid is None

    def test_own_family_includes_this_process(self):
        assert tvtop_max_bridge._own_family() >= {__import__("os").getpid()}


# ---------------------------------------------------------------------------
# The launcher and the package
# ---------------------------------------------------------------------------


class TestLauncher:
    def test_without_bun_it_says_so_and_names_tvtop_pro(self, monkeypatch, capsys):
        def no_bun(*a, **k):
            raise otui.LaunchError(otui.INSTALL_BUN_HINT, exit_code=127)

        monkeypatch.setattr(otui, "find_bun", no_bun)
        assert tvtop_max.cli_main([]) == 127
        err = capsys.readouterr().err
        assert "tvtop-max runs on Bun" in err and "tvtop-pro" in err and "hyped" not in err

    def test_it_runs_the_app_with_this_interpreter(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(otui, "find_bun", lambda **k: "/b/bun")
        monkeypatch.setattr(otui, "prepare_app", lambda source, **k: seen.setdefault("prep", (source, k)) and source)
        monkeypatch.setattr(tvtop_max.subprocess, "call", lambda cmd, env: seen.setdefault("run", (cmd, env)) and 0)
        assert tvtop_max.cli_main(["-s", "--debug"]) == 0
        source, kwargs = seen["prep"]
        assert source == tvtop_max.APP_DIR
        assert (kwargs["name"], kwargs["fallback"]) == ("tvtop-max", "tvtop-pro")
        cmd, env = seen["run"]
        assert cmd == ["/b/bun", "run", str(tvtop_max.APP_DIR / "src" / "index.ts"), "-s"]
        assert env["TVTOP_MAX_PYTHON"] == sys.executable

    def test_prepare_app_uses_its_own_runtime_folder(self, tmp_path):
        assert otui.runtime_root({"HOME": str(tmp_path)}, name="tvtop-max") == tmp_path / ".hypernix" / "tvtop-max"


class TestPackage:
    def test_the_commands_exist(self):
        scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
        assert scripts["tvtop-max"] == "hypernix.monitoring.tvtop_max:cli_main"
        assert scripts["tvtop-pro"] == "hypernix.monitoring.tvtoppro:cli_main"  # still there

    def test_the_app_ships_and_its_node_modules_do_not(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for item in ("package.json", "bun.lock", "tsconfig.json", "src/*.ts"):
            assert f'"monitoring/tvtop_max_app/{item}"' in pyproject
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        assert "recursive-include src/hypernix/monitoring/tvtop_max_app" in manifest
        assert "prune src/hypernix/monitoring/tvtop_max_app/node_modules" in manifest
        assert (APP / "src" / "index.ts").is_file() and (APP / "bun.lock").is_file()

    def test_the_app_version_is_the_package_version(self):
        package = json.loads((APP / "package.json").read_text(encoding="utf-8"))
        assert package["version"] == otui.semver_of(hypernix.__version__)
        assert f'export const VERSION = "{package["version"]}"' in (APP / "src" / "app.ts").read_text(encoding="utf-8")

    def test_the_release_bumps_and_commits_it(self):
        workflow = (ROOT / ".github" / "workflows" / "public-release.yml").read_text(encoding="utf-8")
        bump = workflow[workflow.index("Bump versions in source"):]
        bump = bump[:bump.index("- name: Lint")]
        assert 'src/hypernix/monitoring/tvtop_max_app")' in bump
        commit = workflow[workflow.index("- name: Commit version bump"):]
        staged = commit[commit.index("git add"):commit.index("git diff --cached")]
        for path in ("src/hypernix/monitoring/tvtop_max_app/package.json",
                     "src/hypernix/monitoring/tvtop_max_app/src/app.ts"):
            assert path in staged

    def test_ci_runs_its_tests(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        job = ci[ci.index("  tvtop-max:"):]
        assert "working-directory: src/hypernix/monitoring/tvtop_max_app" in job[:600]
        assert "bun test" in job[:900]

    def test_the_palette_is_hyped_pros(self):
        """One product, one set of colours: the palettes cannot drift."""
        def palette(path):
            text = path.read_text(encoding="utf-8")
            block = text[text.index("export const palette"):text.index("} as const")]
            return dict(re.findall(r"(\w+): \"(#[0-9a-f]{6})\"", block))

        ours = palette(APP / "src" / "theme.ts")
        theirs = palette(ROOT / "src" / "hypernix" / "interfaces" / "hyped_pro_app" / "src" / "theme.ts")
        assert ours and ours == theirs
