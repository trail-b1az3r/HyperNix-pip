"""Linking llama.cpp into HyperLink, checked without a Mac.

The iOS app gained a real inference engine in 0.72.4.post11. None of it
can be compiled here — an xcframework is produced by `xcodebuild`, which
does not cross-compile, and there is no Swift toolchain in CI either. So
these tests check the things that are checkable offline and are also the
things most likely to be wrong:

* every `llama_*` symbol `LlamaRunner.swift` calls exists in the real
  header at the pinned ref
* the pinned ref matches the one the desktop engine uses
* the project actually references the framework, and degrades when it
  is absent

The symbol check is the important one. llama.cpp's C API churns hard —
`llama_load_model_from_file` became `llama_model_load_from_file`,
`use_mmap` became `load_mode`, `llama_kv_cache_clear` became
`llama_memory_clear` — and code written from memory compiles into
nothing on a machine nobody in CI has.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
IOS = REPO_ROOT / "ios"
RUNNER = IOS / "HyperLink" / "Sources" / "OnDevice" / "LlamaRunner.swift"
MANIFEST = IOS / "vendor" / "llama-api-b10883.json"
BUILD_SCRIPT = IOS / "scripts" / "build_llama_xcframework.sh"
PROJECT = IOS / "project.yml"
XCCONFIG = IOS / "vendor" / "LocalLlama.xcconfig"
DESKTOP_BUILD = REPO_ROOT / "native" / "ggml-hnx" / "build.sh"
PREPARE = IOS / "scripts" / "prepare_project.py"


def _prepare():
    """Load prepare_project.py by path — ios/scripts is not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("prepare_project", PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def api() -> dict:
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runner_code() -> str:
    """LlamaRunner with its comments stripped.

    The comments name the *old* API deliberately, to say what moved. A
    check that read them would find `llama_load_model_from_file` and
    conclude the code calls a function that no longer exists.
    """
    source = RUNNER.read_text(encoding="utf-8")
    return "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("//")
    )


class TestEverySymbolExists:
    """The check that would have caught code written from memory."""

    def test_every_function_called_is_declared(self, api, runner_code):
        called = set(re.findall(r"\b(llama_[a-z0-9_]+)\s*\(", runner_code))
        # Swift constructs a C struct with call syntax, so the regex
        # catches struct names too. They are checked separately below.
        called -= set(api["structs"])
        missing = sorted(called - set(api["functions"]))
        assert not missing, (
            f"LlamaRunner calls functions that do not exist at "
            f"{api['llama_ref']}: {missing}"
        )

    def test_every_struct_used_is_declared(self, api, runner_code):
        used = set(re.findall(r"\b(llama_[a-z0-9_]+)\s*\(", runner_code))
        used &= set(api["structs"]) | set(api["functions"])
        structs = sorted(s for s in used if s in api["structs"])
        assert structs, "no llama structs used at all — is the file empty?"
        for name in structs:
            assert name in api["structs"]

    def test_every_constant_used_is_declared(self, api, runner_code):
        used = set(re.findall(r"\bLLAMA_[A-Z0-9_]+\b", runner_code))
        # The compilation condition is ours, not llama.cpp's.
        used.discard("LLAMA_LOCAL")
        missing = sorted(used - set(api["enum_constants"]))
        assert not missing, (
            f"LlamaRunner uses constants that do not exist at "
            f"{api['llama_ref']}: {missing}"
        )

    def test_it_uses_the_current_names_not_the_retired_ones(self, runner_code):
        """Each of these was the name a year ago and is gone now."""
        retired = [
            "llama_load_model_from_file",
            "llama_new_context_with_model",
            "llama_free_model",
            "llama_kv_cache_clear",
            "llama_n_ctx_train(",
        ]
        found = [name for name in retired if name in runner_code]
        assert not found, f"retired llama.cpp API still called: {found}"

    def test_it_does_not_set_use_mmap(self, runner_code):
        """`use_mmap` and `use_mlock` were replaced by `load_mode`.

        Setting a field that no longer exists is a compile error on a
        machine nobody in CI has.
        """
        assert "use_mmap" not in runner_code
        assert "use_mlock" not in runner_code
        assert "load_mode" in runner_code

    def test_the_manifest_says_where_it_came_from(self, api):
        assert api["source"].startswith("https://raw.githubusercontent.com/")
        assert api["llama_ref"] in api["source"]
        assert len(api["functions"]) > 100, "the manifest looks truncated"


class TestTheEngineMatchesTheDesktop:
    """A phone on a different llama.cpp would disagree with the desktop
    about the HyperNix tensor types, and the failure would look like a
    corrupt model rather than a version skew."""

    @staticmethod
    def _ref(path: Path) -> str:
        found = re.search(r'LLAMA_REF="\$\{LLAMA_REF:-([^}"]+)\}"', path.read_text())
        assert found, f"no LLAMA_REF in {path}"
        return found.group(1)

    def test_the_ios_and_desktop_refs_agree(self):
        assert self._ref(BUILD_SCRIPT) == self._ref(DESKTOP_BUILD)

    def test_the_manifest_matches_the_pinned_ref(self, api):
        assert api["llama_ref"] == self._ref(BUILD_SCRIPT)

    def test_the_runner_names_the_ref_it_was_written_against(self):
        assert self._ref(BUILD_SCRIPT) in RUNNER.read_text(encoding="utf-8")

    def test_the_ios_build_applies_the_hypernix_patch(self):
        """Or the phone cannot read the sub-bit models hyprslug makes."""
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "patch_llamacpp.py" in script
        assert 'HNX_PATCH="${HNX_PATCH:-1}"' in script, "patched must be the default"

    def test_it_reverts_before_re_patching(self):
        """A reused checkout would otherwise be patched twice."""
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "--revert" in script


class TestItDegradesWithoutTheEngine:
    """Someone changing a view must not need a twenty-minute compile."""

    def test_the_runner_is_behind_a_compilation_condition(self):
        source = RUNNER.read_text(encoding="utf-8")
        assert source.lstrip().startswith(("//", "#if")) 
        assert "#if HNX_LOCAL_LLAMA" in source
        assert source.rstrip().endswith("#endif")

    def test_the_flag_ships_off(self):
        """Committed off, so a fresh checkout builds."""
        assert XCCONFIG.is_file()
        body = XCCONFIG.read_text(encoding="utf-8")
        setting = re.search(r"^HNX_LOCAL_LLAMA\s*=(.*)$", body, re.M)
        assert setting, "no HNX_LOCAL_LLAMA in the xcconfig"
        assert not setting.group(1).strip(), "the engine flag is committed ON"

    def test_the_build_script_turns_it_on(self):
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "HNX_LOCAL_LLAMA = HNX_LOCAL_LLAMA" in script

    def test_there_is_a_runner_for_a_build_without_it(self):
        local = (IOS / "HyperLink" / "Sources" / "OnDevice" / "LocalRunner.swift").read_text(
            encoding="utf-8"
        )
        assert "actor EchoRunner: ModelRunner" in local
        assert "notBuiltIn" in local

    def test_the_committed_spec_names_no_framework(self):
        """Because XcodeGen cannot skip one that is absent.

        This test used to assert the dependency was declared with
        `optional: true`, on the belief that "optional" meant "skip when
        the file is missing". It does not — it sets weak *linking* — so
        the framework still had to exist at build time, and a checkout
        without it failed with "There is no XCFramework found at ...".

        The test passed the whole time. It checked the spelling in the
        YAML, not the behaviour, which is the useful lesson: the
        decision now lives in prepare_project.py, where it can be
        driven both ways from Python.
        """
        import yaml
        project = yaml.safe_load(PROJECT.read_text(encoding="utf-8"))
        deps = project["targets"]["HyperLink"].get("dependencies", [])
        assert not [d for d in deps if "llama" in str(d)], (
            "project.yml names the framework directly, so a checkout "
            "without it cannot build"
        )

    def test_the_generator_adds_it_when_the_engine_is_there(self, tmp_path):
        spec = _prepare().render(
            PROJECT.read_text(encoding="utf-8"), with_engine=True
        )
        import yaml
        deps = yaml.safe_load(spec)["targets"]["HyperLink"]["dependencies"]
        assert any("llama.xcframework" in d["framework"] for d in deps)

    def test_the_generator_leaves_it_out_when_it_is_not(self):
        spec = _prepare().render(
            PROJECT.read_text(encoding="utf-8"), with_engine=False
        )
        import yaml
        assert not yaml.safe_load(spec)["targets"]["HyperLink"].get("dependencies")

    @pytest.mark.parametrize("with_engine", [True, False])
    def test_both_branches_are_valid_yaml(self, with_engine):
        """The first version of the generator replaced the marker text
        and left its indentation behind, which merged into the next line
        and turned `    settings:` into `        settings:`. The spec
        stopped parsing, and only running it showed that."""
        import yaml
        spec = _prepare().render(
            PROJECT.read_text(encoding="utf-8"), with_engine=with_engine
        )
        parsed = yaml.safe_load(spec)
        assert "settings" in parsed["targets"]["HyperLink"], "indentation broke"
        assert parsed["targets"]["HyperLink"]["settings"]["base"][
            "PRODUCT_BUNDLE_IDENTIFIER"
        ] == "com.hypernix.hyperlink"

    def test_the_framework_is_linked_not_embedded(self):
        """Upstream builds it with BUILD_SHARED_LIBS=OFF, so it is a
        static framework: its code goes into the app binary. Embedding
        one copies a static archive into the bundle, which App Store
        validation rejects and which is pure size in the meantime."""
        import yaml
        spec = _prepare().render(
            PROJECT.read_text(encoding="utf-8"), with_engine=True
        )
        deps = yaml.safe_load(spec)["targets"]["HyperLink"]["dependencies"]
        framework = next(d for d in deps if "llama" in d["framework"])
        assert framework.get("embed") is False

    def test_an_empty_directory_is_not_an_engine(self):
        """An interrupted copy leaves one behind, and xcodebuild's
        complaint about that is much less clear than ours."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "llama.xcframework"
            empty.mkdir()
            assert not _prepare().engine_present(empty)
            (empty / "Info.plist").write_text("<plist/>")
            assert _prepare().engine_present(empty)

    def test_ci_runs_the_generator_before_generating(self):
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "ios.yml"
        ).read_text(encoding="utf-8")
        assert "prepare_project.py" in workflow
        assert "--spec project.generated.yml" in workflow, (
            "CI generates from the committed spec, which has no dependency "
            "and no way to gain one"
        )

    def test_the_project_reads_the_xcconfig(self):
        import yaml
        project = yaml.safe_load(PROJECT.read_text(encoding="utf-8"))
        configs = project["targets"]["HyperLink"].get("configFiles", {})
        assert set(configs) == {"Debug", "Release"}
        assert all("LocalLlama.xcconfig" in path for path in configs.values())


class TestTheBuildScript:
    def test_it_parses(self):
        import subprocess
        subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)

    def test_it_refuses_a_non_mac_rather_than_failing_late(self):
        """An xcframework needs xcodebuild, which does not
        cross-compile. Saying so before a clone is better than a cmake
        error twenty minutes in."""
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert 'uname -s' in script and "Darwin" in script
        assert "xcodebuild" in script

    def test_it_points_at_the_desktop_build_on_linux(self):
        assert "native/ggml-hnx/build.sh" in BUILD_SCRIPT.read_text(encoding="utf-8")

    def test_it_uses_upstreams_own_apple_build(self):
        """Rather than hand-listing sources into an Xcode target, which
        breaks on every llama.cpp restructure."""
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "build-xcframework.sh" in script

    def test_it_checks_that_script_exists_before_relying_on_it(self):
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert 'has no build-xcframework.sh' in script

    def test_it_embeds_the_metal_library(self):
        """With it off the shaders ship as a separate .metallib the app
        has to find at runtime, and failing to find it is a silent fall
        back to CPU rather than an error."""
        assert "GGML_METAL_EMBED_LIBRARY=ON" in BUILD_SCRIPT.read_text(encoding="utf-8")

    def test_it_records_what_it_built(self):
        """xcodebuild gives no way to ask an xcframework which commit
        produced it."""
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "llama.xcframework.json" in script
        assert "hnx_patched" in script


class TestAReleaseShipsTheEngine:
    """Every release shipped an IPA that could not run a model.

    `ios.yml` builds llama.cpp only when `local_engine` is set, and
    `release.yml` called it without setting anything — so the input took
    its `false` default and the artifact people install linked no engine
    and used `EchoRunner`. Every on-device feature was inert in it.

    Nothing reported this, and that is the interesting part: falling back
    to EchoRunner is the *correct* behaviour for a PR build that did not
    ask for an engine, so the release path was quietly taking the right
    branch for the wrong reason.
    """

    @staticmethod
    def _workflow(name: str) -> dict:
        import yaml

        return yaml.safe_load(
            (Path(__file__).resolve().parent.parent / ".github" / "workflows" / name)
            .read_text(encoding="utf-8")
        )

    def test_the_release_asks_for_the_engine(self):
        release = self._workflow("release.yml")
        job = release["jobs"]["ios"]
        assert job["with"].get("local_engine") is True, (
            "release.yml calls ios.yml without local_engine, so the shipped "
            "IPA links no llama.cpp"
        )

    def test_a_caller_that_forgets_still_gets_one(self):
        """The only reason to call this workflow is to ship the result.
        The cheap default belongs on the PR path, where it still is."""
        ios = self._workflow("ios.yml")
        # PyYAML parses the `on:` key as the boolean True.
        triggers = ios.get("on") or ios.get(True)
        assert triggers["workflow_call"]["inputs"]["local_engine"]["default"] is True

    def test_a_pull_request_still_does_not_pay_for_it(self):
        """15-25 minutes on a hosted macOS runner is not worth paying on
        a PR that touched a view."""
        ios = self._workflow("ios.yml")
        triggers = ios.get("on") or ios.get(True)
        assert triggers["workflow_dispatch"]["inputs"]["local_engine"]["default"] is False

    def test_the_engine_step_is_still_conditional(self):
        ios = self._workflow("ios.yml")
        steps = ios["jobs"]["build"]["steps"]
        engine = next(s for s in steps if s.get("name") == "Build the inference engine")
        assert engine.get("if"), "the engine step must stay conditional"

    def test_the_generate_step_requires_what_was_asked_for(self):
        """The guard against the next version of this bug: an engine step
        that runs and fails would otherwise still produce an IPA, because
        prepare_project.py treats a missing framework as "no engine
        wanted"."""
        ios = self._workflow("ios.yml")
        steps = ios["jobs"]["build"]["steps"]
        generate = next(s for s in steps if s.get("name") == "Generate the Xcode project")
        assert "--require-engine" in generate["run"]

    def test_the_condition_does_not_read_inputs_directly(self):
        """The bug that survived the first fix, and it looks correct.

        The `inputs` context exists *only* for `workflow_dispatch` and
        `workflow_call`. On a `push` or a `pull_request` it is not
        populated, so `if: ${{ inputs.local_engine }}` is null, null is
        falsy, and the engine was skipped on every commit to main — with
        a grey "skipped" in the log indistinguishable from a deliberate
        one. Setting the `workflow_call` default to true fixed the
        release path and could not have fixed this one, because the
        release path was never the one skipping.
        """
        ios = self._workflow("ios.yml")
        steps = ios["jobs"]["build"]["steps"]
        for name in ("Build the inference engine", "Generate the Xcode project"):
            step = next(s for s in steps if s.get("name") == name)
            text = str(step.get("if", "")) + step.get("run", "")
            assert "inputs.local_engine" not in text, (
                f"{name} reads inputs.local_engine directly, which is null "
                f"on push and pull_request"
            )


class TestTheEngineDecisionIsMadeForEveryTrigger:
    """The decision step, run as the shell actually runs it.

    Parsing the YAML and asserting on the text of a condition is what
    let this through the first time: the old test checked that the `if`
    *mentioned* `local_engine`, which it did, and not that it was ever
    true on a push, which it never was. So this extracts the real script
    and runs it with each combination a trigger can produce.
    """

    @staticmethod
    def _script() -> str:
        import yaml

        ios = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ios.yml")
            .read_text(encoding="utf-8")
        )
        steps = ios["jobs"]["build"]["steps"]
        step = next(
            s for s in steps if s.get("name") == "Decide whether to build the engine"
        )
        return step["run"]

    @staticmethod
    def _decide(requested: str, event: str) -> bool:
        import os
        import subprocess
        import tempfile

        script = TestTheEngineDecisionIsMadeForEveryTrigger._script()
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / "github_output"
            output.touch()
            subprocess.run(
                ["bash", "-c", script],
                env={
                    **os.environ,
                    "REQUESTED": requested,
                    "EVENT": event,
                    "GITHUB_OUTPUT": str(output),
                },
                check=True, capture_output=True, text=True,
            )
            written = output.read_text()
        assert "build=" in written, "the step wrote no decision at all"
        return "build=true" in written

    def test_a_push_to_main_builds_the_engine(self):
        """The one that was broken. `inputs` does not exist here, so
        REQUESTED is the empty string — which the old expression read as
        false on every commit."""
        assert self._decide(requested="", event="push")

    def test_a_pull_request_does_not(self):
        """15-25 minutes on a hosted macOS runner is not worth paying on
        a PR that touched a view, and paying it anyway is what makes
        somebody turn this off permanently."""
        assert not self._decide(requested="", event="pull_request")

    def test_a_release_gets_what_it_asked_for(self):
        assert self._decide(requested="true", event="push")

    def test_an_explicit_no_is_honoured(self):
        """Including on a push, where the trigger default would say yes.
        A caller that said false meant false."""
        assert not self._decide(requested="false", event="push")

    def test_a_manual_run_can_ask_for_one(self):
        assert self._decide(requested="true", event="workflow_dispatch")

    def test_a_manual_run_without_the_box_ticked_does_not(self):
        assert not self._decide(requested="false", event="workflow_dispatch")

    def test_an_unknown_event_does_not_silently_build(self):
        """A trigger nobody thought about should cost nothing, not 25
        minutes of runner time."""
        assert not self._decide(requested="", event="schedule")

    def test_the_decision_is_announced(self):
        """A skipped step looks the same whether it was skipped on
        purpose or by a null. The log has to say which."""
        assert "Engine requested" in self._script()


class TestRequireEngine:
    def test_it_refuses_when_the_framework_is_absent(self, tmp_path, monkeypatch):
        module = _prepare()
        monkeypatch.setattr(module, "FRAMEWORK", tmp_path / "llama.xcframework")
        with pytest.raises(SystemExit) as refused:
            module.main(["--require-engine", "--check"])
        assert "require-engine" in str(refused.value)

    def test_it_says_how_to_fix_it(self, tmp_path, monkeypatch):
        module = _prepare()
        monkeypatch.setattr(module, "FRAMEWORK", tmp_path / "llama.xcframework")
        with pytest.raises(SystemExit) as refused:
            module.main(["--require-engine", "--check"])
        assert "build_llama_xcframework.sh" in str(refused.value)

    def test_it_is_quiet_when_the_framework_is_there(self, tmp_path, monkeypatch):
        module = _prepare()
        framework = tmp_path / "llama.xcframework"
        framework.mkdir()
        (framework / "Info.plist").write_text("<plist/>")
        monkeypatch.setattr(module, "FRAMEWORK", framework)
        assert module.main(["--require-engine", "--check"]) == 0

    def test_without_the_flag_a_missing_framework_is_still_fine(self, tmp_path, monkeypatch):
        """A checkout that has never built the engine must still be able
        to build the app — that is the whole reason the fallback exists."""
        module = _prepare()
        monkeypatch.setattr(module, "FRAMEWORK", tmp_path / "nothing")
        assert module.main(["--check"]) == 0
