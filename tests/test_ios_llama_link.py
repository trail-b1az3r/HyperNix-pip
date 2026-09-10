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

    def test_the_framework_dependency_is_optional(self):
        """`xcodegen generate` must succeed without it on disk."""
        import yaml
        project = yaml.safe_load(PROJECT.read_text(encoding="utf-8"))
        deps = project["targets"]["HyperLink"].get("dependencies", [])
        frameworks = [d for d in deps if "framework" in d]
        assert frameworks, "the app links no framework"
        assert any("llama.xcframework" in d["framework"] for d in frameworks)
        assert all(d.get("optional") for d in frameworks)

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
