"""``hnx runtime`` — using the HyperNix llama.cpp from other applications.

Two routes with very different risk, and the tests are weighted to
match. `serve` starts a process and changes nothing. `install` copies
libraries over the ones LM Studio ships, which is surgery on somebody
else's application, so most of what is below is about the guards on it:
refused without --yes, backed up before anything is written, restorable
from a manifest after this process is gone, and refused outright on a
directory that does not look like a runtime.

The LM Studio layout here is a fake. It has to be -- LM Studio is
proprietary and not installable in CI -- and that is worth saying
plainly rather than implying the real thing was tested: what is verified
is that the *guards* hold and that install/restore round-trips exactly.
Whether LM Studio then loads the result is version-specific and is the
part `hnx runtime status` reports rather than promises.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypernix.quant import runtime_bridge as bridge
from hypernix.quant.runtime_bridge_cli import main as runtime_main


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never touch the real home directory or a real LM Studio."""
    monkeypatch.setenv("HNX_RUNTIME_BRIDGE_HOME", str(tmp_path / "bridge"))
    monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "no-lmstudio"))
    monkeypatch.delenv("HNX_LLAMA_BUILD", raising=False)


def _suffix() -> str:
    return bridge._library_suffix()


def fake_build(root: Path, *, patched: bool = True) -> Path:
    """A directory shaped like a built llama.cpp."""
    bin_dir = root / "build" / "bin"
    bin_dir.mkdir(parents=True)
    marker = b"hnx_ggml_to_float_iq0_5" if patched else b"nothing to see"
    for stem in bridge.CORE_LIBRARIES:
        (bin_dir / f"{stem}{_suffix()}").write_bytes(b"ELF" + marker)
    server = bin_dir / "llama-server"
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    return root / "build"


def fake_lmstudio(root: Path) -> Path:
    """A directory shaped like an LM Studio runtime."""
    runtime = root / "extensions" / "backends" / "vendor-llama-cpp-linux"
    runtime.mkdir(parents=True)
    for stem in bridge.CORE_LIBRARIES:
        (runtime / f"{stem}{_suffix()}").write_bytes(
            f"ORIGINAL {stem}".encode())
    return runtime


# ---------------------------------------------------------------------------
# Finding a build
# ---------------------------------------------------------------------------


class TestFindingTheBuild:
    def test_it_finds_one_and_reads_the_libraries(self, tmp_path):
        build_dir = fake_build(tmp_path / "llama.cpp")

        build = bridge.find_build(build_dir)

        assert set(build.libraries) == set(bridge.CORE_LIBRARIES)
        assert build.server is not None

    def test_the_checkout_above_the_build_is_accepted_too(self, tmp_path):
        """`--build ~/llama.cpp` is what people type."""
        fake_build(tmp_path / "llama.cpp")

        build = bridge.find_build(tmp_path / "llama.cpp")

        assert build.libraries

    def test_a_patched_build_is_recognised(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "p", patched=True))

        assert build.patched is True
        assert build.note == ""

    def test_an_unpatched_build_is_not_mistaken_for_one(self, tmp_path):
        """Read out of the binary, not inferred from the path. A
        directory called llama.cpp beside this repository is not evidence
        that anybody ran the patcher on it, and installing an unpatched
        runtime over LM Studio's would swap one that cannot read sub-bit
        models for another that cannot -- while looking like a fix."""
        build = bridge.find_build(fake_build(tmp_path / "u", patched=False))

        assert build.patched is False
        assert "does not carry the HyperNix decoder" in build.note

    def test_nothing_there_says_how_to_get_one(self, tmp_path):
        with pytest.raises(bridge.BridgeError) as caught:
            bridge.find_build(tmp_path / "empty")

        assert "build.sh" in str(caught.value)
        assert "HNX_LLAMA_BUILD" in str(caught.value)


# ---------------------------------------------------------------------------
# Serving — changes nothing
# ---------------------------------------------------------------------------


class TestServing:
    def test_the_command_line_is_what_it_should_be(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")

        argv = bridge.serve_argv(build, model, host="0.0.0.0", port=1234,
                                 gpu_layers=20, context=4096, alias="local")

        assert argv[0] == str(build.server)
        assert argv[argv.index("-m") + 1] == str(model)
        assert argv[argv.index("--port") + 1] == "1234"
        assert argv[argv.index("--host") + 1] == "0.0.0.0"
        assert argv[argv.index("-ngl") + 1] == "20"
        assert argv[argv.index("-c") + 1] == "4096"
        assert argv[argv.index("--alias") + 1] == "local"

    def test_the_optional_flags_stay_off_when_not_asked_for(self, tmp_path):
        """`-ngl 0` and `-c 0` are not the same as leaving them out --
        llama.cpp reads the model's own context when the flag is absent."""
        build = bridge.find_build(fake_build(tmp_path / "b"))
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")

        argv = bridge.serve_argv(build, model)

        assert "-ngl" not in argv
        assert "-c" not in argv
        assert "--alias" not in argv

    def test_a_missing_model_is_refused_before_anything_starts(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))

        with pytest.raises(bridge.BridgeError, match="No such model"):
            bridge.serve_argv(build, tmp_path / "absent.gguf")

    def test_it_names_clients_that_can_use_it(self):
        """The point of serving: it works with applications this has
        never heard of, because the base URL is the whole contract."""
        names = {name for name, _ in bridge.OPENAI_CLIENTS}

        assert "LM Studio" in names
        assert len(names) >= 5


# ---------------------------------------------------------------------------
# Installing — the guards
# ---------------------------------------------------------------------------


class TestInstallRefuses:
    def test_without_confirmation_it_only_plans(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")
        before = (runtime / f"libllama{_suffix()}").read_bytes()

        result = bridge.install(build, runtime, confirmed=False)

        assert result["applied"] is False
        assert "not confirmed" in result["reason"]
        assert result["planned"]
        assert (runtime / f"libllama{_suffix()}").read_bytes() == before

    def test_dry_run_changes_nothing_even_when_confirmed(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")
        before = (runtime / f"libllama{_suffix()}").read_bytes()

        result = bridge.install(build, runtime, confirmed=True, dry_run=True)

        assert result["applied"] is False
        assert (runtime / f"libllama{_suffix()}").read_bytes() == before

    def test_a_directory_that_is_not_a_runtime_is_refused(self, tmp_path):
        """Refusing beats scattering four shared objects into somebody's
        Documents folder."""
        build = bridge.find_build(fake_build(tmp_path / "b"))
        ordinary = tmp_path / "Documents"
        ordinary.mkdir()
        (ordinary / "notes.txt").write_text("hello")

        with pytest.raises(bridge.BridgeError, match="does not look like"):
            bridge.install(build, ordinary, confirmed=True)

    def test_an_unpatched_build_is_refused(self, tmp_path):
        """It would replace a runtime that cannot read sub-bit models
        with another that cannot, and look like a fix."""
        build = bridge.find_build(fake_build(tmp_path / "u", patched=False))
        runtime = fake_lmstudio(tmp_path / "lms")

        with pytest.raises(bridge.BridgeError, match="does not carry"):
            bridge.install(build, runtime, confirmed=True)

    def test_a_missing_target_is_refused(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))

        with pytest.raises(bridge.BridgeError, match="not a directory"):
            bridge.install(build, tmp_path / "nowhere", confirmed=True)


class TestInstallAndRestore:
    def test_it_installs_and_backs_up_first(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")

        result = bridge.install(build, runtime, confirmed=True)

        assert result["applied"] is True
        assert len(result["written"]) == len(bridge.CORE_LIBRARIES)
        assert result["backed_up"] == len(bridge.CORE_LIBRARIES)
        # The originals are somewhere, not overwritten into oblivion.
        saved = list(Path(result["backup"]).iterdir())
        assert len(saved) == len(bridge.CORE_LIBRARIES)
        assert b"ORIGINAL" in (Path(result["backup"]) /
                               f"libllama{_suffix()}").read_bytes()
        # And the target now has ours.
        assert b"hnx_ggml_to_float_iq0_5" in (
            runtime / f"libllama{_suffix()}").read_bytes()

    def test_restore_puts_every_file_back_exactly(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")
        originals = {
            p.name: p.read_bytes() for p in runtime.iterdir() if p.is_file()
        }

        bridge.install(build, runtime, confirmed=True)
        result = bridge.restore(confirmed=True)

        assert result["restored"] == len(originals)
        for name, content in originals.items():
            assert (runtime / name).read_bytes() == content

    def test_restore_without_confirmation_only_reports(self, tmp_path):
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")
        bridge.install(build, runtime, confirmed=True)

        result = bridge.restore(confirmed=False)

        assert result["restored"] == 0
        assert result["would_restore"]
        assert b"hnx_" in (runtime / f"libllama{_suffix()}").read_bytes()

    def test_restore_works_from_the_manifest_alone(self, tmp_path):
        """The case it exists for: the process that installed is long
        gone, and something has to know what was there before."""
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")
        bridge.install(build, runtime, confirmed=True)

        # Nothing carried over in memory -- read it back off disk.
        record = json.loads(bridge.manifest_path().read_text(encoding="utf-8"))
        assert record["installs"][0]["target"] == str(runtime)
        assert record["installs"][0]["backed_up"]

        assert bridge.restore(confirmed=True)["restored"] > 0

    def test_a_library_we_added_is_removed_on_restore(self, tmp_path):
        """Putting the directory back means removing what was not there
        before, not leaving ours behind beside the originals."""
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")
        extra = runtime / f"libggml-cpu{_suffix()}"
        extra.unlink()                         # LM Studio did not ship this one

        bridge.install(build, runtime, confirmed=True)
        assert extra.is_file()

        bridge.restore(confirmed=True)

        assert not extra.is_file()

    def test_installing_twice_still_restores_the_original(self, tmp_path):
        """Newest first, or the second restore would put back the first
        install's copy rather than LM Studio's."""
        build = bridge.find_build(fake_build(tmp_path / "b"))
        runtime = fake_lmstudio(tmp_path / "lms")
        original = (runtime / f"libllama{_suffix()}").read_bytes()

        bridge.install(build, runtime, confirmed=True)
        bridge.install(build, runtime, confirmed=True)
        bridge.restore(confirmed=True)

        assert (runtime / f"libllama{_suffix()}").read_bytes() == original

    def test_restoring_with_nothing_installed_is_not_an_error(self):
        result = bridge.restore(confirmed=True)

        assert result["restored"] == 0
        assert "nothing" in result["note"]


class TestDetection:
    def test_it_finds_a_runtime_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "lms"))
        runtime = fake_lmstudio(tmp_path / "lms")

        targets = bridge.detect_targets()

        assert [t["runtime_dir"] for t in targets] == [str(runtime)]
        assert targets[0]["application"] == "LM Studio"

    def test_a_directory_with_no_libraries_is_not_a_runtime(self, tmp_path,
                                                            monkeypatch):
        monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "lms"))
        empty = tmp_path / "lms" / "extensions" / "backends" / "nothing"
        empty.mkdir(parents=True)
        (empty / "manifest.json").write_text("{}")

        assert bridge.detect_targets() == []


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


class TestTheCLI:
    def test_status_runs_with_nothing_installed(self, capsys):
        assert runtime_main(["status"]) == 0
        assert capsys.readouterr().out

    def test_a_bare_invocation_is_status(self, capsys):
        runtime_main([])
        bare = capsys.readouterr().out
        runtime_main(["status"])
        assert bare.split()[:2] == capsys.readouterr().out.split()[:2]

    def test_status_json_is_parseable(self, capsys):
        runtime_main(["status", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert "targets" in payload and "installs" in payload

    def test_path_prints_a_bare_directory(self, tmp_path, capsys):
        """So it can go in a shell substitution."""
        build_dir = fake_build(tmp_path / "b")

        assert runtime_main(["--build", str(build_dir), "path"]) == 0

        printed = capsys.readouterr().out.strip()
        assert Path(printed).is_dir()
        assert printed.endswith("bin")

    def test_path_without_a_build_fails_with_advice(self, capsys):
        assert runtime_main(["--build", "/nowhere", "path"]) == 1
        assert "build.sh" in capsys.readouterr().err

    def test_serve_print_only_starts_nothing(self, tmp_path, capsys):
        build_dir = fake_build(tmp_path / "b")
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")

        code = runtime_main(["--build", str(build_dir), "serve", str(model),
                             "--print-only"])

        captured = capsys.readouterr()
        assert code == 0
        assert "llama-server" in captured.out
        # The base URL and where to paste it go to stderr, so stdout
        # stays a command line something can run.
        assert "base URL" in captured.err
        assert "LM Studio" in captured.err

    def test_install_without_yes_exits_2(self, tmp_path, capsys, monkeypatch):
        """Distinct from 0, so a script can tell "nothing happened
        because you did not confirm" from "done"."""
        monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "lms"))
        fake_lmstudio(tmp_path / "lms")
        build_dir = fake_build(tmp_path / "b")

        code = runtime_main(["--build", str(build_dir), "install"])

        assert code == 2
        assert "--yes" in capsys.readouterr().err

    def test_install_and_restore_through_the_cli(self, tmp_path, monkeypatch,
                                                 capsys):
        monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "lms"))
        runtime = fake_lmstudio(tmp_path / "lms")
        original = (runtime / f"libllama{_suffix()}").read_bytes()
        build_dir = fake_build(tmp_path / "b")

        assert runtime_main(["--build", str(build_dir), "install", "--yes"]) == 0
        assert (runtime / f"libllama{_suffix()}").read_bytes() != original

        assert runtime_main(["restore", "--yes"]) == 0
        assert (runtime / f"libllama{_suffix()}").read_bytes() == original

    def test_install_with_no_target_detected_says_so(self, tmp_path, capsys):
        build_dir = fake_build(tmp_path / "b")

        code = runtime_main(["--build", str(build_dir), "install", "--yes"])

        assert code == 1
        assert "LMSTUDIO_HOME" in capsys.readouterr().err


class TestWiredIn:
    def test_the_subcommand_reaches_the_module(self):
        from hypernix.interfaces import cli

        assert "runtime" in cli._SUBCOMMANDS

    def test_it_reaches_the_cli_through_the_real_entry_point(self, capsys):
        """`hnx` is version_launcher, not cli. A subcommand registered in
        cli._SUBCOMMANDS and unreachable from the console script is the
        bug that made `hnx gather` print the usage table."""
        from hypernix.interfaces.version_launcher import main as launcher

        assert launcher(["runtime", "status"]) == 0
        assert capsys.readouterr().out

    @pytest.mark.parametrize("subcommand", ["gather", "fusebox", "runtime"])
    def test_every_recent_subcommand_is_reachable(self, subcommand):
        """Not just registered -- reachable. Checked for each of the ones
        added since the launcher started re-execing."""
        from hypernix.interfaces import cli

        assert subcommand in cli._SUBCOMMANDS
        source = Path(cli.__file__).read_text(encoding="utf-8")
        assert f'cmd == "{subcommand}"' in source or \
               f'cmd in ("{subcommand}"' in source
