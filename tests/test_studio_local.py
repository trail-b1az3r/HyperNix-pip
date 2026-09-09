"""HyperNix Studio running a model on this machine, with no server.

Studio began as a client: chat went over HTTP to a HyperNix server, and
a laptop with a GGUF sitting on it still needed something running
somewhere. This covers the other path.

Three things get checked here that the C++ suites cannot check alone:

**The build without llama.cpp is the one most people get**, and it has
to be honest rather than broken. `local_engine_test` is compiled and run
in that configuration on every pass.

**The GGUF parser's input is hostile.** A model file is something
somebody downloaded, and every length in its header is a 64-bit number
the parser would otherwise be told to allocate. `model_catalogue_test`
builds headers that lie and checks each one is refused.

**The tool boundary does not move.** A local model gets exactly the
reach a remote one had -- file operations inside the workspace, each
approved -- so the source scan for process spawning covers the new files
too, and the C++ that runs the model must not have acquired a way to run
anything else.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parent.parent / "desktop"
SRC = DESKTOP / "src"
TESTS = DESKTOP / "tests"
QML = DESKTOP / "qml"

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Studio ships on Linux; see tests/test_studio_core.py for why the "
           "compile-and-run halves are POSIX-only.",
)


def _compiler() -> str | None:
    for name in ("c++", "g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _build(name: str, sources: list[str], tmp_path: Path) -> Path:
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++ compiler on this machine")
    out = tmp_path / name
    result = subprocess.run(
        [compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
         str(TESTS / f"{name}.cpp"),
         *[str(SRC / s) for s in sources],
         "-pthread", "-o", str(out)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"{name} does not compile cleanly:\n{result.stderr}")
    return out


@_POSIX_ONLY
class TestTheCatalogue:
    """Reading a GGUF header without loading the model."""

    def test_it_compiles_and_every_check_passes(self, tmp_path):
        binary = _build("model_catalogue_test", ["ModelCatalogue.cpp"], tmp_path)
        result = subprocess.run([str(binary)], capture_output=True, text=True,
                                check=False)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "0 failures" in result.stdout

    def test_it_needs_neither_qt_nor_llama_cpp(self):
        """The point of it being separate.

        A person browsing models on a machine with no llama.cpp should
        still be told what is on the disk, and told that an IQ0.5_XXXL
        file is an IQ0.5_XXXL file -- the build that cannot run it is
        exactly the build where a bare "type 202" is least useful.
        """
        source = (SRC / "ModelCatalogue.cpp").read_text(encoding="utf-8")
        header = (SRC / "ModelCatalogue.h").read_text(encoding="utf-8")

        for banned in ("<QtCore", "<QString", "llama.h", "ggml.h", "QObject"):
            assert banned not in source, f"{banned} in ModelCatalogue.cpp"
            assert banned not in header, f"{banned} in ModelCatalogue.h"

    def test_it_names_the_hypernix_types_itself(self):
        """Not by asking ggml, which may not be linked in."""
        source = (SRC / "ModelCatalogue.cpp").read_text(encoding="utf-8")

        for name in ("IQ0.9_L", "IQ0.75_M", "IQ0.5_XXXL", "IQ0.25_UXL", "INT1"):
            assert f'"{name}"' in source, name

    def test_every_header_number_is_bounded_before_it_is_used(self):
        """The property the hostile-header tests exercise, asserted from
        the other side: the constants that bound it are actually there.

        A parser that reserved on the strength of a length in the file
        could be made to allocate 16 exabytes by a malformed download.
        """
        source = (SRC / "ModelCatalogue.cpp").read_text(encoding="utf-8")

        for limit in ("kMaxHeaderBytes", "kMaxTensors", "kMaxKeys",
                      "kMaxStringBytes", "kMaxArrayItems"):
            assert limit in source, limit
        # And the string read is checked against what is *left* of the
        # buffer, not only against a constant -- "longer than this file"
        # and "2^63" are the same mistake.
        assert "size_ - pos_" in source


@_POSIX_ONLY
class TestTheEngineWithoutLlamaCpp:
    """The build nearly every machine gets."""

    def test_it_compiles_and_the_stub_is_honest(self, tmp_path):
        binary = _build("local_engine_test",
                        ["LocalEngine.cpp", "ModelCatalogue.cpp"], tmp_path)
        result = subprocess.run([str(binary)], capture_output=True, text=True,
                                check=False)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "0 failures" in result.stdout
        assert "local inference is OFF" in result.stdout

    def test_the_reason_says_what_to_do(self):
        """"Not available" on its own sends someone to the issue tracker."""
        source = (SRC / "LocalEngine.cpp").read_text(encoding="utf-8")

        assert "STUDIO_LOCAL_LLAMA" in source
        assert "LLAMA_ROOT" in source

    def test_local_inference_is_off_by_default(self):
        """The server path does not need llama.cpp. Making the harder
        dependency mandatory would stop Studio building for everyone who
        only wants to connect to a HyperNix box."""
        cmake = (DESKTOP / "CMakeLists.txt").read_text(encoding="utf-8")

        assert 'option(STUDIO_LOCAL_LLAMA' in cmake
        block = cmake.split("option(STUDIO_LOCAL_LLAMA", 1)[1][:200]
        assert "OFF)" in block


class TestTheToolBoundaryDidNotMove:
    """A local model gets no more reach than a remote one had."""

    def test_the_new_files_cannot_run_a_command(self):
        """test_studio_core.py sweeps src/*.cpp already; this says out
        loud that the files added for local models are in that sweep,
        so removing them from it would fail here rather than silently."""
        forbidden = ("system(", "popen(", "execv", "execl", "execp", "fork(",
                     "QProcess", "posix_spawn")
        for name in ("LocalEngine.cpp", "LocalEngine.h", "ModelCatalogue.cpp",
                     "ModelCatalogue.h", "LocalSession.cpp", "LocalSession.h"):
            text = (SRC / name).read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"{name} contains {token!r}"

    def test_running_a_model_locally_still_goes_through_the_policy(self):
        """The approval flow is untouched: onToolCall and runApproved are
        the only route to a mutating tool, whichever end the model is
        at. A second path for local models would be a second place to
        get the boundary wrong."""
        bridge = (SRC / "StudioBridge.cpp").read_text(encoding="utf-8")

        assert bridge.count("runner_->policy().Decide(") == 1
        assert "sendToLocal" in bridge


class TestTheBridgeRoutes:
    def test_send_dispatches_on_the_source(self):
        bridge = (SRC / "StudioBridge.cpp").read_text(encoding="utf-8")

        assert "void StudioBridge::send(" in bridge
        assert "sendToLocal(text)" in bridge
        assert "sendToServer(text)" in bridge

    def test_the_engine_runs_off_the_gui_thread(self):
        """Loading a 7B model takes seconds and generating takes as long
        as it takes. On the GUI thread the window stops repainting and
        the desktop offers to kill it."""
        session = (SRC / "LocalSession.cpp").read_text(encoding="utf-8")

        assert "moveToThread" in session
        assert "Qt::QueuedConnection" in session

    def test_cancel_is_not_queued(self):
        """The one call that must not be. The worker thread is *inside*
        Generate, so a queued cancel would sit in its event queue until
        the generation it is meant to stop had finished."""
        session = (SRC / "LocalSession.cpp").read_text(encoding="utf-8")
        body = session.split("void LocalSession::cancel()", 1)[1][:400]

        assert "QueuedConnection" not in body
        assert "worker_->cancel()" in body

    def test_the_streaming_index_is_bounds_checked(self):
        """The conversation can be cleared while a generation is in
        flight. Writing past the end of the list would be the last thing
        this process did."""
        bridge = (SRC / "StudioBridge.cpp").read_text(encoding="utf-8")

        assert "streamingIndex_ >= messages_.size()" in bridge
        assert "streamingIndex_ = -1;" in bridge


class TestTheUiDoesNotAskForAServerItDoesNotNeed:
    def test_the_composer_gates_on_ready_not_connected(self):
        """`connected` is false forever in local mode. Gating the input
        on it left the composer dead with a model loaded and running."""
        chat = (QML / "ChatView.qml").read_text(encoding="utf-8")

        assert "studio.ready && !studio.busy" in chat
        assert "enabled: studio.connected && !studio.busy" not in chat

    def test_the_connect_form_is_hidden_in_local_mode(self):
        sidebar = (QML / "Sidebar.qml").read_text(encoding="utf-8")

        assert 'studio.source !== "local"' in sidebar

    def test_a_generating_model_can_be_stopped(self):
        """A long answer on a slow machine is a minute of watching, and
        closing the window should not be the way out."""
        chat = (QML / "ChatView.qml").read_text(encoding="utf-8")

        assert "studio.stopGenerating()" in chat

    def test_the_models_view_reads_both_shapes(self):
        """A server model calls it displayName and a local one calls it
        name. An undefined binding here does not throw -- it leaves the
        property at its *previous* value, which is how a list showed the
        last model's name against this model's row."""
        models = (QML / "ModelsView.qml").read_text(encoding="utf-8")

        assert "modelData.displayName !== undefined" in models
        assert "studio.localModels" in models

    def test_an_unreadable_model_is_shown_not_hidden(self):
        """A file that is there and broken is something you want to see."""
        models = (QML / "ModelsView.qml").read_text(encoding="utf-8")

        assert "modelData.ok === false" in models
        assert "unreadable" in models
