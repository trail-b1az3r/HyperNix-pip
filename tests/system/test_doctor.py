"""``hnx doctor`` checks what the package needs now, not what it needed.

It had fallen behind: Python 3.14 read as unsupported, ``rich`` (a hard
dependency) was neither checked nor installed by ``--fix``, a missing
``llama-quantize`` failed the whole check though ``quantize`` fetches
one itself, and the things people now ask it about -- the GPU and the
preset it suits, the patched llama.cpp, the models folder -- were not
there. The Model Training Guide says doctor "reports what the machine
has and what that supports"; these tests hold it to that.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from hypernix.system import doctor

tomllib = pytest.importorskip("tomllib")

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture(scope="module")
def project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


class TestItAgreesWithPyproject:
    def test_the_python_range_is_requires_python(self, project):
        spec = project["requires-python"]
        low = re.search(r">=\s*3\.(\d+)", spec).group(1)
        below = re.search(r"<\s*3\.(\d+)", spec).group(1)
        assert doctor.PYTHON_RANGE == ((3, int(low)), (3, int(below) - 1))

    def test_fix_installs_every_dependency_but_torch(self, project):
        wanted = sorted(d for d in project["dependencies"] if not d.startswith("torch"))
        assert sorted(doctor._RUNTIME_DEPS) == wanted

    def test_every_dependency_is_checked(self, project):
        def import_name(spec: str) -> str:
            name = re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0]
            return name.replace("-", "_")
        wanted = {import_name(d) for d in project["dependencies"]} - {"torch"}
        assert set(doctor._REQUIRED_IMPORTS) == wanted

    def test_fix_installs_the_train_extra(self, project):
        train = set(project["optional-dependencies"]["train"])
        assert train <= set(doctor._OPTIONAL_DEPS)

    def test_every_extra_it_reports_exists(self, project):
        assert set(doctor.EXTRAS) <= set(project["optional-dependencies"])


class TestTheGpuLine:
    @pytest.mark.parametrize("gib,preset,optimizer", [
        (None, "cpu-nano … cpu-small", "pressure_cooker_v6"),
        (8, "33m / micro", "pressure_cooker_v5s"),
        (12, "small", "pressure_cooker_v5"),
        (24, "medium", "pressure_cooker_v6"),
        (80, "large", "pressure_cooker_v6"),
    ])
    def test_the_preset_matches_the_training_guide(self, gib, preset, optimizer):
        assert doctor._preset_for(gib) == (preset, optimizer)

    def test_it_never_fails_the_check(self):
        ok, message = doctor._check_gpu()
        assert ok and "preset" in message or "MPS" in message or "failed" in message


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
class TestTheModelsFolder:
    def test_it_is_not_created_by_looking(self, tmp_path, monkeypatch):
        folder = tmp_path / "models"
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(folder))
        ok, message = doctor._check_models_dir()
        assert ok and "not created yet" in message
        assert not folder.exists()

    def test_models_through_links_are_counted(self, tmp_path, monkeypatch):
        disk = tmp_path / "disk" / "qwen"
        disk.mkdir(parents=True)
        (disk / "qwen.gguf").write_bytes(b"GGUF")
        folder = tmp_path / "models"
        folder.mkdir()
        (folder / "qwen").symlink_to(disk, target_is_directory=True)
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(folder))
        ok, message = doctor._check_models_dir()
        assert ok and "1 GGUF" in message

    def test_a_broken_link_is_reported_with_its_target(self, tmp_path, monkeypatch):
        folder = tmp_path / "models"
        folder.mkdir()
        (folder / "gone").symlink_to(tmp_path / "unmounted" / "model")
        monkeypatch.setenv("HYPERNIX_MODELS_DIR", str(folder))
        ok, message = doctor._check_models_dir()
        assert not ok
        assert "gone ->" in message and "unmounted" in message


class TestWhatDecidesTheExitCode:
    def test_no_llama_quantize_is_not_a_failure(self, monkeypatch, capsys):
        """`hypernix quantize` downloads one on first use."""
        monkeypatch.setattr(doctor, "_check_llama_quantize",
                            lambda: (False, "not found -- fetched on first use"))
        monkeypatch.setattr(doctor, "_check_import", lambda mod, minver=None: (True, mod))
        monkeypatch.setattr(doctor, "_check_python", lambda: (True, "python"))
        monkeypatch.setattr(doctor, "_check_torch_version", lambda: (True, "torch"))
        assert doctor.run(fix=False) == 0
        assert "[--] llama-quantize" in capsys.readouterr().out

    def test_a_missing_required_package_is(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor, "_check_import",
                            lambda mod, minver=None: (mod != "rich", mod))
        monkeypatch.setattr(doctor, "_check_python", lambda: (True, "python"))
        monkeypatch.setattr(doctor, "_check_torch_version", lambda: (True, "torch"))
        assert doctor.run(fix=False) == 1
        assert "[!!] rich" in capsys.readouterr().out

    def test_the_report_names_every_section(self, capsys):
        doctor.run(fix=False)
        out = capsys.readouterr().out
        for label in ("Python", "torch", "rich", "GPU", "llama-quantize",
                      "patched llama.cpp", "models folder", "extras"):
            assert label in out
