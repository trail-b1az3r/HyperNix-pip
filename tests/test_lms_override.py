"""`hypernix-t1 override lms move-dir` — repointing LM Studio's models.

What these tests are for
------------------------
This edits a file another application owns. The failures that matter
are the ones that damage it: overwriting a settings file that did not
parse (throwing away settings we cannot see), dropping the keys we did
not touch, writing while LM Studio is running (so it undoes us on
exit), and moving files onto files that already exist. Each is asserted
as a refusal or a preservation, against a fake LM Studio home.
"""
from __future__ import annotations

import json

import pytest

from hypernix.t1api import lmsoverride as lo


@pytest.fixture
def lms_home(tmp_path, monkeypatch):
    home = tmp_path / "lmstudio"
    home.mkdir()
    monkeypatch.setenv("LMSTUDIO_HOME", str(home))
    monkeypatch.setattr(lo, "lmstudio_running", lambda: [])
    monkeypatch.setattr(lo.Path, "home", classmethod(lambda cls: tmp_path))
    return home


def write_settings(home, data):
    path = home / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestFinding:
    def test_lmstudio_home_wins(self, lms_home):
        assert lo.lmstudio_homes() == [lms_home]

    def test_the_settings_file_is_found(self, lms_home):
        path = write_settings(lms_home, {})
        assert lo.find_settings() == path

    def test_the_older_internal_location(self, lms_home):
        (lms_home / ".internal").mkdir()
        path = lms_home / ".internal" / "app-settings.json"
        path.write_text("{}")
        assert lo.find_settings() == path

    def test_no_explicit_folder_means_home_models(self, lms_home):
        write_settings(lms_home, {"theme": "dark"})
        _, folder = lo.current_models_dir()
        assert folder == lms_home / "models"

    def test_the_lms_pointer_is_respected(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_HOME", raising=False)
        monkeypatch.setattr(lo.Path, "home", classmethod(lambda cls: tmp_path))
        elsewhere = tmp_path / "somewhere-else"
        (tmp_path / ".lmstudio-home-pointer").write_text(str(elsewhere))
        assert lo.lmstudio_homes()[0] == elsewhere


class TestSetting:
    def test_it_points_lmstudio_at_the_folder(self, lms_home, tmp_path):
        settings = write_settings(lms_home, {})
        target = tmp_path / "shared-models"
        change = lo.set_models_dir(target)
        assert json.loads(settings.read_text())[lo.SETTING_KEY] == str(target)
        assert change.after == str(target)

    def test_the_default_is_the_hypernix_models_folder(self, lms_home, tmp_path):
        """One folder both LM Studio and HyperNix read, instead of two
        copies of every GGUF."""
        write_settings(lms_home, {})
        change = lo.set_models_dir()
        assert change.after == str(tmp_path / ".hypernix" / "models")
        assert (tmp_path / ".hypernix" / "models").is_dir()

    def test_every_other_key_is_kept(self, lms_home, tmp_path):
        settings = write_settings(lms_home, {"theme": "dark", "nested": {"a": [1, 2]},
                                             "gpuOffload": 33})
        lo.set_models_dir(tmp_path / "m")
        data = json.loads(settings.read_text())
        assert data["theme"] == "dark"
        assert data["nested"] == {"a": [1, 2]}
        assert data["gpuOffload"] == 33

    def test_a_backup_is_made_first(self, lms_home, tmp_path):
        settings = write_settings(lms_home, {lo.SETTING_KEY: "/old"})
        change = lo.set_models_dir(tmp_path / "m")
        assert change.backup is not None
        assert json.loads(change.backup.read_text())[lo.SETTING_KEY] == "/old"
        assert change.before == "/old"
        assert settings.exists()

    def test_revert_restores_it(self, lms_home, tmp_path):
        settings = write_settings(lms_home, {lo.SETTING_KEY: "/old", "theme": "dark"})
        lo.set_models_dir(tmp_path / "m")
        lo.revert()
        assert json.loads(settings.read_text()) == {lo.SETTING_KEY: "/old", "theme": "dark"}

    def test_revert_with_nothing_to_revert_says_so(self, lms_home):
        write_settings(lms_home, {})
        with pytest.raises(lo.LMSOverrideError):
            lo.revert()

    def test_an_unreadable_settings_file_is_not_overwritten(self, lms_home, tmp_path):
        """Either LM Studio is mid-write or it is damaged, and replacing
        it with ours throws away settings we cannot see."""
        settings = lms_home / "settings.json"
        settings.write_text("{not json")
        with pytest.raises(lo.LMSOverrideError):
            lo.set_models_dir(tmp_path / "m")
        assert settings.read_text() == "{not json"

    def test_it_refuses_while_lmstudio_runs(self, lms_home, tmp_path, monkeypatch):
        """LM Studio can write its settings on exit, which would quietly
        undo this."""
        write_settings(lms_home, {})
        monkeypatch.setattr(lo, "lmstudio_running", lambda: ["LM Studio"])
        with pytest.raises(lo.LMSOverrideError) as caught:
            lo.set_models_dir(tmp_path / "m")
        assert "--force" in str(caught.value)

    def test_force_goes_ahead(self, lms_home, tmp_path, monkeypatch):
        write_settings(lms_home, {})
        monkeypatch.setattr(lo, "lmstudio_running", lambda: ["LM Studio"])
        assert lo.set_models_dir(tmp_path / "m", force=True).after

    def test_a_file_where_the_folder_should_be_is_refused(self, lms_home, tmp_path):
        write_settings(lms_home, {})
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        with pytest.raises(lo.LMSOverrideError):
            lo.set_models_dir(blocker)

    def test_no_settings_file_creates_one_in_the_home(self, lms_home, tmp_path):
        change = lo.set_models_dir(tmp_path / "m")
        assert change.created
        assert change.backup is None

    def test_not_installed_at_all(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "nope"))
        monkeypatch.setattr(lo, "lmstudio_running", lambda: [])
        with pytest.raises(lo.LMSOverrideError) as caught:
            lo.set_models_dir(tmp_path / "m")
        assert "LMSTUDIO_HOME" in str(caught.value)

    def test_the_write_is_atomic(self, lms_home, tmp_path, monkeypatch):
        """A crash mid-write must leave the old file, not half a new one."""
        settings = write_settings(lms_home, {"theme": "dark"})

        def explode(*a, **k):
            raise OSError("disk went away")
        monkeypatch.setattr(lo.os, "replace", explode)
        with pytest.raises(OSError):
            lo.set_models_dir(tmp_path / "m")
        assert json.loads(settings.read_text()) == {"theme": "dark"}
        assert not list(lms_home.glob("*.tmp"))


class TestMovingFiles:
    def make_models(self, root):
        (root / "pub" / "repo").mkdir(parents=True)
        (root / "pub" / "repo" / "a.gguf").write_bytes(b"x" * 10)
        (root / "b.gguf").write_bytes(b"y" * 20)
        return root

    def test_the_plan_moves_nothing(self, tmp_path):
        src = self.make_models(tmp_path / "old")
        plan = lo.plan_move(src, tmp_path / "new")
        assert len(plan.move) == 2
        assert (src / "b.gguf").exists()

    def test_moving_keeps_the_layout(self, tmp_path):
        """LM Studio expects publisher/repo/file; flattening it would
        leave the models unfindable from LM Studio."""
        src = self.make_models(tmp_path / "old")
        lo.move_models(lo.plan_move(src, tmp_path / "new"))
        assert (tmp_path / "new" / "pub" / "repo" / "a.gguf").exists()
        assert not (src / "b.gguf").exists()

    def test_it_never_overwrites(self, tmp_path):
        src = self.make_models(tmp_path / "old")
        (tmp_path / "new").mkdir()
        (tmp_path / "new" / "b.gguf").write_bytes(b"KEEP")
        plan = lo.plan_move(src, tmp_path / "new")
        lo.move_models(plan)
        assert (tmp_path / "new" / "b.gguf").read_bytes() == b"KEEP"
        assert (src / "b.gguf").exists()
        assert len(plan.skip) == 1

    def test_a_target_inside_the_source_is_refused(self, tmp_path):
        src = self.make_models(tmp_path / "old")
        with pytest.raises(lo.LMSOverrideError):
            lo.plan_move(src, src / "nested")

    def test_the_same_folder_is_nothing_to_do(self, tmp_path):
        src = self.make_models(tmp_path / "old")
        assert lo.plan_move(src, src).move == []


class TestTheCli:
    def test_move_dir_without_move_files_leaves_models(self, lms_home, tmp_path, capsys):
        write_settings(lms_home, {})
        (lms_home / "models").mkdir()
        (lms_home / "models" / "x.gguf").write_bytes(b"x")
        assert lo.main(["move-dir", str(tmp_path / "new")]) == 0
        out = capsys.readouterr().out
        assert "--move-files" in out
        assert (lms_home / "models" / "x.gguf").exists()

    def test_move_files_needs_yes(self, lms_home, tmp_path, capsys):
        write_settings(lms_home, {})
        (lms_home / "models").mkdir()
        (lms_home / "models" / "x.gguf").write_bytes(b"x")
        lo.main(["move-dir", str(tmp_path / "new"), "--move-files"])
        assert "Nothing has moved" in capsys.readouterr().out
        assert (lms_home / "models" / "x.gguf").exists()

    def test_move_files_with_yes_moves(self, lms_home, tmp_path):
        write_settings(lms_home, {})
        (lms_home / "models").mkdir()
        (lms_home / "models" / "x.gguf").write_bytes(b"x")
        lo.main(["move-dir", str(tmp_path / "new"), "--move-files", "--yes"])
        assert (tmp_path / "new" / "x.gguf").exists()

    def test_show(self, lms_home, capsys):
        from pathlib import Path
        write_settings(lms_home, {lo.SETTING_KEY: "/somewhere"})
        lo.main(["show"])
        # Printed as a path, so with the platform's separator.
        assert str(Path("/somewhere")) in capsys.readouterr().out

    def test_the_bash_wrapper_routes_override(self):
        from pathlib import Path
        text = (Path(__file__).resolve().parents[1] / "bin" / "hypernix-t1").read_text()
        assert "override)          cmd_override" in text
        assert "hypernix.t1api.lmsoverride" in text
