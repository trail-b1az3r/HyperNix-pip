"""Symlinked models in HyperLink: found, loadable, and linkable from the app.

``Path.rglob`` does not descend into a symlinked folder, so a model kept
on another disk and linked into ``~/.hypernix/models`` never reached the
model picker, the runner or ``hypernix-t1 index``. These cover the walk
that replaced it, the catalogue built on it, and ``/hyperlink/models/link``,
which makes the link from the app.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hypernix.hyperlink.catalogue import local_models
from hypernix.hyperlink.modellinks import ModelLinkError, link_model, unlink_model
from hypernix.system.linkwalk import broken_links, walk_files

pytestmark = pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * 60)
    return path


@pytest.fixture
def disks(tmp_path):
    """A models folder, and a model on 'another disk' outside it."""
    models = tmp_path / "models"
    models.mkdir()
    elsewhere = tmp_path / "data"
    _gguf(elsewhere / "qwen" / "qwen3-8b-q4_k_m.gguf")
    _gguf(elsewhere / "loose.gguf")
    return models, elsewhere


class TestTheWalk:
    def test_it_goes_through_a_symlinked_folder(self, disks):
        models, elsewhere = disks
        (models / "qwen").symlink_to(elsewhere / "qwen", target_is_directory=True)
        assert list(models.rglob("*.gguf")) == []          # the bug
        found = list(walk_files(models, ".gguf"))
        assert found == [models / "qwen" / "qwen3-8b-q4_k_m.gguf"]

    def test_a_symlinked_file_is_found_under_its_link_name(self, disks):
        models, elsewhere = disks
        (models / "mine.gguf").symlink_to(elsewhere / "loose.gguf")
        assert list(walk_files(models, ".gguf")) == [models / "mine.gguf"]

    def test_a_link_back_up_the_tree_does_not_loop(self, disks):
        models, _ = disks
        _gguf(models / "a" / "one.gguf")
        (models / "a" / "again").symlink_to(models, target_is_directory=True)
        found = list(walk_files(models, ".gguf"))
        assert found == [models / "a" / "one.gguf"]

    def test_a_dangling_link_is_reported_not_walked(self, disks):
        models, _ = disks
        (models / "gone.gguf").symlink_to("/nowhere/gone.gguf")
        assert list(walk_files(models, ".gguf")) == []
        assert broken_links(models) == [(models / "gone.gguf", "/nowhere/gone.gguf")]


class TestTheCatalogue:
    def test_a_linked_model_is_listed_with_where_it_points(self, disks):
        models, elsewhere = disks
        (models / "qwen").symlink_to(elsewhere / "qwen", target_is_directory=True)
        found, report = local_models(models)
        assert report.count == 1
        model = found[0]
        assert model.path == str(models / "qwen" / "qwen3-8b-q4_k_m.gguf")
        assert model.link_name == "qwen"
        assert model.linked_to == str(elsewhere / "qwen")
        assert model.to_dict()["link_name"] == "qwen"

    def test_a_plain_file_has_no_link(self, disks):
        models, _ = disks
        _gguf(models / "plain.gguf")
        found, _ = local_models(models)
        assert (found[0].link_name, found[0].linked_to) == ("", "")

    def test_a_broken_link_is_listed_as_not_runnable_with_the_reason(self, disks):
        models, _ = disks
        (models / "gone.gguf").symlink_to("/mnt/usb/gone.gguf")
        found, _ = local_models(models)
        assert len(found) == 1
        assert found[0].runnable is False
        assert "/mnt/usb/gone.gguf" in found[0].detail
        assert found[0].link_name == "gone.gguf"


class TestLinking:
    def test_a_gguf_is_linked_in(self, disks):
        models, elsewhere = disks
        made = link_model(elsewhere / "loose.gguf", models)
        assert made["kind"] == "gguf"
        assert (models / "loose.gguf").is_symlink()
        assert os.readlink(models / "loose.gguf") == str((elsewhere / "loose.gguf").resolve())

    def test_a_folder_is_linked_in_under_a_chosen_name(self, disks):
        models, elsewhere = disks
        made = link_model(elsewhere / "qwen", models, "qwen3-8b")
        assert made["kind"] == "folder"
        assert (models / "qwen3-8b").is_symlink()
        assert list(walk_files(models, ".gguf"))

    def test_a_brewer_folder_is_a_model_too(self, disks, tmp_path):
        models, _ = disks
        brew = tmp_path / "brew"
        brew.mkdir()
        (brew / "config.json").write_text(json.dumps(
            {"d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "vocab_size": 64}))
        (brew / "model.safetensors").write_bytes(b"x")
        assert link_model(brew, models)["kind"] == "hypernix"

    @pytest.mark.parametrize("bad,match", [
        ("", "Give the path"),
        ("relative/model.gguf", "relative"),
        ("/definitely/not/here.gguf", "Nothing at"),
    ])
    def test_it_refuses_what_it_cannot_link(self, disks, bad, match):
        models, _ = disks
        with pytest.raises(ModelLinkError, match=match):
            link_model(bad, models)

    def test_it_refuses_a_file_that_is_not_a_model(self, disks, tmp_path):
        models, _ = disks
        (tmp_path / "passwords.txt").write_text("x")
        with pytest.raises(ModelLinkError, match="not a .gguf"):
            link_model(tmp_path / "passwords.txt", models)

    def test_it_refuses_a_folder_of_safetensors_and_says_how_to_convert(self, disks, tmp_path):
        models, _ = disks
        hf = tmp_path / "hf"
        hf.mkdir()
        (hf / "model.safetensors").write_bytes(b"x")
        with pytest.raises(ModelLinkError, match="hnx convert"):
            link_model(hf, models)

    @pytest.mark.parametrize("name", ["../escape", ".hidden", "a/b", "x" * 200])
    def test_the_name_is_one_plain_segment(self, disks, name):
        models, elsewhere = disks
        with pytest.raises(ModelLinkError, match="cannot be a model name"):
            link_model(elsewhere / "loose.gguf", models, name)

    def test_it_will_not_replace_what_is_there(self, disks):
        models, elsewhere = disks
        _gguf(models / "loose.gguf")
        with pytest.raises(ModelLinkError) as caught:
            link_model(elsewhere / "loose.gguf", models)
        assert caught.value.conflict

    def test_something_already_in_the_folder_needs_no_link(self, disks):
        models, _ = disks
        inside = _gguf(models / "sub" / "m.gguf")
        with pytest.raises(ModelLinkError, match="already in"):
            link_model(inside, models)


class TestUnlinking:
    def test_it_removes_the_link_and_not_the_model(self, disks):
        models, elsewhere = disks
        link_model(elsewhere / "loose.gguf", models)
        removed = unlink_model("loose.gguf", models)
        assert removed["kind"] == "removed"
        assert not (models / "loose.gguf").exists()
        assert (elsewhere / "loose.gguf").is_file()

    def test_it_refuses_to_delete_a_real_file(self, disks):
        models, _ = disks
        _gguf(models / "real.gguf")
        with pytest.raises(ModelLinkError, match="not a link") as caught:
            unlink_model("real.gguf", models)
        assert caught.value.conflict
        assert (models / "real.gguf").is_file()

    def test_a_broken_link_can_be_removed(self, disks):
        models, _ = disks
        (models / "gone.gguf").symlink_to("/nowhere/gone.gguf")
        unlink_model("gone.gguf", models)
        assert not (models / "gone.gguf").is_symlink()

    def test_the_name_cannot_walk_out_of_the_folder(self, disks):
        models, _ = disks
        with pytest.raises(ModelLinkError):
            unlink_model("../models", models)


class TestOverHttp:
    @pytest.fixture
    def server(self, disks, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
        from hypernix.t1api.app import create_app
        from hypernix.t1api.config import T1APIConfig

        models, elsewhere = disks
        km = Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)
        gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)
        config = T1APIConfig(
            token_secret="test-secret-value-that-is-long-enough",
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "modules"),
            hyperlink_files_dir=str(tmp_path / "files"),
            hf_download_dir=str(models),
            lmstudio_enabled=False,
            default_plan="free",
        )
        client = TestClient(create_app(config=config, keymaster=km, gatekeeper=gk),
                            client=("127.0.0.1", 5000))
        admin = km.create(key_type=KeyType.ADMIN,
                          scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key
        user = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key
        return client, {"Authorization": f"Bearer {admin}"}, {"Authorization": f"Bearer {user}"}, \
            models, elsewhere

    def test_an_admin_links_a_model_and_it_is_in_the_list(self, server):
        client, admin, _user, models, elsewhere = server
        made = client.post("/hyperlink/models/link", headers=admin,
                           json={"path": str(elsewhere / "qwen"), "name": "qwen"})
        assert made.status_code == 200, made.text
        assert made.json()["linked_to"] == str((elsewhere / "qwen").resolve())
        listed = client.get("/hyperlink/models", headers=admin).json()["models"]
        local = [m for m in listed if m["source"] == "local"]
        assert [m["link_name"] for m in local] == ["qwen"]
        downloaded = client.get("/hyperlink/models/downloaded", headers=admin).json()["models"]
        assert downloaded[0]["has_gguf"] is True
        assert downloaded[0]["linked_to"] == str((elsewhere / "qwen").resolve())

    def test_and_removes_it(self, server):
        client, admin, _user, models, elsewhere = server
        client.post("/hyperlink/models/link", headers=admin,
                    json={"path": str(elsewhere / "loose.gguf")})
        removed = client.delete("/hyperlink/models/link/loose.gguf", headers=admin)
        assert removed.status_code == 200, removed.text
        assert (elsewhere / "loose.gguf").is_file()
        assert not (models / "loose.gguf").exists()

    def test_a_non_admin_cannot(self, server):
        client, _admin, user, models, elsewhere = server
        made = client.post("/hyperlink/models/link", headers=user,
                           json={"path": str(elsewhere / "loose.gguf")})
        assert made.status_code == 403, made.text
        assert not (models / "loose.gguf").exists()

    def test_the_errors_have_the_right_status(self, server):
        client, admin, _user, models, _elsewhere = server
        missing = client.post("/hyperlink/models/link", headers=admin,
                              json={"path": "/definitely/not/here.gguf"})
        assert missing.status_code == 404
        _gguf(models / "real.gguf")
        conflict = client.delete("/hyperlink/models/link/real.gguf", headers=admin)
        assert conflict.status_code == 409
        assert (models / "real.gguf").is_file()
