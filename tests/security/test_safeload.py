"""Checkpoints load without running the file (hypernix.security.safeload)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hypernix.security import safeload  # noqa: E402

RAN: list[str] = []


def _ran(message: str) -> None:
    RAN.append(message)


class Payload:
    """What a malicious checkpoint carries: unpickling calls this."""

    def __reduce__(self):
        return (_ran, ("the file ran code",))


def test_a_plain_checkpoint_loads(tmp_path, monkeypatch):
    monkeypatch.delenv(safeload.TRUST_ENV, raising=False)
    path = tmp_path / "plain.pt"
    torch.save({"config": {"d_model": 8, "name": "x"}, "model_state_dict": {"w": torch.ones(2)},
                "step": 3, "loss": None}, path)
    loaded = safeload.load_checkpoint(path)
    assert loaded["config"]["d_model"] == 8 and loaded["step"] == 3
    assert torch.equal(loaded["model_state_dict"]["w"], torch.ones(2))


def test_code_in_the_file_is_refused_and_does_not_run(tmp_path, monkeypatch):
    monkeypatch.delenv(safeload.TRUST_ENV, raising=False)
    RAN.clear()
    path = tmp_path / "evil.pt"
    torch.save({"x": Payload()}, path)
    with pytest.raises(safeload.UntrustedCheckpoint, match=safeload.TRUST_ENV):
        safeload.load_checkpoint(path)
    assert RAN == []


def test_the_person_can_choose_to_trust_a_file(tmp_path, monkeypatch):
    RAN.clear()
    path = tmp_path / "theirs.pt"
    torch.save({"x": Payload()}, path)
    monkeypatch.setenv(safeload.TRUST_ENV, "1")
    safeload.load_checkpoint(path)
    assert RAN == ["the file ran code"]
    RAN.clear()
    monkeypatch.delenv(safeload.TRUST_ENV)
    safeload.load_checkpoint(path, trust_pickle=True)
    assert RAN == ["the file ran code"]
