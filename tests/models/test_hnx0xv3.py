"""HyperNix0xV3: the architecture, and brewer building it only behind -E."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from hypernix.models import brewer_adapter  # noqa: E402
from hypernix.models.hnx0xv3 import (  # noqa: E402
    PRESETS,
    HyperNix0xV3Config,
    HyperNix0xV3Model,
    is_v3_config,
    preset,
)
from hypernix.training import brewer  # noqa: E402


def _tiny(**over):
    base = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                max_seq_len=96, window=16, chunk=8, n_slots=4, slot_dim=16, name="t")
    base.update(over)
    return HyperNix0xV3Config(**base)


def test_causal_and_reaches_past_the_window():
    torch.manual_seed(0)
    model = HyperNix0xV3Model(_tiny()).eval()
    ids = torch.randint(0, 64, (1, 80))
    base = model(ids)
    changed = ids.clone()
    changed[0, 70] = (changed[0, 70] + 1) % 64
    out = model(changed)
    # Nothing before the edit moves.
    assert torch.allclose(base[0, :70], out[0, :70], atol=1e-5)
    # A token far outside the 16-token window still changes the last logits
    # (through the chunk summaries and the Keel state).
    far = ids.clone()
    far[0, 2] = (far[0, 2] + 1) % 64
    assert not torch.allclose(base[0, -1], model(far)[0, -1], atol=1e-6)


def test_gradients_are_finite_and_moe_has_aux_loss():
    torch.manual_seed(0)
    model = HyperNix0xV3Model(_tiny(n_experts=4, experts_active=2))
    logits = model(torch.randint(0, 64, (2, 40)))
    loss = logits.float().logsumexp(-1).mean() + model.aux_loss
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.aux_loss.requires_grad


def test_presets_share_v2_names_and_xl_is_moe():
    assert {"cpu-nano", "cpu-tiny", "cpu-small", "33m", "small", "medium", "large", "xl"} <= set(PRESETS)
    xl = preset("xl")
    assert xl.n_experts > 1 and xl.active_params() < xl.approx_params()
    nano = preset("cpu-nano")
    assert nano.approx_params() == HyperNix0xV3Model(nano).num_params()


def test_config_roundtrip_carries_the_arch(tmp_path):
    cfg = _tiny()
    cfg.save(tmp_path / "config.json")
    data = json.loads((tmp_path / "config.json").read_text())
    assert is_v3_config(data)
    assert brewer.load_config(tmp_path / "config.json") == cfg


def test_brew_new_defaults_to_v2_and_E_builds_v3(tmp_path, capsys):
    brewer.cli_main(["new", "--preset", "cpu-nano", "--name", "v2", "--save-dir", str(tmp_path)])
    brewer.cli_main(["new", "--preset", "cpu-nano", "-E", "--name", "v3", "--save-dir", str(tmp_path)])
    v2 = json.loads((tmp_path / "v2" / "config.json").read_text())
    v3 = json.loads((tmp_path / "v3" / "config.json").read_text())
    assert not is_v3_config(v2)
    assert is_v3_config(v3)
    with pytest.raises(SystemExit):
        brewer.cli_main(["new", "--preset", "xl", "--save-dir", str(tmp_path)])
    assert "-E" in capsys.readouterr().err


def test_save_reload_adapter_and_no_gguf(tmp_path):
    torch.manual_seed(0)
    b = brewer.Brewer(_tiny(), save_dir=tmp_path / "m")
    b.build()
    folder = b.save()
    again = brewer.Brewer.from_dir(folder)
    ids = torch.randint(0, 64, (1, 20))
    with torch.no_grad():
        assert torch.allclose(b.model(ids), again.model(ids), atol=1e-6)
        adapted, cfg = brewer_adapter.load(folder)
        assert adapted.arch == "hyperNix0x-v3" and is_v3_config(cfg.to_dict())
        out = adapted(ids, labels=ids)
        assert torch.allclose(out["logits"], b.model(ids), atol=1e-6)
        assert torch.isfinite(out["loss"])
    with pytest.raises(ValueError, match="no GGUF"):
        b.export(tmp_path / "x.gguf", fmt="gguf")


def test_brewer_trains_v3(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 200)
    torch.manual_seed(0)
    b = brewer.Brewer(_tiny(), save_dir=tmp_path / "m")
    losses: list[float] = []
    b.train(corpus, steps=200, lr=3e-3, batch_size=4, device="cpu",
            log_callback=lambda step, loss: losses.append(loss))
    assert losses and losses[-1] < losses[0]
