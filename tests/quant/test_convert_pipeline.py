"""``hypernix convert -P -Q``: safetensors to a quantised GGUF in one step."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("gguf")

from safetensors.torch import save_file  # noqa: E402

from hypernix.interfaces import cli  # noqa: E402
from hypernix.quant.convertq import (  # noqa: E402
    ConvertQuantizeError,
    convert_and_quantize,
    default_output,
    source_kind,
)
from hypernix.quant.gguf import GGMLType, GGUFFile  # noqa: E402
from hypernix.quant.hyprslug import HyprslugError  # noqa: E402

D, FF, LAYERS, VOCAB = 256, 512, 4, 320


@pytest.fixture
def snapshot(tmp_path) -> Path:
    """A Hugging Face llama folder, small but with rows of 256."""
    folder = tmp_path / "snap"
    folder.mkdir()
    torch.manual_seed(0)
    state = {"model.embed_tokens.weight": torch.randn(VOCAB, D) * 0.02,
             "model.norm.weight": torch.ones(D),
             "lm_head.weight": torch.randn(VOCAB, D) * 0.02}
    for i in range(LAYERS):
        prefix = f"model.layers.{i}."
        for name, shape in (("self_attn.q_proj", (D, D)), ("self_attn.k_proj", (D, D)),
                            ("self_attn.v_proj", (D, D)), ("self_attn.o_proj", (D, D)),
                            ("mlp.gate_proj", (FF, D)), ("mlp.up_proj", (FF, D)),
                            ("mlp.down_proj", (D, FF))):
            state[f"{prefix}{name}.weight"] = torch.randn(*shape) * 0.02
        state[f"{prefix}input_layernorm.weight"] = torch.ones(D)
        state[f"{prefix}post_attention_layernorm.weight"] = torch.ones(D)
    save_file(state, str(folder / "model.safetensors"))
    (folder / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": D, "intermediate_size": FF,
        "num_hidden_layers": LAYERS, "num_attention_heads": 4,
        "num_key_value_heads": 4, "vocab_size": VOCAB,
    }))
    return folder


def _types(path: Path) -> dict[str, int]:
    return {t.name: int(t.ggml_type) for t in GGUFFile.read(path).tensors}


class TestWhatItAccepts:
    def test_a_folder_and_a_file_in_it_are_safetensors(self, snapshot):
        assert source_kind(snapshot) == "safetensors"
        assert source_kind(snapshot / "model.safetensors") == "safetensors"

    def test_a_brewer_folder_is_brewer(self, tmp_path):
        folder = tmp_path / "brew"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(
            {"d_model": 256, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "vocab_size": 64}))
        (folder / "model.safetensors").write_bytes(b"")
        assert source_kind(folder) == "brewer"
        assert source_kind(folder / "model.safetensors") == "brewer"

    def test_other_things_are_refused(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        assert source_kind(tmp_path / "notes.txt") == ""
        assert source_kind(tmp_path) == ""
        with pytest.raises(ConvertQuantizeError, match="not safetensors"):
            convert_and_quantize(tmp_path, "Q8_0")

    def test_the_default_output_is_beside_the_model(self, snapshot):
        expected = snapshot.parent / "snap.Q4_K_M.gguf"
        assert default_output(snapshot, "Q4_K_M") == expected
        assert default_output(snapshot / "model.safetensors", "Q4_K_M") == expected


class TestTheRun:
    def test_safetensors_become_the_target(self, snapshot):
        result = convert_and_quantize(snapshot / "model.safetensors", "q6h4")
        assert result.output == snapshot.parent / "snap.hnx_Q6_H_4.gguf"
        assert result.kind == "safetensors"
        types = _types(result.output)
        assert types["blk.2.ffn_up.weight"] == int(GGMLType.HNX_INT3)
        assert types["blk.0.attn_v.weight"] == int(GGMLType.Q6_K)
        # The F16 staging copy is gone.
        assert sorted(p.name for p in snapshot.parent.iterdir()) == ["snap", "snap.hnx_Q6_H_4.gguf"]

    def test_the_intermediate_can_be_kept(self, snapshot, tmp_path):
        out = tmp_path / "out" / "m.q8.gguf"
        result = convert_and_quantize(snapshot, "Q8_0", out, keep_intermediate=True)
        assert result.intermediate is not None and result.intermediate.is_file()
        assert set(_types(out).values()) >= {int(GGMLType.Q8_0)}

    def test_a_gguf_is_quantised_as_it_is(self, snapshot, tmp_path):
        first = convert_and_quantize(snapshot, "FP16", tmp_path / "m.f16.gguf")
        second = convert_and_quantize(first.output, "FP8", tmp_path / "m.fp8.gguf")
        assert second.kind == "gguf"
        assert int(GGMLType.HNX_FP8) in set(_types(second.output).values())

    def test_a_bad_target_is_refused_before_converting(self, snapshot):
        with pytest.raises(HyprslugError, match="Unknown target"):
            convert_and_quantize(snapshot, "Q9_Z")
        assert sorted(p.name for p in snapshot.parent.iterdir()) == ["snap"]


class TestTheCommand:
    def test_minus_p_minus_q(self, snapshot, capsys):
        out = snapshot.parent / "x.gguf"
        assert cli.main(["convert", str(snapshot), "-P", "-Q", "Q4_K_M", "-o", str(out)]) == 0
        assert out.is_file()
        assert str(out) in capsys.readouterr().out

    def test_minus_q_implies_minus_p(self, snapshot):
        assert cli.main(["convert", str(snapshot), "-Q", "q8_0"]) == 0
        assert (snapshot.parent / "snap.Q8_0.gguf").is_file()

    def test_minus_p_without_a_target_says_what_to_add(self, snapshot, capsys):
        assert cli.main(["convert", str(snapshot), "-P"]) == 2
        assert "-Q" in capsys.readouterr().err

    def test_a_bad_target_is_an_error_not_a_traceback(self, snapshot, capsys):
        assert cli.main(["convert", str(snapshot), "-P", "-Q", "nope"]) == 1
        assert "Unknown target" in capsys.readouterr().err

    def test_the_old_form_still_needs_an_output(self, snapshot):
        with pytest.raises(SystemExit):
            cli.main(["convert", "--model-dir", str(snapshot)])

    def test_the_old_form_still_works(self, snapshot, tmp_path):
        out = tmp_path / "plain.f16.gguf"
        assert cli.main(["convert", "--model-dir", str(snapshot), "--output", str(out)]) == 0
        assert out.is_file()
