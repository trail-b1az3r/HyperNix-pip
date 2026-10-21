"""hnx0xv2image: an image-to-text model fused onto LFM2.5, Qwen3.5 and HyperNix text models.

Tiny text models are built from transformers' own configs (and a Brewer
config) with a character tokenizer, so nothing is downloaded. Two
coloured squares and their names are enough to show the image reaches
the text model: after training, red is called red and blue blue.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")
pytest.importorskip("tokenizers")

from hypernix.models import hnx0xv2image as hi  # noqa: E402
from hypernix.models import hnx0xv2image_cli as cli  # noqa: E402

CHARS = [chr(c) for c in range(32, 127)]
VOCAB = {"<eos>": 0, **{c: i + 1 for i, c in enumerate(CHARS)}}


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")


TEXT_MODELS = {
    "lfm2": dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, layer_types=["conv", "full_attention"]),
    "qwen3_5_text": dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, layer_types=["linear_attention", "full_attention"],
                         linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                         linear_value_head_dim=16),
}


def _text_model(folder, model_type):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    config = AutoConfig.for_model(model_type, vocab_size=len(VOCAB), **TEXT_MODELS[model_type])
    AutoModelForCausalLM.from_config(config).save_pretrained(folder)
    _tokenizer().save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from PIL import Image

    root = tmp_path_factory.mktemp("images")
    for colour in ("red", "blue"):
        Image.new("RGB", (80, 50), colour).save(root / f"{colour}.png")
    rows = [{"image": "red.png", "text": "red"}, {"image": "blue.png", "text": "blue"}]
    (root / "captions.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return root


class TestTheImageModel:
    def test_tokens_per_image(self):
        model = hi.ImageModel(hi.preset("tiny", text_hidden_size=32))
        out = model(torch.zeros(2, 3, 64, 64))
        assert out.shape == (2, model.config.tokens_per_image, 32) == (2, 16, 32)

    def test_a_bad_shape_is_refused(self):
        with pytest.raises(hi.ImageError, match="multiple of patch_size"):
            hi.ImageConfig(image_size=100, patch_size=16)

    def test_letterboxed_not_stretched(self, data):
        pixels = hi.preprocess(data / "red.png", hi.preset("tiny"))
        assert pixels.shape == (1, 3, 64, 64)
        # The top rows are padding (mean grey, 0 after normalising); the middle is red.
        assert float(pixels[0, :, 0, 32].abs().max()) < 0.01
        assert float(pixels[0, 0, 32, 32]) > 0.9

    def test_saves_and_loads(self, tmp_path):
        model = hi.ImageModel(hi.preset("tiny"))
        model.save(tmp_path / "eye")
        again = hi.load_image_model(tmp_path / "eye")
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values(), strict=True))


class TestFamilies:
    @pytest.mark.parametrize("config, family", [
        ({"model_type": "lfm2"}, "lfm2"), ({"model_type": "qwen3_5"}, "qwen3_5"),
        ({"model_type": "qwen3_5_moe_text"}, "qwen3_5_moe_text"),
        ({"d_model": 64, "n_layers": 2, "n_heads": 4}, "hypernix0x-v2"),
        ({"model_type": "hypernix"}, "hypernix"),
    ])
    def test_fusable(self, config, family):
        assert hi.family_of(config) == family

    def test_a_vision_model_is_refused(self):
        with pytest.raises(hi.ImageError, match="already an image model"):
            hi.family_of({"model_type": "lfm2_vl"})

    def test_another_family_is_refused_unless_allowed(self):
        with pytest.raises(hi.ImageError, match="not fusable"):
            hi.family_of({"model_type": "gpt2"})


@pytest.mark.parametrize("model_type", sorted(TEXT_MODELS))
def test_fuse_train_caption_reload(tmp_path, data, model_type):
    text = _text_model(tmp_path / model_type, model_type)
    torch.manual_seed(0)
    fused = hi.fuse(hi.ImageModel(hi.preset("tiny")), text)
    assert fused.image.output_size == 64          # the projector was re-headed to the text width
    before = {k: v.clone() for k, v in fused.text_model.state_dict().items()}
    report = hi.train(fused, data / "captions.jsonl", stage="projector", steps=10, lr=3e-3)
    assert all(torch.equal(before[k], v) for k, v in fused.text_model.state_dict().items()), \
        "training the projector changed the text model"
    assert report.trainable < sum(p.numel() for p in fused.parameters())
    hi.train(fused, data / "captions.jsonl", stage="all", steps=60, lr=3e-3)
    assert fused.generate(data / "red.png", "", max_new_tokens=6) == "red"
    assert fused.generate(data / "blue.png", "", max_new_tokens=6) == "blue"
    fused.save(tmp_path / "fused")
    assert hi.load(tmp_path / "fused").generate(data / "blue.png", "", max_new_tokens=6) == "blue"


def test_a_brewer_text_model(tmp_path, data):
    from hypernix.training.brewer import Brewer, BrewerConfig

    brewer = Brewer(BrewerConfig(vocab_size=len(CHARS) + 1, n_layers=2, n_heads=4, n_kv_heads=2, d_model=64,
                                 max_seq_len=128, use_sliding_window=False), name="tiny-brew",
                    save_dir=tmp_path / "brew")
    brewer.build()
    brewer.char_vocab = ["\x00", *CHARS]
    brewer.save(tmp_path / "brew")
    fused = hi.fuse(hi.ImageModel(hi.preset("tiny")), tmp_path / "brew")
    assert fused.family == "hypernix0x-v2"
    report = hi.train(fused, data / "captions.jsonl", stage="all", steps=20, lr=3e-3)
    assert report.last < report.first


def test_the_prompt_must_match_the_images(tmp_path):
    text = _text_model(tmp_path / "lfm2", "lfm2")
    fused = hi.fuse(hi.ImageModel(hi.preset("tiny")), text)
    with pytest.raises(hi.ImageError, match="1 <image> but 2"):
        fused._segments("look: <image>", 2)


def test_the_cli(tmp_path, data, capsys):
    text = _text_model(tmp_path / "lfm2", "lfm2")
    assert cli.cli_main(["new", "-o", str(tmp_path / "eye"), "--preset", "tiny"]) == 0
    assert cli.cli_main(["fuse", str(tmp_path / "eye"), str(text), "-o", str(tmp_path / "f"), "--device", "cpu"]) == 0
    assert cli.cli_main(["train", str(tmp_path / "f"), "--data", str(data / "captions.jsonl"),
                         "--steps", "2", "--device", "cpu"]) == 0
    assert cli.cli_main(["caption", str(tmp_path / "f"), str(data / "red.png"), "--max-new-tokens", "2",
                         "--device", "cpu"]) == 0
    assert cli.cli_main(["split", str(tmp_path / "f"), "-o", str(tmp_path / "eye2")]) == 0
    assert cli.cli_main(["--json", "info", str(tmp_path / "eye2")]) == 0
    assert '"tokens_per_image": 16' in capsys.readouterr().out
    assert cli.cli_main(["caption", str(tmp_path / "eye2"), str(data / "red.png"), "--device", "cpu"]) == 1
