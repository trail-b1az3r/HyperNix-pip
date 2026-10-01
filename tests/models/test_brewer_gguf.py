"""Brewer (hyperNix0x-v2) models as GGUF files llama.cpp runs.

Brewer's ``fmt="gguf"`` used to write an ``HNXG`` file no llama.cpp could
open. These check the real export: the llama.cpp names, the RoPE
permutation (against llama.cpp's own rotation, not against itself), the
tokenizer, the sliding-window context cap and the FFN padding.

The export was also run against llama.cpp built from source, stock and
with native/ggml-hnx patched in: logits matched PyTorch to 0.0025 at every
position in f32 and f16, and tokenization matched exactly, multi-byte characters
included. That needs a llama.cpp build, so it is recorded in the changelog
rather than repeated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from hypernix.models import brewer_gguf  # noqa: E402
from hypernix.models.brewer_gguf import (  # noqa: E402
    CHAR_VOCAB_FILE,
    GGUFExportError,
    brewer_to_llama_name,
    export_gguf,
    write_char_vocab,
)
from hypernix.training.brewer import (  # noqa: E402
    Brewer,
    BrewerConfig,
    BrewerModel,
    _apply_rope,
    _build_rope_cache,
    _SimpleTextDataset,
    train_model,
)


def _config(**overrides) -> BrewerConfig:
    base = dict(vocab_size=320, n_layers=2, n_heads=4, n_kv_heads=2, d_model=32, d_ff=40,
                max_seq_len=128, rope_theta=10000.0, use_sliding_window=False,
                sliding_window_size=32, name="tiny")
    base.update(overrides)
    return BrewerConfig(**base)


def _bpe_tokenizer(folder: Path, vocab_size: int = 300) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog " * 20],
        trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|pad|>"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tok.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps(
        {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}))


def _folder(tmp_path: Path, cfg: BrewerConfig, *, tokenizer: bool = True) -> tuple[Path, BrewerModel]:
    folder = tmp_path / cfg.name
    folder.mkdir()
    torch.manual_seed(0)
    model = BrewerModel(cfg)
    cfg.save(folder / "config.json")
    state = {k: v.contiguous() for k, v in model.state_dict().items()
             if not (cfg.tie_embeddings and k == "lm_head.weight")}
    save_file(state, str(folder / "model.safetensors"))
    if tokenizer:
        _bpe_tokenizer(folder)
    return folder, model


def _read(path: Path):
    reader = gguf.GGUFReader(str(path))
    fields = {}
    for name, f in reader.fields.items():
        if not f.types:
            continue
        if f.types[0] == gguf.GGUFValueType.STRING and len(f.types) == 1:
            fields[name] = bytes(f.parts[f.data[0]]).decode("utf-8")
        elif f.types[0] == gguf.GGUFValueType.ARRAY:
            if f.types[-1] == gguf.GGUFValueType.STRING:
                fields[name] = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
            else:
                fields[name] = [int(f.parts[i][0]) for i in f.data]
        else:
            fields[name] = f.parts[f.data[0]][0].item()
    tensors = {t.name: t for t in reader.tensors}
    return fields, tensors


# ---------------------------------------------------------------------------
# The mathematics: llama.cpp's rotation on the permuted weights is Brewer's
# ---------------------------------------------------------------------------

def _rope_pairs(x: np.ndarray, theta: float) -> np.ndarray:
    """llama.cpp's LLAMA_ROPE_TYPE_NORM: rotate (2i, 2i+1) pairs. x: (T, H, D)."""
    T, _, D = x.shape
    pos = np.arange(T)[:, None]
    inv = theta ** (-np.arange(0, D, 2) / D)
    ang = pos * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


@pytest.mark.parametrize("n_heads,n_kv", [(4, 4), (4, 2), (8, 1)])
def test_permuted_weights_give_brewers_attention_scores(n_heads, n_kv):
    torch.manual_seed(1)
    d_model, T, theta = 64, 7, 10000.0
    hd = d_model // n_heads
    x = torch.randn(T, d_model)
    wq = torch.randn(n_heads * hd, d_model)
    wk = torch.randn(n_kv * hd, d_model)

    # Brewer: half rotation, as BrewerAttention does it.
    q = (x @ wq.T).view(T, n_heads, hd).transpose(0, 1)[None]
    k = (x @ wk.T).view(T, n_kv, hd).transpose(0, 1)[None]
    cos, sin = _build_rope_cache(T, hd, theta, x.device, x.dtype)
    q, k = _apply_rope(q, k, cos, sin)
    k = k.repeat_interleave(n_heads // n_kv, dim=1)
    brewer_scores = (q @ k.transpose(-1, -2))[0].numpy()

    # llama.cpp: pair rotation on the exported (permuted) weights.
    pq = brewer_gguf._permute(wq, n_heads).numpy()
    pk = brewer_gguf._permute(wk, n_kv).numpy()
    lq = _rope_pairs((x.numpy() @ pq.T).reshape(T, n_heads, hd), theta)
    lk = _rope_pairs((x.numpy() @ pk.T).reshape(T, n_kv, hd), theta)
    lk = np.repeat(lk, n_heads // n_kv, axis=1)
    llama_scores = np.einsum("thd,shd->hts", lq, lk)

    np.testing.assert_allclose(llama_scores, brewer_scores, rtol=1e-4, atol=1e-3)


def test_unpermuted_weights_would_be_wrong():
    """The check above can fail: without the permutation the scores differ."""
    torch.manual_seed(2)
    T, hd, theta = 5, 8, 10000.0
    x = torch.randn(T, 16)
    wq, wk = torch.randn(2 * hd, 16), torch.randn(2 * hd, 16)
    q = (x @ wq.T).view(T, 2, hd).transpose(0, 1)[None]
    k = (x @ wk.T).view(T, 2, hd).transpose(0, 1)[None]
    cos, sin = _build_rope_cache(T, hd, theta, x.device, x.dtype)
    q, k = _apply_rope(q, k, cos, sin)
    brewer_scores = (q @ k.transpose(-1, -2))[0].numpy()
    lq = _rope_pairs((x.numpy() @ wq.numpy().T).reshape(T, 2, hd), theta)
    lk = _rope_pairs((x.numpy() @ wk.numpy().T).reshape(T, 2, hd), theta)
    assert not np.allclose(np.einsum("thd,shd->hts", lq, lk), brewer_scores, atol=1e-2)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def test_names_map_to_llama_cpp():
    assert brewer_to_llama_name("embed.embed.weight") == "token_embd.weight"
    assert brewer_to_llama_name("blocks.3.attn.q_proj.weight") == "blk.3.attn_q.weight"
    assert brewer_to_llama_name("blocks.0.ffn.down_proj.weight") == "blk.0.ffn_down.weight"
    assert brewer_to_llama_name("norm_out.weight") == "output_norm.weight"
    assert brewer_to_llama_name("blocks.0.attn.rope_cache") is None


def test_a_folder_exports_as_llama_with_its_tokenizer(tmp_path):
    folder, _model = _folder(tmp_path, _config())
    report = export_gguf(folder, tmp_path / "out.gguf", outtype="f32")
    fields, tensors = _read(tmp_path / "out.gguf")

    assert fields["general.architecture"] == "llama"
    assert fields["llama.block_count"] == 2
    assert fields["llama.attention.head_count"] == 4
    assert fields["llama.attention.head_count_kv"] == 2
    assert fields["llama.rope.dimension_count"] == 8
    assert fields["llama.rope.freq_base"] == pytest.approx(10000.0)
    assert fields["tokenizer.ggml.model"] == "gpt2"
    assert fields["tokenizer.ggml.pre"] == "gpt-2"
    assert fields["tokenizer.ggml.tokens"][0] == "<|endoftext|>"
    assert fields["tokenizer.ggml.token_type"][0] == int(gguf.TokenType.CONTROL)
    assert fields["tokenizer.ggml.eos_token_id"] == 0
    assert fields["tokenizer.ggml.padding_token_id"] == 1
    assert len(fields["tokenizer.ggml.tokens"]) == 320     # padded to the embedding rows
    assert fields["tokenizer.ggml.tokens"][-1] == "[PAD319]"
    assert json.loads(fields["hypernix.brewer.config"])["name"] == "tiny"
    assert "output.weight" not in tensors                   # tied head: llama.cpp reuses the table
    assert report.tokenizer == "tokenizer.json" and report.n_tensors == len(tensors)


def test_an_untied_head_is_exported(tmp_path):
    folder, _ = _folder(tmp_path, _config(tie_embeddings=False, name="untied"))
    export_gguf(folder, tmp_path / "u.gguf", outtype="f32")
    _fields, tensors = _read(tmp_path / "u.gguf")
    assert "output.weight" in tensors


def test_q_and_k_are_permuted_and_the_rest_are_not(tmp_path):
    folder, model = _folder(tmp_path, _config())
    export_gguf(folder, tmp_path / "p.gguf", outtype="f32", pad_ffn_to=0)
    _fields, tensors = _read(tmp_path / "p.gguf")
    state = model.state_dict()

    def arr(name):
        t = tensors[name]
        return np.asarray(t.data).reshape([int(x) for x in reversed(t.shape)])

    q = state["blocks.0.attn.q_proj.weight"]
    np.testing.assert_array_equal(arr("blk.0.attn_q.weight"), brewer_gguf._permute(q, 4).numpy())
    np.testing.assert_array_equal(arr("blk.0.attn_v.weight"), state["blocks.0.attn.v_proj.weight"].numpy())


def test_the_ffn_is_zero_padded_for_quantisation(tmp_path):
    folder, model = _folder(tmp_path, _config(d_ff=40))
    report = export_gguf(folder, tmp_path / "f.gguf", outtype="f32")
    fields, tensors = _read(tmp_path / "f.gguf")
    assert report.n_ff == 256 and report.n_ff_trained == 40
    assert fields["llama.feed_forward_length"] == 256
    down = np.asarray(tensors["blk.0.ffn_down.weight"].data).reshape(32, 256)
    gate = np.asarray(tensors["blk.0.ffn_gate.weight"].data).reshape(256, 32)
    np.testing.assert_array_equal(down[:, 40:], 0)
    np.testing.assert_array_equal(gate[40:], 0)
    np.testing.assert_array_equal(down[:, :40], model.state_dict()["blocks.0.ffn.down_proj.weight"].numpy())


def test_zero_padding_does_not_change_the_output():
    torch.manual_seed(3)
    cfg = _config(d_ff=40)
    padded_cfg = _config(d_ff=256)
    model, padded = BrewerModel(cfg).eval(), BrewerModel(padded_cfg).eval()
    state = model.state_dict()
    new = {}
    for k, v in state.items():
        if k.endswith(("gate_proj.weight", "up_proj.weight")):
            v = torch.cat([v, v.new_zeros(216, v.shape[1])])
        elif k.endswith("down_proj.weight"):
            v = torch.cat([v, v.new_zeros(v.shape[0], 216)], dim=1)
        new[k] = v
    padded.load_state_dict(new)
    ids = torch.randint(0, 320, (1, 12))
    with torch.no_grad():
        torch.testing.assert_close(padded(ids), model(ids))


def test_sliding_window_caps_the_context_by_default(tmp_path):
    folder, _ = _folder(tmp_path, _config(use_sliding_window=True, sliding_window_size=32,
                                           max_seq_len=128, name="swa"))
    auto = export_gguf(folder, tmp_path / "a.gguf")
    full = export_gguf(folder, tmp_path / "b.gguf", context="full")
    assert auto.context_length == 32 and "sliding window" in " ".join(auto.notes)
    assert full.context_length == 128 and "further back" in " ".join(full.notes)
    assert export_gguf(folder, tmp_path / "c.gguf", context=64).context_length == 64


@pytest.mark.parametrize("outtype,expected", [
    ("f32", "F32"), ("f16", "F16"), ("bf16", "BF16"), ("q8_0", "Q8_0"),
])
def test_outtypes(tmp_path, outtype, expected):
    folder, _ = _folder(tmp_path, _config(d_model=64, d_ff=64))
    export_gguf(folder, tmp_path / "t.gguf", outtype=outtype)
    _fields, tensors = _read(tmp_path / "t.gguf")
    assert tensors["blk.0.attn_q.weight"].tensor_type.name == expected
    assert tensors["blk.0.attn_norm.weight"].tensor_type.name == "F32"


def test_q8_0_keeps_rows_that_are_not_whole_blocks_in_f16(tmp_path):
    folder, _ = _folder(tmp_path, _config(d_ff=40))
    export_gguf(folder, tmp_path / "q.gguf", outtype="q8_0", pad_ffn_to=0)
    _fields, tensors = _read(tmp_path / "q.gguf")
    assert tensors["blk.0.ffn_down.weight"].tensor_type.name == "F16"   # rows of 40
    assert tensors["blk.0.ffn_up.weight"].tensor_type.name == "Q8_0"    # rows of 32


def test_a_character_vocabulary_becomes_byte_level_bpe(tmp_path):
    folder, _ = _folder(tmp_path, _config(name="chars"), tokenizer=False)
    write_char_vocab(folder / CHAR_VOCAB_FILE, [" ", "a", "é", "✓", "\n"])
    report = export_gguf(folder, tmp_path / "c.gguf")
    fields, _ = _read(tmp_path / "c.gguf")
    tokens = fields["tokenizer.ggml.tokens"]
    assert tokens[:5] == ["Ġ", "a", "Ã©", "âľĵ", "Ċ"]      # GPT-2's byte alphabet
    assert fields["tokenizer.ggml.merges"] == ["Ã ©", "â ľ", "âľ ĵ"]
    assert report.tokenizer == "char"


def test_the_tokenizer_must_fit_the_embeddings(tmp_path):
    folder, _ = _folder(tmp_path, _config(vocab_size=100))
    with pytest.raises(GGUFExportError, match="not a pair"):
        export_gguf(folder, tmp_path / "x.gguf")


def test_no_tokenizer_is_an_error_that_says_what_to_do(tmp_path):
    folder, _ = _folder(tmp_path, _config(name="bare"), tokenizer=False)
    with pytest.raises(GGUFExportError, match="char_vocab.json"):
        export_gguf(folder, tmp_path / "x.gguf")


def test_a_config_without_weights_is_refused(tmp_path):
    folder = tmp_path / "shape"
    folder.mkdir()
    _config().save(folder / "config.json")
    with pytest.raises(GGUFExportError, match="no weights"):
        export_gguf(folder)


def test_a_compiled_models_prefix_is_stripped(tmp_path):
    cfg = _config(name="compiled")
    model = BrewerModel(cfg)
    state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    folder = tmp_path / "c"
    folder.mkdir()
    _bpe_tokenizer(folder)
    report = export_gguf(folder, tmp_path / "c.gguf", state_dict=state, config=cfg)
    assert report.n_tensors == 2 * 9 + 2


# ---------------------------------------------------------------------------
# Brewer itself: the vocabulary is kept, the folder is the model
# ---------------------------------------------------------------------------

def test_the_dataset_keeps_a_models_ids_and_appends_new_characters(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("abcabc zz" * 20)
    ds = _SimpleTextDataset(corpus, 8, vocab=["z", "a"])
    assert ds.chars[:2] == ["z", "a"]
    assert set(ds.chars) == set("abc z")


def test_training_returns_the_vocabulary_and_refuses_overflow(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("hello world " * 200)
    cfg = _config(vocab_size=64, max_seq_len=16)
    chars = train_model(BrewerModel(cfg), cfg, corpus, steps=2, batch_size=2, device="cpu")
    assert set(chars) == set("helo wrd")
    small = _config(vocab_size=4, max_seq_len=16)
    with pytest.raises(ValueError, match="vocab_size"):
        train_model(BrewerModel(small), small, corpus, steps=1, device="cpu")


def test_save_and_from_dir_round_trip(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("round trip text " * 100)
    brewer = Brewer(_config(vocab_size=64, max_seq_len=16, name="rt"), save_dir=tmp_path / "rt")
    brewer.train(corpus, steps=2, batch_size=2, device="cpu")
    folder = brewer.save()
    assert {p.name for p in folder.iterdir()} >= {"config.json", "model.safetensors", CHAR_VOCAB_FILE}
    again = Brewer.from_dir(folder)
    assert again.char_vocab == brewer.char_vocab
    for k, v in brewer.model.state_dict().items():
        torch.testing.assert_close(again.model.state_dict()[k], v.cpu())
    out = again.export(fmt="gguf", outtype="f32")
    assert out.name == "rt.f32.gguf" and out.is_file()


def test_the_old_format_is_still_there_as_hnxg(tmp_path):
    brewer = Brewer(_config(name="old"), save_dir=tmp_path / "old")
    brewer.build()
    out = brewer.export(tmp_path / "old.hnxg", fmt="hnxg")
    assert out.read_bytes()[:4] == b"HNXG"


def test_python_loads_a_character_vocabulary(tmp_path):
    from hypernix.models.generate import _load_tokenizer

    write_char_vocab(tmp_path / CHAR_VOCAB_FILE, ["h", "i", "!"])
    tok, kind = _load_tokenizer(tmp_path)
    assert kind == "byte"
    assert tok.encode("hi!?") == [0, 1, 2]
    assert tok.decode([1, 0, 2]) == "ih!"


# ---------------------------------------------------------------------------
# The brew commands work across processes
# ---------------------------------------------------------------------------

def test_brew_new_train_export_across_separate_runs(tmp_path, monkeypatch, capsys):
    from hypernix.training import brewer as brewer_mod

    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "c.txt"
    corpus.write_text("brew it and run it " * 200)
    brewer_mod.cli_main(["new", "--preset", "cpu-nano", "--name", "cli"])
    brewer_mod.BREWER_REGISTRY.clear()              # a new process knows nothing

    with pytest.raises(SystemExit):
        brewer_mod.cli_main(["export", "--name", "cli"])
    assert "no trained weights" in capsys.readouterr().err

    brewer_mod.cli_main(["train", "--name", "cli", "--data", str(corpus), "--steps", "2",
                         "--batch-size", "2", "--device", "cpu"])
    brewer_mod.BREWER_REGISTRY.clear()
    folder = tmp_path / "brewer_models" / "cli"
    assert (folder / "model.safetensors").is_file() and (folder / CHAR_VOCAB_FILE).is_file()

    brewer_mod.cli_main(["export", "--name", "cli", "--outtype", "f16"])
    assert (folder / "cli.f16.gguf").is_file()
    brewer_mod.cli_main(["list"])
    assert "gguf=cli.f16.gguf" in capsys.readouterr().out


def test_brew_gguf_converts_a_downloaded_folder(tmp_path, capsys):
    from hypernix.training import brewer as brewer_mod

    folder, _ = _folder(tmp_path, _config(name="HyperNix.3-mini-like"))
    brewer_mod.cli_main(["gguf", str(folder), "--outtype", "q8_0"])
    assert (folder / "HyperNix.3-mini-like.q8_0.gguf").is_file()
    assert "llama-cli -m" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# A model saved without config.json (as a training run uploads it)
# ---------------------------------------------------------------------------
#
# From a report: HyperNix.3-mini ran in a patched llama.cpp and the newer
# 3.1-mini did not -- "unknown model architecture: 'hypernix'". The newer
# model's folder, like hypernix.3-mini-Beta on the Hub, has
# model.safetensors, a model.pt checkpoint and tokenizer/, but no
# config.json. It was not recognised as Brewer, went to the generic
# converter, and came out labelled `hypernix` -- a name no llama.cpp has.


def _run_folder(tmp_path: Path, cfg: BrewerConfig, *, checkpoint: bool = True) -> tuple[Path, BrewerModel]:
    """model.safetensors + model.pt (a training checkpoint) + tokenizer/, no config.json."""
    folder, model = _folder(tmp_path, cfg)
    (folder / "config.json").unlink()
    tok = folder / "tokenizer"
    tok.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (folder / name).rename(tok / name)
    if checkpoint:
        torch.save({"arch": "hypernix0x-v2", "config": cfg.to_dict(),
                    "model_state_dict": model.state_dict(), "step": 7}, folder / "model.pt")
    return folder, model


def test_a_folder_without_config_json_is_still_brewer(tmp_path):
    from hypernix.quant.convertq import source_kind
    from hypernix.quant.hyprslug import is_brewer_source

    folder, _ = _run_folder(tmp_path, _config())
    assert is_brewer_source(folder)
    assert source_kind(folder) == "brewer"


def test_it_exports_the_same_file_as_with_config_json(tmp_path):
    cfg = _config()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_config, _ = _folder(tmp_path / "a", cfg)
    without, _ = _run_folder(tmp_path / "b", cfg)
    export_gguf(with_config, tmp_path / "a.gguf", outtype="f32")
    export_gguf(without, tmp_path / "b.gguf", outtype="f32")
    assert (tmp_path / "a.gguf").read_bytes() == (tmp_path / "b.gguf").read_bytes()
    fields, _ = _read(tmp_path / "b.gguf")
    assert fields["general.architecture"] == "llama"


def test_the_generic_converter_hands_brewer_models_over(tmp_path):
    """`hnx convert FOLDER -o x.gguf` (no -P) used convert_to_gguf, which
    wrote `hypernix` and did not permute Q/K. It now writes llama."""
    from hypernix.quant.convert import convert_to_gguf

    for name, make in (("with", _folder), ("without", _run_folder)):
        (tmp_path / name).mkdir()
        folder, _ = make(tmp_path / name, _config())
        out = convert_to_gguf(folder, tmp_path / f"{name}.gguf", dtype="fp16")
        fields, tensors = _read(out)
        assert fields["general.architecture"] == "llama", name
        assert "blk.0.attn_q.weight" in tensors


def test_a_config_in_the_safetensors_header(tmp_path):
    cfg = _config()
    folder, model = _run_folder(tmp_path, cfg, checkpoint=False)
    state = {k: v.contiguous() for k, v in model.state_dict().items()
             if not (cfg.tie_embeddings and k == "lm_head.weight")}
    save_file(state, str(folder / "model.safetensors"), metadata={"config": json.dumps(cfg.to_dict())})
    export_gguf(folder, tmp_path / "m.gguf", outtype="f32")
    fields, _ = _read(tmp_path / "m.gguf")
    assert fields["llama.block_count"] == 2


def test_a_training_runs_latest_checkpoint(tmp_path):
    """runs/<name>/checkpoints/latest.pt, with the tokenizer in runs/<name>/tokenizer."""
    cfg = _config()
    run = tmp_path / "runs" / "mini"
    (run / "checkpoints").mkdir(parents=True)
    (run / "tokenizer").mkdir()
    _bpe_tokenizer(run / "tokenizer")
    torch.manual_seed(0)
    model = BrewerModel(cfg)
    torch.save({"config": cfg.to_dict(), "model_state_dict": model.state_dict(), "step": 3},
               run / "checkpoints" / "latest.pt")
    report = export_gguf(run / "checkpoints" / "latest.pt", tmp_path / "r.gguf", outtype="f32")
    fields, _ = _read(tmp_path / "r.gguf")
    assert fields["general.architecture"] == "llama"
    assert report.tokenizer == "tokenizer.json"


def test_no_config_anywhere_says_where_it_looked(tmp_path):
    folder, _ = _run_folder(tmp_path, _config(), checkpoint=False)
    with pytest.raises(GGUFExportError) as caught:
        export_gguf(folder, tmp_path / "x.gguf")
    message = str(caught.value)
    assert "config.json" in message and "latest.pt" in message


def test_a_hugging_face_folder_is_not_taken_for_brewer(tmp_path):
    from hypernix.models.brewer_gguf import has_brewer_weights

    folder = tmp_path / "hf"
    folder.mkdir()
    save_file({"model.embed_tokens.weight": torch.zeros(4, 4),
               "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4)},
              str(folder / "model.safetensors"))
    assert not has_brewer_weights(folder)
