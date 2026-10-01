"""hyperNix0x-v2 models, in every place NeoOven works.

``hypernix.training.brewer`` builds the ``hyperNix0x-v2`` family with its
own config, presets and checkpoint format. ``NeoOven`` is the front end
for everything else. Until now the two did not meet: a brewed model
could be trained by ``brewer.train_model`` and by nothing else, and none
of NeoOven's generation, sampling or chat templating reached it.

The bug this is really about
----------------------------
``BrewerModel.forward(input_ids, attn_mask)`` returns a bare logits
tensor. ``NeoOven`` calls ``model(ids, labels=labels)["loss"]``.

Passed straight through, ``labels`` binds to ``attn_mask`` and is used as
an *additive attention mask*. A tensor of token ids added to attention
scores does not raise, does not produce NaN, and does not fail any check
— it returns finite logits from a forward pass that means nothing, and
trains the model into noise.

That is why this is an adapter and not a couple of ``getattr`` checks at
the call sites, and it is the first thing tested below: the adapter makes
that call impossible rather than unlikely.

Everything here runs on CPU with the ``cpu-nano`` preset (2.1M params),
so it needs no GPU and takes about a second.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from hypernix.models import brewer_adapter  # noqa: E402


@pytest.fixture(scope="module")
def config():
    """The smallest preset. 2.1M params: a real model, instantly built."""
    return brewer_adapter.preset_config("cpu-nano")


@pytest.fixture
def raw_model(config):
    from hypernix.training import brewer

    torch.manual_seed(0)
    return brewer.BrewerModel(config)


@pytest.fixture
def model(raw_model):
    return brewer_adapter.wrap(raw_model)


class TestThePresets:
    @pytest.mark.parametrize("name", brewer_adapter.PRESETS)
    def test_every_advertised_preset_resolves(self, name):
        assert brewer_adapter.preset_config(name) is not None

    def test_the_fully_qualified_name_round_trips(self):
        """A saved config's ``name`` is ``hypernix0x-v2-small``, and
        loading it back has to find the same preset."""
        plain = brewer_adapter.preset_config("small")
        qualified = brewer_adapter.preset_config(plain.name)

        assert qualified.name == plain.name
        assert qualified.d_model == plain.d_model

    def test_an_unknown_preset_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="cpu-nano"):
            brewer_adapter.preset_config("enormous")

    def test_the_advertised_list_matches_what_brewer_has(self):
        """PRESETS is a second copy of brewer's factory names. The
        previous pattern in this repository was a hand-maintained
        duplicate that went stale, so it is checked."""
        from hypernix.training import brewer

        actual = {
            name[len("hypernix0x_v2_"):].replace("_", "-")
            for name in dir(brewer)
            if name.startswith("hypernix0x_v2_")
        }

        assert set(brewer_adapter.PRESETS) == actual


class TestTheAdapterPreventsTheSilentBug:
    """The reason this module exists."""

    def test_labels_reach_the_loss_and_not_the_attention_mask(self, model):
        ids = torch.randint(0, 100, (2, 16))
        out = model(ids, labels=ids)

        assert "loss" in out and "logits" in out
        # A real loss for a randomly initialised model over a vocab of
        # 8000 is about ln(8000) ~= 9. If `labels` had gone to the mask
        # instead there would be no loss key at all -- and before the
        # adapter, there was no loss key *and no error*.
        assert 5.0 < out["loss"].detach().item() < 14.0

    def test_the_raw_model_really_does_accept_ids_as_a_mask(self, raw_model):
        """Recorded because it is the whole justification for wrapping.

        This is not a hypothetical: passing labels positionally to the
        unwrapped model runs, returns finite numbers, and is meaningless.
        """
        ids = torch.randint(0, 100, (1, 8))
        mask_shaped_labels = torch.randint(0, 100, (8, 8)).float()

        logits = raw_model(ids, mask_shaped_labels)

        assert torch.isfinite(logits).all(), (
            "if this ever raises or produces NaN, the adapter's rationale "
            "has changed and this test should be revisited"
        )
        # And it is a bare tensor, not a dict -- so NeoOven's
        # `model(ids)["logits"]` would raise a TypeError on a tensor
        # rather than doing something wrong. It is the *training* call
        # that fails silently, which is the dangerous half.
        assert not isinstance(logits, dict)

    def test_the_adapted_model_answers_neo_ovens_generation_call(self, model):
        """``self.model(ctx)["logits"][:, -1, :]`` -- the exact
        expression NeoOven.generate uses."""
        ctx = torch.randint(0, 100, (1, 12))

        logits = model(ctx)["logits"][:, -1, :].float()

        assert logits.shape == (1, model.config.vocab_size)

    def test_padding_is_ignored_in_the_loss(self, model):
        """-100 is torch's ignore value and what every HF collator emits
        for padding. Without ignore_index the model is trained to predict
        the pad token."""
        ids = torch.randint(0, 100, (1, 16))
        padded = ids.clone()
        padded[:, 8:] = -100

        full = model(ids, labels=ids)["loss"].detach().item()
        partial = model(ids, labels=padded)["loss"].detach().item()

        assert full != partial
        assert 0.0 < partial < 20.0, "an ignored label must not poison the loss"

    def test_a_loss_backpropagates(self, model):
        """The end of the contract: NeoOven's trainer calls
        ``out_dict["loss"].backward()``."""
        ids = torch.randint(0, 100, (1, 16))
        model(ids, labels=ids)["loss"].backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no gradients reached the parameters"
        assert all(torch.isfinite(g).all() for g in grads)


class TestItIsStillANormalModule:
    """An nn.Module subclass, so everything that walks a model works."""

    def test_parameters_are_visible(self, model, raw_model):
        assert sum(p.numel() for p in model.parameters()) == raw_model.num_params()

    def test_it_moves_between_devices_and_dtypes(self, model):
        moved = model.to(torch.float64)

        assert next(moved.parameters()).dtype is torch.float64

    def test_the_state_dict_round_trips(self, model, config):
        from hypernix.training import brewer

        state = model.state_dict()
        fresh = brewer_adapter.wrap(brewer.BrewerModel(config))
        fresh.load_state_dict(state)

        ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            assert torch.allclose(
                model(ids)["logits"], fresh(ids)["logits"], atol=1e-6
            )

    def test_wrapping_is_idempotent(self, model):
        assert brewer_adapter.wrap(model) is model

    def test_it_leaves_other_models_alone(self):
        """So a caller does not have to know which kind it holds."""
        class Pretend(torch.nn.Module):
            def forward(self, ids, labels=None):
                return {"logits": ids}

        other = Pretend()
        assert brewer_adapter.wrap(other) is other
        assert brewer_adapter.wrap(None) is None

    def test_it_says_what_it_is(self, model):
        assert model.arch == brewer_adapter.ARCH_NAME
        assert "M params" in repr(model)


class TestRecognisingACheckpoint:
    def test_a_save_directory_is_recognised(self, tmp_path, config):
        config.save(tmp_path / "config.json")

        assert brewer_adapter.is_brewer_checkpoint(tmp_path)

    def test_a_hypernix_snapshot_is_not(self, tmp_path):
        """HyperNixConfig uses hidden_size / num_hidden_layers; Brewer
        uses d_model / n_layers. Both write config.json, so the keys are
        what distinguishes them."""
        (tmp_path / "config.json").write_text(
            json.dumps({"hidden_size": 1024, "num_hidden_layers": 16}),
            encoding="utf-8",
        )

        assert not brewer_adapter.is_brewer_checkpoint(tmp_path)

    def test_a_checkpoint_file_is_recognised_without_unpickling_it(
        self, tmp_path, model, config
    ):
        """``torch.load`` on an untrusted file executes a pickle. The
        member names of the zip are enough to recognise the layout."""
        path = tmp_path / "ckpt.pt"
        torch.save(
            {"config": config.to_dict(), "model_state_dict": model.inner.state_dict()},
            path,
        )

        assert brewer_adapter.is_brewer_checkpoint(path)

    def test_a_random_file_is_not(self, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("hello", encoding="utf-8")

        assert not brewer_adapter.is_brewer_checkpoint(junk)
        assert not brewer_adapter.is_brewer_checkpoint(tmp_path / "missing")

    def test_a_broken_config_is_not(self, tmp_path):
        (tmp_path / "config.json").write_text("{ truncated", encoding="utf-8")

        assert not brewer_adapter.is_brewer_checkpoint(tmp_path)


class TestLoading:
    def test_a_checkpoint_loads_and_matches(self, tmp_path, model, config):
        path = tmp_path / "ckpt.pt"
        torch.save(
            {"config": config.to_dict(), "model_state_dict": model.inner.state_dict()},
            path,
        )

        loaded, loaded_config = brewer_adapter.load(path)

        assert loaded.arch == brewer_adapter.ARCH_NAME
        assert loaded_config.d_model == config.d_model
        ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            assert torch.allclose(
                model(ids)["logits"], loaded(ids)["logits"], atol=1e-6
            )

    def test_a_directory_loads(self, tmp_path, model, config):
        config.save(tmp_path / "config.json")
        torch.save(model.inner.state_dict(), tmp_path / "model.pt")

        loaded, _ = brewer_adapter.load(tmp_path)

        ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            assert torch.allclose(
                model(ids)["logits"], loaded(ids)["logits"], atol=1e-6
            )

    def test_a_config_with_no_weights_warns_and_still_loads(
        self, tmp_path, config, caplog
    ):
        """`hnx brew new` writes exactly this -- a shape with no weights.
        A legitimate thing to load, and a terrible thing to hand back
        silently as though it were trained."""
        config.save(tmp_path / "config.json")

        with caplog.at_level("WARNING"):
            loaded, _ = brewer_adapter.load(tmp_path)

        assert loaded is not None
        assert "randomly initialised" in caplog.text

    def test_a_file_that_is_not_a_checkpoint_is_refused(self, tmp_path):
        path = tmp_path / "other.pt"
        torch.save({"something": "else"}, path)

        with pytest.raises(ValueError, match="model_state_dict"):
            brewer_adapter.load(path)


class TestNeoOvenReachesIt:
    def test_new_brewed_builds_a_usable_oven(self, tmp_path):
        from hypernix.models import neo_oven

        oven = neo_oven.new_brewed(tmp_path / "fresh", preset="cpu-nano")

        assert oven.model.arch == brewer_adapter.ARCH_NAME
        # And it wrote both halves, so preheat_brewed can read it back.
        assert (tmp_path / "fresh" / "config.json").is_file()
        assert (tmp_path / "fresh" / "model.pt").is_file()

    def test_preheat_brewed_reads_what_new_brewed_wrote(self, tmp_path):
        from hypernix.models import neo_oven

        neo_oven.new_brewed(tmp_path / "fresh", preset="cpu-nano")
        oven = neo_oven.preheat_brewed(tmp_path / "fresh")

        ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits = oven.model(ids)["logits"]
        assert logits.shape[-1] == oven.model.config.vocab_size

    def test_preheat_recognises_a_brewed_local_dir(self, tmp_path):
        """Without this, pointing preheat(local_dir=...) at a brewed
        model fails verify_snapshot and then re-downloads from the Hub --
        wrong, and slow."""
        from hypernix.models import neo_oven

        neo_oven.new_brewed(tmp_path / "fresh", preset="cpu-nano")
        oven = neo_oven.preheat(local_dir=tmp_path / "fresh")

        assert oven.model.arch == brewer_adapter.ARCH_NAME

    def test_a_missing_tokenizer_is_a_warning_not_a_crash(self, tmp_path, caplog):
        """A brewed model has no tokenizer of its own. Loading without
        one is useful for measuring shapes and useless for generating
        text, so it says which."""
        from hypernix.models import neo_oven

        neo_oven.new_brewed(tmp_path / "fresh", preset="cpu-nano")
        with caplog.at_level("WARNING"):
            oven = neo_oven.preheat_brewed(tmp_path / "fresh")

        assert oven.model is not None
        assert "nonsense" in caplog.text or oven.tokenizer is not None


# ---------------------------------------------------------------------------
# Every layout a Brewer model is found in
# ---------------------------------------------------------------------------
#
# preheat_brewed loaded one layout -- a folder with config.json -- and
# failed or misbehaved on the others a training run produces: a folder
# uploaded with no config.json, a model.pt that is a whole training
# checkpoint, weights from a torch.compile'd model, and a run's
# checkpoints/latest.pt (which loaded with the byte fallback tokenizer
# and generated nonsense). Every model also reported a 2048-token
# context, whatever it was trained at.

def _tiny():
    from hypernix.training.brewer import BrewerConfig
    return BrewerConfig(vocab_size=300, n_layers=2, n_heads=4, n_kv_heads=2, d_model=32,
                        d_ff=40, max_seq_len=64, use_sliding_window=False, name="tiny")


def _tokenizer(folder):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    folder.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["the quick brown fox " * 20], trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(folder / "tokenizer.json"))


def _weights(cfg, prefix=""):
    from hypernix.training.brewer import BrewerModel
    torch.manual_seed(0)
    model = BrewerModel(cfg)
    return model, {prefix + k: v.contiguous() for k, v in model.state_dict().items()}


def _layout(tmp_path, kind):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    cfg = _tiny()
    model, state = _weights(cfg, "_orig_mod." if kind == "compiled" else "")
    folder = tmp_path / kind
    folder.mkdir()
    untied = {k: v for k, v in state.items() if not k.endswith("lm_head.weight")}
    if kind == "uploaded":            # no config.json; config in the checkpoint; tokenizer/
        save_file(untied, str(folder / "model.safetensors"))
        torch.save({"config": cfg.to_dict(), "model_state_dict": state}, folder / "model.pt")
        _tokenizer(folder / "tokenizer")
        return folder, model
    cfg.save(folder / "config.json")
    _tokenizer(folder)
    if kind == "checkpoint_as_model_pt":
        torch.save({"config": cfg.to_dict(), "model_state_dict": state, "step": 9},
                   folder / "model.pt")
    else:
        save_file(untied, str(folder / "model.safetensors"))
    return folder, model


@pytest.mark.parametrize("kind", ["plain", "uploaded", "checkpoint_as_model_pt", "compiled"])
def test_every_layout_loads_the_same_weights(tmp_path, kind):
    folder, original = _layout(tmp_path, kind)
    loaded, _cfg = brewer_adapter.load(folder)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(loaded.inner(ids), original(ids), atol=1e-6)


def test_an_uploaded_folder_is_recognised(tmp_path):
    folder, _ = _layout(tmp_path, "uploaded")
    assert brewer_adapter.is_brewer_checkpoint(folder)


@pytest.mark.parametrize("kind", ["plain", "uploaded"])
def test_preheat_finds_the_tokenizer_and_the_trained_context(tmp_path, kind):
    from hypernix.models.neo_oven import preheat_brewed

    folder, _ = _layout(tmp_path, kind)
    oven = preheat_brewed(folder, device="cpu")
    assert oven.tokenizer_kind == "hf"
    assert oven._max_context() == 64


def test_a_runs_latest_checkpoint_uses_the_runs_tokenizer(tmp_path):
    from hypernix.models.neo_oven import preheat_brewed

    cfg = _tiny()
    _model, state = _weights(cfg)
    run = tmp_path / "runs" / "mini"
    (run / "checkpoints").mkdir(parents=True)
    _tokenizer(run / "tokenizer")
    torch.save({"config": cfg.to_dict(), "model_state_dict": state},
               run / "checkpoints" / "latest.pt")
    oven = preheat_brewed(run / "checkpoints" / "latest.pt", device="cpu")
    assert oven.tokenizer_kind == "hf"


def test_a_folder_with_no_config_anywhere_says_where_it_looked(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    _model, state = _weights(_tiny())
    folder = tmp_path / "bare"
    folder.mkdir()
    save_file({k: v for k, v in state.items() if not k.endswith("lm_head.weight")},
              str(folder / "model.safetensors"))
    with pytest.raises(ValueError, match="config.json"):
        brewer_adapter.load(folder)
