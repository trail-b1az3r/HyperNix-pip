"""hypernix.c4: explode, stem, dial, restore, reap, sow and the c4 optimizers.

A tiny Llama (4 layers, 32 wide) is built with transformers' own code
and saved locally, so nothing is downloaded. Random-token calibration is
enough to check the machinery; it is not a claim about any real model.
"""
from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from hypernix import c4  # noqa: E402
from hypernix.c4 import dial, optim, reap, restore, sow, stem  # noqa: E402
from hypernix.c4.explode import Dial, MDial, PDial, address, explode  # noqa: E402
from hypernix.c4.model import C4Error, layer_of, load, pick_device, role_of  # noqa: E402
from hypernix.c4.probe import Calibration, Prober  # noqa: E402

UP = "model.layers.0.mlp.up_proj.weight"


def _tiny(folder, seed=0, layers=4):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(seed)
    LlamaForCausalLM(cfg).save_pretrained(str(folder), safe_serialization=True)
    return folder


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _tiny(tmp_path_factory.mktemp("c4") / "tiny")


@pytest.fixture
def ex(tiny):
    return explode(tiny, device="cpu")


@pytest.fixture
def prober(ex):
    calibration = Calibration.random_tokens(96, batches=2, batch_size=2, seq_len=16)
    from hypernix.c4.probe import build_module

    return Prober(ex.model, calibration, module=build_module(ex.model))


class TestLoading:
    def test_a_safetensors_folder(self, tiny):
        model = load(tiny, device="cpu")
        assert model.format == "safetensors" and model.n_layers == 4 and model.parameters > 0

    def test_a_single_file_needs_its_config(self, tiny, tmp_path):
        import shutil

        shutil.copy(tiny / "model.safetensors", tmp_path / "w.safetensors")
        with pytest.raises(C4Error, match="config.json"):
            load(tmp_path / "w.safetensors", device="cpu")
        shutil.copy(tiny / "config.json", tmp_path / "config.json")
        assert load(tmp_path / "w.safetensors", device="cpu").n_layers == 4

    def test_a_pt_state_dict(self, tiny, tmp_path):
        model = load(tiny, device="cpu")
        restore.write(model, tmp_path / "pt", format="pt")
        again = load(tmp_path / "pt", device="cpu")
        assert again.format == "pt" and set(again.tensors) == set(model.tensors) - {"lm_head.weight"} | (
            {"lm_head.weight"} & set(again.tensors))

    def test_no_gpu_here_is_said_plainly(self):
        if torch.cuda.is_available():
            pytest.skip("this machine has a GPU")
        assert pick_device("auto") in ("cpu", "mps")
        with pytest.raises(C4Error, match="no CUDA GPU"):
            pick_device("cuda")

    @pytest.mark.parametrize("name, layer, role", [
        ("model.layers.3.mlp.down_proj.weight", 3, "ffn_down"),
        ("blk.7.attn_q.weight", 7, "attn_q"), ("token_embd.weight", -1, "embed"),
        ("model.norm.weight", -1, "output_norm"), ("output.weight", -1, "output"),
    ])
    def test_names(self, name, layer, role):
        assert layer_of(name) == layer and role_of(name) == role


class TestExplode:
    def test_the_pieces(self, ex):
        counts = ex.counts()
        assert counts["mstems"] == 4 and counts["dials"] == ex.model.parameters
        assert counts["mdials"] == counts["dials"] // 2
        assert {s.role for s in ex.mstems[0].stems} >= {"attn_q", "ffn_up", "attn_norm"}
        assert {s.role for s in ex.loose_stems} >= {"embed", "output_norm"}

    def test_pdials_are_three_or_more_parameters(self, ex):
        rows = list(ex.pdials(UP))
        assert len(rows) == 64 and all(p.size(ex.stems[UP].shape) == 32 for p in rows)
        heads = list(ex.pdials("model.layers.0.self_attn.q_proj.weight", by="head"))
        assert len(heads) == 4 and heads[0].stop - heads[0].start == 8
        norm = list(ex.pdials("model.layers.0.input_layernorm.weight"))
        assert all(p.axis == "block" and p.size((32,)) >= 3 for p in norm)

    def test_mdials_are_pairs(self, ex):
        pair = next(ex.mdials(UP))
        assert pair.first != pair.second and len(ex.value(pair)) == 2
        with pytest.raises(C4Error):
            MDial(UP, (0, 0), (0, 0))

    @pytest.mark.parametrize("text, kind", [
        (f"{UP}[3, 5]", Dial), (f"{UP}[3, 5]+[3, 6]", MDial), (f"{UP}[3, :]", PDial),
        (f"{UP}[:, 7]", PDial), (f"{UP}{{0:9}}", PDial),
    ])
    def test_addresses_round_trip(self, ex, text, kind):
        piece = address(text, ex)
        assert isinstance(piece, kind)
        assert address(str(piece), ex) == piece

    def test_saving_one_file_per_mstem(self, ex, tmp_path):
        ex.save(tmp_path / "out")
        manifest = json.loads((tmp_path / "out" / "c4-explode.json").read_text())
        assert manifest["mstems"][0]["index"] == 0
        assert (tmp_path / "out" / "mstem-0003.safetensors").is_file()
        assert (tmp_path / "out" / "loose.safetensors").is_file()


class TestRestore:
    def test_an_ablation_is_undone_exactly(self, ex):
        before = ex.tensor(UP).clone()
        with restore.ablated(ex, f"{UP}[3, :]", "zero"):
            assert float(ex.tensor(UP)[3].abs().sum()) == 0
        assert torch.equal(ex.tensor(UP), before)

    def test_every_ablation_kind(self, ex):
        before = ex.tensor(UP).clone()
        for how in restore.ABLATIONS:
            with restore.ablated(ex, ex.stems[UP], how):
                assert not torch.equal(ex.tensor(UP), before)
        assert torch.equal(ex.tensor(UP), before)

    def test_a_whole_layer(self, ex):
        before = {s.name: ex.tensor(s).clone() for s in ex.mstems[1].stems}
        with restore.ablated(ex, ex.mstems[1]):
            assert all(float(ex.tensor(s).abs().sum()) == 0 for s in ex.mstems[1].stems)
        assert all(torch.equal(ex.tensor(n), t) for n, t in before.items())


class TestStem:
    def test_measuring_a_known_matrix(self, ex):
        with restore.ablated(ex, ex.stems[UP], "zero"):
            torch.nn.init.eye_(ex.tensor(UP))
            stats = stem.measure(ex, UP)
        assert stats.effective_rank == pytest.approx(32, rel=1e-3) and stats.stable_rank == pytest.approx(32)
        assert stats.sparsity > 0.9

    def test_an_inert_layer_is_found(self, ex, prober):
        """Zero a layer's outputs and it adds nothing: removing it changes
        nothing, and its block influence is the lowest. (On a random model
        removing another layer can *lower* the loss, so the ablation score
        alone need not put it last.)"""
        dead = ex.mstems[2]
        snap = restore.Snapshot.take(ex, [dead])
        try:
            for s in dead.stems:
                if s.role in ("attn_output", "ffn_down"):
                    restore.apply_ablation(ex, s, "zero")
            report = stem.layer_report(ex, prober)
            assert min(report, key=lambda v: v.influence).index == 2
            assert report[2].ablation_delta == pytest.approx(0.0, abs=1e-6)
            assert report[2].influence == pytest.approx(0.0, abs=1e-5)
        finally:
            snap.restore()

    def test_tests_leave_the_model_as_found(self, ex, prober):
        base = prober.loss()
        stem.layer_report(ex, prober)
        assert prober.loss() == pytest.approx(base, abs=1e-6)


class TestDial:
    @pytest.mark.parametrize("kind", dial.SCORES)
    def test_every_score_matches_its_weights(self, ex, prober, kind):
        scores = dial.score(ex, prober if kind != "magnitude" else None, kind=kind)
        assert scores.scores[UP].shape == ex.tensor(UP).shape
        assert all(float(v.float().min()) >= 0 for v in scores.scores.values())

    def test_unneeded_takes_the_lowest_share(self, ex, prober):
        scores = dial.score(ex, prober, kind="taylor")
        cut = dial.unneeded(scores, fraction=0.25)
        mask = cut.masks[UP]
        assert mask.float().mean() == pytest.approx(0.25, abs=0.02)
        assert float(scores.scores[UP][mask].float().max()) <= float(scores.scores[UP][~mask].float().min()) + 1e-3

    def test_a_disturbed_stem_needs_retraining(self, ex, prober):
        """Scramble one stem: its gradient is now large for its weights."""
        target = "model.layers.1.mlp.down_proj.weight"
        with restore.ablated(ex, ex.stems[target], "noise"):
            ex.tensor(target).mul_(8.0)
            scores = dial.score(ex, prober, kind="taylor")
            ranked = dial.train_priority(ex, scores, granularity="stem", top=50)
        assert ranked[0].gain >= ranked[-1].gain
        by_name = {p.piece: p for p in ranked}
        assert target in by_name
        layers = dial.train_priority(ex, scores, granularity="mstem")
        assert {p.piece for p in layers} <= {m.name for m in ex.mstems}

    def test_an_mdial_interaction(self, ex, prober):
        found = dial.interaction(ex, prober, next(ex.mdials(UP)))
        assert set(found) >= {"first", "second", "both", "synergy"}
        assert prober.loss() == pytest.approx(found["base_loss"], abs=1e-6)


class TestReapAndSow:
    def test_a_smaller_model_that_runs(self, ex, prober, tmp_path):
        scores = dial.score(ex, prober, kind="taylor")
        cut = dial.unneeded(scores, fraction=0.1)
        report = stem.layer_report(ex, prober)
        harvest = reap.reap(ex, report=report, drop_layers=1, unneeded=cut, dial_scores=scores, ffn_keep=0.5)
        assert len(harvest.layers) == 3 and len(harvest.channels[0]) == 32
        assert not any(ex.stems[s].layer < 0 for _, s in harvest.masks)   # embeddings left alone
        harvest.save(tmp_path / "h")
        again = reap.Harvest.load(tmp_path / "h")
        assert again.layers == harvest.layers and again.zeroed == harvest.zeroed

        result = sow.sow(again, tmp_path / "new", explodeds=[ex], prober=prober)
        config = json.loads((tmp_path / "new" / "config.json").read_text())
        assert config["num_hidden_layers"] == 3 and config["intermediate_size"] == 32
        assert result.parameters_after < result.parameters_before and result.loss_after is not None
        assert math.isfinite(result.loss_after)
        model = transformers.AutoModelForCausalLM.from_pretrained(str(tmp_path / "new"))
        assert model.model.layers[0].mlp.up_proj.weight.shape == (32, 32)

    def test_cherry_picking_layers_across_models(self, tiny, tmp_path, prober):
        """Donor B is A with layer 1 ruined: layer 1 must come from A."""
        import shutil

        shutil.copytree(tiny, tmp_path / "b")
        a = explode(tiny, device="cpu")
        b = explode(tmp_path / "b", device="cpu")
        for s in b.mstems[1].stems:
            if s.kind == "matrix":
                b.tensor(s).mul_(25.0)
        cal = prober.calibration
        from hypernix.c4.probe import build_module

        pa = Prober(a.model, cal, module=build_module(a.model))
        pb = Prober(b.model, cal, module=build_module(b.model))
        harvest = reap.reap_best([a, b], probers=[pa, pb])
        assert harvest.layers[1] == (0, 1)
        out = sow.sow(harvest, tmp_path / "picked", explodeds=[a, b])
        assert out.layers == 4

    def test_shapes_must_match_to_cherry_pick(self, tiny, tmp_path):
        other = explode(_tiny(tmp_path / "three", layers=3), device="cpu")
        with pytest.raises(C4Error, match="numbers of layers"):
            reap.reap_best([explode(tiny, device="cpu"), other])

    def test_a_gguf_weights_only(self, tiny, tmp_path):
        """No forward pass for a GGUF transformers cannot build: explode,
        measure, reap and sow still work, and the result is a GGUF."""
        from hypernix.quant.gguf import GGMLType, GGUFWriter

        source = load(tiny, device="cpu")
        names = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight"}
        writer = GGUFWriter(tmp_path / "m.gguf")
        writer.set_metadata("general.architecture", "hnxtest")
        writer.set_metadata("hnxtest.block_count", 4)
        payload = {}
        for name, value in source.tensors.items():
            if name == "lm_head.weight":
                continue
            gname = names.get(name) or name.replace("model.layers.", "blk.").replace(
                "self_attn.q_proj", "attn_q").replace("mlp.up_proj", "ffn_up")
            array = value.numpy()
            writer.add_tensor(gname, tuple(reversed(array.shape)), int(GGMLType.F32))
            payload[gname] = array.astype("<f4").tobytes()
        writer.write(lambda t: payload[t.name])
        g = explode(tmp_path / "m.gguf", device="cpu")
        assert g.model.format == "gguf" and len(g.mstems) == 4
        assert g.tensor("blk.0.ffn_up.weight").shape == (64, 32)
        report = stem.layer_report(g)
        harvest = reap.reap(g, report=report, drop_layers=2)
        out = sow.sow(harvest, tmp_path / "small.gguf", explodeds=[g])
        assert out.path.endswith(".gguf") and len(explode(out.path, device="cpu").mstems) == 2


class TestOptimizer:
    def _setup(self, ex, prober, **kwargs):
        module = prober.module
        module.train()
        return module, optim.C4Optimizer.for_exploded(ex, module, **kwargs)

    def _step(self, module, opt, batch):
        opt.zero_grad()
        module(input_ids=batch, labels=batch).loss.backward()

    @pytest.mark.parametrize("kind", ["hybrid", "v5s", "adamw"])
    def test_it_trains(self, ex, prober, kind):
        snap = restore.Snapshot.take(ex, list(ex.stems.values()))
        try:
            module, opt = self._setup(ex, prober, kind=kind, lr=1e-2)
            batch = prober.calibration.batches[0]
            first = float(module(input_ids=batch, labels=batch).loss)
            for _ in range(15):
                self._step(module, opt, batch)
                opt.step()
            assert float(module(input_ids=batch, labels=batch).loss) < first
        finally:
            snap.restore()

    def test_a_bad_section_is_skipped_not_the_step(self, ex, prober):
        snap = restore.Snapshot.take(ex, list(ex.stems.values()))
        try:
            module, opt = self._setup(ex, prober, kind="hybrid", lr=1e-3)
            batch = prober.calibration.batches[0]
            self._step(module, opt, batch)
            params = dict(module.named_parameters())
            params[UP].grad[0, 0] = float("nan")
            up_before = params[UP].detach().clone()
            down = "model.layers.0.mlp.down_proj.weight"
            down_before = params[down].detach().clone()
            opt.step()
            assert torch.equal(params[UP], up_before)
            assert not torch.equal(params[down], down_before)
            assert opt.report()["skipped"][UP] == 1
        finally:
            snap.restore()

    def test_only_the_chosen_sections_move(self, ex, prober):
        snap = restore.Snapshot.take(ex, list(ex.stems.values()))
        try:
            module, opt = self._setup(ex, prober, kind="hybrid", lr=1e-3, only=[UP])
            params = dict(module.named_parameters())
            before = {n: p.detach().clone() for n, p in params.items()}
            self._step(module, opt, prober.calibration.batches[0])
            opt.step()
            moved = [n for n, p in params.items() if not torch.equal(p, before[n])]
            assert moved == [UP]
        finally:
            for p in prober.module.parameters():
                p.requires_grad_(True)
            snap.restore()

    @pytest.mark.parametrize("kind", ["hybrid", "v5s"])
    def test_held_dials_do_not_move(self, ex, prober, kind):
        snap = restore.Snapshot.take(ex, list(ex.stems.values()))
        try:
            mask = torch.zeros_like(ex.tensor(UP), dtype=torch.bool)
            mask[:8] = True
            module, opt = self._setup(ex, prober, kind=kind, lr=1e-2, masks={UP: mask})
            held = dict(module.named_parameters())[UP].detach()[:8].clone()
            for _ in range(3):
                self._step(module, opt, prober.calibration.batches[0])
                opt.step()
            assert torch.equal(dict(module.named_parameters())[UP].detach()[:8], held)
        finally:
            snap.restore()

    def test_a_priority_scales_a_section(self, ex, prober):
        snap = restore.Snapshot.take(ex, list(ex.stems.values()))
        try:
            module, opt = self._setup(ex, prober, kind="adamw", lr=1e-3, priorities={UP: 0.0})
            before = dict(module.named_parameters())[UP].detach().clone()
            self._step(module, opt, prober.calibration.batches[0])
            opt.step()
            assert torch.equal(dict(module.named_parameters())[UP], before)
        finally:
            snap.restore()

    def test_pdial_rows_are_checked_one_by_one(self, ex, prober):
        snap = restore.Snapshot.take(ex, list(ex.stems.values()))
        try:
            module, opt = self._setup(ex, prober, kind="hybrid", lr=1e-3, granularity="pdial")
            self._step(module, opt, prober.calibration.batches[0])
            params = dict(module.named_parameters())
            params[UP].grad[5, 3] = float("inf")
            row_before = params[UP].detach()[5].clone()
            other_before = params[UP].detach()[6].clone()
            opt.step()
            assert torch.equal(params[UP][5], row_before)
            assert not torch.equal(params[UP][6], other_before)
        finally:
            snap.restore()

    def test_a_spike_is_clipped(self):
        p = torch.nn.Parameter(torch.ones(4))
        opt = optim.C4Optimizer([optim.Section("p", [p])], torch.optim.SGD([p], lr=0.1),
                                policy=optim.CheckPolicy(spike_z=1.0, warmup=3))
        for _ in range(4):
            p.grad = torch.full_like(p, 0.1)
            opt.step()
        p.grad = torch.full_like(p, 100.0)
        before = p.detach().clone()
        opt.step()
        assert float((before - p).abs().max()) < 1.0
        assert opt.report()["clipped"]["p"] == 1


class TestCli:
    def test_explode_json(self, tiny, capsys):
        from hypernix.c4.cli import cli_main

        assert cli_main(["explode", str(tiny), "--device", "cpu", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["mstems"] == 4

    def test_hnx_routes_to_it(self, tiny, capsys):
        from hypernix.interfaces import cli

        assert cli.main(["c4", "explode", str(tiny), "--device", "cpu"]) == 0
        assert "4 mstems" in capsys.readouterr().out

    def test_a_missing_model_is_an_error_not_a_traceback(self, tmp_path, capsys):
        from hypernix.c4.cli import cli_main

        assert cli_main(["explode", str(tmp_path / "nope"), "--device", "cpu"]) == 1
        assert "No such model" in capsys.readouterr().err

    def test_the_package_exposes_the_sections(self):
        assert c4.explode.explode is explode and callable(c4.load)
