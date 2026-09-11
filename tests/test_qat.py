"""Quantisation-aware training for the HyperNix tiers.

The one thing that has to be true: **the fake quantiser must be the real
one**. If training simulates a packing that differs from what ``hyprslug``
writes, the model spends its capacity adapting to boundaries that never
ship, and the result is worse than not training at all.

That is not a hypothetical in this package. The row-length bug survived
because the writer and the reader shared a misconception and agreed with
each other; a QAT that drifts from its packer is the same mistake with a
training run attached. So the first class here compares
:func:`fake_quantize` against the byte packer element for element, and
the rest are about the training actually helping.
"""
from __future__ import annotations

import numpy as np
import pytest

from hypernix.quant.subbit import BLOCK_SIZE, PACKINGS, dequantize_array, quantize_tensor

torch = pytest.importorskip("torch", reason="QAT needs PyTorch")
nn = torch.nn

from hypernix.quant.qat import (  # noqa: E402
    QATConfig,
    QATError,
    block_scales,
    clamp_weight_,
    fake_quantize,
    finalize_qat,
    prepare_qat,
    quantization_error,
)

SIGN_PACKINGS = sorted(PACKINGS)


class TestTheFakeQuantiserIsTheRealOne:
    @pytest.mark.parametrize("packing", SIGN_PACKINGS)
    def test_it_matches_the_byte_packer(self, packing):
        """Element for element, against the bytes hyprslug would write.

        Agreement to a float32 ulp, not approximately: both compute the
        same value, and anything larger is a difference in the
        arithmetic rather than in the rounding.
        """
        rng = np.random.default_rng(0)
        values = np.concatenate(
            [rng.normal(0, s, 16) for s in rng.uniform(0.005, 0.5, 16 * 4)]
        ).astype(np.float32)
        values[::37] = 0.0                      # exact zeros: >= 0 is positive
        packed = bytes(quantize_tensor(values.tolist(), packing))
        reference = np.asarray(dequantize_array(packed, packing), np.float32)[: values.size]
        got = fake_quantize(torch.from_numpy(values), packing).numpy()
        assert np.array_equal(got, reference), (
            f"{packing}: QAT would train against a different quantiser than "
            f"the one hyprslug writes.\n"
            f"  max difference {np.max(np.abs(got - reference)):.6g}\n"
            f"Bit-exact is reachable and was reached; anything less means the "
            f"two have drifted, and an ulp of drift is how this started."
        )

    @pytest.mark.parametrize(
        "packing",
        [name for name in SIGN_PACKINGS if not PACKINGS[name].has_sub_magnitude],
    )
    def test_the_scale_is_rounded_to_fp16(self, packing):
        """The file holds an FP16 scale, so on these tiers every weight
        reconstructs to +/- an FP16 number.

        Training against the float32 scale optimises for values the
        format cannot hold -- a ~3e-4 relative offset on every weight, in
        the one place where matching the deployed quantiser is the point.
        It was doing exactly that until this test was written.
        """
        rng = np.random.default_rng(3)
        values = rng.normal(0, 0.37, BLOCK_SIZE).astype(np.float32)
        got = fake_quantize(torch.from_numpy(values), packing)
        for magnitude in torch.unique(got.abs()):
            assert magnitude == magnitude.half().float(), magnitude

    def test_the_sub_magnitude_tier_is_a_scale_times_an_index(self):
        """hnx_1375bit reconstructs to ``fp16_scale * index / 31``, which
        is not itself an FP16 number -- so the claim above is about the
        scale, and here it is, recovered from the output."""
        rng = np.random.default_rng(4)
        values = rng.normal(0, 0.37, BLOCK_SIZE).astype(np.float32)
        got = fake_quantize(torch.from_numpy(values), "hnx_1375bit")
        spec = PACKINGS["hnx_1375bit"]
        biggest = float(got.abs().max())
        assert biggest > 0
        # The largest sub-block takes the top index, so the scale is
        # recoverable from the output -- and has to be FP16.
        scale = torch.tensor(biggest)
        assert scale == scale.half().float()
        for magnitude in torch.unique(got.abs()):
            index = float(magnitude) / biggest * spec.levels
            assert abs(index - round(index)) < 1e-4, index

    def test_zero_is_positive_like_the_packer(self):
        """``w >= 0`` in the packer, so a stored zero decodes to +scale.
        A QAT that treated it as negative would train the model to place
        weights on the wrong side of a boundary."""
        values = torch.zeros(BLOCK_SIZE)
        values[0] = 0.5
        got = fake_quantize(values, "int1_binary")
        assert (got[1:] > 0).all()

    @pytest.mark.parametrize("packing", SIGN_PACKINGS)
    def test_the_output_is_finite(self, packing):
        values = torch.randn(BLOCK_SIZE * 2)
        values[0] = float("inf")
        assert torch.isfinite(fake_quantize(values, packing)).all()

    def test_a_ragged_layer_is_refused(self):
        with pytest.raises(QATError, match="whole number"):
            fake_quantize(torch.randn(100), "int1_binary")

    def test_an_unknown_packing_is_refused(self):
        from hypernix.quant.subbit import SubBitError

        with pytest.raises(SubBitError, match="Unknown packing"):
            fake_quantize(torch.randn(BLOCK_SIZE), "not-a-packing")


class TestItCanBeTrainedThrough:
    def test_gradients_reach_the_float_weight(self):
        """Sign extraction and rounding have zero derivative almost
        everywhere; without the straight-through estimator there is
        nothing to train on at all."""
        weight = torch.randn(BLOCK_SIZE, requires_grad=True)
        fake_quantize(weight, "hnx_1375bit").sum().backward()
        assert weight.grad is not None
        assert float(weight.grad.abs().sum()) > 0

    def test_the_forward_value_is_the_quantised_one(self):
        weight = torch.randn(BLOCK_SIZE, requires_grad=True)
        got = fake_quantize(weight, "int1_binary")
        assert torch.equal(
            got.detach(), fake_quantize(weight.detach(), "int1_binary")
        )

    def test_the_estimator_is_the_identity(self):
        """Straight-through: d(quantised)/d(weight) == 1."""
        weight = torch.randn(BLOCK_SIZE, requires_grad=True)
        fake_quantize(weight, "int1_binary").sum().backward()
        assert torch.allclose(weight.grad, torch.ones_like(weight))


class TestClampingIsWhatMakesItWork:
    """Without it QAT was worse than not doing QAT, in every run.

    A straight-through estimator puts no pressure on a weight's
    magnitude -- only its sign reaches the output -- so weights drift
    outward, and the block scale, being their mean absolute value,
    drifts with them.
    """

    def test_clamping_bounds_the_weight_by_the_block_scale(self):
        weight = torch.randn(BLOCK_SIZE * 2) * 0.1
        weight[5] = 50.0
        # Against the scales as they were *at clamp time*: clamping lowers
        # the weights, which lowers their mean, which lowers the bound, so
        # one pass is a contraction rather than a fixed point.
        scales = block_scales(weight, "int1_binary")
        bound = (1.5 * scales).expand(-1, BLOCK_SIZE).reshape(weight.shape)
        clamp_weight_(weight, "int1_binary", 1.5)
        assert (weight.abs() <= bound + 1e-6).all()
        assert float(weight[5]) < 50.0

    def test_a_limit_of_zero_disables_it(self):
        weight = torch.randn(BLOCK_SIZE) * 0.1
        weight[5] = 50.0
        before = weight.clone()
        clamp_weight_(weight, "int1_binary", 0.0)
        assert torch.equal(weight, before)

    def test_it_is_on_by_default(self):
        assert QATConfig().clamp > 0
        layer = prepare_qat(nn.Sequential(nn.Linear(BLOCK_SIZE, BLOCK_SIZE)))[0][0]
        assert layer.clamp > 0

    def test_it_keeps_the_scale_from_running_away(self):
        """The mechanism, isolated: repeated STE steps with no clamp
        inflate the block scale; with the clamp they do not."""
        def drift(clamp):
            torch.manual_seed(0)
            weight = (torch.randn(BLOCK_SIZE * 4) * 0.05).requires_grad_(True)
            opt = torch.optim.Adam([weight], lr=0.02)
            for _ in range(200):
                if clamp:
                    clamp_weight_(weight, "int1_binary", 1.5)
                # a loss that rewards large weights, which is what an
                # unconstrained STE effectively permits
                loss = -fake_quantize(weight, "int1_binary").abs().sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            return float(block_scales(weight, "int1_binary").mean())

        assert drift(clamp=True) < drift(clamp=False)


class TestPreparingAModel:
    def _model(self):
        return nn.Sequential(
            nn.Linear(BLOCK_SIZE, BLOCK_SIZE), nn.ReLU(),
            nn.Linear(BLOCK_SIZE, 128), nn.Linear(100, 10),
        )

    def test_eligible_layers_are_wrapped(self):
        _model, report = prepare_qat(self._model(), tier="HNX_1375BIT")
        assert len(report.wrapped) == 2
        assert report.parameters_quantised > 0

    def test_a_row_shorter_than_a_block_is_left_alone(self):
        """The file format's rule. Simulating a layer the quantiser will
        decline to pack teaches the model to compensate for damage that
        never happens."""
        _model, report = prepare_qat(self._model())
        assert any("under one" in reason for _n, reason in report.skipped)

    def test_embeddings_and_the_head_are_skipped_by_name(self):
        model = nn.ModuleDict({
            "token_embd": nn.Linear(BLOCK_SIZE, BLOCK_SIZE),
            "lm_head": nn.Linear(BLOCK_SIZE, BLOCK_SIZE),
            "body": nn.Linear(BLOCK_SIZE, BLOCK_SIZE),
        })
        _model, report = prepare_qat(model)
        assert report.wrapped == ["body"]

    def test_a_row_that_does_not_divide_is_skipped(self):
        model = nn.Sequential(nn.Linear(BLOCK_SIZE + 1, BLOCK_SIZE))
        _model, report = prepare_qat(model)
        assert report.wrapped == []
        assert "do not divide" in report.skipped[0][1]

    def test_an_unknown_tier_says_which_exist(self):
        with pytest.raises(QATError, match="Unknown tier"):
            prepare_qat(self._model(), tier="IQ9.9_ENORMOUS")

    def test_a_codebook_tier_is_refused_with_a_reason(self):
        """INT4 and FP2 are fixed codebooks, not sign-and-scale."""
        with pytest.raises(QATError, match="fixed codebook"):
            prepare_qat(self._model(), tier="INT4")

    def test_finalize_gives_back_plain_linears(self):
        model, _ = prepare_qat(self._model())
        model = finalize_qat(model)
        assert all(
            not type(m).__name__.startswith("QAT") for m in model.modules()
        )

    def test_finalize_keeps_the_float_weight_not_the_quantised_one(self):
        """The float weight is what hyprslug should quantise, and it
        carries more than the packed form can. Folding the fake quant in
        here would throw the training away."""
        model, _ = prepare_qat(self._model())
        before = model[0].weight.detach().clone()
        model = finalize_qat(model)
        assert torch.equal(model[0].weight.detach(), before)

    def test_the_report_says_when_nothing_was_wrapped(self):
        _model, report = prepare_qat(nn.Sequential(nn.Linear(8, 8)))
        assert "ordinary training" in report.describe()

    def test_a_wrapped_model_still_runs(self):
        model, _ = prepare_qat(self._model())
        out = model[:3](torch.randn(4, BLOCK_SIZE))
        assert out.shape == (4, 128)
        assert torch.isfinite(out).all()

    def test_the_quantize_flag_turns_it_off(self):
        """A warm-up in plain float: a randomly initialised model
        quantises to noise, and the first steps can be spent fighting
        that rather than learning."""
        model, _ = prepare_qat(nn.Sequential(nn.Linear(BLOCK_SIZE, BLOCK_SIZE)))
        x = torch.randn(2, BLOCK_SIZE)
        model[0].quantize = False
        plain = model(x)
        model[0].quantize = True
        assert not torch.allclose(plain, model(x))


class TestItActuallyHelps:
    """The claim, measured. A QAT that does not beat post-training
    quantisation is not worth the training budget, and this suite would
    rather fail than let that ship unnoticed."""

    PACKING = "int1_binary"
    TIER = "INT1"

    def _task(self):
        torch.manual_seed(0)
        width = BLOCK_SIZE
        teacher = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                                nn.Linear(width, width))
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        train_x, valid_x = torch.randn(1024, width), torch.randn(256, width)
        with torch.no_grad():
            return train_x, teacher(train_x), valid_x, teacher(valid_x)

    def _build(self):
        torch.manual_seed(1)
        return nn.Sequential(nn.Linear(BLOCK_SIZE, BLOCK_SIZE), nn.Tanh(),
                             nn.Linear(BLOCK_SIZE, BLOCK_SIZE))

    def _train(self, model, x, y, steps=400):
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(steps):
            index = torch.randint(0, len(x), (128,))
            loss = nn.functional.mse_loss(model(x[index]), y[index])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def _quantised_loss(self, model, valid_x, valid_y):
        copy = self._build()
        copy.load_state_dict(model.state_dict())
        with torch.no_grad():
            for layer in copy:
                if isinstance(layer, nn.Linear):
                    layer.weight.copy_(fake_quantize(layer.weight, self.PACKING))
            return float(nn.functional.mse_loss(copy(valid_x), valid_y))

    def test_qat_beats_quantising_after_the_fact(self):
        x, y, vx, vy = self._task()
        plain = self._quantised_loss(self._train(self._build(), x, y), vx, vy)
        model, _ = prepare_qat(self._build(), tier=self.TIER)
        trained = self._quantised_loss(
            finalize_qat(self._train(model, x, y)), vx, vy
        )
        assert trained < plain, (
            f"QAT ({trained:.4f}) did not beat post-training quantisation "
            f"({plain:.4f}) -- it is costing a training run for nothing"
        )

    def test_and_clamping_is_why(self):
        x, y, vx, vy = self._task()
        model, _ = prepare_qat(self._build(), config=QATConfig(tier=self.TIER))
        with_clamp = self._quantised_loss(
            finalize_qat(self._train(model, x, y)), vx, vy
        )
        off, _ = prepare_qat(
            self._build(), config=QATConfig(tier=self.TIER, clamp=0.0)
        )
        without = self._quantised_loss(
            finalize_qat(self._train(off, x, y)), vx, vy
        )
        assert with_clamp < without


class TestTheErrorReadout:
    def test_it_reports_what_it_says(self):
        weight = torch.randn(BLOCK_SIZE * 2) * 0.1
        stats = quantization_error(weight, "hnx_1375bit")
        assert 0.0 <= stats["signs_kept"] <= 1.0
        assert stats["rmse"] >= 0
        assert stats["relative"] >= 0

    def test_every_sign_survives_the_tiers_that_keep_them_all(self):
        weight = torch.randn(BLOCK_SIZE * 2) * 0.1
        for packing in ("int1_binary", "hnx_1375bit"):
            assert quantization_error(weight, packing)["signs_kept"] == 1.0

    def test_the_new_tier_beats_int1_on_structured_weights(self):
        """The reason it exists: INT1 reconstructs all 256 weights at one
        magnitude, this one at sixteen, and a row whose magnitudes vary
        is the normal case."""
        rng = np.random.default_rng(0)
        weight = torch.from_numpy(np.concatenate([
            rng.normal(0, s, 16) for s in
            [.01,.3,.02,.15,.05,.4,.01,.2,.08,.02,.3,.01,.12,.25,.03,.18]
        ]).astype(np.float32))
        assert (quantization_error(weight, "hnx_1375bit")["rmse"]
                < quantization_error(weight, "int1_binary")["rmse"])
