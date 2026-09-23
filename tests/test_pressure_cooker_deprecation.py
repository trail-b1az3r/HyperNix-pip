"""Pressure Cooker v1 and v3 deprecated; v4 kept — and made to work.

What these tests are for
------------------------
A deprecation has two ways to go wrong. It can be invisible — a warning
category the default filters hide — so nobody learns anything. Or it
can be loud in the wrong place: fired on import, it tells every V4 user
their V3 is deprecated (V4 imports V3's helpers), and fires for people
who never touched a Pressure Cooker because the package loader imports
these modules for their flat-name aliases.

The third test class is the one that matters most. Moving
`instant_pot`'s default off the deprecated V3 meant moving it onto V4,
and **V4 had never worked through `NeoOven.train`**: `train` passes a
bare `lr` or `peak_lr`, and V4 accepted a rate only inside a
ScheduleConfig, so it raised TypeError. That is now fixed, and tested
end to end rather than at the constructor.
"""
from __future__ import annotations

import importlib
import warnings

import pytest

torch = pytest.importorskip("torch")

from hypernix.optimizers.optimizer_framework import ScheduleConfig  # noqa: E402
from hypernix.optimizers.pressure_cooker_v4 import PressureCookerV4  # noqa: E402


def params():
    return torch.nn.Linear(4, 4).parameters()


def constructed_warnings(factory) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factory()
    return [w for w in caught if "deprecated" in str(w.message)]


class TestTheDeprecatedOnes:
    def test_v1_warns_on_construction(self):
        from hypernix.optimizers.pressure_cooker import PressureCooker

        found = constructed_warnings(lambda: PressureCooker(params()))
        assert len(found) == 1
        assert "V1" in str(found[0].message)

    def test_the_router_warns_only_on_its_legacy_path(self):
        """UniversalCooker routes: its default returns the current V5
        family, and only `variant="legacy"` reaches the V1 tiers.
        Deprecating the router itself would warn people sent to V5."""
        from hypernix.optimizers.pressure_cooker import UniversalCooker

        assert constructed_warnings(lambda: UniversalCooker.select(params())) == []
        legacy = constructed_warnings(
            lambda: UniversalCooker.select(params(), variant="legacy"))
        assert len(legacy) == 1

    def test_v3_warns_on_construction(self):
        from hypernix.optimizers.pressure_cooker_v3 import PressureCookerV3

        found = constructed_warnings(lambda: PressureCookerV3(params()))
        assert len(found) == 1
        assert "V3" in str(found[0].message)

    def test_a_subclass_warns_once_not_twice(self):
        """One construction through super() is one optimizer. Two
        warnings would read as two."""
        from hypernix.optimizers.pressure_cooker_v3 import StovetopV3Cooker

        found = constructed_warnings(lambda: StovetopV3Cooker(params()))
        assert len(found) == 1
        assert "StovetopV3Cooker" in str(found[0].message)

    def test_the_warning_names_the_replacement(self):
        from hypernix.optimizers.pressure_cooker import PressureCooker

        message = str(constructed_warnings(lambda: PressureCooker(params()))[0].message)
        assert "PressureCookerV4" in message

    def test_it_is_a_future_warning(self):
        """DeprecationWarning is hidden by default unless raised from
        `__main__`, and training runs call the optimizer from a module.
        A deprecation nobody sees has not been announced."""
        from hypernix.optimizers.pressure_cooker import PressureCooker

        found = constructed_warnings(lambda: PressureCooker(params()))
        assert found[0].category is FutureWarning

    def test_importing_does_not_warn(self):
        """V4 imports V3's helpers, and the package loader imports these
        modules for their aliases. An import-time warning would reach
        people who never used either."""
        import hypernix.optimizers.pressure_cooker as v1
        import hypernix.optimizers.pressure_cooker_v3 as v3

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(v3)
            importlib.reload(v1)
        assert [w for w in caught if "deprecated" in str(w.message)] == []

    def test_they_are_marked(self):
        from hypernix.optimizers.pressure_cooker import PressureCooker
        from hypernix.optimizers.pressure_cooker_v3 import PressureCookerV3

        assert PressureCooker.__hnx_deprecated__ == "v1"
        assert PressureCookerV3.__hnx_deprecated__ == "v3"

    def test_there_is_no_v2_to_deprecate(self):
        """No PressureCookerV2 has ever existed here — the old v1
        docstring listed one by mistake (wiki/Optimizers.md). Inventing
        a class to deprecate would be worse than saying so."""
        from hypernix.optimizers.deprecation import DEPRECATED_GENERATIONS

        assert "v2" not in DEPRECATED_GENERATIONS
        import hypernix.optimizers as pkg
        assert not any("V2" in name for name in dir(pkg) if name.startswith("PressureCooker"))


class TestV4IsKept:
    def test_v4_does_not_warn(self):
        assert constructed_warnings(lambda: PressureCookerV4(params(), lr=1e-3)) == []

    def test_v4_is_not_marked_deprecated(self):
        assert not hasattr(PressureCookerV4, "__hnx_deprecated__")


class TestV4AcceptsARate:
    """The bug found while moving instant_pot's default onto V4."""

    @pytest.mark.parametrize("key", ["lr", "peak_lr"])
    def test_a_bare_rate(self, key):
        opt = PressureCookerV4(params(), **{key: 1.234e-2})
        assert opt._schedule.lr == pytest.approx(1.234e-2)

    def test_an_explicit_rate_keeps_the_schedules_shape(self):
        opt = PressureCookerV4(params(), lr=5e-3,
                               schedule=ScheduleConfig(lr=1.0, warmup_steps=7))
        assert opt._schedule.lr == pytest.approx(5e-3)
        assert opt._schedule.warmup_steps == 7

    def test_two_rates_that_disagree_are_refused(self):
        with pytest.raises(ValueError):
            PressureCookerV4(params(), lr=1e-3, peak_lr=2e-3)

    def test_no_rate_keeps_the_old_default(self):
        assert PressureCookerV4(params())._schedule.lr == ScheduleConfig().lr


def _optimizer(dotted: str):
    module, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(f"hypernix.optimizers.{module}"), name)


class TestTheWholeBaseClassFamilyThroughTrain:
    """OptimizerBase never seeded `group["lr"]`, and every PyTorch
    scheduler reads it at construction. `NeoOven.train` always wraps the
    optimizer in CosineAnnealingLR — so this was every OptimizerBase
    optimizer, not just V4."""

    @pytest.mark.parametrize("dotted", [
        "pressure_cooker_v4.PressureCookerV4",
        "pressure_cooker_v5.PressureCookerV5",
        "pressure_cooker_v5s.PressureCookerV5S",
        "pressure_cooker_v6.PressureCookerV6",
    ])
    def test_it_trains(self, tmp_path, dotted):
        from hypernix.models.neo_oven import new_oven

        oven = new_oven(
            tmp_path / "oven", hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, vocab_size=256,
            max_position_embeddings=64, seed=0, device="cpu",
        )
        data = tmp_path / "data.txt"
        data.write_text("hello world, this is a tiny corpus. " * 40)
        oven.train(data, tmp_path / "out", steps=2, batch_size=1,
                   context_length=32, lr=1e-3, log_every=100, save_every=0,
                   quiet=True, optimizer_class=_optimizer(dotted))

    def test_every_group_has_an_lr(self):
        opt = PressureCookerV4(params(), lr=2e-3)
        assert all(g["lr"] == pytest.approx(2e-3) for g in opt.param_groups)


class TestV4ThroughTrain:
    def test_neo_oven_trains_with_v4(self, tmp_path):
        """End to end, through the exact path that raised TypeError.
        A constructor test would have passed while this was broken —
        `train` builds the optimizer from its own kwargs."""
        from hypernix.models.neo_oven import new_oven

        oven = new_oven(
            tmp_path / "oven", hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, vocab_size=256,
            max_position_embeddings=64, seed=0, device="cpu",
        )
        data = tmp_path / "data.txt"
        data.write_text("hello world, this is a tiny corpus. " * 40)
        out = oven.train(data, tmp_path / "out", steps=3, batch_size=1,
                         context_length=32, lr=1e-3, log_every=100,
                         save_every=0, quiet=True,
                         optimizer_class=PressureCookerV4)
        assert out is not None

    def test_instant_pot_defaults_to_v4(self):
        """A default HyperNix chose must not produce a warning the user
        has to act on."""
        import inspect

        from hypernix.training import instant_pot

        source = inspect.getsource(instant_pot)
        assert 'recipe.get("use_pressure_cooker_v3", False)' in source
        assert "opt_class = PressureCookerV4" in source
