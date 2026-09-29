"""Regression tests for the v0.51.1 patch (5 bug fixes found across
1 by-hand source-read pass + 2 hand-driven testing passes).

Bugs fixed:

1. ``bell.Bell._iter_from_ids`` — the stop-sequence check ran AFTER
   yielding the offending token, leaking the marker (e.g.
   ``"<|im_end|>"``) into the streamed output.
2. ``countertop.Countertop._trim`` — under-budget trimming could
   wipe the freshly-appended user turn (and any trailing element)
   when ``max_history_tokens`` was set very small.
3. ``cookbook._HYPER_NIX_2`` shared the SAME ``role_prefixes`` /
   ``role_suffixes`` dict objects as ``_CHATML``; mutating one
   silently corrupted the other.
4. ``flour.Flour.process`` raised
   ``RuntimeError: Boolean value of Tensor with more than one
   value is ambiguous`` when called with a torch.Tensor
   ``produced_ids`` (because of the ``if produced_ids`` truthiness
   check).
5. ``pressure_cooker.UniversalCooker.select`` always returned
   ``ProCooker`` / ``InductionCooker`` with ``fused=True`` on any
   CUDA device, breaking Pascal GPUs (sm_61, e.g. GTX 1080) where
   fused AdamW and CUDA graphs require sm_70+.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from hypernix.bell import Bell
from hypernix.cookbook import COOKBOOK
from hypernix.countertop import Countertop
from hypernix.flour import Flour
from hypernix.pressure_cooker import UniversalCooker, _is_pre_volta

# ---------------------------------------------------------------------------
# Shared stub oven that doesn't need real weights
# ---------------------------------------------------------------------------

class _StopOven:
    """Stub oven whose model always produces the same token id and
    whose decoder maps id 5 → ``"END"``.  Used to verify the bell
    stop-sequence behaviour deterministically."""

    tokenizer_kind = "byte"
    tokenizer = None
    device = torch.device("cpu")
    repo_id = None

    def __init__(self) -> None:
        class _M(torch.nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.linear = torch.nn.Linear(8, 8)  # unused, just to be a Module

            def forward(self_inner, x: torch.Tensor):  # noqa: N805
                bsz, seq = x.shape
                logits = torch.zeros(bsz, seq, 16)
                logits[..., 5] = 100.0  # argmax always picks 5
                return type("O", (), {"logits": logits})()

        self.model = _M()
        self.model.eval()

    def _encode(self, text: str) -> list[int]:
        return [1]

    def _decode(self, ids: list[int]) -> str:
        return "".join("END" if int(i) == 5 else chr(int(i) + 65) for i in ids)

    def _format_chat(self, messages: list[dict[str, str]]) -> list[int]:
        return [1]


# ---------------------------------------------------------------------------
# Bug 1 — bell stop-marker leak
# ---------------------------------------------------------------------------

class TestBellStopMarkerLeak:
    def test_stop_marker_does_not_appear_in_streamed_tokens(self) -> None:
        oven = _StopOven()
        bell = Bell(flour=Flour(stop_sequences=["END"]))
        toks = list(bell.iter_complete(oven, "hi", max_new_tokens=5, temperature=0.0))
        full = "".join(toks)
        assert "END" not in full, f"stop marker leaked: {toks!r}"

    def test_stop_marker_does_not_fire_token_callback(self) -> None:
        oven = _StopOven()
        seen: list[str] = []
        bell = Bell(flour=Flour(stop_sequences=["END"]))
        bell.on_token(lambda tok, _idx: seen.append(tok))
        list(bell.iter_complete(oven, "hi", max_new_tokens=5, temperature=0.0))
        assert "END" not in "".join(seen)

    def test_done_callback_still_fires_with_clean_text(self) -> None:
        oven = _StopOven()
        full: list[str] = []
        bell = Bell(flour=Flour(stop_sequences=["END"]))
        bell.on_done(lambda s: full.append(s))
        bell.stream_complete(oven, "hi", max_new_tokens=5, temperature=0.0)
        assert full and "END" not in full[0]


# ---------------------------------------------------------------------------
# Bug 2 — countertop trim wiping user turn
# ---------------------------------------------------------------------------

class _CapturingOven:
    repo_id = None

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:  # noqa: ARG002
        self.last_messages = messages
        return "reply"


class TestCountertopTrimPreservesLastUser:
    def test_freshly_added_user_survives_aggressive_trim(self) -> None:
        oven = _CapturingOven()
        ct = Countertop(oven, system="SYS", template="plain", max_history_tokens=5)
        ct.history = [
            {"role": "user", "content": "OLD_U"},
            {"role": "assistant", "content": "OLD_A"},
        ]
        ct.say("NEW")
        assert oven.last_messages is not None
        roles = [m["role"] for m in oven.last_messages]
        assert "user" in roles
        contents = [m["content"] for m in oven.last_messages if m["role"] == "user"]
        assert "NEW" in contents

    def test_trim_never_drops_below_one_history_element(self) -> None:
        oven = _CapturingOven()
        ct = Countertop(oven, system="X" * 200, template="plain", max_history_tokens=10)
        ct.say("hello")
        # The freshly-added user message must remain after trim.
        assert any(m["role"] == "user" for m in ct.history)


# ---------------------------------------------------------------------------
# Bug 3 — cookbook aliasing
# ---------------------------------------------------------------------------

class TestCookbookNoAliasing:
    def test_chatml_and_hyper_nix_2_have_independent_role_prefix_dicts(self) -> None:
        chatml = COOKBOOK.get("chatml")
        hyper = COOKBOOK.get("hyper-nix.2")
        assert chatml.role_prefixes is not hyper.role_prefixes
        assert chatml.role_suffixes is not hyper.role_suffixes

    def test_mutating_chatml_does_not_corrupt_hyper_nix_2(self) -> None:
        chatml = COOKBOOK.get("chatml")
        hyper = COOKBOOK.get("hyper-nix.2")
        before = hyper.role_prefixes["user"]
        # Cast away the read-only Mapping hint to actually mutate.
        chatml.role_prefixes["user"] = "CORRUPTED: "  # type: ignore[index]
        try:
            assert hyper.role_prefixes["user"] == before
        finally:
            chatml.role_prefixes["user"] = before  # type: ignore[index]


# ---------------------------------------------------------------------------
# Bug 4 — flour with non-list produced_ids
# ---------------------------------------------------------------------------

class TestFlourAcceptsTensorProducedIds:
    def test_process_accepts_torch_tensor(self) -> None:
        f = Flour(repetition_penalty=1.5)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = f.process(logits, torch.tensor([1, 2]))
        # Token 1 was logit 2.0 (positive) → /= 1.5 ≈ 1.333
        assert out[1].item() == pytest.approx(2.0 / 1.5)

    def test_process_accepts_generator(self) -> None:
        f = Flour(repetition_penalty=1.5)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = f.process(logits, (i for i in (1, 2)))  # one-shot generator
        assert out[1].item() == pytest.approx(2.0 / 1.5)

    def test_process_with_empty_tensor_is_noop(self) -> None:
        f = Flour(repetition_penalty=1.5, frequency_penalty=0.5)
        logits = torch.tensor([1.0, 2.0, 3.0])
        out = f.process(logits, torch.tensor([], dtype=torch.long))
        assert torch.allclose(out, logits)


# ---------------------------------------------------------------------------
# Bug 5 — pressure_cooker Pascal awareness
# ---------------------------------------------------------------------------

class TestUniversalCookerPascalAware:
    def test_is_pre_volta_returns_false_on_cpu(self) -> None:
        assert _is_pre_volta(torch.device("cpu")) is False

    def test_is_pre_volta_handles_no_cuda(self) -> None:
        # Whatever the host has, calling with a CUDA device while CUDA
        # is unavailable must not raise.
        assert _is_pre_volta(torch.device("cuda")) in (True, False)

    def test_select_on_cpu_picks_a_cpu_tier_under_legacy_variant(self) -> None:
        params = [torch.nn.Parameter(torch.randn(4, 4))]
        cooker = UniversalCooker.select(params, prefer_speed=True, variant="legacy")
        # Should be one of the CPU tiers.
        from hypernix.pressure_cooker import ElectricCooker, StovetopCooker
        assert isinstance(cooker, ElectricCooker | StovetopCooker)

    def test_select_pascal_path_forces_fused_off_under_legacy_variant(self, monkeypatch) -> None:
        """Simulate a Pascal CUDA device by stubbing _is_pre_volta and
        verifying the legacy selector forces fused=False."""
        from hypernix import pressure_cooker as pc

        monkeypatch.setattr(pc, "_is_pre_volta", lambda _dev: True)

        # Stand-in param whose .device.type reads "cuda" without
        # actually placing anything on a GPU.
        class _Param:
            device = type("D", (), {"type": "cuda"})

        captured: dict[str, Any] = {}

        class _SpyInduction(pc.InductionCooker):
            def __init__(self_inner, params, **kwargs):  # noqa: N805
                captured["fused"] = kwargs.get("fused")
                captured["foreach"] = kwargs.get("foreach")
                # Skip the real Optimizer init (we have no real CUDA params).
                self_inner._captured_kwargs = kwargs

        monkeypatch.setattr(pc, "InductionCooker", _SpyInduction)
        UniversalCooker.select.__func__(UniversalCooker, [_Param()], prefer_speed=True, variant="legacy")
        assert captured.get("fused") is False
        # Foreach is whatever _HAS_FOREACH evaluates to on this torch.
        assert "foreach" in captured


class TestUniversalCookerV5Family:
    """Patch (0.71.1): UniversalCooker now defaults to the V5 / V5S
    optimizer family instead of the legacy device tiers; Pascal is
    still auto-detected and routed to the matching Aged* tier."""

    def test_default_variant_on_cpu_is_v5s(self) -> None:
        from hypernix.pressure_cooker_v5s import PressureCookerV5S

        params = [torch.nn.Parameter(torch.randn(4, 4))]
        cooker = UniversalCooker.select(params)
        assert isinstance(cooker, PressureCookerV5S)

    def test_variant_v5_selects_pressure_cooker_v5(self) -> None:
        from hypernix.pressure_cooker_v5 import PressureCookerV5

        params = [torch.nn.Parameter(torch.randn(4, 4))]
        cooker = UniversalCooker.select(params, variant="v5")
        assert isinstance(cooker, PressureCookerV5)

    def test_variant_v5_plus_selects_pressure_cooker_v5_plus(self) -> None:
        from hypernix.pressure_cooker_v5 import PressureCookerV5Plus

        params = [torch.nn.Parameter(torch.randn(4, 4))]
        cooker = UniversalCooker.select(params, variant="v5-plus")
        assert isinstance(cooker, PressureCookerV5Plus)

    def test_unknown_variant_raises(self) -> None:
        params = [torch.nn.Parameter(torch.randn(2, 2))]
        with pytest.raises(ValueError, match="unknown variant"):
            UniversalCooker.select(params, variant="bogus")

    def test_pascal_device_selects_aged_v5s(self, monkeypatch) -> None:
        from hypernix import pressure_cooker as pc
        from hypernix.pressure_cooker_v5s import Agedcookerv5s

        monkeypatch.setattr(pc, "_is_pre_volta", lambda _dev: True)
        params = [torch.nn.Parameter(torch.randn(4, 4))]
        with pytest.warns(UserWarning, match="Pascal"):
            cooker = UniversalCooker.select(params, variant="v5s")
        assert isinstance(cooker, Agedcookerv5s)

    def test_pascal_device_selects_aged_v5(self, monkeypatch) -> None:
        from hypernix import pressure_cooker as pc
        from hypernix.pressure_cooker_v5 import Agedcookerv5

        monkeypatch.setattr(pc, "_is_pre_volta", lambda _dev: True)
        params = [torch.nn.Parameter(torch.randn(4, 4))]
        with pytest.warns(UserWarning, match="Pascal"):
            cooker = UniversalCooker.select(params, variant="v5")
        assert isinstance(cooker, Agedcookerv5)

    def test_pascal_device_selects_ultra_aged_v5_plus(self, monkeypatch) -> None:
        from hypernix import pressure_cooker as pc
        from hypernix.pressure_cooker_v5 import ULTRAagedcookerv5

        monkeypatch.setattr(pc, "_is_pre_volta", lambda _dev: True)
        params = [torch.nn.Parameter(torch.randn(4, 4))]
        with pytest.warns(UserWarning, match="Pascal"):
            cooker = UniversalCooker.select(params, variant="v5-plus")
        assert isinstance(cooker, ULTRAagedcookerv5)

    def test_universal_cooker_function_defaults_to_v5s(self) -> None:
        from hypernix.pressure_cooker import universal_cooker
        from hypernix.pressure_cooker_v5s import PressureCookerV5S

        params = [torch.nn.Parameter(torch.randn(3, 3))]
        assert isinstance(universal_cooker(params), PressureCookerV5S)

    def test_universal_cooker_function_legacy_opt_out_still_works(self) -> None:
        from hypernix.pressure_cooker import ElectricCooker, StovetopCooker, universal_cooker

        params = [torch.nn.Parameter(torch.randn(3, 3))]
        cooker = universal_cooker(params, variant="legacy")
        assert isinstance(cooker, ElectricCooker | StovetopCooker)
