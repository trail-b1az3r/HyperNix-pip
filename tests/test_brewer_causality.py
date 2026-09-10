"""Brewer attention must not let a token see its own future.

From a report against `BrewerAttention.forward`, which combined the
causal mask and the sliding-window mask with ``torch.maximum``. Both are
*additive* masks -- ``0`` allows, ``finfo.min`` forbids -- so the
elementwise maximum keeps the **less** masked of the two: "allow if
either allows", when the requirement is "mask if either masks".

The failure is quiet in the worst way. A non-causal language model
trains to an excellent loss, because predicting a token it can already
see is easy, and then generates nothing usable. Nothing in a loss curve
says which one you have.

Two things were wrong and they were coupled, so neither could be fixed
alone:

* ``_sliding_mask`` used ``dist = i - j`` (positive = past) but masked
  ``dist >= 0``, so it kept the strictly *future* positions inside the
  window -- an anti-causal band.
* ``maximum`` then unioned that band with the causal mask.

Switching only the combinator masks every position of every row and
sends softmax to NaN; fixing only the mask leaves ``maximum`` making the
window a no-op identical to plain causal attention. The tests below pin
both halves and, more importantly, the property itself.
"""
from __future__ import annotations

import pytest
import torch

from hypernix.training.brewer import BrewerAttention, BrewerConfig

D_MODEL, N_HEADS, N_KV, T = 32, 4, 2, 16


def _cfg(*, window: int = 4, use_sliding_window: bool = True) -> BrewerConfig:
    return BrewerConfig(
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV, d_ff=64,
        n_layers=2, vocab_size=32, dropout=0.0,
        use_sliding_window=use_sliding_window, sliding_window_size=window,
    )


def _moved_by_perturbing(layer, position: int, *, seq: int = T) -> list[int]:
    """Which output positions change when one input token changes.

    The direct test of causality, and the one the reporter's harness
    used: a causal layer can only propagate a change forward, so any
    output *before* the perturbed token that moves is a leak. It tests
    the property rather than the mask's spelling, so it stays true if the
    masking is ever rewritten.
    """
    torch.manual_seed(0)
    x = torch.randn(1, seq, D_MODEL)
    with torch.no_grad():
        before = layer(x)
        perturbed = x.clone()
        perturbed[0, position] += 5.0
        after = layer(perturbed)
    return [
        i for i in range(seq)
        if not torch.allclose(before[0, i], after[0, i], atol=1e-6)
    ]


class TestNothingSeesItsOwnFuture:
    @pytest.mark.parametrize("layer_idx", [0, 1])
    @pytest.mark.parametrize("window", [4, 8, 64])
    def test_a_perturbation_never_moves_an_earlier_output(self, layer_idx, window):
        """Layer 1 is the sliding-window layer; layer 0 is plain causal.
        Both have to be causal, and the bug was only on the odd one."""
        layer = BrewerAttention(_cfg(window=window), layer_idx=layer_idx).eval()
        moved = _moved_by_perturbing(layer, 10)
        assert all(i >= 10 for i in moved), f"leaked into the past: {moved}"

    def test_the_perturbed_position_does_move(self):
        """Otherwise "nothing earlier moved" would pass on a layer that
        ignores its input entirely."""
        layer = BrewerAttention(_cfg(window=4), layer_idx=1).eval()
        assert 10 in _moved_by_perturbing(layer, 10)

    @pytest.mark.parametrize("position", [0, 1, 5, 15])
    def test_it_holds_wherever_the_perturbation_lands(self, position):
        layer = BrewerAttention(_cfg(window=4), layer_idx=1).eval()
        moved = _moved_by_perturbing(layer, position)
        assert all(i >= position for i in moved)

    def test_the_default_window_with_a_short_sequence(self):
        """The configured default is 4096. Every sequence shorter than
        the window made ``dist < -(win - 1)`` unreachable, so the window
        mask forbade nothing and the union with causal was *fully
        bidirectional* -- not merely leaky. That is the shape every
        preset in this module ships with."""
        layer = BrewerAttention(_cfg(window=4096), layer_idx=1).eval()
        moved = _moved_by_perturbing(layer, 10)
        assert all(i >= 10 for i in moved), f"fully bidirectional: {moved}"


class TestTheWindowStillDoesSomething:
    """A causal layer is easy to get by disabling the window. These
    assert the feature survived the fix."""

    def test_a_small_window_limits_how_far_influence_reaches(self):
        window = 4
        layer = BrewerAttention(_cfg(window=window), layer_idx=1).eval()
        moved = _moved_by_perturbing(layer, 4)
        assert max(moved) == 4 + window - 1, moved

    def test_a_wider_window_reaches_further(self):
        narrow = BrewerAttention(_cfg(window=3), layer_idx=1).eval()
        wide = BrewerAttention(_cfg(window=6), layer_idx=1).eval()
        assert max(_moved_by_perturbing(narrow, 4)) < max(_moved_by_perturbing(wide, 4))

    def test_an_even_layer_has_no_window(self):
        """Only odd layers are local; an even one must reach the end."""
        layer = BrewerAttention(_cfg(window=3), layer_idx=0).eval()
        assert max(_moved_by_perturbing(layer, 4)) == T - 1

    def test_the_window_is_not_a_silent_no_op(self):
        """Fixing the mask while keeping ``maximum`` produces attention
        identical to plain causal: the window forbids a subset of what
        causal already forbids, so a union changes nothing. That version
        passes every causality test above."""
        windowed = BrewerAttention(_cfg(window=3), layer_idx=1).eval()
        plain = BrewerAttention(_cfg(window=3, use_sliding_window=False), layer_idx=1).eval()
        plain.load_state_dict(windowed.state_dict())
        torch.manual_seed(0)
        x = torch.randn(1, T, D_MODEL)
        with torch.no_grad():
            assert not torch.allclose(windowed(x), plain(x), atol=1e-5)


class TestTheMaskItself:
    """Read off the tensor, so a failure says which cell is wrong rather
    than only that some output moved."""

    def _allowed(self, layer):
        mask = layer._sliding_mask(T, torch.device("cpu"), torch.float32)
        return mask > torch.finfo(torch.float32).min / 2

    def test_the_window_mask_is_causal_on_its_own(self):
        """It is called a *causal* sliding-window mask. It was not one:
        it kept exactly the future positions and dropped the past."""
        allowed = self._allowed(BrewerAttention(_cfg(window=4), layer_idx=1))
        assert not allowed.triu(diagonal=1).any(), "window mask allows the future"

    def test_every_query_can_see_itself(self):
        allowed = self._allowed(BrewerAttention(_cfg(window=4), layer_idx=1))
        assert allowed.diagonal().all()

    def test_it_reaches_exactly_window_minus_one_into_the_past(self):
        window = 5
        allowed = self._allowed(BrewerAttention(_cfg(window=window), layer_idx=1))
        for i in range(T):
            expected = {j for j in range(T) if 0 <= i - j <= window - 1}
            assert {j for j in range(T) if allowed[i, j]} == expected, i

    def test_no_row_is_entirely_masked(self):
        """``torch.minimum`` against the old anti-causal band masked
        every cell, which is a NaN out of softmax rather than a wrong
        answer -- a different bug that a naive fix walks straight into."""
        window = 4
        layer = BrewerAttention(_cfg(window=window), layer_idx=1)
        neg = torch.finfo(torch.float32).min
        causal = torch.full((T, T), neg).triu(diagonal=1)
        combined = torch.minimum(
            causal, layer._sliding_mask(T, torch.device("cpu"), torch.float32)
        )
        assert (combined > neg / 2).any(dim=-1).all()

    def test_the_output_is_finite(self):
        layer = BrewerAttention(_cfg(window=4), layer_idx=1).eval()
        torch.manual_seed(0)
        with torch.no_grad():
            out = layer(torch.randn(1, T, D_MODEL))
        assert torch.isfinite(out).all()


class TestThePlainCausalFastPath:
    """An even layer with no padding mask hands SDPA ``is_causal=True``
    instead of a mask tensor, so it can use a fused kernel rather than
    materialising B*H*T*T scores to add a mask to."""

    def test_it_takes_the_is_causal_path(self, monkeypatch):
        import hypernix.training.brewer as brewer

        seen = {}
        real = brewer.F.scaled_dot_product_attention

        def spy(q, k, v, **kwargs):
            seen.update(kwargs)
            return real(q, k, v, **kwargs)

        monkeypatch.setattr(brewer.F, "scaled_dot_product_attention", spy)
        layer = BrewerAttention(_cfg(use_sliding_window=False), layer_idx=0).eval()
        with torch.no_grad():
            layer(torch.randn(1, T, D_MODEL))
        assert seen.get("is_causal") is True
        assert seen.get("attn_mask") is None

    def test_a_padding_mask_falls_back_to_an_explicit_mask(self):
        """``is_causal`` and ``attn_mask`` are mutually exclusive, so a
        caller-supplied mask has to take the other path."""
        import hypernix.training.brewer as brewer

        seen = {}
        real = brewer.F.scaled_dot_product_attention

        def spy(q, k, v, **kwargs):
            seen.update(kwargs)
            return real(q, k, v, **kwargs)

        layer = BrewerAttention(_cfg(use_sliding_window=False), layer_idx=0).eval()
        brewer.F.scaled_dot_product_attention = spy
        try:
            with torch.no_grad():
                layer(torch.randn(1, T, D_MODEL), attn_mask=torch.zeros(T, T))
        finally:
            brewer.F.scaled_dot_product_attention = real
        assert not seen.get("is_causal")
        assert seen.get("attn_mask") is not None

    def test_the_fast_path_agrees_with_the_masked_one(self):
        """The optimisation must not change the answer."""
        layer = BrewerAttention(_cfg(use_sliding_window=False), layer_idx=0).eval()
        torch.manual_seed(0)
        x = torch.randn(1, T, D_MODEL)
        with torch.no_grad():
            fast = layer(x)
            explicit = layer(x, attn_mask=torch.zeros(T, T))
        assert torch.allclose(fast, explicit, atol=1e-5)

    def test_the_fast_path_is_still_causal(self):
        layer = BrewerAttention(_cfg(use_sliding_window=False), layer_idx=0).eval()
        assert all(i >= 6 for i in _moved_by_perturbing(layer, 6))
