"""Regression tests for the v0.51.4 patch:

``CodeOven.chat`` used to crash with::

    ValueError: too many dimensions 'str'

deep inside ``torch.tensor([input_ids], dtype=torch.long, ...)``
when the tokenizer's ``apply_chat_template`` returned an
unexpected shape — typically a plain string (a tokenizer that
ignored ``tokenize=True``), a 2-D batched ``torch.Tensor``, or a
``BatchEncoding``-like object.

The fix lives in two places:

* :meth:`CodeOven._coerce_token_ids` normalises every legal
  return shape (str / 1-D Tensor / 2-D Tensor / BatchEncoding /
  list[int] / list[list[int]]) into a flat ``list[int]``.
* :meth:`CodeOven._run` defensively coerces its argument and
  raises a clear ``TypeError`` instead of bubbling up
  ``too many dimensions 'str'`` if anything still slips through.
"""
from __future__ import annotations

import tempfile
from typing import Any

import pytest
import torch

from hypernix import new_oven


def _tiny_oven():
    """Build a minimum-viable HyperNix oven on CPU for fast tests."""
    d = tempfile.mkdtemp(prefix="hypernix-v051-4-")
    return new_oven(
        d, arch="hypernix",
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, device="cpu", seed=0,
    )


class _BaseTok:
    """Minimal HF-tokenizer-shaped stub."""

    chat_template = "{messages}"
    eos_token_id = 0
    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i) % 128) for i in ids)


# ---------------------------------------------------------------------------
# The headline bug: apply_chat_template returns a plain string
# ---------------------------------------------------------------------------

class TestStringReturnFromApplyChatTemplate:
    def test_chat_does_not_raise_too_many_dimensions(self) -> None:
        oven = _tiny_oven()

        class _Buggy(_BaseTok):
            def apply_chat_template(self, msgs, **_kw: Any) -> str:
                # The exact bug — ignores tokenize=True, returns a string.
                return "user: hi\nassistant: "

        oven.tokenizer = _Buggy()
        oven.tokenizer_kind = "hf"
        out = oven.chat(
            [{"role": "user", "content": "hi"}],
            max_new_tokens=2, temperature=0.0,
        )
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Other broken shapes: tensor / BatchEncoding / 2-D batched / mixed strings
# ---------------------------------------------------------------------------

class TestVariousReturnShapes:
    def test_1d_tensor_return(self) -> None:
        oven = _tiny_oven()

        class _T(_BaseTok):
            def apply_chat_template(self, msgs, **_kw: Any) -> torch.Tensor:
                return torch.tensor([1, 2, 3, 4], dtype=torch.long)

        oven.tokenizer = _T()
        oven.tokenizer_kind = "hf"
        assert isinstance(
            oven.chat([{"role": "user", "content": "hi"}], max_new_tokens=2, temperature=0.0),
            str,
        )

    def test_2d_batched_tensor_return(self) -> None:
        oven = _tiny_oven()

        class _T(_BaseTok):
            def apply_chat_template(self, msgs, **_kw: Any) -> torch.Tensor:
                return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        oven.tokenizer = _T()
        oven.tokenizer_kind = "hf"
        assert isinstance(
            oven.chat([{"role": "user", "content": "hi"}], max_new_tokens=2, temperature=0.0),
            str,
        )

    def test_batchencoding_like_return(self) -> None:
        oven = _tiny_oven()

        class _BE:
            def __init__(self, ids: list[int]):
                self.input_ids = ids

        class _T(_BaseTok):
            def apply_chat_template(self, msgs, **_kw: Any):
                return _BE([5, 6, 7])

        oven.tokenizer = _T()
        oven.tokenizer_kind = "hf"
        assert isinstance(
            oven.chat([{"role": "user", "content": "hi"}], max_new_tokens=2, temperature=0.0),
            str,
        )

    def test_list_of_str_falls_back_to_cookbook(self) -> None:
        """``list[str]`` is an unrecoverable shape — _coerce returns
        None and we fall through to the cookbook / plain path
        instead of crashing."""
        oven = _tiny_oven()

        class _T(_BaseTok):
            def apply_chat_template(self, msgs, **_kw: Any) -> list[str]:
                return ["<bos>", "user", ":", " hi"]

        oven.tokenizer = _T()
        oven.tokenizer_kind = "hf"
        # Should not raise; falls back to plain transcript fallback.
        out = oven.chat(
            [{"role": "user", "content": "hi"}],
            max_new_tokens=2, temperature=0.0,
        )
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# _coerce_token_ids unit tests (no model in the loop)
# ---------------------------------------------------------------------------

class TestCoerceTokenIds:
    @pytest.fixture
    def oven(self):
        return _tiny_oven()

    def test_string_input_is_re_encoded(self, oven) -> None:
        assert oven._coerce_token_ids("hi") == oven._encode("hi")

    def test_list_int_passthrough(self, oven) -> None:
        assert oven._coerce_token_ids([1, 2, 3]) == [1, 2, 3]

    def test_tuple_of_int_normalised_to_list(self, oven) -> None:
        assert oven._coerce_token_ids((1, 2, 3)) == [1, 2, 3]

    def test_1d_tensor_flattens(self, oven) -> None:
        assert oven._coerce_token_ids(torch.tensor([1, 2, 3])) == [1, 2, 3]

    def test_2d_tensor_flattens(self, oven) -> None:
        assert oven._coerce_token_ids(torch.tensor([[1, 2, 3]])) == [1, 2, 3]

    def test_empty_tensor(self, oven) -> None:
        assert oven._coerce_token_ids(torch.tensor([], dtype=torch.long)) == []

    def test_empty_list(self, oven) -> None:
        assert oven._coerce_token_ids([]) == []

    def test_batched_list_of_lists(self, oven) -> None:
        assert oven._coerce_token_ids([[1, 2, 3], [4, 5, 6]]) == [1, 2, 3]

    def test_batchencoding_like_with_input_ids(self, oven) -> None:
        class _BE:
            input_ids = [9, 8, 7]
        assert oven._coerce_token_ids(_BE()) == [9, 8, 7]

    def test_list_of_strings_returns_none(self, oven) -> None:
        assert oven._coerce_token_ids(["h", "e", "l", "l", "o"]) is None

    def test_unrecognised_returns_none(self, oven) -> None:
        assert oven._coerce_token_ids(object()) is None


# ---------------------------------------------------------------------------
# _run defensive guard
# ---------------------------------------------------------------------------

class TestRunDefensiveGuard:
    def test_run_accepts_string_input_via_coercion(self) -> None:
        oven = _tiny_oven()
        out = oven._run(
            "hi",  # type: ignore[arg-type]
            max_new_tokens=2, temperature=0.0, top_k=40, top_p=0.95,
        )
        assert isinstance(out, list)
        assert all(isinstance(t, int) for t in out)

    def test_run_accepts_tensor_via_coercion(self) -> None:
        oven = _tiny_oven()
        out = oven._run(
            torch.tensor([1, 2, 3]),  # type: ignore[arg-type]
            max_new_tokens=2, temperature=0.0, top_k=40, top_p=0.95,
        )
        assert isinstance(out, list)

    def test_run_raises_typeerror_on_truly_unrecoverable_input(self) -> None:
        oven = _tiny_oven()
        with pytest.raises(TypeError, match="expected list\\[int\\]"):
            oven._run(
                ["this", "is", "wrong"],  # type: ignore[arg-type, list-item]
                max_new_tokens=2, temperature=0.0, top_k=40, top_p=0.95,
            )
