from __future__ import annotations

import torch

from hypernix.training.brewer import BrewerConfig, BrewerModel
from hypernix.models.hnx0xv3 import HyperNix0xV3Config, HyperNix0xV3Model
from scripts.distill_hypernix0xv3 import (
    distillation_loss, resolve_eos_token_id, transfer_v2_to_v3,
)


def _models():
    v2 = BrewerModel(BrewerConfig(
        vocab_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        d_model=32, d_ff=64, max_seq_len=32, tie_embeddings=True,
        use_sliding_window=False,
    ))
    v3 = HyperNix0xV3Model(HyperNix0xV3Config(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=64, max_seq_len=32, window=8, chunk=4, n_slots=4,
        slot_dim=16, tie_embeddings=True,
    ))
    return v2, v3


def test_transfer_maps_shared_v2_weights_and_keeps_keel_trainable():
    torch.manual_seed(7)
    teacher, student = _models()
    report = transfer_v2_to_v3(teacher, student)
    assert report["copied_tensors"] >= 12
    assert torch.equal(student.embed.weight, teacher.embed.embed.weight)
    assert torch.equal(student.blocks[0].tidepool.q_proj.weight,
                       teacher.blocks[0].attn.q_proj.weight)
    assert torch.equal(student.blocks[0].ffn.down_proj.weight,
                       teacher.blocks[0].ffn.down_proj.weight)
    assert student.blocks[0].keel.init_slots.requires_grad
    assert student.blocks[0].keel.persist.weight.requires_grad


def test_distillation_loss_backpropagates_into_v3_only_layers():
    teacher, student = _models()
    ids = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        teacher_logits = teacher(ids)
    student_logits = student(ids)
    labels = torch.randint(0, 64, (2, 24))
    loss, kd, ce = distillation_loss(student_logits, teacher_logits, labels, 2.0, 0.7)
    assert torch.isfinite(loss) and torch.isfinite(kd) and torch.isfinite(ce)
    loss.backward()
    grad = student.blocks[0].keel.gate.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


class _TokenizerWithoutEOS:
    eos_token_id = None


class _TokenizerWithEOS:
    eos_token_id = 2


def test_resolve_eos_token_id_allows_tokenizer_without_eos():
    assert resolve_eos_token_id(_TokenizerWithoutEOS(), vocab_size=64) is None


def test_resolve_eos_token_id_uses_tokenizer_eos():
    assert resolve_eos_token_id(_TokenizerWithEOS(), vocab_size=64) == 2


def test_resolve_eos_token_id_accepts_explicit_override():
    assert resolve_eos_token_id(_TokenizerWithoutEOS(), override=3, vocab_size=64) == 3


def test_resolve_eos_token_id_rejects_out_of_range_override():
    import pytest
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        resolve_eos_token_id(_TokenizerWithoutEOS(), override=64, vocab_size=64)
