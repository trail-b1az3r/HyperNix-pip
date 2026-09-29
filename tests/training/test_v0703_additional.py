"""Tests for version 0.70.3 lazy_suzan module."""
from __future__ import annotations

from unittest import mock

import torch
import torch.nn as nn

from hypernix.lazy_suzan import LazySusan, LazySusanConfig


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_lazy_susan_config_defaults() -> None:
    config = LazySusanConfig()
    assert config.compression == "fp8"
    assert config.ring_topology is True
    assert config.overlap_comm is True
    assert config.local_steps == 1
    assert config.sparsification_ratio == 0.01


def test_lazy_susan_config_custom() -> None:
    config = LazySusanConfig(
        compression="topk",
        ring_topology=False,
        overlap_comm=False,
        local_steps=4,
        sparsification_ratio=0.05,
    )
    assert config.compression == "topk"
    assert config.ring_topology is False
    assert config.overlap_comm is False
    assert config.local_steps == 4
    assert config.sparsification_ratio == 0.05


def test_lazy_susan_initialization() -> None:
    model = SimpleModel()
    susan = LazySusan(model)
    assert susan.model is model
    assert isinstance(susan.config, LazySusanConfig)
    assert len(susan.hooks) > 0  # parameters have hooks registered by default


def test_lazy_susan_no_overlap_hooks() -> None:
    model = SimpleModel()
    config = LazySusanConfig(overlap_comm=False)
    susan = LazySusan(model, config)
    assert len(susan.hooks) == 0


def test_lazy_susan_compression_none() -> None:
    model = SimpleModel()
    config = LazySusanConfig(compression="none")
    susan = LazySusan(model, config)
    tensor = torch.randn(2, 2)
    compressed, scale = susan.compress(tensor)
    assert torch.equal(compressed, tensor)
    assert scale is None
    decompressed = susan.decompress(compressed, scale, tensor.shape, tensor.dtype)
    assert torch.equal(decompressed, tensor)


def test_lazy_susan_compression_int8() -> None:
    model = SimpleModel()
    config = LazySusanConfig(compression="int8")
    susan = LazySusan(model, config)
    tensor = torch.tensor([[1.0, -2.0], [3.0, 0.0]])
    compressed, scale = susan.compress(tensor)
    assert compressed.dtype == torch.int8
    assert scale == 3.0
    decompressed = susan.decompress(compressed, scale, tensor.shape, tensor.dtype)
    # Check that decompressed values are reasonably close to quantized values
    assert torch.allclose(decompressed, tensor, atol=0.05)


def test_lazy_susan_compression_topk() -> None:
    model = SimpleModel()
    config = LazySusanConfig(compression="topk", sparsification_ratio=0.5)
    susan = LazySusan(model, config)
    tensor = torch.tensor([1.0, 5.0, 2.0, -10.0])  # size 4
    # sparsification_ratio=0.5 means top-2 values by magnitude (5.0, -10.0)
    (sparse_values, indices), original_shape = susan.compress(tensor)
    assert len(sparse_values) == 2
    assert set(indices.tolist()) == {1, 3}
    
    decompressed = susan.decompress((sparse_values, indices), None, original_shape, tensor.dtype)
    assert decompressed[1] == 5.0
    assert decompressed[3] == -10.0
    assert decompressed[0] == 0.0
    assert decompressed[2] == 0.0


def test_lazy_susan_compress_fp8_fallback() -> None:
    model = SimpleModel()
    config = LazySusanConfig(compression="fp8")
    susan = LazySusan(model, config)
    
    # Test fallback behavior when torch does not have float8_e4m3fn
    had_attr = hasattr(torch, "float8_e4m3fn")
    old_attr = getattr(torch, "float8_e4m3fn", None)
    if had_attr:
        delattr(torch, "float8_e4m3fn")
        
    try:
        tensor = torch.tensor([[1.0, -2.0]])
        compressed, scale = susan.compress(tensor)
        assert compressed.dtype == torch.int8
        decompressed = susan.decompress(compressed, scale, tensor.shape, tensor.dtype)
        assert torch.allclose(decompressed, tensor, atol=0.05)
    finally:
        if had_attr:
            torch.float8_e4m3fn = old_attr


def test_lazy_susan_step_counter() -> None:
    model = SimpleModel()
    config = LazySusanConfig(local_steps=3)
    susan = LazySusan(model, config)
    
    with mock.patch.object(susan, "synchronize_parameters") as mock_sync:
        susan.step()
        assert susan.step_counter == 1
        assert not mock_sync.called
        
        susan.step()
        assert susan.step_counter == 2
        assert not mock_sync.called
        
        susan.step()
        assert susan.step_counter == 3
        assert mock_sync.called
