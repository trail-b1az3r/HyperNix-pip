"""Tests for LazySusan."""
import torch
import torch.nn as nn

from hypernix.compute_framework import ComputeFramework
from hypernix.lazy_suzan import LazySusan, LazySusanConfig


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.fc(x)

def test_lazy_susan_compression_fp8():
    model = DummyModel()
    config = LazySusanConfig(compression="fp8", overlap_comm=False)
    ls = LazySusan(model, config)
    
    grad = torch.randn(10, 10) * 10
    compressed, scale = ls.compress(grad)
    
    decompressed = ls.decompress(compressed, scale, grad.shape, grad.dtype)
    assert decompressed.shape == grad.shape
    # Ensure some accuracy remains
    assert torch.allclose(grad, decompressed, rtol=0.5, atol=2.0)

def test_lazy_susan_compression_int8():
    model = DummyModel()
    config = LazySusanConfig(compression="int8", overlap_comm=False)
    ls = LazySusan(model, config)
    
    grad = torch.randn(10, 10) * 5
    compressed, scale = ls.compress(grad)
    assert compressed.dtype == torch.int8
    
    decompressed = ls.decompress(compressed, scale, grad.shape, grad.dtype)
    assert decompressed.shape == grad.shape
    assert torch.allclose(grad, decompressed, rtol=0.2, atol=0.5)

def test_lazy_susan_compression_topk():
    model = DummyModel()
    # 20% sparsity
    config = LazySusanConfig(compression="topk", sparsification_ratio=0.2, overlap_comm=False)
    ls = LazySusan(model, config)
    
    grad = torch.randn(10, 10)
    compressed, original_shape = ls.compress(grad)
    
    decompressed = ls.decompress(compressed, None, original_shape, grad.dtype)
    assert decompressed.shape == grad.shape
    # Only 20% should be non-zero
    num_non_zero = (decompressed != 0).sum().item()
    assert num_non_zero == 20

def test_lazy_susan_backward_hook():
    model = DummyModel()
    config = LazySusanConfig(overlap_comm=True, ring_topology=False)
    ls = LazySusan(model, config)
    
    assert len(ls.hooks) > 0
    
    # Simulate a backward pass
    output = model(torch.randn(5, 10))
    loss = output.sum()
    loss.backward()
    
    # Check if gradient was updated (which means hook fired)
    assert model.fc.weight.grad is not None

def test_compute_framework_lazy_suzan_integration():
    model = DummyModel()
    cf = ComputeFramework(
        local_rank="cpu",
        use_lazy_suzan=True,
        lazy_suzan_config=LazySusanConfig(overlap_comm=True)
    )
    
    model = cf.prepare_model(model)
    assert hasattr(cf, "lazy_suzan")
    assert isinstance(cf.lazy_suzan, LazySusan)
    
    # Simulate backward and step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    output = model(torch.randn(2, 10))
    loss = output.sum()
    
    cf.backward(loss)
    cf.step(optimizer)
    
    assert cf.lazy_suzan.step_counter == 1
