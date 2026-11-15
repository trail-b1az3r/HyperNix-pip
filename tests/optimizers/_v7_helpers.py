"""Shared fixtures for the Pressure Cooker V7 tests."""
from __future__ import annotations

import torch
import torch.nn as nn


def make_model(seed: int = 0, dim: int = 32) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, 48),
                         nn.GELU(), nn.Linear(48, 8))


def make_data(seed: int = 1, dim: int = 32, n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    teacher = torch.randn(dim, 8, generator=g)
    return x, (x @ teacher).argmax(dim=1)


def train(model: nn.Module, optimizer: torch.optim.Optimizer, steps: int = 60,
          report: bool = True) -> list[float]:
    x, y = make_data()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if report and hasattr(optimizer, "report_loss"):
            optimizer.report_loss(loss.item())
        losses.append(loss.item())
    return losses


def flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])
