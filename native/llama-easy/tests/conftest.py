import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    if torch.cuda.is_available():  # CUDA and ROCm
        return torch.device("cuda")
    pytest.skip("No GPU available for Triton tests")
