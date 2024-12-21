"""Test model configurations."""

import torch

from qgsw.configs.models import ModelConfig
from qgsw.specs import DEVICE


def test_h_g_prime() -> None:
    """Test H and g' shapes, dtype and device."""
    DEVICE.set_manually("cpu")

    config = ModelConfig(
        type="QG",
        prefix="",
        layers=[100, 200],
        reduced_gravity=[10, 0.05],
        collinearity_coef=None,
    )
    assert config.h.shape == (2,)
    assert config.g_prime.shape == (2,)
    assert config.h.dtype == torch.float64
    assert config.g_prime.dtype == torch.float64
    assert config.h.device == DEVICE.get()
    assert config.g_prime.device == DEVICE.get()
