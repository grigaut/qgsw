"""Test model configurations."""

import pytest
import torch

from qgsw.configs.models import ModelConfig
from qgsw.specs import DEVICE

DEVICE.set_manually("cpu")


@pytest.fixture
def model_config() -> ModelConfig:
    """Define model config."""
    return ModelConfig(
        type="QG",
        prefix="",
        layers=[100, 200],
        reduced_gravity=[10, 0.05],
        collinearity_coef=None,
    )


def test_h_g_prime_shape(model_config: ModelConfig) -> None:
    """Test H and g' dtype and device."""
    assert model_config.h.shape == (2,)
    assert model_config.g_prime.shape == (2,)


def test_h_g_prime_dtype(model_config: ModelConfig) -> None:
    """Test H and g' dtype."""
    assert model_config.h.dtype == torch.float64
    assert model_config.g_prime.dtype == torch.float64


def test_h_g_prime_device(model_config: ModelConfig) -> None:
    """Test H and g' device."""
    assert model_config.h.device == DEVICE.get()
    assert model_config.g_prime.device == DEVICE.get()