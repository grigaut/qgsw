"""Test least squares regression."""

import pytest
import torch

from qgsw.specs import DEVICE
from qgsw.utils.least_squares_regression import (
    perform_linear_least_squares_regression,
)

testdata = [
    pytest.param(
        torch.rand((3, 1, 50), device=DEVICE.get()),
        torch.rand((1,), device=DEVICE.get()),
    ),
    pytest.param(
        torch.rand((1, 50), device=DEVICE.get()),
        torch.rand((1,), device=DEVICE.get()),
    ),
    pytest.param(
        torch.rand((3, 2, 50), device=DEVICE.get()),
        torch.rand((2,), device=DEVICE.get()),
    ),
    pytest.param(
        torch.rand((2, 50), device=DEVICE.get()),
        torch.rand((2,), device=DEVICE.get()),
    ),
]


@pytest.mark.parametrize(("x", "beta_ref"), testdata)
def test_linear_least_squares_regression(
    x: torch.Tensor,
    beta_ref: torch.Tensor,
) -> None:
    """Test linear least squares regression."""
    y = torch.einsum("l,...ln->...n", beta_ref, x)
    beta = perform_linear_least_squares_regression(x.transpose(-2, -1), y)
    assert torch.isclose(beta, beta_ref).all()
