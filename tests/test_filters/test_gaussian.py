"""Test gaussian filters."""

import numpy as np
import pytest
import torch
from scipy import signal

from qgsw.filters.gaussian import (
    GaussianFilter,
    GaussianFilter1D,
    GaussianFilter2D,
)
from qgsw.specs import DEVICE

testdata = [
    pytest.param((500,), GaussianFilter1D(10), id="1D"),
    pytest.param((500, 500), GaussianFilter2D(10), id="2D"),
]


@pytest.mark.parametrize(("input_shape", "filt"), testdata)
def test_gaussian_filter(
    input_shape: tuple[int, ...],
    filt: GaussianFilter,
) -> None:
    """Test 1D Gaussian filter."""
    y = torch.rand(input_shape, dtype=torch.float64, device=DEVICE.get())

    y_filt = filt(y).cpu().numpy()
    y_filt_ref = signal.convolve(
        y.cpu().numpy(),
        filt.kernel.cpu().numpy(),
        mode="same",
    )
    assert y_filt.shape == y.shape
    assert np.isclose(y_filt, y_filt_ref).all()
