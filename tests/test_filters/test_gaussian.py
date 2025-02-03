"""Test gaussian filters."""

import numpy as np
import pytest
import torch
from scipy import signal

from qgsw.filters.base import _Filter
from qgsw.filters.gaussian import (
    GaussianFilter1D,
    GaussianFilter2D,
)
from qgsw.filters.high_pass import (
    GaussianHighPass1D,
    GaussianHighPass2D,
    SpectralGaussianHighPass1D,
    SpectralGaussianHighPass2D,
)
from qgsw.filters.low_pass import (
    GaussianLowPass1D,
    GaussianLowPass2D,
    SpectralGaussianLowPass1D,
    SpectralGaussianLowPass2D,
)
from qgsw.specs import DEVICE


def test_gaussian_filter_1d() -> None:
    """Test 1D Gaussian filter."""
    y = torch.rand((50,), dtype=torch.float64, device=DEVICE.get())
    filt = GaussianFilter1D.from_span(1)
    y_filt = filt(y).cpu().numpy()
    y_filt_ref = signal.convolve(
        y.cpu().numpy(),
        filt.kernel.cpu().numpy(),
        mode="same",
    )
    assert y_filt.shape == y.shape
    np.testing.assert_allclose(y_filt, y_filt_ref)


def test_gaussian_filter_2d() -> None:
    """Test 1D Gaussian filter."""
    y = torch.rand((50, 50), dtype=torch.float64, device=DEVICE.get())
    filt = GaussianFilter2D.from_span(1)
    y_filt = filt(y).cpu().numpy()
    y_filt_ref = signal.convolve2d(
        y.cpu().numpy(),
        filt.kernel.cpu().numpy(),
        mode="same",
    )
    assert y_filt.shape == y.shape
    np.testing.assert_allclose(y_filt, y_filt_ref)


testdata = [
    pytest.param(
        (500,),
        GaussianLowPass1D(10),
        GaussianHighPass1D(10),
        id="1D",
    ),
    pytest.param(
        (500,),
        SpectralGaussianLowPass1D(1),
        SpectralGaussianHighPass1D(1),
        id="1D-spectral",
    ),
    pytest.param(
        (500, 500),
        GaussianLowPass2D(10),
        GaussianHighPass2D(10),
        id="2D",
    ),
    pytest.param(
        (500, 500),
        SpectralGaussianLowPass2D(1),
        SpectralGaussianHighPass2D(1),
        id="2D-spectral",
    ),
]


@pytest.mark.parametrize(("input_shape", "filt_lp", "filt_hp"), testdata)
def test_lp_hp(
    input_shape: tuple[int, ...],
    filt_lp: _Filter,
    filt_hp: _Filter,
) -> None:
    """Test low-pass and high-pass reconstruction."""
    y = torch.rand(input_shape, dtype=torch.float64, device=DEVICE.get())

    y_filt_lp = filt_lp(y)
    y_filt_hp = filt_hp(y)
    torch.testing.assert_close(y_filt_hp + y_filt_lp, y)
