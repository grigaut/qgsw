"""Low-Pass filters."""

from qgsw.filters.gaussian import GaussianFilter1D, GaussianFilter2D
from qgsw.filters.spectral import (
    SpectralGaussianFilter1D,
    SpectralGaussianFilter2D,
)


class GaussianLowPass1D(GaussianFilter1D):
    """1D Gaussian Low-Pass filter."""


class SpectralGaussianLowPass1D(SpectralGaussianFilter1D):
    """1D Spectral Gaussian Low-Pass filter."""


class GaussianLowPass2D(GaussianFilter2D):
    """2D Gaussian Low-Pass filter."""


class SpectralGaussianLowPass2D(SpectralGaussianFilter2D):
    """2D Spectral Gaussian Low-Pass filter."""
