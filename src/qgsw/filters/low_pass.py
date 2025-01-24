"""Low-Pass filters."""

from qgsw.filters.gaussian import GaussianFilter1D, GaussianFilter2D


class GaussianLowPass1D(GaussianFilter1D):
    """1D Gaussian Low-Pass filter."""


class GaussianLowPass2D(GaussianFilter2D):
    """2D Gaussian Low-Pass filter."""
