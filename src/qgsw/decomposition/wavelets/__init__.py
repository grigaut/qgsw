"""Wavelet decomposition."""

from qgsw.decomposition.wavelets.core import (
    WaveletBasis,
    generate_space_params,
    generate_time_params,
)

__all__ = ["WaveletBasis", "generate_space_params", "generate_time_params"]
