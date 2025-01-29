"""High-Pass filters."""

import torch
from torch import Tensor

from qgsw.filters.gaussian import GaussianFilter1D, GaussianFilter2D


class GaussianHighPass1D(GaussianFilter1D):
    """1D Gaussian High-Pass filter."""

    @property
    def kernel(self) -> torch.Tensor:
        """Convolution kernel."""
        id_ = torch.zeros_like(self._kernel)
        id_[self.window_radius] = 1
        return (id_ - self._kernel) / self._normalization_factor()

    def __call__(self, to_filter: Tensor) -> Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter, (p,q)-shaped.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        return (
            to_filter - super().__call__(to_filter)
        ) / self._normalization_factor()


class GaussianHighPass2D(GaussianFilter2D):
    """2D Gaussian High-Pass filter."""

    @property
    def kernel(self) -> torch.Tensor:
        """Convolution kernel."""
        id_ = torch.zeros_like(self._kernel)
        id_[self.window_radius, self.window_radius] = 1
        return id_ - self._kernel

    def __call__(self, to_filter: Tensor) -> Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter, (p,q)-shaped.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        return to_filter - super().__call__(to_filter)
