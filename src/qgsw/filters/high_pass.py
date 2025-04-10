"""High-Pass filters."""

import torch
from torch import Tensor

from qgsw.filters.gaussian import GaussianFilter1D, GaussianFilter2D
from qgsw.filters.spectral import (
    SpectralGaussianFilter1D,
    SpectralGaussianFilter2D,
)


class GaussianHighPass1D(GaussianFilter1D):
    """1D Gaussian High-Pass filter."""

    @property
    def kernel(self) -> torch.Tensor:
        """Convolution kernel."""
        id_ = torch.zeros_like(self._kernel)
        id_[self.window_radius] = 1
        return id_ - self._kernel

    def __repr__(self) -> str:
        """String representation for 1D gaussian filters."""
        return "High pass " + super().__repr__()

    def __call__(self, to_filter: Tensor) -> Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter, (p,q)-shaped.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        return to_filter - super().__call__(to_filter)


class SpectralGaussianHighPass1D(SpectralGaussianFilter1D):
    """1D Spectral Gaussian High-Pass filter."""

    def __repr__(self) -> str:
        """String representation for 1D gaussian filters."""
        return "High pass " + super().__repr__()

    @classmethod
    def compute_kernel(
        cls,
        sigma: float,
        *,
        n: int,
        d: float = 1,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Tensor:
        """Compute the spectral kernel.

        Args:
            sigma (float): Standard deviation.
            n (int): Input signal length.
            d (float, optional): Real signal sample spacing. Defaults to 1.
            dtype (torch.dtype, optional): Dtype. Defaults to None.
            device (torch.device, optional): Device. Defaults to None.

        Returns:
            torch.Tensor: Spectral kernel.
        """
        low_pass = super().compute_kernel(
            sigma,
            n=n,
            d=d,
            dtype=dtype,
            device=device,
        )
        return torch.ones_like(low_pass) - low_pass


class GaussianHighPass2D(GaussianFilter2D):
    """2D Gaussian High-Pass filter."""

    def __repr__(self) -> str:
        """String representation for 1D gaussian filters."""
        return "High pass " + super().__repr__()

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


class SpectralGaussianHighPass2D(SpectralGaussianFilter2D):
    """2D Spectral Gaussian High-Pass filter."""

    def __repr__(self) -> str:
        """String representation for 1D gaussian filters."""
        return "High pass " + super().__repr__()

    @classmethod
    def compute_kernel(
        cls,
        sigma: float,
        *,
        nx: int,
        ny: int,
        dx: float = 1,
        dy: float = 1,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Compute spectral kernel.

        Args:
            sigma (float): Standard deviation.
            nx (int): Number of sample along x.
            ny (int): Number of samples along y.
            dx (float, optional): Sample spacing for x. Defaults to 1.
            dy (float, optional): Sample spacing for y. Defaults to 1.
            dtype (torch.dtype, optional): Dtype. Defaults to None.
            device (torch.device, optional): Device. Defaults to None.

        Returns:
            torch.Tensor: Spectral kernel
        """
        low_pass = super().compute_kernel(
            sigma,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            dtype=dtype,
            device=device,
        )
        return torch.ones_like(low_pass) - low_pass
