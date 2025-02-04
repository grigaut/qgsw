"""Spectral Filters."""

import torch

from qgsw import fft
from qgsw.filters.base import _SpectralFilter


class SpectralGaussianFilter(_SpectralFilter):
    """Spectral Gaussian filter."""

    def __init__(self, sigma: float) -> None:
        """Instantiate the filter.

        Args:
            sigma (float): Standard deviation.
        """
        self.sigma = sigma

    @property
    def sigma(self) -> float:
        """Standard deviation."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        self._sigma = value


class SpectralGaussianFilter1D(SpectralGaussianFilter):
    """1D Spectral Gaussian Filter."""

    @classmethod
    def compute_kernel(
        cls,
        sigma: float,
        *,
        n: int,
        d: float = 1,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> torch.Tensor:
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
        freqs = fft.dstI1Dffreq(
            n=n,
            d=d,
            dtype=dtype,
            device=device,
        )
        return torch.exp(-(freqs**2) / (2 * sigma**2))

    def compute_kernel_for(
        self,
        to_filter: torch.Tensor,
        *,
        d: float = 1,
    ) -> torch.Tensor:
        """Compute spectral kernel for a given tensor.

        Args:
            to_filter (torch.Tensor): Tensor to compute kernel for.
            d (float, optional): Smaple spacing. Defaults to 1.

        Returns:
            torch.Tensor: kernel
        """
        return self.compute_kernel(
            self.sigma,
            n=to_filter.shape[-1],
            d=d,
            dtype=to_filter.dtype,
            device=to_filter.device,
        )

    def __call__(
        self,
        to_filter: torch.Tensor,
        *,
        d: float = 1,
    ) -> torch.Tensor:
        """Perform filtering on a given tensor.

        Args:
            to_filter (torch.Tensor): Tensor to filter.
            d (float, optional): Sample spacing. Defaults to 1.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        kernel = self.compute_kernel_for(
            to_filter=to_filter,
            d=d,
        )

        return fft.dstI1D(fft.dstI1D(to_filter) * kernel)


class SpectralGaussianFilter2D(SpectralGaussianFilter):
    """2D Spectral Gaussain Filter."""

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
        freqs_x = fft.dstI1Dffreq(
            n=nx,
            d=dx,
            dtype=dtype,
            device=device,
        )
        freqs_y = fft.dstI1Dffreq(
            n=ny,
            d=dy,
            dtype=dtype,
            device=device,
        )
        fx, fy = torch.meshgrid(freqs_x, freqs_y, indexing="ij")
        return torch.exp(-(fx**2 + fy**2) / (2 * sigma**2))

    def compute_kernel_for(
        self,
        to_filter: torch.Tensor,
        *,
        dx: float = 1,
        dy: float = 1,
    ) -> torch.Tensor:
        """Compute spectral kernel for a given tensor.

        Args:
            to_filter (torch.Tensor): Tensor to compute kernel for.
            dx (float, optional): Sample spacing in x. Defaults to 1.
            dy (float, optional): Sample spacing in y. Defaults to 1.

        Returns:
            torch.Tensor: Spectral kernel.
        """
        return self.compute_kernel(
            self.sigma,
            nx=to_filter.shape[-2],
            ny=to_filter.shape[-1],
            dx=dx,
            dy=dy,
            dtype=to_filter.dtype,
            device=to_filter.device,
        )

    def __call__(
        self,
        to_filter: torch.Tensor,
        *,
        dx: float = 1,
        dy: float = 1,
    ) -> torch.Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter.
            dx (float, optional): Sample spacing in x. Defaults to 1.
            dy (float, optional): Sample spacing in y. Defaults to 1.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        kernel = self.compute_kernel_for(
            to_filter,
            dx=dx,
            dy=dy,
        )

        return fft.dstI2D(fft.dstI2D(to_filter) * kernel)
