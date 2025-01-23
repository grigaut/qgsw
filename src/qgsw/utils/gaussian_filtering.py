"""Gaussian filtering tools."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from qgsw.specs import DEVICE


class GaussianFilter(ABC):
    """Gaussian Filter."""

    _windows_width = 4  # nbs of sigma
    _null_threshold = 0.05
    dtype = torch.float64

    def __init__(self, sigma: float) -> None:
        """Instantiate the filter.

        Args:
            sigma (float): Standard deviation.
        """
        self._mu = self._set_null_mu()
        self.sigma = sigma

    @property
    def kernel(self) -> torch.Tensor:
        """Gaussian kernel."""
        return self._kernel

    @property
    def sigma(self) -> float:
        """Standard deviation."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        self._sigma = value
        self._kernel = self.compute_kernel(
            self._sigma,
            mu=self._mu,
            window_radius=self.window_radius,
        )

    @property
    def window_radius(self) -> int:
        """Window radius."""
        return int(self._windows_width * self._sigma) + 1

    @property
    def window_width(self) -> int:
        """Window radius."""
        return self.compute_window_width(self.window_radius)

    @abstractmethod
    def _set_null_mu(self) -> torch.Tensor:
        """Compute null mu.

        Returns:
            torch.Tensor: mu
        """

    @staticmethod
    def compute_window_width(window_radius: int) -> int:
        """Compute the wisndow width.

        Args:
            window_radius (int): Window radius.

        Returns:
            int: Window width
        """
        return window_radius * 2 + 1

    @staticmethod
    @abstractmethod
    def compute_kernel(
        sigma: float,
        mu: torch.Tensor,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
            mu (torch.Tensor): Expected value.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel.
        """


class GaussianFilter1D(GaussianFilter):
    """1D Gaussian Filter."""

    def _set_null_mu(self) -> torch.Tensor:
        return torch.zeros(
            (1,),
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @staticmethod
    def compute_kernel(
        sigma: float,
        mu: torch.Tensor,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
            mu (torch.Tensor): Expected value.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel.
        """
        x = torch.arange(
            -window_radius,
            window_radius + 1,
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        r = x - mu
        kernel = torch.exp(-(r**2 / (2 * sigma**2)))
        kernel /= torch.sum(kernel)
        return kernel


class GaussianFilter2D(GaussianFilter):
    """2D Gaussian Filter."""

    def _set_null_mu(self) -> torch.Tensor:
        return torch.zeros(
            (2, 1, 1),
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @staticmethod
    def compute_kernel(
        sigma: float,
        mu: torch.Tensor,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
            mu (torch.Tensor): Expected value.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel.
        """
        x = torch.arange(
            -window_radius,
            window_radius + 1,
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        r = torch.norm(torch.stack(torch.meshgrid(x, x)) - mu, dim=0)

        kernel = torch.exp(-(r**2 / (2 * sigma**2)))
        kernel /= torch.sum(kernel)
        return kernel
