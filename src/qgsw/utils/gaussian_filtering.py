"""Gaussian filtering tools."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import overload

import numpy as np
import torch
from scipy import signal

from qgsw.specs import DEVICE


class GaussianFilterBase(metaclass=ABCMeta):
    """Base Class for Gaussian Filters."""

    def __init__(self, sigma: float, radius: int) -> None:
        """Instantiates the Gaussian filter.

        Args:
            sigma (float): Standard deviation.
            radius (int): Kernel Radius.
        """
        self.sigma = sigma
        self.radius = radius

    @property
    def sigma(self) -> float:
        """Sigma."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        if value <= 0:
            msg = "Sigma must be greater than 0."
            raise ValueError(msg)
        self._sigma = value

    @property
    def radius(self) -> int:
        """Kernel Width."""
        return self._radius

    @radius.setter
    def radius(self, value: int) -> None:
        if value <= 0:
            msg = "Kernel width must be greater than 0."
            raise ValueError(msg)
        if not isinstance(value, int):
            msg = "Kernel width must be an integer."
            raise TypeError(msg)
        self._radius = value
        self._kernel = self._generate_kernel()

    @abstractmethod
    def _generate_kernel(self) -> torch.Tensor: ...

    @abstractmethod
    def _smooth(self, y: np.ndarray) -> np.ndarray: ...

    @overload
    def smooth(self, y: np.ndarray) -> np.ndarray: ...

    @overload
    def smooth(self, y: torch.Tensor) -> torch.Tensor: ...

    def smooth(
        self,
        y: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        """Smooth the given array.

        Args:
            y (torch.Tensor | np.ndarray): Array to smooth.

        Returns:
            torch.Tensor | np.ndarray: Smoothed array.
        """
        if isinstance(y, np.ndarray):
            return self._smooth(y)
        return torch.tensor(
            self._smooth(y.numpy()),
            dtype=torch.float64,
            device=DEVICE.get(),
        )


class GaussianFilter1D(GaussianFilterBase):
    """1D Gaussian Filtering."""

    def _generate_kernel(self) -> torch.Tensor:
        """Generate gaussian kernel.

        Returns:
            torch.Tensor: gaussian kernel.
        """
        x = torch.linspace(-1, 1, int(self.radius))
        kernel = (
            1
            / (self.sigma * np.sqrt(2 * torch.pi))
            * torch.exp(-torch.pow(x, 2) / (2 * self.sigma**2))
        )
        return kernel / torch.sum(kernel)

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        """Smooth coefficients using a gaussian kernel.

        Args:
            y (np.ndarray): tensor to smooth.

        Returns:
            np.ndarray: Smoothed tensor.
        """
        # Padding
        i_left = self._kernel.shape[0] - 1
        i_right = self._kernel.shape[0] - 1
        pad_width = (i_left, i_right)
        y_padded = np.pad(y, pad_width=pad_width, mode="edge")
        # Convolution
        convolved = signal.convolve(y_padded, self._kernel, mode="same")
        return convolved[i_left:-i_right]


class GaussianFilter2d(GaussianFilterBase):
    """1D Gaussian Filtering."""

    def _generate_kernel(self) -> torch.Tensor:
        """Generate gaussian kernel.

        Returns:
            torch.Tensor: gaussian kernel.
        """
        x_cord = torch.linspace(
            -1,
            1,
            self._radius * 2,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        x_grid = x_cord.repeat(self._radius * 2).view(
            self._radius * 2,
            self._radius * 2,
        )
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1.0 / (self._sigma**2 * 2 * torch.pi)) * torch.exp(
            -torch.sum((xy_grid) ** 2.0, dim=-1) / (2 * self._sigma**2),
        )
        # Make sure sum of values in gaussian kernel equals 1.
        return gaussian_kernel / torch.sum(gaussian_kernel)

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        """Smooth coefficients using a gaussian kernel.

        Args:
            y (np.ndarray): tensor to smooth.

        Returns:
            np.ndarray: Smoothed tensor.
        """
        # Padding
        i_left = self._kernel.shape[0] - 1
        i_right = self._kernel.shape[0] - 1
        j_left = self._kernel.shape[0] - 1
        j_right = self._kernel.shape[0] - 1
        pad_width = ((i_left, i_right), (j_left, j_right))
        y_padded = np.pad(y, pad_width=pad_width, mode="edge")
        # Convolution
        convolved = signal.convolve2d(y_padded, self._kernel, mode="same")
        return convolved[i_left:-i_right, j_left:-j_right]
