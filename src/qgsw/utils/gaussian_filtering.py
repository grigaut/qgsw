"""Gaussian filtering tools."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import overload

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from qgsw.specs import DEVICE


class GaussianFilterBase(metaclass=ABCMeta):
    """Base Class for Gaussian Filters."""

    def __init__(self, sigma: float, kernel_width: int) -> None:
        """Instantiates the Gaussian filter.

        Args:
            sigma (float): _description_
            kernel_width (int): _description_
        """
        self.sigma = sigma
        self.kernel_width = kernel_width

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
    def kernel_width(self) -> int:
        """Kernel Width."""
        return self._kernel_width

    @kernel_width.setter
    def kernel_width(self, value: int) -> None:
        if value <= 0:
            msg = "Kernel width must be greater than 0."
            raise ValueError(msg)
        if not isinstance(value, int):
            msg = "Kernel width must be an integer."
            raise TypeError(msg)
        self._kernel_width = value
        self._kernel = self._generate_kernel()

    @abstractmethod
    def _generate_kernel(self) -> torch.Tensor: ...

    @abstractmethod
    def _smooth(self, y: torch.Tensor) -> torch.Tensor: ...

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
            return self._smooth(
                torch.tensor(y, dtype=torch.float64, device=DEVICE.get()),
            )
        return torch.tensor(
            self._smooth(y),
            dtype=torch.float64,
            device=DEVICE.get(),
        )


class GaussianFilter1D(GaussianFilterBase):
    """1D Gaussian Filetring."""

    def _generate_kernel(self) -> torch.Tensor:
        """Generate filtering kernel.

        Returns:
            torch.Tensor: Filtering kernel.
        """
        x = torch.linspace(-1, 1, int(self.kernel_width))
        kernel = (
            1
            / (self.sigma * np.sqrt(2 * torch.pi))
            * torch.exp(-torch.pow(x, 2) / (2 * self.sigma**2))
        )
        return kernel / torch.sum(kernel)

    def _smooth(self, y: torch.Tensor) -> torch.Tensor:
        """Smooth coefficients using a gaussian kernel.

        Args:
            y (torch.Tensor): tensor to smooth.

        Returns:
            torch.Tensor: Smoothed tensor.
        """
        pad_size = self._kernel.shape[0] - 1
        pad_y_left = F.pad(y, (pad_size, 0), mode="constant", value=y[0])
        pad_y = F.pad(
            pad_y_left,
            (0, pad_size),
            mode="constant",
            value=pad_y_left[-1],
        )
        convolved = np.convolve(pad_y, self._kernel, mode="same")
        return convolved[pad_size:-pad_size]
