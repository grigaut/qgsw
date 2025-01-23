"""Gaussian filtering tools."""

from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from abc import ABC, abstractmethod

import torch

from qgsw.specs import DEVICE


class GaussianFilter(ABC):
    """Gaussian Filter."""

    _windows_width_factor = 4  # nbs of sigma
    _span_threshold = 0.05
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
    def span(self) -> float:
        """Gaussian span."""
        return self.compute_span_from_sigma(self._sigma, self.span_threshold)

    @property
    def span_threshold(self) -> float:
        """Span bounding threshold."""
        return self._span_threshold

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
        return int(self._windows_width_factor * self._sigma) + 1

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
    def compute_sigma_from_span(span: float, threshold: float) -> float:
        """Compute the standard deviation givent the span.

        Args:
            span (float): Desired span (in px).
            threshold (float): Threshold to bound the span.

        Returns:
            float: Corresponding standard deviation.
        """
        thres = torch.tensor(
            threshold,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        return (span / torch.sqrt(-8 * torch.log(thres))).item()

    @staticmethod
    def compute_span_from_sigma(sigma: float, threshold: float) -> float:
        """Compute the standard deviation givent the span.

        Args:
            sigma (float): Standard deviation.
            threshold (float): Threshold to bound the span.

        Returns:
            float: Corresponding standard deviation.
        """
        thres = torch.tensor(
            threshold,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        return (sigma * torch.sqrt(-8 * torch.log(thres))).item()

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

    @classmethod
    def from_span(cls, span: float) -> Self:
        """Instantiate the filter given the span of the gaussian.

        Args:
            span (float): Span (in px).

        Returns:
            Self: Gaussian filter.
        """
        return cls(
            sigma=cls.compute_sigma_from_span(
                span,
                threshold=cls._span_threshold,
            ),
        )


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
            mu (torch.Tensor): Expected value, (1,)-shaped.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel (2*windows_radius+1,)-shaped.
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
            (2,),
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
            mu (torch.Tensor): Expected value, (2,)-shaped.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel
            (2*windows_radius+1,2*windows_radius+1)-shaped.
        """
        x = torch.arange(
            -window_radius,
            window_radius + 1,
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        xy = torch.stack(torch.meshgrid(x, x))
        mu_reshaped = mu.reshape((-1, 1, 1))

        r = torch.norm(xy - mu_reshaped, dim=0)

        kernel = torch.exp(-(r**2 / (2 * sigma**2)))
        kernel /= torch.sum(kernel)
        return kernel
