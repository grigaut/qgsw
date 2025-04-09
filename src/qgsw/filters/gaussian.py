"""Gaussian filtering tools."""

from __future__ import annotations

from typing import Generic, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from abc import ABC, abstractmethod

import torch

from qgsw.filters.base import _Filter
from qgsw.specs import DEVICE

T = TypeVar("T", bound=torch.nn.modules.conv._ConvNd)  # noqa: SLF001


class GaussianFilter(_Filter, ABC, Generic[T]):
    """Gaussian Filter."""

    _windows_width_factor = 1  # nbs of sigma
    _span_threshold = 0.5

    def __init__(self, sigma: float) -> None:
        """Instantiate the filter.

        Args:
            sigma (float): Standard deviation.
        """
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
            window_radius=self.window_radius,
        )
        self._conv = self._build_convolution()
        self._conv.weight.data = self._kernel.unsqueeze(0).unsqueeze(0)

    @property
    def window_radius(self) -> int:
        """Window radius."""
        return self._windows_width_factor * int(self.span) + 1

    def __repr__(self) -> str:
        """String representation for gaussian filters."""
        return f"Gaussian filter\n\t└── sigma = {self.sigma}"

    @abstractmethod
    def _build_convolution(self) -> T:
        """Build the convolution.

        Returns:
            T: Convolution.
        """

    def __call__(self, to_filter: torch.Tensor) -> torch.Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter, (p,q)-shaped.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        to_filter_expanded = to_filter.unsqueeze(0).unsqueeze(0)
        filtered: torch.Tensor = self._conv(to_filter_expanded).detach()
        return filtered.squeeze(0).squeeze(0)

    @classmethod
    def compute_sigma_from_span(cls, span: float, threshold: float) -> float:
        """Compute the standard deviation given the span.

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
        return (span / torch.sqrt(-2 * torch.log(thres))).item()

    @classmethod
    def compute_span_from_sigma(cls, sigma: float, threshold: float) -> float:
        """Compute the span given the satndard deviation.

        Args:
            sigma (float): Standard deviation.
            threshold (float): Threshold to bound the span.

        Returns:
            float: Corresponding span.
        """
        thres = torch.tensor(
            threshold,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        return (sigma * torch.sqrt(-2 * torch.log(thres))).item()

    @classmethod
    @abstractmethod
    def compute_kernel(
        cls,
        sigma: float,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
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


class GaussianFilter1D(GaussianFilter[torch.nn.Conv1d]):
    """1D Gaussian Filter."""

    def __repr__(self) -> str:
        """String representation for 1D gaussian filters."""
        return "1D " + super().__repr__()

    def _build_convolution(self) -> torch.nn.Conv1d:
        return torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=(self.window_width,),
            padding=self.window_width // 2,
            bias=False,
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @classmethod
    def compute_kernel_profile(
        cls,
        sigma: float,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
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
        return torch.exp(-(x**2 / (2 * sigma**2)))

    @classmethod
    def compute_kernel(
        cls,
        sigma: float,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel (2*windows_radius+1,)-shaped.
        """
        kernel = cls.compute_kernel_profile(
            sigma=sigma,
            window_radius=window_radius,
        )
        return kernel / torch.sum(kernel)


class GaussianFilter2D(
    GaussianFilter[tuple[torch.nn.Conv2d, torch.nn.Conv2d]],
):
    """2D Gaussian Filter."""

    def __repr__(self) -> str:
        """String representation for 2D gaussian filters."""
        return "2D " + super().__repr__()

    def _build_convolution(self) -> tuple[torch.nn.Conv2d, torch.nn.Conv2d]:
        conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(self.window_width, 1),
            padding=(self.window_width // 2, 0),
            bias=False,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        conv2 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, self.window_width),
            padding=(0, self.window_width // 2),
            bias=False,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        return conv1, conv2

    @GaussianFilter.sigma.setter
    def sigma(self, value: float) -> None:
        """Setter for sigma."""
        self._sigma = value
        self._kernel = self.compute_kernel(
            self._sigma,
            window_radius=self.window_radius,
        )
        kernel_1d = self.compute_kernel_profile(
            self._sigma,
            window_radius=self.window_radius,
        )
        self._conv = self._build_convolution()
        self._conv[0].weight.data = kernel_1d.unsqueeze(0).unsqueeze(0)
        self._conv[1].weight.data = kernel_1d.T.unsqueeze(0).unsqueeze(0)

    def __call__(self, to_filter: torch.Tensor) -> torch.Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter, (p,q)-shaped.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        conv1, conv2 = self._conv
        to_filter_expanded = to_filter.unsqueeze(0).unsqueeze(0)
        filtered: torch.Tensor = conv2(conv1(to_filter_expanded)).detach()
        return filtered.squeeze(0).squeeze(0)

    @classmethod
    def compute_kernel_profile(
        cls,
        sigma: float,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel (2*windows_radius+1,)-shaped.
        """
        r = torch.arange(
            -window_radius,
            window_radius + 1,
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        kernel_1d = torch.exp(-(r**2 / (2 * sigma**2))).unsqueeze(-1)
        kernel_1d /= (kernel_1d @ kernel_1d.T).sum().sqrt()
        return kernel_1d

    @classmethod
    def compute_kernel(
        cls,
        sigma: float,
        window_radius: int,
    ) -> torch.Tensor:
        """Compute the gaussian kernel.

        Args:
            sigma (float): Standard deviation.
            window_radius (int): Window radius.

        Returns:
            torch.Tensor: Gaussian kernel
            (2*windows_radius+1,2*windows_radius+1)-shaped.
        """
        kernel = cls.compute_kernel_profile(
            sigma=sigma,
            window_radius=window_radius,
        )
        return kernel @ kernel.T
