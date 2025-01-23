"""Base filter class."""

from abc import ABC, abstractmethod

import torch


class _Filter(ABC):
    @property
    @abstractmethod
    def kernel(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def window_radius(self) -> int: ...

    @property
    def window_width(self) -> int:
        """Window radius."""
        return self.compute_window_width(self.window_radius)

    @abstractmethod
    def __call__(self, to_filter: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    def compute_window_width(window_radius: int) -> int:
        """Compute the window width.

        Args:
            window_radius (int): Window radius.

        Returns:
            int: Window width
        """
        return window_radius * 2 + 1
