"""Interpolation tools."""

from collections.abc import Iterator
from functools import cached_property
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class LinearInterpolation(Generic[T]):
    """Linear Interpolation."""

    def __init__(
        self,
        xs: Iterator[float],
        ys: list[T],
        *,
        remove_x_offset: bool = True,
    ) -> None:
        """Instantiate the linear interpolation.

        Args:
            xs (Iterator[float]): X values.
            ys (list[T]): Y values.
            remove_x_offset (bool, optional): Whether to remove an eventual x
            offset or not. Defaults to True.
        """
        xx = torch.tensor(list(xs), dtype=torch.float64)
        argsort = torch.argsort(xx)
        self._xs = xx[argsort]
        if remove_x_offset:
            self._xs = self._xs - self._xs[0]
        self._xmin = self._xs[0]
        self._xmax = self._xs[-1]
        self._ys = ys

    @cached_property
    def xmax(self) -> float:
        """Maximum x of the interpolation."""
        return self._xmax.item()

    @cached_property
    def xmin(self) -> float:
        """Minimum x of the interpolation."""
        return self._xmin.item()

    def __call__(self, x: float) -> T:
        """Get the y value at a specific x.

        Args:
            x (float): X value.

        Returns:
            T: Interpolation of ys evaluated at x.
        """
        if x < self._xmin or x > self._xmax:
            msg = (
                f"X must be greater than {self._xmin} "
                f"and lower than {self._xmax}"
            )
            raise ValueError(msg)
        # Find the two surrounding time points
        i = torch.searchsorted(self._xs, x, right=True)
        x0, x1 = self._xs[i - 1].item(), self._xs[i].item()
        y0, y1 = self._ys[i - 1], self._ys[i]
        # Linear interpolation
        alpha = (x - x0) / (x1 - x0)
        return y0 * (1 - alpha) + y1 * alpha
