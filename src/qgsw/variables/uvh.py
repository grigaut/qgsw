"""UVH object."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    import torch


class UVH(NamedTuple):
    """Zonal velocity, meridional velocity and layer thickness."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor

    def __mul__(self, other: float) -> UVH:
        """Left mutlitplication."""
        return UVH(self.u * other, self.v * other, self.h * other)

    def __rmul__(self, other: float) -> UVH:
        """Right multiplication."""
        return self.__mul__(other)

    def __add__(self, other: UVH) -> UVH:
        """Addition."""
        return UVH(self.u + other.u, self.v + other.v, self.h + other.h)

    def __sub__(self, other: UVH) -> UVH:
        """Substraction."""
        return self.__add__(-1 * other)
