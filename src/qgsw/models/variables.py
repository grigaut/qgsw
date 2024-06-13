"""Variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch


class UVH(NamedTuple):
    """Zonal velocity, meridional velocity and layer thickness."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor

    def __mul__(self, value: float) -> UVH:
        """Left mutlitplication."""
        return UVH(self.u * value, self.v * value, self.h * value)

    def __rmul__(self, value: float) -> UVH:
        """Right multiplication."""
        return self.__mul__(value)

    def __add__(self, value: UVH) -> UVH:
        """Addition."""
        return UVH(self.u + value.u, self.v + value.v, self.h + value.h)

    def __sub__(self, value: UVH) -> UVH:
        """Substraction."""
        return UVH(self.u - value.u, self.v - value.v, self.h - value.h)
