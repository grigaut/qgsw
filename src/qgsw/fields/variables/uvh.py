"""UVH object."""

from __future__ import annotations

import numpy as np

from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    LayerDepthAnomaly,
    MeridionalVelocity,
    ZonalVelocity,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from pathlib import Path


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

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate a steady UVH with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVH.
        """
        h = torch.zeros(
            (n_ens, nl, nx, ny),
            dtype=dtype,
            device=device,
        )
        u = torch.zeros(
            (n_ens, nl, nx + 1, ny),
            dtype=dtype,
            device=device,
        )
        v = torch.zeros(
            (n_ens, nl, nx, ny + 1),
            dtype=dtype,
            device=device,
        )
        return cls(u=u, v=v, h=h)

    @classmethod
    def from_file(
        cls,
        file: Path,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate UVH from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVH.
        """
        data = np.load(file)
        u_name = ZonalVelocity.get_name()
        u = torch.tensor(data[u_name]).to(dtype=dtype, device=device)
        v_name = MeridionalVelocity.get_name()
        v = torch.tensor(data[v_name]).to(dtype=dtype, device=device)
        h_name = LayerDepthAnomaly.get_name()
        h = torch.tensor(data[h_name]).to(dtype=dtype, device=device)
        return cls(u=u, v=v, h=h)


class UVHalpha(NamedTuple):
    """Zonal velocity, meridional velocity and layer thickness."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor
    alpha: torch.Tensor

    def __mul__(self, other: float) -> UVHalpha:
        """Left mutlitplication."""
        return UVHalpha(
            self.u * other,
            self.v * other,
            self.h * other,
            self.alpha * other,
        )

    def __rmul__(self, other: float) -> UVHalpha:
        """Right multiplication."""
        return self.__mul__(other)

    def __add__(self, other: UVHalpha) -> UVHalpha:
        """Addition."""
        return UVHalpha(
            self.u + other.u,
            self.v + other.v,
            self.h + other.h,
            self.alpha + other.alpha,
        )

    def __sub__(self, other: UVHalpha) -> UVHalpha:
        """Substraction."""
        return self.__add__(-1 * other)

    @classmethod
    def steady(
        cls,
        alpha: torch.Tensor,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate a steady UVHalpha with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVHalpha.
        """
        h = torch.zeros(
            (n_ens, nl, nx, ny),
            dtype=dtype,
            device=device,
        )
        u = torch.zeros(
            (n_ens, nl, nx + 1, ny),
            dtype=dtype,
            device=device,
        )
        v = torch.zeros(
            (n_ens, nl, nx, ny + 1),
            dtype=dtype,
            device=device,
        )
        return cls(u=u, v=v, h=h, alpha=alpha)

    @classmethod
    def from_file(
        cls,
        file: Path,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate UVHalpha from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVHalpha.
        """
        data = np.load(file)
        u_name = ZonalVelocity.get_name()
        u = torch.tensor(data[u_name]).to(dtype=dtype, device=device)
        v_name = MeridionalVelocity.get_name()
        v = torch.tensor(data[v_name]).to(dtype=dtype, device=device)
        h_name = LayerDepthAnomaly.get_name()
        h = torch.tensor(data[h_name]).to(dtype=dtype, device=device)
        alpha_name = CollinearityCoefficient.get_name()
        alpha = torch.tensor(data[alpha_name]).to(dtype=dtype, device=device)
        return cls(u=u, v=v, h=h, alpha=alpha)
