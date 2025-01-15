"""UVH object."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    LayerDepthAnomaly,
    MeridionalVelocity,
    Time,
    ZonalVelocity,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from typing import TYPE_CHECKING, NamedTuple, TypeVar, Union

import torch

if TYPE_CHECKING:
    from pathlib import Path


class BasePrognosticTuple(ABC):
    """Prognostic tuple base class."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor

    @property
    @abstractmethod
    def uvh(self) -> UVH:
        """UVH."""

    def __mul__(self, other: float) -> Self:
        """Left multiplication."""
        return self.__class__(*[v * other for v in self])

    def __rmul__(self, other: float) -> Self:
        """Right multiplication."""
        return self.__mul__(other)

    def __add__(self, other: Self) -> Self:
        """Addition."""
        return self.__class__(*[s + o for s, o in zip(self, other)])

    def __sub__(self, other: Self) -> Self:
        """Substraction."""
        return self.__add__(-1 * other)


class _UVH(NamedTuple):
    """Zonal velocity, meridional velocity and layer thickness."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor


class _UVHT(NamedTuple):
    """Zonal velocity, meridional velocity and layer thickness."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor
    t: torch.Tensor


class _UVHTAlpha(NamedTuple):
    """Zonal velocity, meridional velocity, layer thickness and alpha."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor
    t: torch.Tensor
    alpha: torch.Tensor


class UVH(BasePrognosticTuple, _UVH):
    """Zonal velocity, meridional velocity and layer thickness."""

    @property
    def uvh(self) -> UVH:
        """UVH."""
        return self

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
        data: dict[str, torch.Tensor] = torch.load(file, weights_only=True)
        u = data[ZonalVelocity.get_name()].to(dtype=dtype, device=device)
        v = data[MeridionalVelocity.get_name()].to(dtype=dtype, device=device)
        h = data[LayerDepthAnomaly.get_name()].to(dtype=dtype, device=device)
        return cls(u=u, v=v, h=h)


class UVHT(BasePrognosticTuple, _UVHT):
    """Time, Zonal velocity, meridional velocity and layer thickness."""

    @property
    def uvh(self) -> UVH:
        """UVH."""
        return UVH(self.u, self.v, self.h)

    def __mul__(self, other: float) -> UVHT:
        """Left multiplication."""
        return UVHT.from_uvh(self.t, self.uvh.__mul__(other))

    def __add__(self, other: UVHT) -> UVHT:
        """Addition."""
        return UVHT.from_uvh(self.t, self.uvh.__add__(other))

    def increment_time(self, dt: float) -> UVHT:
        """Increment time.

        Args:
            dt (float): Timestep.

        Returns:
            UVHT: UVHT
        """
        return UVHT.from_uvh(self.t + dt, self.uvh)

    @classmethod
    def from_uvh(cls, t: torch.Tensor, uvh: UVH) -> Self:
        """Instantiate from UVH.

        Args:
            t (torch.Tensor): Time.
            uvh (UVH): UVH.

        Returns:
            Self: UVHT
        """
        return cls(uvh.u, uvh.v, uvh.h, t)

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
        """Instantiate a UVHT with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVHT.
        """
        return cls.from_uvh(
            torch.zeros((n_ens,), dtype=dtype, device=device),
            UVH.steady(n_ens, nl, nx, ny, dtype, device),
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate UVHT from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVHT.
        """
        data: dict[str, torch.Tensor] = torch.load(file, weights_only=True)
        t = data[Time.get_name()].to(dtype=dtype, device=device)
        return cls.from_uvh(t, UVH.from_file(file, dtype, device))


class UVHTAlpha(BasePrognosticTuple, _UVHTAlpha):
    """Time, alpha, Zonal velocity,meridional velocity and layer thickness."""

    @property
    def uvh(self) -> UVH:
        """UVH."""
        return UVH(self.u, self.v, self.h)

    @property
    def uvht(self) -> UVHT:
        """UVHT."""
        return UVHT(self.u, self.v, self.h, self.t)

    def __mul__(self, other: float) -> UVHTAlpha:
        """Left multiplication."""
        return UVHTAlpha.from_uvh(
            self.t,
            self.alpha,
            UVHT.from_uvh(self.t, self.uvh.__mul__(other)),
        )

    def __add__(self, other: UVHTAlpha) -> UVHTAlpha:
        """Addition."""
        return UVHTAlpha.from_uvh(
            self.t,
            self.alpha,
            UVHT.from_uvh(self.t, self.uvh.__add__(other)),
        )

    def increment_time(self, dt: float) -> UVHTAlpha:
        """Increment time.

        Args:
            dt (float): Timestep.

        Returns:
            UVHTAlpha: UVHTAlpha.
        """
        return UVHTAlpha.from_uvht(self.alpha, self.uvht.increment_time(dt))

    @classmethod
    def from_uvh(cls, t: torch.Tensor, alpha: torch.Tensor, uvh: UVH) -> Self:
        """Instantiate from UVH.

        Args:
            t (torch.Tensor): Time.
            alpha (torch.Tensor): Alpha.
            uvh (UVH): UVH.

        Returns:
            Self: UVHTAlpha
        """
        return cls(t=t, u=uvh.u, v=uvh.v, h=uvh.h, alpha=alpha)

    @classmethod
    def from_uvht(cls, alpha: torch.Tensor, uvht: UVHT) -> Self:
        """Instantiate from UVHT.

        Args:
            alpha (torch.Tensor): Alpha.
            uvht (UVHT): UVHT.

        Returns:
            Self: UVHTAlpha
        """
        return cls(t=uvht.t, u=uvht.u, v=uvht.v, h=uvht.h, alpha=alpha)

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
        """Instantiate a UVHTAlpha with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVHTAlpha.
        """
        return cls.from_uvht(
            alpha,
            UVHT.steady(n_ens, nl, nx, ny, dtype, device),
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate UVHTAlpha from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: UVHTAlpha.
        """
        data: dict[str, torch.Tensor] = torch.load(file, weights_only=True)
        alpha = data[CollinearityCoefficient.get_name()].to(
            dtype=dtype,
            device=device,
        )
        return cls.from_uvht(alpha, UVHT.from_file(file, dtype, device))


PrognosticTuple = TypeVar("PrognosticTuple", bound=Union[UVHT, UVHTAlpha])
