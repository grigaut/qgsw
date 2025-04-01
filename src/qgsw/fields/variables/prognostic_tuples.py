"""UVH object."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    LayerDepthAnomaly,
    MeridionalVelocity,
    PrognosticPotentialVorticity,
    PrognosticStreamFunction,
    Time,
    ZonalVelocity,
)
from qgsw.specs import defaults

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from pathlib import Path


class BasePrognosticTuple:
    """Prognostic tuple base class."""

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


class BasePrognosticPSIQ(BasePrognosticTuple, metaclass=ABCMeta):
    """Base Prognostic tuple for PSIQ tuple."""

    psi: torch.Tensor
    q: torch.Tensor

    @property
    @abstractmethod
    def psiq(self) -> PSIQ:
        """PSIQ."""


class BasePrognosticUVH(BasePrognosticTuple, metaclass=ABCMeta):
    """Base Prognostic tuple for UVH tuple."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor

    @property
    @abstractmethod
    def uvh(self) -> UVH:
        """UVH."""


class _PSIQ(NamedTuple):
    """Stream function and potential vorticity."""

    psi: torch.Tensor
    q: torch.Tensor


class _PSIQT(NamedTuple):
    """Stream function, potential vorticity and time."""

    psi: torch.Tensor
    q: torch.Tensor
    t: torch.Tensor


class PSIQ(BasePrognosticPSIQ, _PSIQ):
    """Stream function and potential vorticity."""

    @property
    def psiq(self) -> PSIQ:
        """PSIQ."""
        return self

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a steady PSIQ with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQ.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        psi = torch.zeros(
            (n_ens, nl, nx + 1, ny + 1),
            dtype=dtype,
            device=device,
        )
        q = torch.zeros(
            (n_ens, nl, nx, ny),
            dtype=dtype,
            device=device,
        )
        return cls(psi=psi, q=q)

    @classmethod
    def from_file(
        cls,
        file: Path,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate PSIQ from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQ.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        data: dict[str, torch.Tensor] = torch.load(file)
        psi_name = PrognosticStreamFunction.get_name()
        psi = data[psi_name].to(dtype=dtype, device=device)
        q_name = PrognosticPotentialVorticity.get_name()
        q = data[q_name].to(dtype=dtype, device=device)
        return cls(psi=psi, q=q)


class PSIQT(BasePrognosticPSIQ, _PSIQT):
    """Stream function, potential vorticity and time."""

    @property
    def psiq(self) -> PSIQ:
        """PSIQ."""
        return PSIQ(self.psi, self.q)

    def __mul__(self, other: float) -> PSIQT:
        """Left multiplication."""
        return PSIQT.from_psiq(self.t, self.psiq.__mul__(other))

    def __add__(self, other: PSIQT) -> PSIQT:
        """Addition."""
        return PSIQT.from_psiq(self.t, self.psiq.__add__(other))

    def increment_time(self, dt: float) -> PSIQT:
        """Increment time.

        Args:
            dt (float): Timestep.

        Returns:
            PSIQT: PSIQT
        """
        return PSIQT.from_psiq(self.t + dt, self.psiq)

    @classmethod
    def from_psiq(cls, t: torch.Tensor, psiq: PSIQ) -> Self:
        """Instantiate from PSIQ.

        Args:
            t (torch.Tensor): Time.
            psiq (PSIQ): PSIQ.

        Returns:
            Self: PSIQT
        """
        return cls(psiq.psi, psiq.q, t)

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a PSIQT with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQT.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        return cls.from_psiq(
            torch.zeros((n_ens,), dtype=dtype, device=device),
            PSIQ.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate PSIQT from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQT.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        data: dict[str, torch.Tensor] = torch.load(file)
        t = data[Time.get_name()].to(dtype=dtype, device=device)
        return cls.from_psiq(
            t,
            PSIQ.from_file(file, dtype=dtype, device=device),
        )


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


class UVH(BasePrognosticUVH, _UVH):
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
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a steady UVH with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: UVH.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
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
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate UVH from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: UVH.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        data: dict[str, torch.Tensor] = torch.load(file)
        u = data[ZonalVelocity.get_name()].to(dtype=dtype, device=device)
        v = data[MeridionalVelocity.get_name()].to(dtype=dtype, device=device)
        h = data[LayerDepthAnomaly.get_name()].to(dtype=dtype, device=device)
        return cls(u=u, v=v, h=h)


class UVHT(BasePrognosticUVH, _UVHT):
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
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a UVHT with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: UVHT.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        return cls.from_uvh(
            torch.zeros((n_ens,), dtype=dtype, device=device),
            UVH.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate UVHT from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: UVHT.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        data: dict[str, torch.Tensor] = torch.load(file)
        t = data[Time.get_name()].to(dtype=dtype, device=device)
        return cls.from_uvh(t, UVH.from_file(file, dtype=dtype, device=device))


class UVHTAlpha(BasePrognosticUVH, _UVHTAlpha):
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
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a UVHTAlpha with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: UVHTAlpha.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        return cls.from_uvht(
            torch.zeros((n_ens, 1, nx, ny), dtype=dtype, device=device),
            UVHT.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate UVHTAlpha from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: UVHTAlpha.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        data: dict[str, torch.Tensor] = torch.load(file)
        alpha = data[CollinearityCoefficient.get_name()].to(
            dtype=dtype,
            device=device,
        )
        return cls.from_uvht(
            alpha,
            UVHT.from_file(file, dtype=dtype, device=device),
        )
