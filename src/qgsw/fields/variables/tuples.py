"""UVH object."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing_extensions import Self

from qgsw.exceptions import ParallelSlicingError
from qgsw.fields.variables.covariant import (
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalZonalVelocity,
)
from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    PrognosticPotentialVorticity,
    PrognosticSST,
    PrognosticStreamFunction,
    Time,
)
from qgsw.specs import defaults
from qgsw.utils import tensorio

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from typing import (
    TYPE_CHECKING,
    Generic,
    NamedTuple,
    TypeVar,
    overload,
)

import torch

if TYPE_CHECKING:
    from pathlib import Path


class BaseTuple:
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


P = TypeVar("P", bound=BaseTuple)


class ParallelSlice(Generic[P]):
    """Parallele slicing for prognostic tuples."""

    def __init__(self, prognostic: P, slice_depth: int | None = None) -> None:
        """Instantiate the parallel slice.

        Args:
            prognostic (P): Prognostic tuple.
            slice_depth (int | None, optional): Slice depth. Defaults to None.
        """
        self._prognostic = prognostic
        self._slice_depth = slice_depth

    def _raise_if_invalid_depth(self, key: tuple[int | slice, ...]) -> None:
        """Raise an error if the key depth is greater than self._slice_depth.

        Args:
            key (tuple[int  |  slice, ...]): Slicing tuple.

        Raises:
            ParallelSlicingError: If the key depth is greater than
            self._slice_depth
        """
        if self._slice_depth is None:
            return
        if len(key) <= self._slice_depth:
            return
        msg = (
            f"Slicing can only be done on the first {self._slice_depth}"
            " dimension(s)."
        )
        raise ParallelSlicingError(msg)

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> P:
        """__getitem__ magic method for parallel slicing.

        Args:
            key (int | slice | tuple[int  |  slice, ...]): Key.

        Returns:
            P: Sliced prognostic tuple.
        """
        self._raise_if_invalid_depth(key)
        return self._prognostic.__class__(*[v[key] for v in self._prognostic])


class BasePSIQ(BaseTuple, metaclass=ABCMeta):
    """Base Prognostic tuple for PSIQ tuple."""

    psi: torch.Tensor
    q: torch.Tensor

    @property
    @abstractmethod
    def psiq(self) -> PSIQ:
        """PSIQ."""


class BasePSIQSST(BaseTuple, metaclass=ABCMeta):
    """Base Prognostic tuple for PSIQ tuple."""

    psi: torch.Tensor
    q: torch.Tensor
    sst: torch.Tensor

    @property
    @abstractmethod
    def psiq(self) -> PSIQ:
        """PSIQ."""

    @property
    @abstractmethod
    def psiqsst(self) -> PSIQSST:
        """PSIQSST."""


class BaseUVH(BaseTuple, metaclass=ABCMeta):
    """Base Prognostic tuple for UVH tuple."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor

    @property
    @abstractmethod
    def uvh(self) -> UVH:
        """UVH."""

    @abstractmethod
    def with_uvh(self, uvh: UVH) -> Self:
        """Update the tuple by specifying new UVH.

        Usefult to switch from covariant to physical and reverse.

        Args:
            uvh (UVH): UVH.

        Returns:
            Self: New tuple.
        """


class _PSIQ(NamedTuple):
    """Stream function and potential vorticity."""

    psi: torch.Tensor
    q: torch.Tensor


class _PSIQSST(NamedTuple):
    """Stream function, potential vorticity and sst."""

    psi: torch.Tensor
    q: torch.Tensor
    sst: torch.Tensor


class _PSIQT(NamedTuple):
    """Stream function, potential vorticity and time."""

    psi: torch.Tensor
    q: torch.Tensor
    t: torch.Tensor


class _PSIQSSTT(NamedTuple):
    """Stream function, potential vorticity, sst and time."""

    psi: torch.Tensor
    q: torch.Tensor
    sst: torch.Tensor
    t: torch.Tensor


class _PSIQTAlpha(NamedTuple):
    """Stream function, potential vorticity and time."""

    psi: torch.Tensor
    q: torch.Tensor
    t: torch.Tensor
    alpha: torch.Tensor


class _PSIQSSTTAlpha(NamedTuple):
    """Stream function, potential vorticity, sst, time and alpha."""

    psi: torch.Tensor
    q: torch.Tensor
    sst: torch.Tensor
    t: torch.Tensor
    alpha: torch.Tensor


class PSIQ(BasePSIQ, _PSIQ):
    """Stream function and potential vorticity."""

    @property
    def psiq(self) -> PSIQ:
        """PSIQ."""
        return self

    @property
    def parallel_slice(self) -> ParallelSlice[PSIQ]:
        """Parallel slicing object."""
        return ParallelSlice(self, slice_depth=2)

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
        data: dict[str, torch.Tensor] = tensorio.load(
            file,
            dtype=dtype,
            device=device,
        )
        psi_name = PrognosticStreamFunction.get_name()
        psi = data[psi_name]
        q_name = PrognosticPotentialVorticity.get_name()
        q = data[q_name]
        return cls(psi=psi, q=q)

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return (
            PrognosticStreamFunction.get_name() in keys
            and PrognosticPotentialVorticity.get_name() in keys
        )


class PSIQSST(BasePSIQSST, _PSIQSST):
    """Stream function, potential vorticity and sst."""

    steady_sst_value: int = 18

    @property
    def psiq(self) -> PSIQ:
        """PSIQ."""
        return PSIQ(self.psi, self.q)

    @property
    def psiqsst(self) -> PSIQSST:
        """PSIQSST."""
        return self

    @property
    def parallel_slice(self) -> ParallelSlice[PSIQSST]:
        """Parallel slicing object."""
        return ParallelSlice(self, slice_depth=2)

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
        """Instantiate a steady PSIQSST with zero psi and q and constant sst.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQSST.
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
        sst = (
            torch.ones(
                (n_ens, 1, nx, ny),
                dtype=dtype,
                device=device,
            )
            * cls.steady_sst_value
        )
        return cls(psi=psi, q=q, sst=sst)

    @classmethod
    def from_file(
        cls,
        file: Path,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate PSIQSST from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQSST.
        """
        data: dict[str, torch.Tensor] = tensorio.load(
            file,
            dtype=dtype,
            device=device,
        )
        psi_name = PrognosticStreamFunction.get_name()
        psi = data[psi_name]
        q_name = PrognosticPotentialVorticity.get_name()
        q = data[q_name]
        sst_name = PrognosticSST.get_name()
        sst = data[sst_name]
        return cls(psi=psi, q=q, sst=sst)

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return (
            PrognosticStreamFunction.get_name() in keys
            and PrognosticPotentialVorticity.get_name() in keys
            and PrognosticSST.get_name() in keys
        )

    @classmethod
    def from_psiq(cls, sst: torch.Tensor, psiq: PSIQ) -> Self:
        """Instantiate from PSIQ.

        Args:
            sst (torch.Tensor): Sea surface temperature field.
            psiq (PSIQ): PSIQ.

        Returns:
            Self: PSIQSST
        """
        return cls(psiq.psi, psiq.q, sst)


class PSIQT(BasePSIQ, _PSIQT):
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
        data = tensorio.load(file, dtype=dtype, device=device)
        t = data[Time.get_name()]
        return cls.from_psiq(
            t,
            PSIQ.from_file(file, dtype=dtype, device=device),
        )

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return Time.get_name() in keys and PSIQ.is_file_readable(file)


class PSIQSSTT(BasePSIQSST, _PSIQSSTT):
    """Stream function, potential vorticity, sst and time."""

    @property
    def psiq(self) -> PSIQ:
        """PSIQ."""
        return PSIQ(self.psi, self.q)

    @property
    def psiqsst(self) -> PSIQSSTT:
        """PSIQ."""
        return PSIQSST(self.psi, self.q, self.sst)

    def __mul__(self, other: float) -> PSIQT:
        """Left multiplication."""
        return PSIQSSTT.from_psiqsst(self.t, self.psiqsst.__mul__(other))

    def __add__(self, other: PSIQT) -> PSIQT:
        """Addition."""
        return PSIQSSTT.from_psiqsst(self.t, self.psiqsst.__add__(other))

    def increment_time(self, dt: float) -> PSIQT:
        """Increment time.

        Args:
            dt (float): Timestep.

        Returns:
            PSIQT: PSIQT
        """
        return PSIQSSTT.from_psiqsst(self.t + dt, self.psiqsst)

    @classmethod
    def from_psiqsst(cls, t: torch.Tensor, psiqsst: PSIQSST) -> Self:
        """Instantiate from PSIQSSTT.

        Args:
            t (torch.Tensor): Time.
            psiqsst (PSIQSST): PSIQSST.

        Returns:
            Self: PSIQT
        """
        return cls(psiqsst.psi, psiqsst.q, psiqsst.sst, t)

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
            Self: PSIQSSTT.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        return cls.from_psiqsst(
            torch.zeros((n_ens,), dtype=dtype, device=device),
            PSIQSST.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
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
            Self: PSIQSSTT.
        """
        data = tensorio.load(file, dtype=dtype, device=device)
        t = data[Time.get_name()]
        return cls.from_psiqsst(
            t,
            PSIQSST.from_file(file, dtype=dtype, device=device),
        )

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return Time.get_name() in keys and PSIQSST.is_file_readable(file)


class PSIQTAlpha(BasePSIQ, _PSIQTAlpha):
    """Time, alpha, stream function and potential vorticity."""

    @property
    def psiq(self) -> PSIQ:
        """PSIQ."""
        return PSIQ(self.psi, self.q)

    @property
    def psiqt(self) -> PSIQT:
        """PSIQT."""
        return PSIQT(self.psi, self.q, self.t)

    def __mul__(self, other: float) -> PSIQTAlpha:
        """Left multiplication."""
        return PSIQTAlpha.from_psiqt(
            self.alpha,
            PSIQT.from_psiq(self.t, self.psiq.__mul__(other)),
        )

    def __add__(self, other: PSIQTAlpha) -> PSIQTAlpha:
        """Addition."""
        return PSIQTAlpha.from_psiqt(
            self.alpha,
            PSIQT.from_psiq(self.t, self.psiq.__add__(other)),
        )

    def increment_time(self, dt: float) -> PSIQTAlpha:
        """Increment time.

        Args:
            dt (float): Timestep.

        Returns:
            PSIQTAlpha: PSIQTAlpha.
        """
        return PSIQTAlpha.from_psiqt(self.alpha, self.psiqt.increment_time(dt))

    @classmethod
    def from_psiq(
        cls,
        t: torch.Tensor,
        alpha: torch.Tensor,
        psiq: PSIQ,
    ) -> Self:
        """Instantiate from PSIQ.

        Args:
            t (torch.Tensor): Time.
            alpha (torch.Tensor): Alpha.
            psiq (PSIQ): PSIQ.

        Returns:
            Self: PSIQTAlpha
        """
        return cls(t=t, psi=psiq.psi, q=psiq.q, alpha=alpha)

    @classmethod
    def from_psiqt(cls, alpha: torch.Tensor, psiqt: PSIQT) -> Self:
        """Instantiate from PSIQT.

        Args:
            alpha (torch.Tensor): Alpha.
            psiqt (PSIQT): PSIQT.

        Returns:
            Self: PSIQTAlpha
        """
        return cls(t=psiqt.t, psi=psiqt.psi, q=psiqt.q, alpha=alpha)

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
        """Instantiate a PSIQTAlpha with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQTAlpha.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        return cls.from_psiqt(
            torch.zeros(
                (n_ens, 1, nx + 1, ny + 1),
                dtype=dtype,
                device=device,
            ),
            PSIQT.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate PSIQTAlpha from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQTAlpha.
        """
        data = tensorio.load(file, dtype=dtype, device=device)
        alpha = data[CollinearityCoefficient.get_name()]
        return cls.from_psiqt(
            alpha,
            PSIQT.from_file(file, dtype=dtype, device=device),
        )

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return (
            CollinearityCoefficient.get_name() in keys
            and PSIQT.is_file_readable(file)
        )


class PSIQSSTTAlpha(BasePSIQSST, _PSIQSSTTAlpha):
    """Time, alpha, stream function and potential vorticity."""

    @property
    def psiq(self) -> PSIQ:
        """PSIQ."""
        return PSIQ(self.psi, self.q)

    @property
    def psiqt(self) -> PSIQT:
        """PSIQT."""
        return PSIQT(self.psi, self.q, self.t)

    @property
    def psiqsst(self) -> PSIQSST:
        """PSIQSST."""
        return PSIQSST(self.psi, self.q, self.sst)

    @property
    def psiqsstt(self) -> PSIQSSTT:
        """PSIQSSTT."""
        return PSIQSSTT(self.psi, self.q, self.sst, self.t)

    def __mul__(self, other: float) -> PSIQSSTTAlpha:
        """Left multiplication."""
        return PSIQSSTTAlpha.from_psiqsstt(
            self.alpha,
            PSIQSSTT.from_psiqsst(self.t, self.psiqsst.__mul__(other)),
        )

    def __add__(self, other: PSIQSSTTAlpha) -> PSIQSSTTAlpha:
        """Addition."""
        return PSIQSSTTAlpha.from_psiqsstt(
            self.alpha,
            PSIQSSTT.from_psiqsst(self.t, self.psiqsst.__add__(other)),
        )

    def increment_time(self, dt: float) -> PSIQSSTTAlpha:
        """Increment time.

        Args:
            dt (float): Timestep.

        Returns:
            PSIQSSTTAlpha: PSIQSSTTAlpha.
        """
        return PSIQSSTTAlpha.from_psiqsstt(
            self.alpha, self.psiqsstt.increment_time(dt)
        )

    @classmethod
    def from_psiqsst(
        cls,
        t: torch.Tensor,
        alpha: torch.Tensor,
        psiqsst: PSIQSST,
    ) -> Self:
        """Instantiate from PSIQSST.

        Args:
            t (torch.Tensor): Time.
            alpha (torch.Tensor): Alpha.
            psiqsst (PSIQSST): PSIQSST.

        Returns:
            Self: PSIQSSTTAlpha
        """
        return cls(
            t=t,
            psi=psiqsst.psi,
            q=psiqsst.q,
            sst=psiqsst.sst,
            alpha=alpha,
        )

    @classmethod
    def from_psiqsstt(cls, alpha: torch.Tensor, psiqsstt: PSIQSSTT) -> Self:
        """Instantiate from PSIQSSTT.

        Args:
            alpha (torch.Tensor): Alpha.
            psiqsstt (PSIQSSTT): PSIQSSTT.

        Returns:
            Self: PSIQSSTTAlpha
        """
        return cls(
            t=psiqsstt.t,
            psi=psiqsstt.psi,
            q=psiqsstt.q,
            sst=psiqsstt.sst,
            alpha=alpha,
        )

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
        """Instantiate a PSIQSSTTAlpha with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQSSTTAlpha.
        """
        dtype = defaults.get_dtype(dtype)
        device = defaults.get_device(device)
        return cls.from_psiqsstt(
            torch.zeros(
                (n_ens, 1, nx + 1, ny + 1),
                dtype=dtype,
                device=device,
            ),
            PSIQSSTT.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate PSIQSSTTAlpha from a file.

        Args:
            file (Path): File to read.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: PSIQSSTTAlpha.
        """
        data = tensorio.load(file, dtype=dtype, device=device)
        alpha = data[CollinearityCoefficient.get_name()]
        return cls.from_psiqsstt(
            alpha,
            PSIQSSTT.from_file(file, dtype=dtype, device=device),
        )

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return (
            CollinearityCoefficient.get_name() in keys
            and PSIQSSTT.is_file_readable(file)
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


class UVH(BaseUVH, _UVH):
    """Zonal velocity, meridional velocity and layer thickness."""

    @property
    def uvh(self) -> UVH:
        """UVH."""
        return self

    @property
    def parallel_slice(self) -> ParallelSlice[UVH]:
        """Parallel slicing object."""
        return ParallelSlice(self, slice_depth=2)

    def with_uvh(self, uvh: UVH) -> UVH:
        """Update the tuple by specifying new UVH.

        Usefult to switch from covariant to physical and reverse.

        Args:
            uvh (UVH): UVH.

        Returns:
            UVH: New tuple.
        """
        return uvh

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
        data = tensorio.load(file, dtype=dtype, device=device)
        u = data[PhysicalZonalVelocity.get_name()]
        v = data[PhysicalMeridionalVelocity.get_name()]
        h = data[PhysicalLayerDepthAnomaly.get_name()]
        return cls(u=u, v=v, h=h)

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return (
            PhysicalZonalVelocity.get_name() in keys
            and PhysicalMeridionalVelocity.get_name() in keys
            and PhysicalLayerDepthAnomaly.get_name() in keys
        )


class UVHT(BaseUVH, _UVHT):
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

    def with_uvh(self, uvh: UVH) -> UVHT:
        """Update the tuple by specifying new UVH.

        Usefult to switch from covariant to physical and reverse.

        Args:
            uvh (UVH): UVH.

        Returns:
            UVHT: New tuple.
        """
        return UVHT.from_uvh(self.t, uvh)

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
        data: dict[str, torch.Tensor] = tensorio.load(
            file,
            dtype=dtype,
            device=device,
        )
        t = data[Time.get_name()]
        return cls.from_uvh(t, UVH.from_file(file, dtype=dtype, device=device))

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return Time.get_name() in keys and UVH.is_file_readable(file)


class UVHTAlpha(BaseUVH, _UVHTAlpha):
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

    def with_uvh(self, uvh: UVH) -> UVHT:
        """Update the tuple by specifying new UVH.

        Usefult to switch from covariant to physical and reverse.

        Args:
            uvh (UVH): UVH.

        Returns:
            UVHT: New tuple.
        """
        return UVHTAlpha.from_uvh(self.t, self.alpha, uvh)

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
        data: dict[str, torch.Tensor] = tensorio.load(
            file,
            dtype=dtype,
            device=device,
        )
        alpha = data[CollinearityCoefficient.get_name()].to(
            dtype=dtype,
            device=device,
        )
        return cls.from_uvht(
            alpha,
            UVHT.from_file(file, dtype=dtype, device=device),
        )

    @classmethod
    def is_file_readable(cls, file: str | Path) -> bool:
        """Check whether a file is readable.

        Args:
            file (str | Path): File to read.

        Returns:
            bool: True if from_file can read the file.
        """
        keys = tensorio.load(file).keys()
        return (
            CollinearityCoefficient.get_name() in keys
            and UVHT.is_file_readable(file)
        )


@overload
def rand_like(prognostic: PSIQ) -> PSIQ: ...
@overload
def rand_like(prognostic: PSIQT) -> PSIQT: ...
@overload
def rand_like(prognostic: PSIQTAlpha) -> PSIQTAlpha: ...
@overload
def rand_like(prognostic: PSIQSST) -> PSIQSST: ...
@overload
def rand_like(prognostic: PSIQSSTT) -> PSIQSSTT: ...
@overload
def rand_like(prognostic: PSIQSSTTAlpha) -> PSIQSSTTAlpha: ...


def rand_like(
    prognostic: P,
) -> P:
    """Generate random tuple similar to given one.

    Args:
        prognostic (P): Given tuple.

    Returns:
        P: Output.
    """
    return prognostic.__class__(
        **{
            f: torch.rand_like(getattr(prognostic, f))
            for f in prognostic._fields
        }
    )
