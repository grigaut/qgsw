"""Basis VarDyn connector."""

from typing import Any, Self

import numpy as np
import torch
from scipy.sparse import csc_matrix

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.space.vd import SupportLayout, VDSpaceSupport
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.decomposition.supports.time.vd import VDTimeSupport
from qgsw.eNATL60.interpolation import xy_to_lonlat
from qgsw.logging.core import getLogger
from qgsw.specs import defaults

logger = getLogger(__name__)


class VDBasis(SpaceTimeDecomposition[VDSpaceSupport, VDTimeSupport]):
    """Basis using VarDyn's basis specs."""

    type = "wavelets-vd"
    _n_theta = 10

    @property
    def n_theta(self) -> int:
        """Number of orientations to consider."""
        return self._n_theta

    @n_theta.setter
    def n_theta(self, n_theta: int) -> None:
        self._n_theta = n_theta
        self._thetas = (
            torch.linspace(0, 1, self.n_theta + 1, **self._specs)[:-1]
            * torch.pi
        )

    def __init__(
        self,
        space_params: dict[int, dict[str, Any]],
        time_params: dict[int, dict[str, Any]],
        *,
        lons0: torch.Tensor,
        lat0: torch.Tensor,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate the basis.

        Args:
            space_params (dict[int, csc_matrix]): Space parameters.
            time_params (dict[int, list[torch.Tensor]]): Time parameters.
            lons0 (torch.tensor): Reference longitudes.
            lat0 (torch.Tensor): Reference latitude.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        self._check_validity(space_params, time_params)
        msg = f"Instantiating {self.__class__.__name__}."
        logger.info(msg)

        self._specs = defaults.get(dtype=dtype, device=device)
        self._space = space_params
        self._time = time_params

        self.lons0 = lons0
        self.lat0 = lat0

    @property
    def order(self) -> int:
        """Basis order."""
        return len(self._time)

    def _check_validity(
        self,
        space_params: dict[int, Any],
        time_params: dict[int, Any],
    ) -> None:
        """Check parameters validity."""
        if space_params.keys() != time_params.keys():
            msg = "Mismatching keys between space and time parameters."
            raise ValueError(msg)

    def convert_to_lonlat(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert cartesian coordinates to longitude / latitudes.

        WARNING: the lons0 tensor must be centered on the same points as xx.

        Args:
            xx (torch.Tensor): X coordinates.
            yy (torch.Tensor): Y coordinates.

        Raises:
            ValueError: If there not enough values in lons0

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Longitudes, latitudes._
        """
        nx = xx.shape[1]
        nlons0 = self.lons0.shape[1]

        if nlons0 < nx:
            msg = (
                f"Not enough reference longitudes ({nlons0})"
                f" to convert cartesian coordinates ({nx} points)."
            )
            raise ValueError(msg)
        if (nlons0 - nx) % 2 == 1:
            lons0 = (self.lons0[:, 1:] + self.lons0[:, :-1]) / 2
        else:
            lons0 = self.lons0

        b = (nlons0 - nx) // 2
        if b != 0:
            lons0 = lons0[:, b:-b]

        return xy_to_lonlat(xx, yy, lons0, self.lat0)

    def set_coefs(self, coefs: DecompositionCoefs) -> None:
        """Set coefficients values.

        To ensure consistent coefficients shapes, best is to use
        self.generate_random_coefs().

        Args:
            coefs (DecompositionCoefs): Level -> coefficients.
                ├── 0: coefficents for level 0.
                ├── ...
                └── order-1: coefficient for level order-1
        """
        self._coefs = coefs

    def localize(self, xx: torch.Tensor, yy: torch.Tensor) -> VDTimeSupport:  # noqa: D102
        space_fields, layouts = self._build_space(xx=xx, yy=yy)

        return self.generate_time_support(
            self._time, space_fields, layouts=layouts, shape=xx.shape
        )

    localize.__doc__ = SpaceTimeDecomposition[
        VDSpaceSupport, VDTimeSupport
    ].localize.__doc__

    def localize_dx(self, xx: torch.Tensor, yy: torch.Tensor) -> VDTimeSupport:  # noqa: D102
        space_fields, layouts = self._build_space_dx(xx=xx, yy=yy)

        return self.generate_time_support(
            self._time, space_fields, layouts=layouts, shape=xx.shape
        )

    localize_dx.__doc__ = SpaceTimeDecomposition[
        VDSpaceSupport, VDTimeSupport
    ].localize_dx.__doc__

    def localize_dy(self, xx: torch.Tensor, yy: torch.Tensor) -> VDTimeSupport:  # noqa: D102
        space_fields, layouts = self._build_space_dy(xx=xx, yy=yy)

        return self.generate_time_support(
            self._time, space_fields, layouts=layouts, shape=xx.shape
        )

    localize_dy.__doc__ = SpaceTimeDecomposition[
        VDSpaceSupport, VDTimeSupport
    ].localize_dy.__doc__

    def localize_dx2(  # noqa: D102
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> VDTimeSupport:
        space_fields, layouts = self._build_space_dx2(xx=xx, yy=yy)

        return self.generate_time_support(
            self._time, space_fields, layouts=layouts, shape=xx.shape
        )

    localize_dx2.__doc__ = SpaceTimeDecomposition[
        VDSpaceSupport, VDTimeSupport
    ].localize_dx2.__doc__

    def localize_dy2(  # noqa: D102
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> VDTimeSupport:
        space_fields, layouts = self._build_space_dy2(xx=xx, yy=yy)

        return self.generate_time_support(
            self._time, space_fields, layouts=layouts, shape=xx.shape
        )

    localize_dy2.__doc__ = SpaceTimeDecomposition[
        VDSpaceSupport, VDTimeSupport
    ].localize_dy2.__doc__

    def localize_laplacian(  # noqa: D102
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> VDTimeSupport:
        dx2, layouts = self._build_space_dx2(xx=xx, yy=yy)
        dy2, _ = self._build_space_dy2(xx=xx, yy=yy)

        lap = {
            lvl: (dx2[lvl][0] + dy2[lvl][0], dx2[lvl][1])
            for lvl in range(self.order)
        }

        return self.generate_time_support(
            self._time, lap, layouts=layouts, shape=xx.shape
        )

    localize_laplacian.__doc__ = SpaceTimeDecomposition[
        VDSpaceSupport, VDTimeSupport
    ].localize_laplacian.__doc__

    def _retrieve_space_supports(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, VDSpaceSupport]:

        lons, lats = self.convert_to_lonlat(xx, yy)

        space_fields = {}

        for lvl, params in self._space.items():
            centers = params["centers"]
            lonsc = torch.tensor([c[0] for c in centers], **self._specs)
            latsc = torch.tensor([c[1] for c in centers], **self._specs)

            k = params["k"]
            Lx = params["Lx"]

            space = VDSpaceSupport(
                lons, lats, lonsc, latsc, k, Lx, self._thetas
            )

            space_fields[lvl] = space

        return space_fields

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, SupportLayout]
    ]:
        supports = self._retrieve_space_supports(xx, yy)

        fields = {}

        for lvl, s in supports.items():
            fields[lvl] = (s.field, self._coefs[lvl])

        return fields, {lvl: s.layout for lvl, s in supports.items()}

    def _build_space_dx(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, SupportLayout]
    ]:
        supports = self._retrieve_space_supports(xx, yy)

        fields = {}

        for lvl, s in supports.items():
            fields[lvl] = (s.dx, self._coefs[lvl])

        return fields, {lvl: s.layout for lvl, s in supports.items()}

    def _build_space_dy(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, SupportLayout]
    ]:
        supports = self._retrieve_space_supports(xx, yy)

        fields = {}

        for lvl, s in supports.items():
            fields[lvl] = (s.dy, self._coefs[lvl])

        return fields, {lvl: s.layout for lvl, s in supports.items()}

    def _build_space_dx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, SupportLayout]
    ]:
        supports = self._retrieve_space_supports(xx, yy)

        fields = {}

        for lvl, s in supports.items():
            fields[lvl] = (s.dx2, self._coefs[lvl])

        return fields, {lvl: s.layout for lvl, s in supports.items()}

    def _build_space_dy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, SupportLayout]
    ]:
        supports = self._retrieve_space_supports(xx, yy)

        fields = {}

        for lvl, s in supports.items():
            fields[lvl] = (s.dy2, self._coefs[lvl])

        return fields, {lvl: s.layout for lvl, s in supports.items()}

    def generate_time_support(
        self,
        time_params: dict[int, list[torch.Tensor]],
        space_fields: dict[int, torch.Tensor],
        layouts: dict[int:SupportLayout],
        shape: tuple[int, ...],
    ) -> VDTimeSupport:
        """Generate time support.

        Args:
            time_params (dict[int, list[torch.Tensor]]): Time params.
            space_fields (dict[int, torch.Tensor]): Space fields.
            layouts (dict[int, SupportLayout]): Layouts.
            shape (tuple[int, ...]): Output shape (nx, ny).

        Returns:
            VDTimeSupport: Time support function.
        """
        return VDTimeSupport(
            time_params,
            space_fields,
            layouts,
            shape,
        )

    def shapes(self) -> dict[int, torch.Size]:  # noqa: D102
        return {
            k: (
                len(self._time[k]["centers"][0]),
                2 * self.n_theta * len(self._space[k]["centers"]),
            )
            for k in self._space
        }

    shapes.__doc__ = SpaceTimeDecomposition[
        VDSpaceSupport, VDTimeSupport
    ].shapes.__doc__

    def get_params(self) -> dict[str, Any]:
        """Return decomposition params as dict.

        Returns:
            dict[str, Any]: Decomposition params.
        """
        return {
            "type": self.type,
            "space": self._space,
            "time": self._time,
            "kwargs": {
                "lons0": self.lons0,
                "lat0": self.lat0,
            },
            "n_theta": self.n_theta,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        """Build class from params dict.

        Args:
            params (dict[str, Any]): Decomposition params.

        Returns:
            Self: Instance of class.
        """
        n_theta = params["n_theta"]
        basis = cls(
            space_params=params["space"],
            time_params=params["time"],
            **params["kwargs"],
        )
        basis.n_theta = n_theta
        return basis


class MIOSTBasis(
    SpaceTimeDecomposition["MIOSTSpaceSupport", "MIOSTTimeSupport"]
):
    """Basis using VarDyn's basis specs."""

    type = "basis-miost"

    def __init__(
        self,
        space_params: dict[int, csc_matrix],
        time_params: dict[int, list[torch.Tensor]],
        *,
        xx: np.ndarray,
        yy: np.ndarray,
        dt: float,
        Nt: dict[int, list[int]],
        Nx: list[int],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate the basis.

        Args:
            space_params (dict[int, csc_matrix]): Space parameters.
            time_params (dict[int, list[torch.Tensor]]): Time parameters.
            xx (np.ndarray): Cartesian x locations.
            yy (np.ndarray): Cartesian y locations.
            dt (float): Timestep.
            Nt (dict[int, list[int]]): Time specs.
            Nx (list[int]): X specs.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        self._check_validity(space_params, time_params)
        msg = f"Instantiating {self.__class__.__name__}."
        logger.info(msg)

        self._specs = defaults.get(dtype=dtype, device=device)
        self._space = space_params
        self._time = time_params
        self._xx = xx
        self._yy = yy
        self._dt = dt
        self._Nt = Nt
        self._Nx = Nx

    @property
    def order(self) -> int:
        """Basis order."""
        return len(self._time)

    def _check_validity(
        self,
        space_params: dict[int, Any],
        time_params: dict[int, Any],
    ) -> None:
        """Check parameters validity."""
        if space_params.keys() != time_params.keys():
            msg = "Mismatching keys between space and time parameters."
            raise ValueError(msg)

    def set_coefs(self, coefs: DecompositionCoefs) -> None:
        """Set coefficients values.

        To ensure consistent coefficients shapes, best is to use
        self.generate_random_coefs().

        Args:
            coefs (DecompositionCoefs): Level -> coefficients.
                ├── 0: coefficents for level 0.
                ├── ...
                └── order-1: coefficient for level order-1
        """
        self._coefs = coefs

    def localize(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> "MIOSTTimeSupport":
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        assert torch.all(xx == self._xx)
        assert torch.all(yy == self._yy)

        return {
            k: torch.from_numpy(v.toarray()).to(**self._specs)
            for k, v in self._space.items()
        }

    def generate_time_support(
        self,
        time_params: dict[int, list[torch.Tensor]],
        space_fields: dict[int, torch.Tensor],
    ) -> "MIOSTTimeSupport":
        """Generate time support.

        Args:
            time_params (dict[int, list[torch.Tensor]]): Time params.
            space_fields (dict[int, torch.Tensor]): Space fields.

        Returns:
            MIOSTTimeSupport: Time support function.
        """
        return MIOSTTimeSupport(
            time_params,
            space_fields,
            self._dt,
            self._xx.shape[-2:],
            self._Nt,
            self._Nx,
            self._coefs,
        )

    def shapes(self) -> dict[int, torch.Size]:  # noqa: D102
        return {k: torch.Size((t[0].shape[0],)) for k, t in self._time.items()}

    shapes.__doc__ = SpaceTimeDecomposition[
        "MIOSTSpaceSupport", "MIOSTTimeSupport"
    ].shapes.__doc__

    def get_params(self) -> dict[str, Any]:
        """Return decomposition params as dict.

        Returns:
            dict[str, Any]: Decomposition params.
        """
        return {
            "type": self.type,
            "space": self._space,
            "time": self._time,
            "kwargs": {
                "xx": self._xx,
                "yy": self._yy,
                "dt": self._dt,
                "Nt": self._Nt,
                "Nx": self._Nx,
            },
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        """Build class from params dict.

        Args:
            params (dict[str, Any]): Decomposition params.

        Returns:
            Self: Instance of class.
        """
        return cls(
            space_params=params["space"],
            time_params=params["time"],
            **params["kwargs"],
        )


class MIOSTTimeSupport(TimeSupportFunction):
    """Time support function."""

    def __init__(
        self,
        time_params: dict[int, list[torch.Tensor]],
        space_fields: dict[int, torch.Tensor],
        dt: float,
        shape: tuple[int, int],
        Nt: dict[int, list[int]],
        Nx: list[int],
        coefs: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate time support function.

        Args:
            time_params (dict[int, list[torch.Tensor]]): Time parames.
            space_fields (dict[int, torch.Tensor]): Space params.
            dt (float): Timestep.
            shape (tuple[int, int]): Output shape.
            Nt (dict[int, list[int]]): Time specs.
            Nx (list[int]): Space specs
            coefs (torch.Tensor): Coefs.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        self._coefs = coefs
        self._space = space_fields
        self._time = time_params
        self._time_params = time_params
        self._dt = dt
        self._shape = shape
        self._specs = defaults.get(dtype=dtype, device=device)
        self._Nt = Nt
        self._Nx = Nx

    def __call__(self, t: torch.Tensor) -> torch.Tensor:  # noqa: D102
        fields = self.decompose(t)
        return sum(fields.values())

    __call__.__doc__ = TimeSupportFunction.__call__.__doc__

    def decompose(self, t: torch.Tensor) -> dict[int, torch.Tensor]:  # noqa: D102

        i_ = t.item() / self._dt

        if (i := round(i_)) != i_:
            msg = f"Evaluated at wrong time, using t={i * self._dt} s instead."
            logger.warning(msg)

        fields = {}

        for iff, space in self._space.items():
            Xf = self._coefs[iff]
            GtXf = self._time[iff][i] * Xf
            indNoNan = ~torch.isnan(self._time[iff][i])
            if indNoNan.numel() > 0:
                GtXf = GtXf[indNoNan].reshape(self._Nt[iff][i], self._Nx[iff])
                fields[iff] = torch.einsum(
                    "ij,j->i", space, GtXf.sum(axis=0)
                ).reshape(self._shape)
        return fields

    decompose.__doc__ = TimeSupportFunction.decompose.__doc__

    def decompose_dt(self, t: torch.Tensor) -> dict[int, torch.Tensor]:  # noqa: D102
        fields = self.decompose(t)
        fields_ = self.decompose(t + self._dt)
        return {k: (fields_[k] - fields[k]) / self._dt for k in fields_}

    decompose_dt.__doc__ = TimeSupportFunction.decompose_dt.__doc__

    def dt(self, t: torch.Tensor) -> torch.Tensor:  # noqa: D102
        fields = self.decompose_dt(t)
        return sum(fields.values())

    dt.__doc__ = TimeSupportFunction.dt.__doc__


class MIOSTSpaceSupport(SpaceSupportFunction):
    """Dummy class."""

    def _compute(self) -> torch.Tensor:
        """Method to compute field."""
        raise NotImplementedError
