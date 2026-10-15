"""Basis MIOST connector."""

from typing import Any, Self

import numpy as np
import torch
from scipy.sparse._csc import csc_matrix

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.logging.core import getLogger
from qgsw.specs import defaults

logger = getLogger(__name__)


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

    def numel(self) -> int:
        """Number of elements."""
        return sum([t[0].shape[0] for t in self._time.values()])

    def generate_random_coefs(self) -> DecompositionCoefs:
        """Generate random coefs."""
        return DecompositionCoefs(
            *[
                torch.rand((t[0].shape[0],), **self._specs)
                for t in self._time.values()
            ]
        )

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
