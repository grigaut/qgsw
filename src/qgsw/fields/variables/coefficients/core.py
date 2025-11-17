"""Collinearity Coefficients."""

from __future__ import annotations

import contextlib
from copy import copy
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy import optimize

from qgsw.exceptions import (
    InappropriateShapeError,
    UnmatchingShapesError,
    UnsetCentersError,
    UnsetSigmaError,
    UnsetValuesError,
)
from qgsw.fields.scope import Scope
from qgsw.fields.variables.coefficients.coef_names import CoefficientName
from qgsw.logging import getLogger
from qgsw.utils.named_object import NamedObject

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Generic, TypeVar

import torch

from qgsw.fields.variables.base import Variable
from qgsw.specs import defaults
from qgsw.utils.units._units import Unit

if TYPE_CHECKING:
    from qgsw.configs.space import SpaceConfig

Values = TypeVar("Values")
logger = getLogger(__name__)


class Coefficient(
    NamedObject[CoefficientName],
    Variable,
    Generic[Values],
    metaclass=ABCMeta,
):
    """Coefficient base class."""

    _unit = Unit._
    _scopt = Scope.POINT_WISE

    _nl = 1
    _core: torch.Tensor

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        n_ens: int = 1,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        """Instantiate the coefficient.

        Args:
            nx (int): Points in the x direction.
            ny (int): Points in the y direction.
            n_ens (int, optional): Number of ensemble. Defaults to 1.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device. Defaults to None.
        """
        self._shape = (n_ens, self._nl, nx, ny)
        self._nx = nx
        self._ny = ny
        self._specs = defaults.get(dtype=dtype, device=device)

    @property
    def values(self) -> Values:
        """Values."""
        try:
            return self._values
        except AttributeError as e:
            raise UnsetValuesError from e

    @values.setter
    def values(self, values: Values) -> None:
        self._values = values
        self._update()

    @abstractmethod
    def with_optimal_values(
        self,
        p: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> None:
        """Sets optimal values to the coefficient."""
        if p.shape != self._shape[-2:]:
            msg = f"p shape must be {self._shape[-2:]}, not {p.shape}"
            raise InappropriateShapeError(msg)
        if p_ref.shape != self._shape[-2:]:
            msg = f"p_ref shape must be {self._shape[-2:]}, not {p_ref.shape}"
            raise InappropriateShapeError(msg)

    @abstractmethod
    def _update(self) -> None:
        """Update the Coefficient value."""

    def update(self, values: Values) -> None:
        """Update values.

        Args:
            values (Values): New values to consider.
        """
        with contextlib.suppress(AttributeError):
            del self._values
        self.values = values

    def get(self) -> torch.Tensor:
        """Get the coefficient value.

        Returns:
            torch.Tensor: Coefficient value.
        """
        try:
            return self._core
        except AttributeError as e:
            msg = (
                "All parameters have not been properly set."
                "Thus the coefficient has not been instantiated."
            )
            raise AttributeError(msg) from e

    @classmethod
    def from_config(cls, space_config: SpaceConfig) -> Self:
        """Instantiate the coefficient from configuration.

        Args:
            space_config (SpaceConfig): Space configuration.

        Returns:
            Self: Coefficient
        """
        return cls(nx=space_config.nx, ny=space_config.ny)

    @classmethod
    @abstractmethod
    def compute_optimal_values(
        cls,
        p: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> Values:
        """Compute the optimal values given pressure."""


class UniformCoefficient(Coefficient[float]):
    """Space-uniform coefficient."""

    _type = CoefficientName.UNIFORM
    _description = "Space-uniform Collinearity coefficient"

    def _update(self) -> None:
        """Update core value.

        Args:
            values (float): Float value.
        """
        self._core = self.values * torch.ones(
            self._shape,
            **self._specs,
        )

    def with_optimal_values(
        self,
        p: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> None:
        """Set optimal values for `values`.

        Optimal values are inferred using Least Square Regression.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx, ny)-shaped
            p_ref (torch.Tensor): Reference pressure to approximate.
                └── (n_ens, nl, nx, ny)-shaped

        Raises:
            InappropriateShapeError: If the pressure shape does not match with
            nx and ny.
        """
        super().with_optimal_values(p, p_ref)
        optimal = self.compute_optimal_values(
            p,
            p_ref,
        )
        msg = f"Optimal coefficient inferred: {round(optimal, 2)}"
        logger.detail(msg)
        self.values = optimal

    @classmethod
    def compute_optimal_values(
        cls,
        p: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> float:
        """Compute optimal values.

        Optimal values are inferred using Least Square Regression.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx, ny)-shaped
            p_ref (torch.Tensor): Reference pressure to approximate.
                └── (n_ens, nl, nx, ny)-shaped

        Raises:
            UnmatchingShapesError: If p and p_ref shapes don't match.

        Returns:
            float: Optimal values.
        """
        if p.shape != p_ref.shape:
            msg = (
                f"p ({p.shape}) and p_ref ({p_ref.shape})"
                " must have the same shape."
            )
            raise UnmatchingShapesError(msg)
        if len(p.shape) != len(p_ref.shape):
            msg = (
                f"p ({p.shape}) and p_ref ({p_ref.shape})"
                " must have the same dimension."
            )
            raise UnmatchingShapesError(msg)

        solution = optimize.lsq_linear(
            p.flatten().reshape((-1, 1)).cpu().numpy(),
            p_ref.flatten().cpu().numpy(),
            bounds=(-1, 1),
        )
        msg = f"Optimal coefficient inferred with a cost of {solution.cost}."
        logger.detail(msg)
        return solution.x.item()


class NonUniformCoefficient(Coefficient[torch.Tensor]):
    """Non-space-uniform coefficient."""

    _type = CoefficientName.NON_UNIFORM
    _description = "Non-space-uniform Collinearity coefficient"

    def _update(
        self,
    ) -> None:
        """Update core value."""
        self._core = self._values

    @Coefficient.values.setter
    def values(self, values: torch.Tensor) -> None:
        """Setter for values."""
        if values.shape != self._shape:
            msg = f"Invalid shape, it should be {self._shape}-shaped."
            raise ValueError(msg)
        if values.dtype != (dtype := self._specs["dtype"]):
            msg = f"Invalid dtype, it should be {dtype}."
            raise ValueError(msg)
        if values.device.type != (device := self._specs["device"].type):
            msg = f"Invalid device type, it should be {device}."
            raise ValueError(msg)
        self._values = values
        self._update()

    def with_optimal_values(
        self,
        p: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> None:
        """Set optimal values for `values`.

        Optimal values are inferred using Least Square Regression.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx, ny)-shaped
            p_ref (torch.Tensor): Reference pressure to approximate.
                └── (n_ens, nl, nx, ny)-shaped
        """
        msg = "NonUniformCoefficient does not support optimal value."
        raise NotImplementedError(msg)

    @classmethod
    def compute_optimal_values(
        cls,
        p: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> float:
        """Compute optimal values.

        Optimal values are inferred using Least Square Regression.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx, ny)-shaped
            p_ref (torch.Tensor): Reference pressure to approximate.
                └── (n_ens, nl, nx, ny)-shaped
        """
        msg = "NonUniformCoefficient does not support optimal value."
        raise NotImplementedError(msg)


class SmoothNonUniformCoefficient(Coefficient[Iterable[float]]):
    """Non-space-uniform coefficient smoothed by gaussians."""

    _type = CoefficientName.SMOOOTH_NON_UNIFORM
    _description = "Smoothed-non-space-uniform Collinearity coefficient"

    _sigma = 1

    @property
    def values(self) -> list[float]:
        """Values."""
        try:
            return self._values
        except AttributeError as e:
            raise UnsetValuesError from e

    @values.setter
    def values(self, values: Iterable[float]) -> None:
        with contextlib.suppress(UnsetCentersError):
            if len(vs := list(values)) != len(self.centers):
                msg = "There must be as many values as centers."
                raise ValueError(msg)
        self._values = list(vs)
        with contextlib.suppress(UnsetSigmaError, UnsetCentersError):
            self._update()

    @property
    def centers(self) -> list[tuple[int, int]]:
        """Values centers."""
        try:
            return self._centers
        except AttributeError as e:
            raise UnsetCentersError from e

    @centers.setter
    def centers(self, centers: Iterable[tuple[int, int]]) -> None:
        with contextlib.suppress(UnsetValuesError):
            if len(cs := list(centers)) != len(self.values):
                msg = "There must be as many centers as values."
                raise ValueError(msg)
        self._centers = list(cs)
        with contextlib.suppress(UnsetSigmaError, UnsetValuesError):
            self._update()

    @property
    def sigma(self) -> float:
        """Standard deviation of gaussian kernel."""
        try:
            return self._sigma
        except AttributeError as e:
            raise UnsetSigmaError from e

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if sigma <= 0:
            msg = f"Standard deviation must be > 0 thus cannot be {sigma}."
            raise ValueError(msg)
        self._sigma = sigma
        with contextlib.suppress(UnsetCentersError, UnsetValuesError):
            self._update()

    def with_optimal_values(
        self,
        p: torch.Tensor,
        p_ref: torch.Tensor,
        centers: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        """Set optimal values for `values`.

        Optimal values are inferred using Least Square Regression.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx, ny)-shaped
            p_ref (torch.Tensor): Reference pressure to approximate.
                └── (n_ens, nl, nx, ny)-shaped
            centers (Iterable[tuple[int, int]] | None, optional): Values
            center locations. If None, the actual centers will be reused
            (if already set). Defaults to None.

        Raises:
            InappropriateShapeError: If the pressure shape does not match with
            nx and ny.
            np.linalg.LinAlgError: If the least square regression does not
            converge.
        """
        super().with_optimal_values(p, p_ref)
        if centers is not None:
            with contextlib.suppress(AttributeError):
                del self._values
                del self._centers
            self.centers = centers
        try:
            optimal = self.compute_optimal_values(
                p,
                p_ref,
                self.sigma,
                self.centers,
            )
        except np.linalg.LinAlgError as e:
            msg = (
                "Convergence issue might be solved by increasing the value "
                f"of sigma.\n Currently, {self.__class__.__name__}.sigma"
                f" = {self.sigma}."
            )
            raise np.linalg.LinAlgError(msg) from e
        msg = f"Optimal coefficient inferred: {[round(r, 2) for r in optimal]}"
        logger.detail(msg)
        self.values = optimal

    def update(
        self,
        values: Iterable[float],
        centers: Iterable[tuple[int, int]],
    ) -> None:
        """Manually update.

        This removes both attributes before setting them to final values.

        Args:
            values (Iterable[float]): Values.
            centers (Iterable[tuple[int, int]]): Values center locations.
        """
        with contextlib.suppress(AttributeError):
            del self._values
            del self._centers
        self.values = values
        self.centers = centers

    def _update(self) -> None:
        """Update the core."""
        supports = self.compute_supports(
            self.sigma,
            self._nx,
            self._ny,
            self.centers,
            **self._specs,
        )

        core = supports @ torch.tensor(
            self.values,
            **self._specs,
        )
        self._core = (
            core.reshape((self._nx, self._ny)).unsqueeze(0).unsqueeze(0)
        )

    @classmethod
    def compute_supports(
        cls,
        sigma: float,
        nx: int,
        ny: int,
        centers: Iterable[tuple[int, int]],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute gaussian support for the coefficient.

        Args:
            sigma (float): Standard deviation for gaussian smoothing.
            nx (int): Number of points along x.
            ny (int): Number of points along y.
            centers (Iterable[tuple[int, int]]): Values center locations.
            dtype (torch.dtype): Data type.
            device (torch.device): Data device.

        Returns:
            torch.Tensor: Gaussian supports
                └── (nx*ny, n)-shaped (n is the number of centers)
        """
        specs = defaults.get(dtype=dtype, device=device)
        X, Y = torch.meshgrid(
            torch.arange(nx, **specs),
            torch.arange(ny, **specs),
            indexing="ij",
        )

        supports: list[torch.Tensor] = []

        norm = torch.zeros((nx, ny), **specs)

        for x0, y0 in copy(centers):
            kernel = torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / 2 / sigma**2)
            norm += kernel
            supports.append(kernel)

        return torch.cat(
            [(s / norm).reshape((-1, 1)) for s in supports],
            dim=-1,
        )

    @classmethod
    def compute_optimal_values(
        cls,
        p: torch.Tensor,
        p_ref: torch.Tensor,
        sigma: float,
        centers: Iterable[tuple[int, int]],
    ) -> Iterable[float]:
        """Compute optimal values.

        Optimal values are inferred using Least Square Regression.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx, ny)-shaped
            p_ref (torch.Tensor): Reference pressure to approximate.
                └── (n_ens, nl, nx, ny)-shaped
            sigma (float): Standard deviation for gaussian smoothing.
            filt (_Filter): Filter to apply to p[0,0].
            centers (Iterable[tuple[int, int]]): Values centers locations.

        Raises:
            UnmatchingShapesError: If p and p_ref shapes don't match.

        Returns:
            Iterable[float]: Optimal values.
        """
        if p.shape != p_ref.shape:
            msg = (
                f"p ({p.shape}) and p_ref ({p_ref.shape})"
                " must have the same shape."
            )
            raise UnmatchingShapesError(msg)
        if len(p.shape) != len(p_ref.shape):
            msg = (
                f"p ({p.shape}) and p_ref ({p_ref.shape})"
                " must have the same dimension."
            )
            raise UnmatchingShapesError(msg)

        nx, ny = p.shape

        supports = cls.compute_supports(
            sigma=sigma,
            nx=nx,
            ny=ny,
            centers=centers,
            dtype=p.dtype,
            device=p.device.type,
        )

        weighted_p = supports * (p.reshape((-1, 1)))

        solution = optimize.lsq_linear(
            weighted_p.cpu().numpy(),
            p_ref.flatten().cpu().numpy(),
            bounds=(-1, 1),
        )
        msg = f"Optimal coefficient inferred with a cost of {solution.cost}."
        logger.detail(msg)
        return solution.x.tolist()


CoefType = Union[
    UniformCoefficient,
    NonUniformCoefficient,
    SmoothNonUniformCoefficient,
]
