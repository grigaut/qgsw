"""Collinearity Coefficients."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Union

from qgsw.exceptions import (
    UnsetCentersError,
    UnsetSigmaError,
    UnsetValuesError,
)
from qgsw.fields.variables.coefficients.coef_names import CoefficientName
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
from qgsw.utils.least_squares_regression import (
    perform_linear_least_squares_regression,
)
from qgsw.utils.units._units import Unit

if TYPE_CHECKING:
    from qgsw.configs.space import SpaceConfig

Values = TypeVar("Values")


class Coefficient(
    Generic[Values],
    NamedObject[CoefficientName],
    Variable,
    metaclass=ABCMeta,
):
    """Coefficient base class."""

    _unit = Unit._

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
        self._dtype = defaults.get_dtype(dtype)
        self._device = defaults.get_device(device)

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
                "Thus the coefficient has not been instantiated"
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


class UniformCoefficient(Coefficient[float]):
    """Space-uniform coefficient."""

    _type = CoefficientName.UNIFORM
    _description = "Space-uniform Collinearity coefficient"

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
        super().__init__(
            nx=nx,
            ny=ny,
            n_ens=n_ens,
            dtype=dtype,
            device=device,
        )

    def _update(self) -> None:
        """Update core value.

        Args:
            values (float): Float value.
        """
        self._core = self.values * torch.ones(
            self._shape,
            device=self._device,
            dtype=self._dtype,
        )


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
        if values.dtype != self._dtype:
            msg = f"Invalid dtype, it should be {self._dtype}."
            raise ValueError(msg)
        if values.device.type != self._device.type:
            msg = f"Invalid device type, it should be {self._device.type}."
            raise ValueError(msg)
        self._values = values
        self._update()


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
        core = torch.zeros(self._shape, dtype=self._dtype, device=self._device)

        x, y = torch.meshgrid(
            torch.arange(
                0,
                core.shape[-2],
                dtype=core.dtype,
                device=core.device,
            ),
            torch.arange(
                0,
                core.shape[-1],
                dtype=core.dtype,
                device=core.device,
            ),
            indexing="ij",
        )

        norm = torch.zeros_like(core)

        for alpha, loc in zip(self.values, self.locations):
            i, j = loc
            exp_factor = torch.exp(
                -((x - i) ** 2 + (y - j) ** 2) / 2 / self.sigma**2,
            )
            norm[..., :, :] += exp_factor
            core[..., :, :] += alpha * exp_factor

        self._core = core / norm


class LSRUniformCoefficient(UniformCoefficient):
    """Inferred collinearity from the streamfunction.

    Performs linear least squares regression to infer alpha.
    """

    _type = CoefficientName.LSR_INFERRED_UNIFORM
    _name = "alpha_lsr_sf"
    _description = "LSR-Stream function inferred coefficient"

    @classmethod
    def compute_coefficient(
        cls,
        p: torch.Tensor,
    ) -> float:
        """Compute collinearity coefficient.

        Args:
           p (torch.Tensor): Reference pressure (2-layered at least).

        Returns:
            Self: Coefficient.
        """
        p_1 = p[0, 0, ...]  # (nx,ny)-shaped
        p_2 = p[0, 1, ...]  # (nx,ny)-shaped

        x = p_1.flatten(-2, -1).unsqueeze(-1)  # (nx*ny,1)-shaped
        y = p_2.flatten(-2, -1)  # (nx*ny)-shaped

        try:
            return perform_linear_least_squares_regression(x, y).item()
        except torch.linalg.LinAlgError:
            return 0


CoefType = Union[
    UniformCoefficient,
    NonUniformCoefficient,
    SmoothNonUniformCoefficient,
    LSRUniformCoefficient,
]
