"""Collinearity Coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.coef_names import CoefficientName
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
    from qgsw.configs.core import Configuration
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
        self._core = torch.ones(
            (n_ens, self._nl, nx, ny),
            **defaults.get(dtype=dtype, device=device),
        )

    @abstractmethod
    def update(self, values: Values) -> None:
        """Update the Coefficient value.

        Args:
            values (Values): Values to use for update.
        """

    def get(self) -> torch.Tensor:
        """Get the coefficient value.

        Returns:
            torch.Tensor: Coefficient value.
        """
        return self._core

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

    def update(self, values: float) -> None:
        """Update core value.

        Args:
            values (float): Float value.
        """
        self._core = values * self._core


class NonUniformCoefficient(Coefficient[torch.Tensor]):
    """Non-space-uniform coefficient."""

    _type = CoefficientName.NON_UNIFORM

    sigma = 1

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

    def update(
        self,
        values: torch.Tensor,
    ) -> None:
        """Update core value.

        Args:
            values (torch.Tensor): Coefficient value.
        """
        if values.shape != self._shape:
            msg = f"Invalid shape, it should be {self._shape}-shaped."
            raise ValueError(msg)
        if values.dtype != self._dtype:
            msg = f"Invalid dtype, it should be {self._dtype}."
            raise ValueError(msg)
        self._core = values


class SmoothNonUniformCoefficient(Coefficient[Iterable[float]]):
    """Non-space-uniform coefficient smoothed by gaussians."""

    _type = CoefficientName.SMOOOTH_NON_UNIFORM

    sigma = 1

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

    def update(
        self,
        values: Iterable[float],
        locations: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        """Update core value.

        Args:
            values (Iterable[float]): Float value.
            locations (Iterable[tuple[int, int]] | None, optional): Center
            points indexes. Defaults to None.
        """
        core = torch.zeros_like(self._core)

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
        )

        norm = torch.zeros_like(core)

        for alpha, loc in zip(values, locations):
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


CoefType = (
    UniformCoefficient
    | NonUniformCoefficient
    | SmoothNonUniformCoefficient
    | LSRUniformCoefficient
)


def create_coefficient(
    config: Configuration,
) -> CoefType:
    """Create the coefficient.

    Args:
        config (Configuration): Model Configuration.

    Raises:
        ValueError: If the coefficient is not valid.

    Returns:
        CoefType: Coefficient
    """
    coef_type = config.model.collinearity_coef.type
    if coef_type == CoefficientName.UNIFORM:
        return UniformCoefficient.from_config(
            space_config=config.space,
        )
    if coef_type == CoefficientName.NON_UNIFORM:
        return NonUniformCoefficient.from_config(
            space_config=config.space,
        )
    if coef_type == CoefficientName.SMOOOTH_NON_UNIFORM:
        return SmoothNonUniformCoefficient.from_config(
            space_config=config.space,
        )
    if coef_type == CoefficientName.LSR_INFERRED_UNIFORM:
        return LSRUniformCoefficient.from_config(
            space_config=config.space,
        )
    msg = "Possible coefficient types are: "
    coef_types = [
        UniformCoefficient.get_name(),
        NonUniformCoefficient.get_name(),
        SmoothNonUniformCoefficient.get_name(),
        LSRUniformCoefficient.get_name(),
    ]
    raise ValueError(msg + ", ".join(coef_types))
