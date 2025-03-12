"""Model Configuration."""

# ruff: noqa: TC001, UP007
from __future__ import annotations

from functools import cached_property
from typing import Generic, Literal, TypeVar, Union

import torch
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from qgsw.fields.variables.coefficients.coef_names import CoefficientName
from qgsw.fields.variables.coefficients.core import NonUniformCoefficient
from qgsw.models.names import ModelName
from qgsw.models.qg.projected.modified.utils import is_modified
from qgsw.specs import DEVICE, defaults
from qgsw.utils.named_object import NamedObjectConfig


class UniformCoefConfig(
    NamedObjectConfig[CoefficientName],
    BaseModel,
):
    """Uniform collinearity configuration."""

    type: Literal[CoefficientName.UNIFORM]
    initial: float
    use_optimal: bool = False


class NonUniformCoefConfig(
    NamedObjectConfig[CoefficientName],
    BaseModel,
):
    """Non uniform collinearity configuration."""

    type: Literal[CoefficientName.NON_UNIFORM]
    initial: list[list[float]]
    use_optimal: bool = False

    @cached_property
    def matrix(self) -> torch.Tensor:
        """Values matrix.

        └── (1, 1, nx, ny) shaped
        """
        matrix = torch.tensor(
            self.initial,
            **defaults.get(),
        )
        return matrix.unsqueeze(0).unsqueeze(0)


class SmoothNonUniformCoefConfig(
    NamedObjectConfig[CoefficientName],
    BaseModel,
):
    """Non uniform collinearity configuration."""

    type: Literal[CoefficientName.SMOOOTH_NON_UNIFORM]
    initial: list[float]
    centers: list[tuple[int, int]]
    sigma: float = 1
    use_optimal: bool = False


CollinearityCoefficientConfig = Union[
    UniformCoefConfig,
    NonUniformCoefficient,
]

CoefConfig = Union[
    UniformCoefConfig,
    NonUniformCoefConfig,
    SmoothNonUniformCoefConfig,
]

CoefConfigVar = TypeVar(
    "CoefConfigVar",
    bound=Union[
        UniformCoefConfig,
        NonUniformCoefConfig,
        SmoothNonUniformCoefConfig,
    ],
)


class ModelConfig(
    NamedObjectConfig[ModelName],
    BaseModel,
    Generic[CoefConfigVar],
):
    """Model configuration."""

    prefix: str
    layers: list[PositiveFloat]
    reduced_gravity: list[PositiveFloat]
    collinearity_coef: Union[CoefConfigVar, None] = Field(
        None,
        discriminator="type",
    )
    sigma: Union[PositiveFloat, None] = None

    @cached_property
    def h(self) -> torch.Tensor:
        """Vertical layers.

        └── (nl,) shaped
        """
        return torch.tensor(
            self.layers,
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @cached_property
    def nl(self) -> torch.Tensor:
        """Number of layers."""
        n_tot = self.h.shape[0]
        return n_tot - int(is_modified(self.type))

    @cached_property
    def g_prime(self) -> torch.Tensor:
        """Reduced gravity.

        └── (nl,) shaped
        """
        return torch.tensor(
            self.reduced_gravity,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
