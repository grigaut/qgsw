"""Model Configuration."""

# ruff: noqa: TC001, UP007
from __future__ import annotations

from functools import cached_property
from typing import Literal, Union

import torch
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from qgsw.fields.variables.coef_names import CoefficientName
from qgsw.models.names import ModelName
from qgsw.models.qg.projected.modified.utils import is_modified
from qgsw.specs import DEVICE
from qgsw.utils.named_object import NamedObjectConfig


class ModelConfig(NamedObjectConfig[ModelName], BaseModel):
    """Model configuration."""

    prefix: str
    layers: list[PositiveFloat]
    reduced_gravity: list[PositiveFloat]
    collinearity_coef: Union[
        ConstantCollinearityCoefConfig,
        InferredCollinearityCoefConfig,
        None,
    ] = Field(None, discriminator="type")
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


class ConstantCollinearityCoefConfig(
    NamedObjectConfig[CoefficientName],
    BaseModel,
):
    """Constant collinearity model configuration."""

    type: Literal[CoefficientName.CONSTANT]
    value: float


class InferredCollinearityCoefConfig(
    NamedObjectConfig[CoefficientName],
    BaseModel,
):
    """Inferred collinearity coeffciient."""

    type: Literal[CoefficientName.LSR_INFERRED]
    initial: float


CollinearityCoefficientConfig = Union[
    ConstantCollinearityCoefConfig,
    InferredCollinearityCoefConfig,
]
