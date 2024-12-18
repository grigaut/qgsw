"""Model Configuration."""

# ruff: noqa: TCH001, UP007
from __future__ import annotations

from functools import cached_property
from typing import Literal, Union

import torch
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    PositiveFloat,
)

from qgsw.specs import DEVICE


class ModelConfig(BaseModel):
    """Model configuration."""

    type: str
    prefix: str
    layers: list[PositiveFloat]
    reduced_gravity: list[PositiveFloat]
    collinearity_coef: Union[
        ConstantCollinearityCoefConfig,
        ChangingCollinearityCoefConfig,
        None,
    ] = Field(None, discriminator="type")

    @cached_property
    def h(self) -> torch.Tensor:
        """Vertical layers."""
        return torch.tensor(
            self.layers,
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @cached_property
    def g_prime(self) -> torch.Tensor:
        """Reduced gravity."""
        return torch.tensor(
            self.reduced_gravity,
            dtype=torch.float64,
            device=DEVICE.get(),
        )


class ConstantCollinearityCoefConfig(BaseModel):
    """Constant collinearity model configuration."""

    type: Literal["constant"]
    value: float


class ChangingCollinearityCoefConfig(BaseModel):
    """Changing collinearity coefficient configuration."""

    type: Literal["changing"]
    source_file: FilePath
