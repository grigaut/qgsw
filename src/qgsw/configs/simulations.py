"""Simulations configurations."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    PositiveFloat,
)

from qgsw.configs.models import ModelConfig  # noqa: TCH001


class ModelRunSimulationConfig(BaseModel):
    """Model run simulaton configuration."""

    kind: Literal["simple-run"]
    duration: PositiveFloat
    dt: PositiveFloat


class AssimilationSimulationConfig(BaseModel):
    """Assimilation simulation configuration."""

    kind: Literal["assimilation"]
    duration: PositiveFloat
    dt: PositiveFloat
    reference: ModelConfig
