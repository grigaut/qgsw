"""Pertubation configuration."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    NonNegativeFloat,
)

from qgsw.perturbations.names import PertubationName
from qgsw.utils.named_object import NamedObjectConfig


class PerturbationConfig(NamedObjectConfig[PertubationName], BaseModel):
    """Perturbation configuration."""

    perturbation_magnitude: NonNegativeFloat
