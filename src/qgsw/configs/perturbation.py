"""Pertubation configuration."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    NonNegativeFloat,
)


class PerturbationConfig(BaseModel):
    """Perturbation configuration."""

    type: str
    perturbation_magnitude: NonNegativeFloat
