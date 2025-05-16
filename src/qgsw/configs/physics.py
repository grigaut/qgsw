"""Physics configuration."""

from __future__ import annotations

from functools import cached_property

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from qgsw.physics.coriolis.beta_plane import BetaPlane


class PhysicsConfig(BaseModel):
    """Physics configuration."""

    rho: PositiveFloat
    slip_coef: float = Field(ge=0, le=1)
    f0: float
    beta: float
    bottom_drag_coefficient: float = Field(ge=0, le=1)
    Ro: PositiveFloat = 0.1

    @cached_property
    def beta_plane(self) -> BetaPlane:
        """Beta Plane."""
        return BetaPlane(f0=self.f0, beta=self.beta)
