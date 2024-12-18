"""Space configuration."""

from __future__ import annotations

from functools import cached_property

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)

from qgsw.spatial.units._units import Unit


class SpaceConfig(BaseModel):
    """Space configuration."""

    nx: PositiveInt
    ny: PositiveInt
    unit_str: str = Field(alias="unit")
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @cached_property
    def unit(self) -> Unit:
        """Space Unit."""
        return Unit(self.unit_str)
