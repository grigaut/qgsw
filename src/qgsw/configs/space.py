"""Space configuration."""

from __future__ import annotations

from functools import cached_property

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)

from qgsw.utils.units._units import Unit


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

    @cached_property
    def dx(self) -> float:
        """Infinitesimal distance in the x direction.

        dx = (x_max - x_min)/nx
        """
        return (self.x_max - self.x_min) / self.nx

    @cached_property
    def dy(self) -> float:
        """Infinitesimal distance in the y direction.

        dy = (y_max - y_min)/ny
        """
        return (self.y_max - self.y_min) / self.ny

    @cached_property
    def ds(self) -> float:
        """Infinitesimal horizontal area.

        ds = dx * dy
        """
        return self.dx * self.dy
