"""Space configuration."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import torch
from pydantic import (
    BaseModel,
    PositiveInt,
    field_serializer,
)

from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.utils.units._units import Unit


class SpaceConfig(BaseModel):
    """Space configuration."""

    nx: PositiveInt
    ny: PositiveInt
    unit: Unit
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @cached_property
    def dx(self) -> torch.Tensor:
        """Infinitesimal distance in the x direction.

        dx = (x_max - x_min)/nx
        """
        return torch.tensor(
            (self.x_max - self.x_min) / self.nx,
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @cached_property
    def dy(self) -> torch.Tensor:
        """Infinitesimal distance in the y direction.

        dy = (y_max - y_min)/ny
        """
        return torch.tensor(
            (self.y_max - self.y_min) / self.ny,
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @cached_property
    def ds(self) -> torch.Tensor:
        """Infinitesimal horizontal area.

        ds = dx * dy
        """
        return self.dx * self.dy

    @field_serializer("unit")
    def serialize_unit_as_str(self, unit: Unit) -> str:
        """Serialize unit as str.

        Args:
            unit (Unit): Unit

        Returns:
            str: Unit value.
        """
        return unit.value
