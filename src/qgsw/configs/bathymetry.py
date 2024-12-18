"""Bathymetry configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)


class BathyConfig(BaseModel):
    """Bathymetry configuration."""

    h_top_ocean: PositiveFloat
    lake_min_area: PositiveFloat
    island_min_area: PositiveFloat
    interpolation_method: str
    data: BathyDataConfig | None = None


class BathyDataConfig(BaseModel):
    """Bathymetry data configuration."""

    url: str
    folder_str: str = Field(alias="folder")
    longitude: str
    latitude: str
    elevation: str

    @property
    def folder(self) -> Path:
        """Data folder."""
