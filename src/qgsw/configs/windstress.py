"""WindStress configuration."""

# ruff: noqa: TCH001, UP007

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
)


class WindStressConfig(BaseModel):
    """Windstress configuration."""

    type: str
    magnitude: Union[NonNegativeFloat, None] = None
    drag_coefficient: Union[NonNegativeFloat, None] = None
    data: WindStressDataConfig | None = None


class WindStressDataConfig(BaseModel):
    """WindStress Data Configuration."""

    url: str
    folder_str: str = Field(alias="folder")
    data_type: str
    longitude: str
    latitude: str
    time: str
    field_1: str
    field_2: str
    method: str

    @property
    def folder(self) -> Path:
        """Data folder."""
