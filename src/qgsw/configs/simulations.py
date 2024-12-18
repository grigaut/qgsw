"""Simulations configurations."""

# ruff: noqa: UP007

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from qgsw.configs.models import ModelConfig  # noqa: TCH001


class SimulationConfig(BaseModel):
    """Simulation configuration."""

    kind: Literal["simple-run", "assimilation"]
    duration: PositiveFloat
    dt: PositiveFloat


class ModelRunSimulationConfig(SimulationConfig):
    """Model run simulaton configuration."""

    kind: Literal["simple-run"]


class AssimilationSimulationConfig(SimulationConfig):
    """Assimilation simulation configuration."""

    kind: Literal["assimilation"]
    fork_interval: PositiveFloat
    reference: ModelConfig
    startup_file_str: Union[str, None] = Field(None, alias="startup_file")

    @property
    def startup_file(self) -> Path | None:
        """Startup file."""
        if self.startup_file_str is None:
            return None
        startup_file = Path(self.startup_file_str)
        if not startup_file.is_file():
            msg = f"{startup_file} does not correspond to any file."
            raise ValueError(msg)
        return startup_file
