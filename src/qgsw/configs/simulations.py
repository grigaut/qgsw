"""Simulations configurations."""

# ruff: noqa: UP007

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from qgsw.simulation.names import SimulationName
from qgsw.utils.named_object import NamedObjectConfig

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig


class SimulationConfig(NamedObjectConfig[SimulationName], BaseModel):
    """Simulation configuration."""

    type: Literal[SimulationName.RUN, SimulationName.ASSIMILATION]
    duration: PositiveFloat
    dt: PositiveFloat


class ModelRunSimulationConfig(SimulationConfig):
    """Model run simulaton configuration."""

    type: Literal[SimulationName.RUN]


class AssimilationSimulationConfig(SimulationConfig):
    """Assimilation simulation configuration."""

    type: Literal[SimulationName.ASSIMILATION]
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
