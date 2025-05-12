"""Simulations configurations."""

# ruff: noqa: UP007

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

from pydantic import (
    BaseModel,
    FilePath,
    PositiveFloat,
    field_serializer,
    field_validator,
)

from qgsw.simulation.names import SimulationName
from qgsw.utils.named_object import NamedObjectConfig

if TYPE_CHECKING:
    from pathlib import Path

    from qgsw.configs.models import ModelConfig


class SimulationConfig(NamedObjectConfig[SimulationName], BaseModel):
    """Simulation configuration."""

    type: Literal[SimulationName.RUN, SimulationName.ASSIMILATION]
    duration: PositiveFloat
    dt: PositiveFloat
    startup: Union[StartupConfig, None] = None


class ModelRunSimulationConfig(SimulationConfig):
    """Model run simulaton configuration."""

    type: Literal[SimulationName.RUN]


class AssimilationSimulationConfig(SimulationConfig):
    """Assimilation simulation configuration."""

    type: Literal[SimulationName.ASSIMILATION]
    fork_interval: PositiveFloat
    reference: ModelConfig


class StartupConfig(BaseModel):
    """Startup file configuration."""

    file: FilePath
    config: FilePath

    @field_serializer("file")
    def serialize_file_as_str(self, file: Path) -> str:
        """Serialize file as str.

        Args:
            file (Path): File path

        Returns:
            str: Path as posix.
        """
        return file.as_posix()

    @field_validator("file", mode="after")
    @classmethod
    def is_pt(cls, value: Path) -> Path:
        """Verify that file is a .pt file.

        Args:
            value (Path): Filepath.

        Raises:
            ValueError: If file is not a .pt file.

        Returns:
            Path: Filepath
        """
        if value.suffix != ".pt":
            msg = f"{value} is not a '.pt' file."
            raise ValueError(msg)
        return value

    @field_serializer("config")
    def serialize_config_as_str(self, config: Path) -> str:
        """Serialize config as str.

        Args:
            config (Path): File path

        Returns:
            str: Path as posix.
        """
        return config.as_posix()

    @field_validator("config", mode="after")
    @classmethod
    def is_toml(cls, value: Path) -> Path:
        """Verify that config is a .toml file.

        Args:
            value (Path): Filepath.

        Raises:
            ValueError: If config is not a .toml file.

        Returns:
            Path: Filepath
        """
        if value.suffix != ".toml":
            msg = f"{value} is not a '.toml' file."
            raise ValueError(msg)
        return value
