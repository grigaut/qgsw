"""Simulations configurations."""

# ruff: noqa: UP007

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Literal, Union

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    PositiveFloat,
    field_serializer,
    field_validator,
)

from qgsw.models.references.names import ReferenceName
from qgsw.simulation.names import SimulationName
from qgsw.utils.named_object import NamedObjectConfig

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig


class StartupConfig(BaseModel):
    """Startup file configuration."""

    file: Path
    config: Path

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


class ModelOutputReferenceConfig(NamedObjectConfig[ReferenceName], BaseModel):
    """Config for ModelOutputReference."""

    type: Literal[ReferenceName.MODEL_OUTPUT]
    prefix: str
    folder: DirectoryPath

    @field_serializer("folder")
    def serialize_folder_as_str(self, folder: Path) -> str:
        """Serialize folder as str.

        Args:
            folder (Path): Directory path

        Returns:
            str: Path as posix.
        """
        return folder.as_posix()

    @field_validator("folder", mode="after")
    @classmethod
    def contains_config(cls, value: Path) -> Path:
        """Verify that the folder contains a _config.toml file.

        Args:
            value (Path): Filepath.

        Raises:
            ValueError: If folder does not have a _config.toml file.

        Returns:
            Path: Filepath
        """
        if not value.joinpath("_config.toml").is_file():
            msg = f"{value} must contain '_config.toml' file."
            raise ValueError(msg)
        return value


class ModelReferenceConfig(NamedObjectConfig[ReferenceName], BaseModel):
    """Config for ModelReference."""

    type: Literal[ReferenceName.MODEL]
    prefix: str
    model: ModelConfig


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
    reference: Union[
        ModelReferenceConfig,
        ModelOutputReferenceConfig,
    ] = Field(discriminator="type")
