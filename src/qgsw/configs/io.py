"""Input/Output Configuration."""

# ruff: noqa: UP007

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from qgsw.utils.storage import get_absolute_storage_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qgsw.simulation.steps import Steps


class IOConfig(BaseModel):
    """Input/Output ."""

    name: str
    output: Union[IntervalSaveConfig, QuantitySaveConfig, None] = Field(
        None,
        discriminator="type",
    )


class IntervalSaveConfig(BaseModel):
    """Interval save configuration."""

    type: Literal["interval"]
    save: bool
    interval_duration: PositiveFloat
    directory_str: str = Field(
        alias="directory",
    )

    @cached_property
    def directory(self) -> Path:
        """Output directory."""
        directory = get_absolute_storage_path(Path(self.directory_str))
        if not directory.is_dir():
            directory.mkdir()
            gitignore = directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")
        return directory

    def get_saving_steps(self, steps: Steps) -> Iterator[bool]:
        """Get saving steps.

        Args:
            steps (Steps): Steps manager.

        Yields:
            Iterator[bool]: Boolean iterator with the same length as the one
            returned by `simulation_steps`, True when the corresponding step
            matches the interval, always ends with True.
        """
        if not self.save:
            return (False for _ in steps.simulation_steps())
        return steps.steps_from_interval(interval=self.interval_duration)


class QuantitySaveConfig(BaseModel):
    """Quantity save configuration."""

    type: Literal["quantity"]
    save: bool
    quantity: PositiveFloat
    directory_str: str = Field(
        alias="directory",
    )

    @cached_property
    def directory(self) -> Path:
        """Output directory."""
        directory = get_absolute_storage_path(Path(self.directory_str))
        if not directory.is_dir():
            directory.mkdir()
            gitignore = directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")
        return directory

    def get_saving_steps(self, steps: Steps) -> Iterator[bool]:
        """Get saving steps.

        Args:
            steps (Steps): Steps manager.

        Yields:
            Iterator[bool]: Boolean iterator with the same length as the one
            returned by `simulation_steps`, True when the corresponding step is
            selected, always ends with True. May include one additional step.
        """
        if not self.save:
            return (False for _ in steps.simulation_steps())
        return steps.steps_from_total(total=self.quantity)
