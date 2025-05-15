"""Model references."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.configs.core import Configuration
from qgsw.models.instantiation import (
    instantiate_model_from_config,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from qgsw.models.names import ModelCategory, ModelName, get_category
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.references.base import Reference
from qgsw.models.references.names import ReferenceName
from qgsw.output import RunOutput, _OutputReader
from qgsw.simulation.steps import Steps
from qgsw.utils.named_object import NamedObject

if TYPE_CHECKING:
    from qgsw.configs.core import Configuration
    from qgsw.fields.variables.prognostic_tuples import BaseUVH
    from qgsw.models.base import ModelUVH


class ModelReference(NamedObject[ReferenceName], Reference):
    """Reference using a running model."""

    __slots__ = ("_core",)
    _type = ReferenceName.MODEL

    def __init__(self, model: ModelUVH) -> None:
        """Instantiate the reference.

        Args:
            model (ModelUVH): Model to use.
        """
        super().__init__()
        self._core = model

    @property
    def time(self) -> float:
        """Reference time."""
        return self._core.time.item()

    def at_time(self, time: float) -> None:
        """Set the model to the reference time.

        Args:
            time (float): Reference time.

        Raises:
            ValueError: If the required time if lower than the model's
                actual time.
        """
        super().at_time(time)
        if time < self.time:
            msg = (
                f"Model reference can only be set to a "
                f"time later than {self.time} (s)."
            )
            raise ValueError(msg)
        steps = Steps(
            t_start=self.time,
            t_end=time,
            dt=self._core.dt,
        )
        verbose.display(
            msg=(
                f"Performing {steps.n_tot} steps with "
                f"reference model: {self._core.name}"
            ),
            trigger_level=2,
        )
        for _ in steps.simulation_steps():
            self._core.step()

    def load(self) -> BaseUVH:
        """Load the data.

        Returns:
            BaseUVH: Model's prognostic variables.
        """
        return self._core.physical

    def retrieve_P(self) -> QGProjector:  # noqa: N802
        """Retrieve projector associated with reference.

        Returns:
            QGProjector: Projector
        """
        return self._core.P

    def retrieve_category(self) -> ModelCategory:
        """Retrieve model category..

        Returns:
            ModelCategory: Model category.
        """
        return self._core.get_category()

    @classmethod
    def from_config(cls, configuration: Configuration) -> Self:
        """Create reference from configuration.

        Args:
            configuration (Configuration): Configuration.

        Returns:
            Self: Reference object.
        """
        model = instantiate_model_from_config(
            configuration.simulation.reference.model,
            configuration.space,
            configuration.windstress,
            configuration.physics,
            configuration.perturbation,
            configuration.simulation,
        )
        return cls(model=model)


class ModelOutputReference(NamedObject[ReferenceName], Reference):
    """Reference using a QGSW model output."""

    __slots__ = ("_config", "_data", "_folder", "_outs", "_ts")
    _type = ReferenceName.MODEL_OUTPUT

    def __init__(self, output_folder: str | Path) -> None:
        """Instantiate the reference.

        Args:
            output_folder (str | Path): Folder storing output.
        """
        super().__init__()
        self._folder = Path(output_folder)
        output = RunOutput(self._folder)
        self._config = output.summary.configuration
        self._ts = torch.tensor(list(output.seconds()))
        self._outs: list[_OutputReader] = list(output.outputs())
        self.at_time(self._ts[0])

    @property
    def time(self) -> float:
        """Actual loaded time."""
        return self._time

    def at_time(self, time: float) -> None:
        """Set the reference time.

        Args:
            time (float): Required time.

        Raises:
            ValueError: If the required time is beyond computed times.
        """
        super().at_time(time)
        if time > self._ts[-1]:
            msg = (
                f"The requested time ({time} s) is beyond "
                f"computed period (ends at {self._ts[-1]})."
            )
            raise ValueError(msg)
        self._time = time
        ts_index = torch.argmin((self._ts - time).abs()).item()
        self._data: _OutputReader = self._outs[ts_index]

    def load(self) -> BaseUVH:
        """Load the data.

        Returns:
            BaseUVH: Stored prognostic variables.
        """
        verbose.display(
            msg=f"Loading reference data from {self._data.path}",
            trigger_level=2,
        )
        return self._data.read()

    def retrieve_P(self) -> QGProjector:  # noqa: N802
        """Retrieve projector associated with reference.

        Returns:
            QGProjector: Projector
        """
        return QGProjector.from_config(
            space_config=self._config.space,
            model_config=self._config.model,
            physics_config=self._config.physics,
        )

    def retrieve_category(self) -> ModelCategory:
        """Retrieve model category..

        Returns:
            ModelCategory: Model category.
        """
        return get_category(ModelName(self._config.model.type))

    @classmethod
    def from_config(cls, configuration: Configuration) -> Self:
        """Create reference from configuration.

        Args:
            configuration (Configuration): Configuration.

        Returns:
            Self: Reference object.
        """
        return cls(
            output_folder=configuration.simulation.reference.folder,
        )
