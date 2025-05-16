"""Base class for references."""

from abc import ABC, abstractmethod

from qgsw.fields.variables.tuples import BaseUVH
from qgsw.models.names import ModelCategory
from qgsw.models.qg.uvh.projectors.core import QGProjector

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from qgsw.configs.core import Configuration


class Reference(ABC):
    """Reference base class."""

    @property
    @abstractmethod
    def time(self) -> float:
        """Actual loaded time."""

    @abstractmethod
    def retrieve_P(self) -> QGProjector:  # noqa: N802
        """Retrieve projector associated with reference.

        Returns:
            QGProjector: Projector
        """

    @abstractmethod
    def retrieve_category(self) -> ModelCategory:
        """Retrieve model category..

        Returns:
            ModelCategory: Model category.
        """

    @abstractmethod
    def at_time(self, time: float) -> None:
        """Set the reference time.

        Args:
            time (float): Required time.

        Raises:
            ValueError: If the required time is negative.
        """
        if time < 0:
            msg = "Time must be a positive number."
            raise ValueError(msg)

    @abstractmethod
    def load(self) -> BaseUVH:
        """Load the data.

        Returns:
            BaseUVH: Stored physical variables.
        """

    @classmethod
    @abstractmethod
    def from_config(cls, configuration: Configuration) -> Self:
        """Create reference from configuration.

        Args:
            configuration (Configuration): Configuration.

        Returns:
            Self: Reference object.
        """
