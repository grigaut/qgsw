"""Scopes."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, overload


class Scope(ABC):
    """Scope."""

    _ens: str = "ensemble_wise"
    _level: str = "level_wise"
    _point: str = "point_wise"

    @property
    @abstractmethod
    def ensemble_wise(self) -> bool:
        """Whether it is ensemble wise."""

    @property
    @abstractmethod
    def level_wise(self) -> bool:
        """Whether it is level wise."""

    @property
    @abstractmethod
    def point_wise(self) -> bool:
        """Whether it is point wise."""

    @cached_property
    def ensemble_wise_at_most(self) -> bool:
        """Whether the scope is no more than ensemble-wise."""
        return (
            self.ensemble_wise
            and (not self.level_wise)
            and (not self.point_wise)
        )

    @cached_property
    def level_wise_at_most(self) -> bool:
        """Whether the scope is no more than level-wise."""
        return self.ensemble_wise and not self.point_wise

    @cached_property
    def point_wise_at_most(self) -> bool:
        """Whether the scope is no more than point-wise."""
        return self.ensemble_wise and self.level_wise and self.point_wise

    @cached_property
    def ensemble_wise_at_least(self) -> bool:
        """Whether the scope is at least ensemble-wise."""
        return self.ensemble_wise

    @cached_property
    def level_wise_at_least(self) -> bool:
        """Whether the scope is at least level-wise."""
        return self.ensemble_wise and self.level_wise

    @cached_property
    def point_wise_at_least(self) -> bool:
        """Whether the scope is at least point-wise."""
        return self.ensemble_wise and self.level_wise and self.point_wise

    @property
    def stricly_ensemble_wise(self) -> bool:
        """Whether the scope is stricly ensemble wise."""
        return self.ensemble_wise_at_least and self.ensemble_wise_at_most

    @property
    def stricly_level_wise(self) -> bool:
        """Whether the scope is stricly level wise."""
        return self.level_wise_at_least and self.level_wise_at_most

    @property
    def stricly_point_wise(self) -> bool:
        """Whether the scope is stricly point wise."""
        return self.point_wise_at_least and self.point_wise_at_most

    def to_dict(self) -> dict[str, bool]:
        """Convert the scope to a dictionnary.

        Returns:
            dict[str, bool]: Dictionnary.
        """
        return {
            self._ens: self.ensemble_wise,
            self._level: self.level_wise,
            self._point: self.point_wise,
        }

    @overload
    @classmethod
    def from_bools(
        cls,
        ensemble: Literal[True],
        level: Literal[True],
        point: Literal[True],
    ) -> "PointWise": ...
    @overload
    @classmethod
    def from_bools(
        cls,
        ensemble: Literal[True],
        level: Literal[True],
        point: Literal[False],
    ) -> "LevelWise": ...
    @overload
    @classmethod
    def from_bools(
        cls,
        ensemble: Literal[True],
        level: Literal[False],
        point: Literal[False],
    ) -> "EnsembleWise": ...

    @classmethod
    def from_bools(
        cls,
        ensemble: bool,  # noqa: FBT001
        level: bool,  # noqa: FBT001
        point: bool,  # noqa: FBT001
    ) -> "EnsembleWise|LevelWise|PointWise":
        """Instantiate the scope from booleans.

        Args:
            ensemble (bool): Whether it is ensemble wise.
            level (bool): Whether it is level wise.
            point (bool): Whether it is point wise.

        Raises:
            ValueError: If the boolean pattern is not valid.

        Returns:
            EnsembleWise|LevelWise|PointWise: Scope.
        """
        if ensemble and level and point:
            return PointWise()
        if ensemble and level:
            return LevelWise()
        if ensemble:
            return EnsembleWise()
        msg = "Invalid arguments pattern."
        raise ValueError(msg)

    @classmethod
    def from_dict(cls, dic: dict) -> "EnsembleWise|LevelWise|PointWise":
        """Instantiate the scope from a dictionnary.

        Args:
            dic (dict): Dictionnary to use.

        Returns:
            EnsembleWise|LevelWise|PointWise: Scope.
        """
        return cls.from_bools(dic[cls._ens], dic[cls._level], dic[cls._point])


class EnsembleWise(Scope):
    """Ensemble-wise scope."""

    @property
    def ensemble_wise(self) -> bool:
        """Whether it is ensemble wise."""
        return True

    @property
    def level_wise(self) -> bool:
        """Whether it is level wise."""
        return False

    @property
    def point_wise(self) -> bool:
        """Whether it is point wise."""
        return False


class LevelWise(EnsembleWise):
    """Level-wise scope."""

    @property
    def level_wise(self) -> bool:
        """Whether it is level wise."""
        return True


class PointWise(LevelWise):
    """Point-wise scope."""

    @property
    def point_wise(self) -> bool:
        """Whether it is point wise."""
        return True
