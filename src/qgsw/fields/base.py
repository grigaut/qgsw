"""Base class for fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from types import EllipsisType
    except ImportError:
        EllipsisType = type(...)

    from qgsw.fields.scope import Scope


class Field:
    """Field."""

    _name: str
    _description: str
    _scope: Scope
    _slices: list[slice | EllipsisType] = None  # type: ignore  # noqa: PGH003

    @property
    def name(self) -> str:
        """Variable name."""
        return self._name

    @property
    def description(self) -> str:
        """Variable description."""
        return self._description

    @property
    def scope(self) -> Scope:
        """Variable scope."""
        return self._scope

    @property
    def slices(self) -> list[slice | EllipsisType]:  # type: ignore  # noqa: PGH003
        """Slice to apply to data."""
        if self._slices is None:
            return [...]
        return self._slices

    @slices.setter
    def slices(self, slices: list[slice | EllipsisType]) -> None:  # type: ignore  # noqa: PGH003
        self._slices = slices

    def __repr__(self) -> str:
        """Variable string representation."""
        return f"{self._description}: {self._name}"

    @classmethod
    def get_name(cls) -> str:
        """Retrieve the name of the variable.

        Returns:
            str: Variable name.
        """
        return cls._name

    @classmethod
    def get_description(cls) -> str:
        """Retrieve the description of the variable.

        Returns:
            str: Variable description.
        """
        return cls._description

    @classmethod
    def get_scope(cls) -> Scope:
        """Retrieve the scope of the variable.

        Returns:
            Scope: Variable scope.
        """
        return cls._scope
