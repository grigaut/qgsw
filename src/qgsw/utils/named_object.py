"""Type-Switch Base object."""

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, field_serializer

T = TypeVar("T", bound=Enum)


class Name(str, Enum):
    """Name."""

    __slots__ = ()


class NamedObject(Generic[T]):
    """Named Object."""

    _type: T

    @classmethod
    def get_type(cls) -> T:
        """Get the object name type.

        Returns:
            T: Name Enum.
        """
        return cls._type


class NamedObjectConfig(BaseModel, Generic[T]):
    """Named object config."""

    type: T

    @field_serializer("type")
    def serialize_type_as_str(self, t: T) -> str:
        """Serialize type as str.

        Args:
            t (T): Type.

        Returns:
            str: Type value.
        """
        return t.value
