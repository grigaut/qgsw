"""Type-Switch Base object."""


class TypeSwitch:
    """Implement type switch."""

    _type: str

    def __init__(self) -> None:
        """Instantiate the Type Switch."""

    @property
    def type(self) -> str:
        """Object's type."""
        return self._type

    @classmethod
    def match_type(cls, type_string: str) -> bool:
        """Check if a string matches the object's type.

        Args:
            type_string (str): String to check.

        Returns:
            bool: True if the string matches the type.
        """
        return cls._type == type_string
