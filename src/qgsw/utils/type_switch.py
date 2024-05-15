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
    def get_type(cls) -> str:
        """Return the string type of the object.

        Use it to validate input from configuration.

        Returns:
            str: Object type string.
        """
        return cls._type
