"""Units."""


class Unit:
    """Unit."""

    def __init__(self, name: str) -> None:
        """Instantiate unit.

        Args:
            name (str): Unit name.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Name."""
        return self._name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Unit):
            return False
        return self.name == value.name

    def __repr__(self) -> str:
        """Unit string representation.

        Returns:
            str: Unit name.
        """
        return self.name


DEGREES = Unit("degree")
METERS = Unit("meters")
KILOMETERS = Unit("kilometers")
