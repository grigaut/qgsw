"""Size."""

from collections.abc import Callable


class Size:
    """Dimension."""

    def __init__(self, size: int) -> None:
        """Instantiate the size.

        Args:
            size (int): Size initial value.
        """
        self._size = size

    def update(self, size: int) -> None:
        """Update the size value.

        Args:
            size (int): New size value.
        """
        self._size = size

    def __call__(self) -> int:
        """Return the size.

        Returns:
            int: Size value.
        """
        return self._size

    def __add__(self, add_to_size: int) -> Callable[[], int]:
        """Magic method for the '+' operator."""
        return lambda: self() + add_to_size
