"""Size."""

from collections.abc import Callable


class Size:
    """Dimension."""

    def __init__(self) -> None:
        """Instantiate the size.

        Args:
            size (int): Size initial value.
        """

    def set_to(self, size: int) -> None:
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
        try:
            return self._size
        except AttributeError as e:
            msg = "Call `set_to` method to set the size first."
            raise AttributeError(msg) from e

    def __add__(self, add_to_size: int) -> Callable[[], int]:
        """Magic method for the '+' operator."""
        return lambda: self() + add_to_size

    def __repr__(self) -> str:
        """String representation of Size."""
        return self.__call__().__repr__()
