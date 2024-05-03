"""Verbose Decorators."""

from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
P = ParamSpec("P")


class VerboseDisplayError(Exception):
    """Verbose Related Exception."""


class VerboseDisplayer:
    """Class for any object displaying verbose."""

    _instance = None
    _min_verbose_allowed: int = 0
    _max_verbose_allowed: int = 3

    def __init__(self, level: int, prefix: str = "\t") -> None:
        r"""Instantiate verbose displayer.

        Args:
            level (int, optional): Level of details:
            - 0 or below: no information displayed
            - 1: minimal informations displayed
            - 2: very complete informations displayed
            - 3 or higher: exhaustive informations displayed

            Defaults to 0.
            prefix (str, optional): Verbose messages prefix.
            Defaults to "\t".
        """
        self.level = level
        self.prefix = prefix

    @property
    def level(self) -> int:
        """Verbose level."""
        return self._level

    @level.setter
    def level(self, level: int) -> None:
        """Set verbose level."""
        self._level = self._keep_within_bounds(level=level)

    @property
    def prefix(self) -> int:
        """Verbose prefix."""
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        """Set verbose prefix.

        Args:
            prefix (str): String to prepend to the
            verbose messages.

        Raises:
            VerboseDisplayError: If the given value is not a string.
        """
        if not isinstance(prefix, str):
            msg = "Indentation must be a string."
            raise VerboseDisplayError(msg)
        self._prefix = prefix

    def _keep_within_bounds(self, level: int) -> int:
        """Project a given verbose level between the verbose bounds.

        Args:
            level (int): Verbose level.

        Returns:
            int: Projected level.
        """
        bound_max_level = min(level, self._max_verbose_allowed)
        return max(bound_max_level, self._min_verbose_allowed)

    def _check_trigger_level(self, trigger_level: int) -> int:
        """Check validity of trigger level and project it within the bounds.

        Args:
            trigger_level (int): Given trigger level.

        Raises:
            VerboseDisplayError: If the trigger levl is below or equal to 0.

        Returns:
            int: Projected trigger level.
        """
        if trigger_level <= self._min_verbose_allowed:
            msg = (
                f"Trigger level must be strictly superior to "
                f"{self._min_verbose_allowed}. "
                f"Given value: {trigger_level}."
            )
            raise VerboseDisplayError(msg)
        return self._keep_within_bounds(level=trigger_level)

    def _indent(self, msg: str, level: int) -> str:
        """Return indentated message.

        Args:
            msg (str): Message to indentate.
            level (int): Level of indentation.

        Returns:
            str: Indentated message.
        """
        indent = "".join([self._prefix] * (level))
        return f"{indent}{msg}"

    def display(self, msg: str, trigger_level: int) -> None:
        """Display verbose message.

        Args:
            msg (str): Message to display.
            trigger_level (int): Trigger level.
        """
        trigger = self._check_trigger_level(trigger_level=trigger_level)
        if self.level >= trigger:
            print(self._indent(msg=msg, level=(trigger - 1)))  # noqa: T201


VERBOSE = VerboseDisplayer(0)


def set_level(level: int) -> None:
    """Set verbose level.

    Args:
        level (int): Level of details:

        - 0 or below: no information displayed
        - 1: minimal informations displayed
        - 2: very complete informations displayed
        - 3 or higher: exhaustive informations displayed
    """
    VERBOSE.level = level


def set_prefix(prefix: str) -> None:
    """Set verbose prefix.

    Args:
        prefix (str): Prefix repeatedly indentated before messages.
    """
    VERBOSE.prefix = prefix


def get_level() -> int:
    """Access verbose level.

    Returns:
        int: Level of verbose:

        - 0 or below: no information displayed
        - 1: minimal informations displayed
        - 2: very complete informations displayed
        - 3 or higher: exhaustive informations displayed
    """
    return VERBOSE.level


def display(msg: str, trigger_level: int) -> None:
    """Display verbose message.

    Args:
        msg (str): Message to display.
        trigger_level (int): Trigger level.
    """
    return VERBOSE.display(msg=msg, trigger_level=trigger_level)
