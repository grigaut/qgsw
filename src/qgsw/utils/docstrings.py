"""Docstrings-related methods."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
Param = ParamSpec("Param")


def with_docstring(
    doc: str | None,
) -> Callable[[Callable[Param, T]], Callable[Param, T]]:
    """Uses another docstring for the decorated function.

    Args:
        doc (str | None): Docstring to use.

    Returns:
        Callable[[Callable[Param, T]], Callable[Param, T]]: Warpper.
    """

    def wrapper(f: Callable[Param, T]) -> Callable[Param, T]:
        @wraps(f)
        def wrapped(*args: Param.args, **kwargs: Param.kwargs) -> T:
            return f(*args, **kwargs)

        wrapped.__doc__ = doc

        return wrapped

    return wrapper
