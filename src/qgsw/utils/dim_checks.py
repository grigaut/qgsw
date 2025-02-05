"""Decorator to check tensor shapes."""

from __future__ import annotations

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

import inspect
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable


class DimValidationError(Exception):
    """Dim Validation Error."""


Param = ParamSpec("Param")
T = TypeVar("T")


def with_dims(
    **dims: int,
) -> Callable[[Callable[Param, T]], Callable[Param, T]]:
    """Validate input dims.

    Parameters:
        **dims (int): Number of dims to enforce for variables.

    Returns:
        Callable[Param, Callable[Param, T]]: Wrapper.
    """

    def wrapper(f: Callable[Param, T]) -> Callable[Param, T]:
        """Wrapper.

        Args:
            f (Callable[Param, T]): Function to wrap.

        Returns:
            Callable[Param, T]: Wrapped function.
        """
        # Signature of f
        signature = inspect.signature(f)
        if any(key not in signature.parameters for key in dims):
            key = filter(
                lambda key: key not in signature.parameters,
                dims,
            ).__next__()
            msg = (
                f"'{key}' is not a valid argument name for {f.__qualname__}()."
            )
            raise ValueError(msg)

        @wraps(f)
        def wrapped(*args: Param.args, **kwargs: Param.kwargs) -> T:
            """Wrapped version of f.

            Raises:
                DimValidationError: If a parameter dim is invalid.

            Returns:
                T: Return value of f.
            """
            # bind arguments to f
            f_params = signature.bind(*args, **kwargs)
            # Apply defaults
            f_params.apply_defaults()
            # Check dims
            if any(f_params.arguments[k].dim() != d for k, d in dims.items()):
                key, dim = filter(
                    lambda ks: f_params.arguments[ks[0]].dim() != ks[1],
                    dims.items(),
                ).__next__()
                msg = (
                    f"{f.__qualname__}(): '{key}'"
                    f" must be a tensor with {dim} dimensions, and not"
                    f" a tensor with {f_params.arguments[key].dim()}"
                    f" dimensions."
                )
                raise DimValidationError(msg)
            # Return result
            return f(*f_params.args, **f_params.kwargs)

        return wrapped

    return wrapper
