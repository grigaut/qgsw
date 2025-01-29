"""Decorator to check tensor shapes."""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar


class ShapeValidationError(Exception):
    """Shape Validation Error."""


Param = ParamSpec("Param")
T = TypeVar("T")


def with_shapes(
    **shapes: tuple[int, ...],
) -> Callable[[Callable[Param, T]], Callable[Param, T]]:
    """Validate input shapes.

    Parameters:
        **shapes (tuple[int,...]): Shapes to enforce for variables.

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
        for key in shapes:
            if key not in signature.parameters:
                msg = (
                    f"'{key}' is not a valid argument"
                    f" name for {f.__qualname__}()."
                )
                raise ValueError(msg)

        @wraps(f)
        def wrapped(*args: Param.args, **kwargs: Param.kwargs) -> T:
            """Wrapped version of f.

            Raises:
                ShapeValidationError: If a parameter shape is invalid.

            Returns:
                T: Return value of f.
            """
            # bind arguments to f
            f_params = signature.bind(*args, **kwargs)
            # Apply defaults
            f_params.apply_defaults()
            # Check shapes
            for key, shape in shapes.items():
                if (arg_shape := f_params.arguments[key].shape) != shape:
                    msg = (
                        f"{f.__qualname__}(): '{key}'"
                        f" must be a {shape}-shaped tensor,"
                        f" and not a {arg_shape}-shaped tensor,"
                    )
                    raise ShapeValidationError(msg)
            # Return result
            return f(*f_params.args, **f_params.kwargs)

        return wrapped

    return wrapper
