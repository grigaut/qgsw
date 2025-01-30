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
        if any(key not in signature.parameters for key in shapes):
            key = filter(
                lambda key: key not in signature.parameters,
                shapes,
            ).__next__()
            msg = (
                f"'{key}' is not a valid argument name for {f.__qualname__}()."
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
            if any(
                f_params.arguments[k].shape != s for k, s in shapes.items()
            ):
                key, shape = filter(
                    lambda ks: f_params.arguments[ks[0]].shape != ks[1],
                    shapes.items(),
                ).__next__()
                msg = (
                    f"{f.__qualname__}(): '{key}'"
                    f" must be a {shape}-shaped tensor, and not"
                    f" a {f_params.arguments[key].shape}-shaped tensor."
                )
                raise ShapeValidationError(msg)
            # Return result
            return f(*f_params.args, **f_params.kwargs)

        return wrapped

    return wrapper
