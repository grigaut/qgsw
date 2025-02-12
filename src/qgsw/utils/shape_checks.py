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

    from qgsw.utils.size import Size


class ShapeValidationError(Exception):
    """Shape Validation Error."""


Param = ParamSpec("Param")
T = TypeVar("T")


def with_shapes(
    **shapes: tuple[int | Size, ...],
) -> Callable[[Callable[Param, T]], Callable[Param, T]]:
    """Validate input shapes.

    Parameters:
        **shapes (tuple[int,...]): Shapes to enforce for variables.

    Returns:
        Callable[Param, Callable[Param, T]]: Wrapper.
    """
    shapes: dict[str, tuple[Callable[[], int], ...]] = {
        k: tuple((lambda s=s: s) if isinstance(s, int) else s for s in v)
        for k, v in shapes.items()
    }

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
            eval_shapes = {k: tuple(e() for e in s) for k, s in shapes.items()}
            # Check shapes
            if any(
                f_params.arguments[k].shape != s
                for k, s in eval_shapes.items()
            ):
                key, shape = filter(
                    lambda ks: f_params.arguments[ks[0]].shape != ks[1],
                    eval_shapes.items(),
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
