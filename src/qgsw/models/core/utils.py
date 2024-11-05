"""Utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import ParamSpec

from qgsw import verbose

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")


class OptimizableFunction:
    """Optimize functions."""

    def __init__(self, func: Callable) -> None:
        """Instantiate the OptimizableFunction.

        Args:
            func (Callable): Function to optimize.
        """
        if torch.__version__[0] == "2":
            verbose.display(
                f"Compiling {func.__name__} using torch.compile.",
                trigger_level=2,
            )
            self._core = torch.compile(func)
        else:
            self._func = func
            self._core = self._trace

    def _trace(self, *args: P.args) -> Any:  # noqa: ANN401
        """Trace the core function.

        Returns:
            Any: Function output.
        """
        verbose.display(
            f"Tracing {self._func.__name__} using torch.jit.trace.",
            trigger_level=2,
        )
        self._core = torch.jit.trace(self._func, args)
        return self._core(*args)

    def __call__(self, *args: P.args) -> Any:  # noqa: ANN401
        """Function call.

        Returns:
            Any: Core function output.
        """
        return self._core(*args)
