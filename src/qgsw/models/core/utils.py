"""Utils."""

from __future__ import annotations

from typing import Any, Callable

import torch
from typing_extensions import ParamSpec

P = ParamSpec("P")


class OptimizableFunction:
    """Optimize functions."""

    def __init__(self, function: Callable) -> None:
        """Instantiate the OptimizableFunction.

        Args:
            function (Callable): Function to optimize.
        """
        if torch.__version__[0] == "2":
            self._core = torch.compile(function)
        else:
            self._func = function
            self._core = self._trace

    def _trace(self, *args: P.args) -> Any:  # noqa: ANN401
        """Trace the core function.

        Returns:
            Any: Function output.
        """
        self._core = torch.jit.trace(self._func, args)
        return self._core(*args)

    def __call__(self, *args: P.args) -> Any:  # noqa: ANN401
        """Function call.

        Returns:
            Any: Core function output.
        """
        return self._core(*args)
