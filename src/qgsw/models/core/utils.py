"""Utils."""

from __future__ import annotations

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from qgsw.logging import getLogger

if TYPE_CHECKING:
    from collections.abc import Callable

Param = ParamSpec("Param")
T = TypeVar("T")


logger = getLogger(__name__)


class OptimizableFunction(Generic[Param, T]):
    """Optimize functions."""

    def __init__(self, func: Callable[Param, T]) -> None:
        """Instantiate the OptimizableFunction.

        Args:
            func (Callable): Function to optimize.
        """
        if torch.__version__[0] == "2":
            msg = f"Compiling {func.__name__} using torch.compile."
            logger.detail(msg)
            self._core = torch.compile(func)
        else:
            self._func = func
            self._core = self._trace

    def _trace(self, *args: Param.args, **kwargs: Param.kwargs) -> T:
        """Trace the core function.

        Returns:
            Any: Function output.
        """
        msg = f"Tracing {self._func.__name__} using torch.jit.trace."
        logger.detail(msg)
        self._core = torch.jit.trace(self._func, args)
        return self._core(*args, **kwargs)

    def __call__(self, *args: Param.args, **kwargs: Param.kwargs) -> T:
        """Function call.

        Returns:
            Any: Core function output.
        """
        return self._core(*args, **kwargs)
