"""Comparison plots."""

from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING, Generic

import matplotlib.pyplot as plt
from typing_extensions import ParamSpec

from qgsw.plots.base.axes import AxesManager
from qgsw.plots.exceptions import (
    ImpossibleAxesUpdateError,
)

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

P = ParamSpec("P")


class ComparisonFigure(Generic[AxesManager], metaclass=ABCMeta):
    """Comparison figure."""

    _n_cols = 3

    def __init__(self, *axes_managers: AxesManager) -> None:
        """Instantiate the compariosn plot.

        Args:
            *axes_managers (AxesManager): Axes Managers for the plot.
        """
        self._axes_nb = len(axes_managers)
        self._axes_ms = axes_managers
        self._figure, axes = self._create_figure_axes()
        self._axes = axes.flatten()
        self._set_axes()

    @property
    def figure(self) -> Figure:
        """Figure."""
        return self._figure

    def _raise_if_inconsistent_datas(self, elements_nb: int) -> None:
        """Raise an error if the number of plot to update is invalid.

        Args:
            elements_nb (int): Number of plots ot update.

        Raises:
            ImpossibleAxesUpdateError: If the number doesn't match
            the number of plots.
        """
        if elements_nb != self._axes_nb:
            msg = (
                "There must be as many elements to update as axes."
                f"{self._axes_nb} were expected, {elements_nb} were given."
            )
            raise ImpossibleAxesUpdateError(msg)

    def _create_figure_axes(self) -> tuple[Figure, list[Axes]]:
        """Create the Figure and the Axes list.

        Returns:
            tuple[Figure, list[Axes]]: Figure, Axes list.
        """
        ncols = min(self._axes_nb, self._n_cols)
        nrows = (self._axes_nb - 1) // self._n_cols + 1
        return plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
        )

    def _set_axes(self) -> None:
        """Set the Axes within the Axes Managers."""
        for i, axes_m in enumerate(self._axes_ms):
            axes_m.set_ax(self._axes[i])
            axes_m.context.reload_axes(axes_m.ax)

    def update(self, *datas: np.ndarray | None, **kwargs: P.kwargs) -> None:
        """Update the Figure."""
        self._raise_if_inconsistent_datas(elements_nb=len(datas))
        for i, data in enumerate(datas):
            if data is None:
                continue
            self._axes_ms[i].update(data, **kwargs)
            plt.pause(0.05)
