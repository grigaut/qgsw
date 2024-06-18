"""Comparison plots."""

from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING, Generic

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import ParamSpec

from qgsw.plots.base.axes import AxesManager
from qgsw.plots.base.figures import BaseFigure
from qgsw.plots.exceptions import (
    AxesUpdateError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qgsw.models.base import Model

P = ParamSpec("P")


class ComparisonFigure(Generic[AxesManager], BaseFigure, metaclass=ABCMeta):
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
        self._figure.canvas.manager.set_window_title("comparison")
        self._axes: np.ndarray = axes.reshape((-1,))
        self._set_axes()

    def _raise_if_inconsistent_length(self, elements_nb: int) -> None:
        """Raise an error if the number of plot to update is invalid.

        Args:
            elements_nb (int): Number of plots ot update.

        Raises:
            AxesUpdateError: If the number doesn't match
            the number of plots.
        """
        if elements_nb != self._axes_nb:
            msg = (
                "There must be as many elements to update as axes."
                f"{self._axes_nb} were expected, {elements_nb} were given."
            )
            raise AxesUpdateError(msg)

    def _create_figure_axes(self) -> tuple[Figure, list[Axes]]:
        """Create the Figure and the Axes list.

        Returns:
            tuple[Figure, list[Axes]]: Figure, Axes list.
        """
        ncols = min(self._axes_nb, self._n_cols)
        nrows = (self._axes_nb - 1) // self._n_cols + 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6 * ncols, 6 * nrows),
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        return fig, axes

    def _set_axes(self) -> None:
        """Set the Axes within the Axes Managers."""
        for i, axes_m in enumerate(self._axes_ms):
            axes_m.set_ax(self._axes[i])
            axes_m.context.reload_axes(axes_m.ax)

    def _update(self, *datas: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update the Figure."""
        self._raise_if_inconsistent_length(elements_nb=len(datas))
        for i, data in enumerate(datas):
            self._axes_ms[i].update(data, **kwargs)

    def update_with_arrays(
        self,
        *arrays: np.ndarray,
        **kwargs: P.kwargs,
    ) -> None:
        """Update the Figure."""
        self._raise_if_inconsistent_length(len(arrays))
        self._update(*arrays, **kwargs)

    def update_with_files(self, *files: Path, **kwargs: P.kwargs) -> None:
        """Update the plot given some NPZ files."""
        self._raise_if_inconsistent_length(len(files))

        datas = [
            self._axes_ms[i].retrieve_array_from_file(files[i])
            for i in range(len(files))
        ]
        self._update(*datas, **kwargs)

    def update_with_models(self, *models: Model, **kwargs: P.kwargs) -> None:
        """Update the plot given some models."""
        self._raise_if_inconsistent_length(len(models))

        arrays = [
            self._axes_ms[i].retrieve_array_from_model(models[i])
            for i in range(len(models))
        ]
        self._update(*arrays, **kwargs)
