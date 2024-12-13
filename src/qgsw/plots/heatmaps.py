"""Plots from files."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import plotly.colors as pco
import plotly.graph_objects as go

from qgsw.plots.animated_plots import BaseAnimatedPlot
from qgsw.run_summary import RunOutput, check_time_compatibility

if TYPE_CHECKING:
    from pathlib import Path


class AnimatedHeatmaps(BaseAnimatedPlot[np.ndarray]):
    """Animated Heatmap with shared colorscale."""

    _color_bar_text = ""

    def __init__(
        self,
        datas: list[list[np.ndarray]],
    ) -> None:
        """Instantiate the plot.

        Args:
            datas (list[list[np.ndarray]]): Data list.
            frame_labels (list[str] | None, optional): Frames names.
            Defaults to None.
        """
        self._colorbar = self._create_colorbar()
        super().__init__(datas)

    @property
    def colorbar(self) -> go.heatmap.ColorBar:
        """Plot colorbar."""
        return self._colorbar

    def _create_colorbar(self) -> go.heatmap.ColorBar:
        """Create the colorbar.

        Returns:
            go.heatmap.ColorBar: Colorbar.
        """
        return go.heatmap.ColorBar(
            exponentformat="e",
            showexponent="all",
            title={"text": self._color_bar_text, "side": "right"},
            thickness=50,
        )

    def _add_traces(self) -> None:
        """Initialize the traces."""
        zmax = self._compute_zmax(self._datas[0][1])
        showscales = self._compute_showscales(self._datas[0][1])
        for subplot_index in range(self.n_subplots):
            row, col = self.map_subplot_index_to_subplot_loc(subplot_index)
            self.figure.add_trace(
                go.Heatmap(
                    z=self._datas[0][1][subplot_index],
                    showscale=showscales[subplot_index],
                    colorscale=pco.diverging.RdBu_r,
                    zmin=-zmax,
                    zmax=zmax,
                    colorbar=self.colorbar,
                ),
                row=row,
                col=col,
            )

    def _compute_frame(self, frame_index: int) -> go.Frame:
        """Compute a frame a at a given index.

        Args:
            frame_index (int): Frame index.

        Returns:
            go.Frame: Frame.
        """
        frame_arrays = self._datas[frame_index][1]
        zmax = self._compute_zmax(frame_arrays)
        showscales = self._compute_showscales(frame_arrays)
        return go.Frame(
            data=[
                go.Heatmap(
                    z=frame_arrays[subplot_index],
                    colorscale=pco.diverging.RdBu_r,
                    showscale=showscales[subplot_index],
                    colorbar=self.colorbar,
                    zmin=-zmax,
                    zmax=zmax,
                )
                for subplot_index in range(self.n_subplots)
            ],
            traces=list(range(self.n_subplots)),
            name=frame_index,
        )

    def _compute_step(self, frame_index: int) -> go.layout.slider.Step:
        """Compute a step at a given index.

        Args:
            frame_index (int): Frame index.

        Returns:
            go.layout.slider.Step: Step..
        """
        return go.layout.slider.Step(
            args=[
                [frame_index],
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            label=self._datas[frame_index][0],
            method="animate",
        )

    def _compute_showscales(self, arrays: list[np.ndarray]) -> list[bool]:
        """Compute which trace to show scale of.

        Args:
            arrays (list[np.ndarray]): List of data arrays.

        Returns:
            list[bool]: True on the index of scale to show, False elsewhere.
        """
        not_empty_arrays = [~np.isnan(array).all() for array in arrays]
        find_first = np.cumsum(not_empty_arrays)
        remove_trailing = np.cumsum(find_first)
        return [bool(res) for res in (remove_trailing == 1)]

    def _compute_zmax(self, arrays: list[np.ndarray]) -> float:
        """Compute maxiumal value over al ist of arrays, without nans.

        Args:
            arrays (list[np.ndarray]): List of arrays to find the max of.

        Returns:
            float: Maximum values over the arrays, np.nan only if all array
            are entirely made of nans.
        """
        not_empty_arrays = [
            array for array in arrays if ~np.isnan(array).all()
        ]
        if not not_empty_arrays:
            return np.nan
        not_empty_vals = [arr[~np.isnan(arr)] for arr in not_empty_arrays]
        return max(np.max(np.abs(arr)) for arr in not_empty_vals)

    @classmethod
    def from_point_wise_output(
        cls,
        folders: list[Path | str],
        field: str,
        ensembles: int | list[int] = 0,
        levels: int | list[int] = 0,
    ) -> Self:
        """Instantiate the plot from a list of folders.

        Args:
            folders (list[Path  |  str]): LIst of folders to use as source.
            field (str): Field to display.
            ensembles (int | list[int], optional): Ensemble(s) to display.
            Defaults to 0.
            levels (int | list[int], optional): Level(s) to display.
            Defaults to 0.

        Raises:
            ValueError: If the timesteps are incompatible.
            TypeError: If levels is neither int or list[int]
            ValueError: If the levels length doesn't match run's

        Returns:
            Self: AnimatedHeatmap.
        """
        runs = [RunOutput(folder=f) for f in folders]
        check_time_compatibility(*runs)

        if not all(run[field].scope.point_wise for run in runs):
            msg = "The fields must be level-wise."
            raise ValueError(msg)

        # Validate ensembles structure
        if not (isinstance(ensembles, (int, list))):
            msg = "`ensembles` parameter should be of type int or list[int]."
            raise TypeError(msg)
        if isinstance(ensembles, list):
            if len(ensembles) != len(runs):
                msg = (
                    "The ensembles list should be of length "
                    f"{len(runs)} instead of {len(ensembles)}."
                )
                raise ValueError(msg)
            es = ensembles
        else:
            es = [ensembles] * len(runs)
        # Validate levels structure
        if not (isinstance(levels, (int, list))):
            msg = "`levels` parameter should be of type int or list[int]."
            raise TypeError(msg)
        if isinstance(levels, list):
            if len(levels) != len(runs):
                msg = (
                    "The levels list should be of length "
                    f"{len(runs)} instead of {len(levels)}."
                )
                raise ValueError(msg)
            ls = levels
        else:
            ls = [levels] * len(runs)

        datas = [
            [data[es[k], ls[k]].T for data in run[field].datas()]
            for k, run in enumerate(runs)
        ]
        cls._slider_prefix = "Time: "
        legend = f"{runs[0][field].description} [{runs[0][field].unit}]"
        cls._color_bar_text = legend
        plot = cls(datas=datas)
        plot.set_subplot_titles(
            [
                f"{run.summary.configuration.model.name} - Layer {ls[i]}"
                for i, run in enumerate(runs)
            ],
        )
        plot.set_frame_labels([f"{t.days} days" for t in runs[0].timesteps()])

        return plot
