"""Plots from files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.colors as pco
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qgsw.plots.animated_plots import BaseAnimatedMaps
from qgsw.run_summary import RunOutput

if TYPE_CHECKING:
    from pathlib import Path

    from qgsw.variables.base import ParsedVariable


class AnimatedHeatmaps(BaseAnimatedMaps[np.ndarray]):
    """Animated Heatmap with shared colorscale."""

    def __init__(
        self,
        datas: list[list[np.ndarray]],
        frame_labels: list[str] | None = None,
    ) -> None:
        """Instantiate the plot.

        Args:
            datas (list[list[np.ndarray]]): Data list.
            frame_labels (list[str] | None, optional): Frames names.
            Defaults to None.
        """
        self._colorbar = self._create_colorbar()
        super().__init__(datas, frame_labels)

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
            title={"text": "", "side": "right"},
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


class AnimatedHeatmapsFromRunFolders(AnimatedHeatmaps):
    """Animated heatmap from folders."""

    def __init__(
        self,
        folders: list[Path | str],
        field: str = "p",
        layers: int | list[int] = 0,
    ) -> None:
        """Instantiate the plot.

        Args:
            folders (list[Path  |  str]): Folder to use for plotting.
            field (str, optional): Field to plot. Defaults to "p".
            layers (int | list[int], optional): Layer to plot.
            If list, correspond to layer for each run and must have the same
            length as folders. Defaults to 0.
        """
        self._runs = [RunOutput(folder=f) for f in folders]
        self._check_compatibilities()
        self._field = field
        self._layers = self._set_layers(layers)
        super().__init__(
            [
                self._read_data(run[field], self._layers[k])
                for k, run in enumerate(self._runs)
            ],
            frame_labels=[f"{t.days} days" for t in self._runs[0].timesteps()],
        )

    @property
    def field(self) -> str:
        """Field to plot."""
        return self._field

    def _make_titles(self) -> list[str]:
        """Create subplots titles.

        Returns:
            list[str]: List of titles.
        """
        return [
            f"{run.summary.configuration.model.name} - Layer {self._layers[i]}"
            for i, run in enumerate(self._runs)
        ]

    def _create_figure(self) -> go.Figure:
        """Create the figure.

        Returns:
            go.Figure: Figure.
        """
        return make_subplots(
            rows=self.n_rows,
            cols=self.n_cols,
            subplot_titles=self._make_titles(),
        )

    def _create_slider_current_value(self) -> go.layout.slider.Currentvalue:
        """Create Slider current value and set prefix to 'Time'.

        Returns:
            go.layout.slider.Currentvalue: Slider current value.
        """
        return super()._create_slider_current_value().update(prefix="Time: ")

    def _create_colorbar(self) -> go.heatmap.ColorBar:
        """Create the colorbar.

        Returns:
            go.heatmap.ColorBar: Colorbar.
        """
        variable = self._runs[0][self._field]
        legend = f"{variable.description} [{variable.unit}]"
        return (
            super()
            ._create_colorbar()
            .update(title={"text": legend, "side": "right"})
        )

    def _check_compatibilities(self) -> None:
        """Ensure that timesteps are compatible.

        Raises:
            ValueError: If timestep don't match.
        """
        dt0 = self._runs[0].summary.configuration.simulation.dt
        dts = [run.summary.configuration.simulation.dt for run in self._runs]
        if all(dt == dt0 for dt in dts):
            return
        msg = "Incompatible timesteps."
        raise ValueError(msg)

    def _set_layers(self, layers: int | list[int]) -> list[int]:
        """Set the layers.

        Args:
            layers (int | list[int]): Layers.

        Raises:
            ValueError: If the layer list is not valid.

        Returns:
            list[int]: List of layer to plot for every given run.
        """
        if isinstance(layers, int):
            return [layers] * len(self._runs)
        if isinstance(layers, list):
            if len(layers) == len(self._runs):
                return layers
            msg = (
                "The layers list should be of length "
                f"{len(self._runs)} instead of {len(layers)}."
            )
            raise ValueError(msg)
        msg = "`layers` parameter should be of type int or list[int]."
        raise ValueError(msg)

    def _read_data(
        self,
        variable: ParsedVariable,
        layer: int,
    ) -> list[np.ndarray]:
        """Read the data from a file.

        Args:
            variable (ParsedVariable): run output.
            layer (int): Layer to consider.

        Returns:
            np.ndarray: Data.
        """
        return [data[0, layer].T for data in variable.datas()]
