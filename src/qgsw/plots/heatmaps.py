"""Plots from files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.colors as pco
import plotly.graph_objects as go

from qgsw.plots.animated_plots import BaseAnimatedPlots
from qgsw.run_summary import RunOutput

if TYPE_CHECKING:
    from pathlib import Path


class AnimatedHeatmaps(BaseAnimatedPlots):
    """Animated Heatmap with shared colorscale."""

    def __init__(
        self,
        datas: list[list[np.ndarray]],
    ) -> None:
        """Instantiate plot."""
        super().__init__(datas)

    def _add_traces(self) -> None:
        """Initialize the traces."""
        for subplot_index in range(self.n_subplots):
            row, col = self.map_subplot_index_to_subplot_loc(subplot_index)
            self.figure.add_trace(
                go.Heatmap(
                    z=self._datas[subplot_index][0],
                    showscale=subplot_index == 0,
                    colorscale=pco.diverging.RdBu_r,
                ),
                row=row,
                col=col,
            )

    def _compute_frame(self, frame_nb: int) -> go.Frame:
        zmax = max(
            np.max(np.abs(data)) for data in [d[frame_nb] for d in self._datas]
        )
        return go.Frame(
            data=[
                go.Heatmap(
                    z=self._datas[subplot_index][frame_nb],
                    colorscale=pco.diverging.RdBu_r,
                    showscale=subplot_index == 0,
                    zmin=-zmax,
                    zmax=zmax,
                )
                for subplot_index in range(self.n_subplots)
            ],
            traces=list(range(self.n_subplots)),
            name=frame_nb,
        )

    def _compute_step(self, frame_nb: int) -> go.Frame:
        return go.layout.slider.Step(
            args=[
                [frame_nb],
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            label=frame_nb,
            method="animate",
        )


class AnimatedHeatmapsFromRunFolders:
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
        self._plot = AnimatedHeatmaps(
            [
                [self._read_data(file, self._layers[k]) for file in run.files]
                for k, run in enumerate(self._runs)
            ],
        )

    @property
    def figure(self) -> go.Figure:
        """Figure."""
        return self._plot.figure

    @property
    def field(self) -> str:
        """Field to plot."""
        return self._field

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

    def show(self) -> None:
        """Show the Figure."""
        return self._plot.show()

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

    def _read_data(self, file: Path, layer: int) -> np.ndarray:
        """Read the data from a file.

        Args:
            file (Path): File to read data from.
            layer (int): Layer to consider.

        Returns:
            np.ndarray: Data.
        """
        return np.load(file)[self.field][0, layer].T
