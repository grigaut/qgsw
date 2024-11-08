"""Plots from files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgsw.plots.animated_plots import AnimatedHeatmaps
from qgsw.run_summary import RunOutput

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go


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
