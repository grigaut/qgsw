"""Line plots."""

from __future__ import annotations

from qgsw.run_summary import RunOutput, check_time_compatibility

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from qgsw.plots.base import BasePlot

if TYPE_CHECKING:
    from pathlib import Path


class ScatterPlot(BasePlot[np.ndarray]):
    """Line plot."""

    def __init__(self, datas: list[np.ndarray]) -> None:
        """Instantiate the plot.

        Args:
            datas (list[np.ndarray]): List of data to use.
        """
        super().__init__(datas)
        self._datas = datas

    def _add_traces(self) -> None:
        for subplot_index in range(self.n_subplots):
            self.figure.add_trace(
                go.Scatter(
                    y=self._datas[subplot_index],
                ),
            )

    @classmethod
    def level_wise_from_folders(
        cls,
        folders: list[Path | str],
        field: str,
        levels: int | list[int] = 0,
    ) -> Self:
        """Instantiate the plot from a list of folders.

        Args:
            folders (list[Path  |  str]): List of folders to use as source.
            field (str): Field to display.
            levels (int | list[int], optional): Layer(s) to display.
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

        if (not isinstance(levels, int)) and (not isinstance(levels, list)):
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
            [data[0, ls[k]] for data in run[field].datas()]
            for k, run in enumerate(runs)
        ]
        cls._xaxis_title = "Time"
        yaxis_title = f"{runs[0][field].description} [{runs[0][field].unit}]"
        cls._yaxis_title = yaxis_title
        return cls(datas=datas)

    @classmethod
    def ensemble_wise_from_folders(
        cls,
        folders: list[Path | str],
        field: str,
    ) -> Self:
        """Instantiate the plot from a list of folders.

        Args:
            folders (list[Path  |  str]): List of folders to use as source.
            field (str): Field to display.

        Raises:
            ValueError: If the timesteps are incompatible.
            TypeError: If levels is neither int or list[int]
            ValueError: If the levels length doesn't match run's

        Returns:
            Self: AnimatedHeatmap.
        """
        runs = [RunOutput(folder=f) for f in folders]
        check_time_compatibility(*runs)
        datas = [[data[0] for data in run[field].datas()] for run in runs]
        cls._xaxis_title = "Time"
        yaxis_title = f"{runs[0][field].description} [{runs[0][field].unit}]"
        cls._yaxis_title = yaxis_title
        return cls(datas=datas)
