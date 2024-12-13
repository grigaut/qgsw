"""Line plots."""

from __future__ import annotations

from qgsw.run_summary import RunOutput, check_time_compatibility
from qgsw.variables.utils import check_unit_compatibility

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
        self._traces_name = [None for _ in range(self.n_traces)]

    def _add_traces(self) -> None:
        for trace_nb in range(self.n_traces):
            self.figure.add_trace(
                go.Scatter(
                    y=self._datas[trace_nb],
                    name=self._traces_name[trace_nb],
                ),
            )

    def set_traces_name(self, traces_name: list[str]) -> None:
        """Set the name of the traces.

        Args:
            traces_name (list[str]): List of names.

        Raises:
            ValueError: If the number of names
            doesn't match the number of traces.
        """
        if len(traces_name) != self.n_traces:
            msg = f"There must be {self.n_traces} names."
            raise ValueError(msg)
        self._traces_name = traces_name

    @classmethod
    def level_wise_from_folders(
        cls,
        folders: list[Path | str],
        fields: str | list[str],
        ensembles: int | list[int] = 0,
        levels: int | list[int] = 0,
    ) -> Self:
        r"""Instantiate the plot from a list of folders.

        Args:
            folders (list[Path  |  str]): List of folders to use as source.
            fields (str | list[str]): Field to display.
            ensembles (int | list[int], optional): Ensemble(s) to display.
            Defaults to 0.
            levels (int | list[int], optional): level(s) to display.
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

        if not isinstance(fields, str | list):
            msg = "`fields` parameter should be of type str or list[str]."
            raise TypeError(msg)
        if isinstance(fields, list):
            if len(fields) != len(runs):
                msg = (
                    "The fields list should be of length "
                    f"{len(runs)} instead of {len(fields)}."
                )
                raise ValueError(msg)
            fs = fields
        else:
            fs = [fields] * len(runs)

        check_unit_compatibility(*[run[fs[k]] for k, run in enumerate(runs)])

        if not isinstance(ensembles, int | list):
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

        if not isinstance(levels, int | list):
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
            [data[es[k], ls[k]] for data in run[fs[k]].datas()]
            for k, run in enumerate(runs)
        ]
        cls._xaxis_title = "Time"
        yaxis_title = f"{runs[0][fs[0]].description} [{runs[0][fs[0]].unit}]"
        cls._yaxis_title = yaxis_title
        plot = cls(datas=datas)
        traces_name = [
            f"{run[fs[k]].description} - Ens: {es[k]} - Level: {ls[k]}"
            for k, run in enumerate(runs)
        ]
        plot.set_traces_name(traces_name)
        return plot

    @classmethod
    def ensemble_wise_from_folders(
        cls,
        folders: list[Path | str],
        fields: str,
        ensembles: int | list[int] = 0,
    ) -> Self:
        """Instantiate the plot from a list of folders.

        Args:
            folders (list[Path  |  str]): List of folders to use as source.
            fields (str | list[str]): Field to display.
            ensembles (int | list[int], optional): Ensemble(s) to display.
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

        if not isinstance(fields, str | list):
            msg = "`fields` parameter should be of type str or list[str]."
            raise TypeError(msg)
        if isinstance(fields, list):
            if len(fields) != len(runs):
                msg = (
                    "The fields list should be of length "
                    f"{len(runs)} instead of {len(fields)}."
                )
                raise ValueError(msg)
            fs = fields
        else:
            fs = [fields] * len(runs)

        check_unit_compatibility(*[run[fs[k]] for k, run in enumerate(runs)])

        if not isinstance(ensembles, int | list):
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

        datas = [
            [data[es[k]] for data in run[fs[k]].datas()]
            for k, run in enumerate(runs)
        ]
        cls._xaxis_title = "Time"
        yaxis_title = f"{runs[0][fs[0]].description} [{runs[0][fs[0]].unit}]"
        cls._yaxis_title = yaxis_title
        plot = cls(datas=datas)
        traces_name = [
            f"{run[fs[k]].description} - Ens: {es[k]}"
            for k, run in enumerate(runs)
        ]
        plot.set_traces_name(traces_name)
        return plot
