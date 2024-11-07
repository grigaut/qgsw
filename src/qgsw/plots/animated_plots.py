"""Base plots."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
import plotly.colors as pco
import plotly.graph_objects as go
from plotly.subplots import make_subplots

T = TypeVar("T")


class BaseAnimatedPlots(ABC, Generic[T]):
    """Animated Plot base class."""

    def __init__(self, datas: list[list[T]]) -> None:
        """Instantiate the plot.

        Args:
            datas (list[list[T]]): Data list.
        """
        self._datas = datas
        self._check_lengths()
        self._fig = self._create_figure()
        self.figure.update_layout(self._create_layout())
        self._add_traces()

    @cached_property
    def n_subplots(self) -> int:
        """NUmber of subplots."""
        return len(self._datas)

    @cached_property
    def n_steps(self) -> int:
        """Number of steps."""
        return len(self._datas[0])

    @cached_property
    def n_cols(self) -> int:
        """Number of columns."""
        return min(3, self.n_subplots)

    @cached_property
    def n_rows(self) -> int:
        """NUmber of rows."""
        return (self.n_subplots - 1) // self.n_cols + 1

    @property
    def figure(self) -> go.Figure:
        """Figure."""
        return self._fig

    def _check_lengths(self) -> None:
        """Check than the data length are matching."""
        if all(len(data) == self.n_steps for data in self._datas):
            return
        msg = "Different lengths for datas."
        raise ValueError(msg)

    def _create_figure(self) -> go.Figure:
        """Create the Figure.

        Returns:
            go.Figure: Figure.
        """
        return make_subplots(rows=self.n_rows, cols=self.n_cols)

    def _create_layout(self) -> go.Layout:
        """Create the layout.

        Returns:
            go.Layout: Layout.
        """
        return {
            "updatemenus": [
                {
                    "type": "buttons",
                    "visible": True,
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None]},
                    ],
                },
            ],
        }

    def map_subplot_index_to_subplot_loc(
        self,
        subplot_index: int,
    ) -> tuple[int, int]:
        """Convert subplot index to subplot location.

        Args:
            subplot_index (int): Subplot index.

        Returns:
            tuple[int, int]: Row and column location within the plot.
        """
        return (subplot_index // self.n_cols + 1, subplot_index % 3 + 1)

    def map_subplot_loc_to_subplot_index(self, loc: tuple[int, int]) -> int:
        """Convert subplot location to subplot index.

        Args:
            loc (tuple[int, int]): Row and column location within the plot.

        Returns:
            int: Subplot index.
        """
        row, col = loc
        return (row - 1) * self.n_cols + (col - 1)

    @abstractmethod
    def _add_traces(self) -> None:
        """Initialize the traces."""

    @abstractmethod
    def _generate_frames(self) -> list[go.Frame]:
        """Generate the frames.

        Returns:
            list[dict]: Frames list.
        """

    def show(self) -> None:
        """Show the Figure."""
        self.figure.update(frames=self._generate_frames())
        self.figure.show()


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

    def _generate_frames(self) -> list[go.Frame]:
        """Generate the frames.

        Returns:
            list[go.Frame]: Frames list.
        """
        zbounds = [
            self._compute_zbounds([d[i] for d in self._datas])
            for i in range(self.n_steps)
        ]
        return [
            go.Frame(
                name=step,
                data=[
                    go.Heatmap(
                        z=self._datas[subplot_index][step],
                        colorscale=pco.diverging.RdBu_r,
                        showscale=subplot_index == 0,
                        zmin=zbounds[step]["min"],
                        zmax=zbounds[step]["max"],
                    )
                    for subplot_index in range(self.n_subplots)
                ],
                traces=list(range(self.n_subplots)),
            )
            for step in range(self.n_steps)
        ]

    def _compute_zbounds(
        self,
        datas_at_step: list[np.ndarray],
    ) -> tuple[int, int]:
        """Compute colorbar bounds.

        Args:
            datas_at_step (list[np.ndarray]): List of data at given step.

        Returns:
            tuple[int, int]: Minimum and maximum value.
        """
        zmax = max(np.max(np.abs(data)) for data in datas_at_step)
        return {"min": -zmax, "max": zmax}
