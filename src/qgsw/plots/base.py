"""Base class for plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import plotly.graph_objects as go
from plotly.subplots import make_subplots

T = TypeVar("T")


class BasePlot(ABC, Generic[T]):
    """Base for all plots."""

    def __init__(
        self,
        datas: list[T],
    ) -> None:
        """Instantiate the plot.

        Args:
            datas (list[T]): List of datas to plot.
        """
        self._check_lengths(datas)
        self._n_subplots = len(datas)
        self._fig = self._create_figure()

    @cached_property
    def n_cols(self) -> int:
        """Number of columns."""
        return min(3, self.n_subplots)

    @property
    def figure(self) -> go.Figure:
        """Figure."""
        return self._fig

    @property
    def n_subplots(self) -> int:
        """Number of subplots."""
        return self._n_subplots

    @cached_property
    def n_rows(self) -> int:
        """NUmber of rows."""
        return (self.n_subplots - 1) // self.n_cols + 1

    def _check_lengths(self, datas: list[T]) -> None:
        """Check than the data length are matching."""
        n_frames = len(datas[0])
        if all(len(data) == n_frames for data in datas):
            return
        msg = "Different lengths for datas."
        raise ValueError(msg)

    def _create_figure(
        self,
        subplot_titles: list[str] | None = None,
    ) -> go.Figure:
        """Create the Figure.

        Returns:
            go.Figure: Figure.
        """
        return make_subplots(
            rows=self.n_rows,
            cols=self.n_cols,
            subplot_titles=subplot_titles,
        )

    def set_subplot_titles(self, subplot_titles: list[str]) -> None:
        """Set the subplot titles.

        Args:
            subplot_titles (list[str]): List of subplot titles.
        """
        self._fig = self._create_figure(subplot_titles)

    def _create_layout(self) -> go.Layout:
        return go.Layout()

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

    def _set_figure(self) -> None:
        """Set the figure traces."""
        if self._is_set:
            return
        self._is_set = True
        self._add_traces()
        self.figure.update_layout(self._create_layout())

    def show(self) -> None:
        """Show the Figure."""
        self._set_figure()
        self.figure.show()

    def retrieve_figure(self) -> go.Figure:
        """Retrieve the figure.

        Returns:
            go.Figure: Figure
        """
        self._set_figure()
        return self.figure
