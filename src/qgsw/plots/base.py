"""Base class for plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import plotly.graph_objects as go

T = TypeVar("T")


class BasePlot(ABC, Generic[T]):
    """Base for all plots."""

    _is_set = False

    _xaxis_title = ""
    _yaxis_title = ""

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

    def _create_figure(self) -> go.Figure:
        """Create the Figure.

        Returns:
            go.Figure: Figure.
        """
        return go.Figure()

    def _create_xaxis(self) -> go.layout.XAxis:
        return go.layout.XAxis(title=self._xaxis_title)

    def _create_yaxis(self) -> go.layout.YAxis:
        return go.layout.YAxis(title=self._yaxis_title)

    def _create_layout(self) -> go.Layout:
        return go.Layout()

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
        self.figure.update_xaxes(self._create_xaxis())
        self.figure.update_yaxes(self._create_yaxis())

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
