"""Base plot."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import ParamSpec

from qgsw.plots.exceptions import InvalidMaskError, MismatchingMaskError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qgsw.models.sw import SW

P = ParamSpec("P")


class BaseAxes(metaclass=ABCMeta):
    """Base class for for plots contents."""

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs: P.kwargs,
    ) -> None:
        """Instantiate the Axes plot.

        Args:
            data (np.ndarray): Data to plot.
            mask (np.ndarray | None, optional): Mask to apply on data.
            Mask will be set to ones if None. Defaults to None.
            **kwargs:  additional arguments to pass to plotting method.
        """
        self._data = self._format_data(data=data)
        self._mask = self._validate_mask(mask=mask)
        self._kwargs = kwargs

    @abstractmethod
    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format data for plotting sake.

        Args:
            data (np.ndarray): Original data.

        Returns:
            np.ndarray: Valid data.
        """
        return data

    def _validate_mask(self, mask: np.ndarray | None) -> np.ndarray:
        """Assert that the mask is valid.

        Args:
            mask (np.ndarray | None): Mask input.

        Raises:
            MismatchingMaskError: If the mask's shape doesn't match data's.
            InvalidMaskError: If the mask does not only contain 0s and 1s.

        Returns:
            np.ndarray: Valid mask.
        """
        if mask is None:
            return np.ones(self._data.shape)
        if mask.shape != self._data.shape:
            msg = "Mask's shape must match data's shape."
            raise MismatchingMaskError(msg)
        if not np.all([(e in {0, 1}) for e in np.unique(mask)]):
            msg = "The mask must contain only 0s and 1s."
            raise InvalidMaskError(msg)
        return mask

    @staticmethod
    @abstractmethod
    def create_axes(figure: Figure) -> Axes:
        """Create a new axes from a given figure.

        Args:
            figure (Figure): Figure to add axes to.

        Returns:
            Axes: Created axes.
        """
        return figure.add_subplot()

    @abstractmethod
    def _style_axes(self, ax: Axes) -> Axes:
        """Style a given axes.

        Args:
            ax (Axes): Axes to style.

        Returns:
            Axes: Styled axes.
        """

    @abstractmethod
    def _set_content(self, ax: Axes) -> Axes:
        """Set the axes content.

        Args:
            ax (Axes): Axes to set the content of.

        Returns:
            Axes: Axes with content.
        """

    def add_to_axes(self, ax: Axes) -> Axes:
        """Add the content to a given Axes.

        Args:
            ax (Axes): Axes to add content to.

        Returns:
            Axes: Final Axes.
        """
        ax.clear()
        ax = self._style_axes(ax=ax)
        return self._set_content(ax=ax)

    def add_to_figure(self, figure: Figure) -> Axes:
        """Add the content to a given Figure.

        Args:
            figure (Figure): Figure to add content to.

        Returns:
            Axes: Axes added to the figure.
        """
        ax = self.create_axes(figure=figure)
        return self.add_to_axes(ax=ax)


class BaseSinglePlot(metaclass=ABCMeta):
    """Base class for a plot rendering a single Axes."""

    def __init__(
        self,
        figure: Figure | None = None,
    ) -> None:
        """Instantiate the plot.

        Args:
            figure (Figure | None, optional): Figure to render plot to.
            The figure will be created if None. Defaults to None.
        """
        plt.ion()
        self._figure = figure
        self._ax = None

    @property
    def figure(self) -> Figure:
        """Figure containing the plot."""
        if self._figure is None:
            self._figure = self._create_figure()
        return self._figure

    @property
    def ax(self) -> Axes:
        """Figure's axes."""
        if self._ax is None:
            self._ax = self._create_axes(figure=self.figure)
        return self._ax

    def _create_figure(self) -> Figure:
        """Create an empty figure.

        Returns:
            Figure: Created figure.
        """
        figure = plt.figure()
        figure.tight_layout()
        return figure

    @abstractmethod
    def _create_axes(self, figure: Figure) -> Axes:
        """Create Axes within the Figure.

        Args:
            figure (Figure): Figure to add axes to.

        Returns:
            Axes: Created axes.
        """

    @abstractmethod
    def _retrieve_data_from_model(self, model: SW) -> np.ndarray:
        """Retrieve relevant data from a SW model.

        Args:
            model (SW): Model to retrieve data from.

        Returns:
            np.ndarray: Retrieved data.
        """

    @abstractmethod
    def _update(
        self, data: np.ndarray, mask: np.ndarray, **kwargs: P.kwargs
    ) -> Axes:
        """Update the plot content.

        Args:
            data (np.ndarray): Data to use.
            mask (np.ndarray): Mask to use.
            **kwargs: additional parameters to give to plotting method.

        Returns:
            Axes: Updated Axes.
        """

    def update_with(
        self, data: np.ndarray, mask: np.ndarray, **kwargs: P.kwargs
    ) -> None:
        """Update and render the plot content.

        Args:
            data (np.ndarray): Data to use.
            mask (np.ndarray): Mask to use.
            **kwargs: additional parameters to give to plotting method.
        """
        self._update(data=data, mask=mask, **kwargs)
        plt.pause(0.05)

    def update_with_model(
        self,
        model: SW,
        mask: np.ndarray,
        **kwargs: P.kwargs,
    ) -> None:
        """Update the plot content with a SW model.

        Args:
            model (SW): Model to use for plot update.
            mask (np.ndarray): Mask to apply on data.
            **kwargs: additional parameters to give to plotting method.
        """
        return self.update_with(
            data=self._retrieve_data_from_model(model=model),
            mask=mask,
            **kwargs,
        )
