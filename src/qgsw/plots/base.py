"""Base plot."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import ParamSpec, Self

from qgsw.plots.exceptions import (
    AxesInstantiationError,
    ImpossibleAxesUpdateError,
    InvalidMaskError,
    MismatchingMaskError,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qgsw.models.sw import SW

P = ParamSpec("P")


class BaseAxesContext(metaclass=ABCMeta):
    """Base class for Axes context management."""

    _default_title: str = ""

    def __init__(
        self,
    ) -> None:
        """Instantiate the AxesContext.

        Args:
            title (str): Axes title.
            xticks (list, optional): X ticks. Defaults to [].
            yticks (list, optional): Y ticks. Defaults to [].
        """
        self.title = self._default_title

    @abstractmethod
    def _create_axes(self, figure: Figure) -> Axes:
        """Create a new Axes from a given Figure.

        Args:
            figure (Figure): Figure to add axes to.

        Returns:
            Axes: Created Axes.
        """
        return figure.add_subplot()

    @abstractmethod
    def _style_axes(self, ax: Axes) -> Axes:
        """Style Axes.

        Args:
            ax (Axes): Axes to style.

        Returns:
            Axes: Styled Axes.
        """
        ax.set_title(self.title)
        return ax

    def add_new_axes(self, figure: Figure) -> Axes:
        """Add and style a new Axes to ag iven figure.

        Args:
            figure (Figure): Figure to add Axes to.

        Returns:
            Axes: Final Axes.
        """
        ax = self._create_axes(figure=figure)
        return self._style_axes(ax=ax)

    def reload_axes(self, ax: Axes) -> None:
        """Reaload given Axes.

        Args:
            ax (Axes): Axes to reload.
        """
        ax.clear()
        return self._style_axes(ax=ax)


class BaseAxesContent(metaclass=ABCMeta):
    """Base for Axes Content Management."""

    def __init__(
        self,
        mask: np.ndarray | None = None,
        **kwargs: P.kwargs,
    ) -> None:
        """Instantiate the AxesContent.

        Args:
            mask (np.ndarray | None, optional): Mask to apply on data.
            Mask will be set to ones if None. Defaults to None.
            **kwargs: Additional arguments to pass to plotting method.
        """
        self._raise_if_invalid_mask(mask=mask)
        self._mask = mask
        self._kwargs = kwargs

    def _raise_if_invalid_mask(self, mask: np.ndarray | None) -> None:
        """Raise error if the Mask is not valid.

        Args:
            mask (np.ndarray | None): Mask to inspect.

        Raises:
            InvalidMaskError: If the mask does not only contain 0s and 1s.
        """
        if mask is None:
            return
        if not np.all([(e in {0, 1}) for e in np.unique(mask)]):
            msg = "The mask must contain only 0s and 1s."
            raise InvalidMaskError(msg)

    @abstractmethod
    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format input data.

        Args:
            data (np.ndarray): Data tot format.

        Returns:
            np.ndarray: Formatted data.
        """
        return data

    @abstractmethod
    def _update(self, ax: Axes, data: np.ndarray, mask: np.ndarray) -> Axes:
        """Update ax content using data from a np.ndarray.

        Args:
            ax (Axes): Axes to update the content of.
            data (np.ndarray): Data to use for content update.
            mask (np.ndarray): Mask to use to show data.

        Returns:
            Axes: Updated axes.
        """

    def update(self, ax: Axes, data: np.ndarray, **kwargs: P.kwargs) -> Axes:
        """Update ax content using data from a np.ndarray.

        Args:
            ax (Axes): Axes to update the content of.
            data (np.ndarray): Data to use for content update.
            **kwargs: Additional arguments to give to the plotting function.

        Returns:
            Axes: Updated axes.
        """
        self._kwargs = self._kwargs | kwargs
        formatted_data = self._format_data(data=data)
        mask = self._validate_mask(data=formatted_data)
        return self._update(ax=ax, data=formatted_data, mask=mask)

    @abstractmethod
    def _retrieve_data_from_model(self, model: SW) -> np.ndarray:
        """Retrieve relevant from a given SW model.

        Args:
            model (SW): Model to retrieve data from.

        Returns:
            np.ndarray: Relevant data.
        """

    def update_with_model(
        self, ax: Axes, model: SW, **kwargs: P.kwargs
    ) -> Axes:
        """Update Axes content using data from a SW model.

        Args:
            ax (Axes): Axes to update the content of.
            model (SW): MOdel to use for the content update.
            **kwargs: Additional arguments to give to the plotting function.

        Returns:
            Axes: Updated Axes.
        """
        return self.update(
            ax=ax,
            data=self._retrieve_data_from_model(model=model),
            **kwargs,
        )

    def _validate_mask(self, data: np.ndarray) -> np.ndarray:
        """Validate Mask.

        Args:
            data (np.ndarray): Data to use as content.

        Raises:
            MismatchingMaskError: If the mask's shape doesn't match
            the data's shape.
            InvalidMaskError: If the mask do

        Returns:
            np.ndarray: Valid Mask.
        """
        if self._mask is None:
            return np.ones(data.shape)
        if self._mask.shape != data.shape:
            msg = "Mask's shape must match data's shape."
            raise MismatchingMaskError(msg)
        return self._mask


AxesContext = TypeVar("AxesContext", bound=BaseAxesContext)
AxesContent = TypeVar("AxesContent", bound=BaseAxesContent)


class BaseAxes(Generic[AxesContext, AxesContent], metaclass=ABCMeta):
    """Base class for for plots contents."""

    def __init__(self, context: AxesContext, content: AxesContent) -> None:
        """Instantiate the Axes.

        Args:
            context (AxesContext): Axes Context Manager.
            content (AxesContent): Axes Content Manager.
        """
        self._content = content
        self._context = context
        self._ax: Axes | None = None
        self._figure: Figure | None = None

    @property
    def context(self) -> AxesContext:
        """Axes Context."""
        return self._context

    @property
    def ax(self) -> Axes:
        """Axes to plot to."""
        if self._ax is None:
            msg = (
                f"Instantiate ax before calling {self.__class__.__name__}.ax"
                f" using {self.__class__.__name__}.set_ax()."
            )
            raise AxesInstantiationError(msg)
        return self._ax

    @ax.setter
    def ax(self, ax: Axes) -> None:
        """Set the ax of the object.

        Recommended Axes to use is the one generated by .create_axes.

        Args:
            ax (Axes): Axes to use for self._ax.
        """
        self.set_ax(ax=ax)

    def create_axes(self, figure: Figure) -> Axes:
        """Create a new axes from a given figure.

        Args:
            figure (Figure): Figure to add axes to.

        Returns:
            Axes: Created axes.
        """
        return self._context.add_new_axes(figure=figure)

    def set_ax(self, ax: Axes) -> None:
        """Set the ax of the object.

        Recommended Axes to use is the one generated by .create_axes.

        Args:
            ax (Axes): Axes to use for self._ax.
        """
        self._ax = ax

    def set_title(self, title: str) -> None:
        """Set the Axes' title.

        Args:
            title (str): New title.
        """
        self._context.title = title
        if self._ax is not None:
            self.ax = self._context.reload_axes(self._ax)

    def update(self, data: np.ndarray, **kwargs: P.kwargs) -> Axes:
        """Update the Axes content.

        Args:
            data (np.ndarray): Data to use for update.
            **kwargs: Additional arguments to give to the plotting function.

        Returns:
            Axes: Updated Axes.
        """
        return self._content.update(ax=self._ax, data=data, **kwargs)

    def update_with_model(self, model: SW, **kwargs: P.kwargs) -> Axes:
        """Update the Axes content using a SW model.

        Args:
            model (SW): Model to use to update Axes.
            **kwargs: Additional arguments to give to the plotting function.

        Returns:
            Axes: Updated Axes.
        """
        return self._content.update_with_model(
            ax=self._ax, model=model, **kwargs
        )

    @classmethod
    @abstractmethod
    def from_mask(
        cls, mask: np.ndarray | None = None, **kwargs: P.kwargs
    ) -> Self:
        """Instantiate Plot only from the mask.

        Args:
            mask (np.ndarray | None, optional): Mask to apply on data.
            Mask will be set to ones if None. Defaults to None.
            **kwargs: Additional arguments to pass to plotting method.

        Returns:
            Self: Instantiated plot.
        """


AxesManager = TypeVar("AxesManager", bound=BaseAxes)


class BaseSinglePlot(Generic[AxesManager], metaclass=ABCMeta):
    """Base class for a plot rendering a single Axes."""

    def __init__(
        self,
        axes_manager: AxesManager,
        figure: Figure | None = None,
    ) -> None:
        """Instantiate the plot.

        Args:
            axes_manager (AxesManager): Axes Manager.
            figure (Figure | None, optional): Figure to render plot to.
            The figure will be created if None. Defaults to None.
        """
        plt.ion()
        self._figure = figure
        self._ax = axes_manager
        self._ax.set_ax(self._ax.create_axes(figure=self.figure))

    @property
    def figure(self) -> Figure:
        """Figure containing the plot."""
        if self._figure is None:
            self._figure = self._create_figure()
        return self._figure

    @property
    def ax(self) -> AxesManager:
        """Figure's axes manager."""
        return self._ax

    @abstractmethod
    def _create_figure(self) -> Figure:
        """Create an empty figure.

        Returns:
            Figure: Created figure.
        """
        figure = plt.figure()
        figure.tight_layout()
        return figure

    def update_with(self, data: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update and render the plot content.

        Args:
            data (np.ndarray): Data to use.
            **kwargs: Additional arguments to give to the plotting function.
        """
        self._ax.update(data=data, **kwargs)
        plt.pause(0.05)

    def update_with_model(
        self,
        model: SW,
        **kwargs: P.kwargs,
    ) -> None:
        """Update the plot content with a SW model.

        Args:
            model (SW): Model to use for plot update.
            **kwargs: Additional arguments to give to the plotting function.
        """
        self._ax.update_with_model(model=model, **kwargs)
        plt.pause(0.05)

    @classmethod
    @abstractmethod
    def from_mask(
        cls, mask: np.ndarray | None = None, **kwargs: P.kwargs
    ) -> Self:
        """Instantiate Plot only from the mask.

        Args:
            mask (np.ndarray | None, optional): Mask to apply on data.
            Mask will be set to ones if None. Defaults to None.
            **kwargs: Additional arguments to pass to plotting method.

        Returns:
            Self: Instantiated plot.
        """
        return cls()


class ComparisonPlot(Generic[AxesManager], metaclass=ABCMeta):
    """Comparison plot."""

    _n_cols = 3

    def __init__(self, *axes_managers: AxesManager) -> None:
        """Instantiate the compariosn plot.

        Args:
            *axes_managers (AxesManager): Axes Managers for the plot.
        """
        self._axes_nb = len(axes_managers)
        self._axes_ms = axes_managers
        self._fig, axes = self._create_figure_axes()
        self._axes = axes.flatten()
        self._set_axes()

    def _raise_if_inconsistent_datas(self, elements_nb: int) -> None:
        """Raise an error if the number of plot to update is invalid.

        Args:
            elements_nb (int): Number of plots ot update.

        Raises:
            ImpossibleAxesUpdateError: If the number doesn't match
            the number of plots.
        """
        if elements_nb != self._axes_nb:
            msg = (
                "There must be as many elements to update as axes."
                f"{self._axes_nb} were expected, {elements_nb} were given."
            )
            raise ImpossibleAxesUpdateError(msg)

    def _create_figure_axes(self) -> tuple[Figure, list[Axes]]:
        """Create the Figure and the Axes list.

        Returns:
            tuple[Figure, list[Axes]]: Figure, Axes list.
        """
        ncols = min(self._axes_nb, self._n_cols)
        nrows = (self._axes_nb - 1) // self._n_cols + 1
        return plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
        )

    def _set_axes(self) -> None:
        """Set the Axes within the Axes Managers."""
        for i, axes_m in enumerate(self._axes_ms):
            axes_m.set_ax(self._axes[i])
            axes_m.context.reload_axes(axes_m.ax)

    def update(self, *datas: np.ndarray | None, **kwargs: P.kwargs) -> None:
        """Update the Figure."""
        self._raise_if_inconsistent_datas(elements_nb=len(datas))
        for i, data in enumerate(datas):
            if data is None:
                continue
            self._axes_ms[i].update(data, **kwargs)
            plt.pause(0.05)
