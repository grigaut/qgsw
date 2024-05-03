"""Axes Managers."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from typing_extensions import ParamSpec, Self

from qgsw.plots.exceptions import (
    AxesInstantiationError,
    InvalidMaskError,
    MismatchingMaskError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qgsw.models.base import Model


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
        mask = self._validate_mask(data=data)
        return self._update(ax=ax, data=data, mask=mask)

    @abstractmethod
    def retrieve_data_from_array(self, array: np.ndarray) -> np.ndarray:
        """Retrieve relevant data from a given array.

        Args:
            array (np.ndarray): Data tot format.

        Returns:
            np.ndarray: Formatted data.
        """

    @abstractmethod
    def retrieve_data_from_model(self, model: Model) -> np.ndarray:
        """Retrieve relevant from a given model.

        Args:
            model (Model): Model to retrieve data from.

        Returns:
            np.ndarray: Relevant data.
        """

    @abstractmethod
    def retrieve_data_from_file(self, file: Path) -> np.ndarray:
        """Retrieve relevant from a given file.

        Args:
            file (Path): File to retrieve data from.

        Returns:
            np.ndarray: Relevant data.
        """

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
            return np.zeros(data.shape)
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
    def content(self) -> AxesContent:
        """Axes Content."""
        return self._content

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

    def retrieve_data_from_array(self, array: np.ndarray) -> np.ndarray:
        """Retrieve data from a given np.ndarray.

        Args:
            array (np.ndarray): array to use data from.

        Returns:
            np.ndarray: Loaded data from the array.
        """
        return self._content.retrieve_data_from_array(array=array)

    def retrieve_data_from_model(self, model: Model) -> np.ndarray:
        """Retrieve data from a given Model.

        Args:
            model (Model): model to use data from.

        Returns:
            np.ndarray: Loaded dat from the model.
        """
        return self._content.retrieve_data_from_model(model=model)

    def retrieve_data_from_file(self, file: Path) -> np.ndarray:
        """Retrieve data from a given npz file.

        Args:
            file (Path): NPZ file path.

        Returns:
            np.ndarray: Loaded data.
        """
        return self._content.retrieve_data_from_file(filepath=file)

    @classmethod
    @abstractmethod
    def from_mask(
        cls,
        mask: np.ndarray | None = None,
        **kwargs: P.kwargs,
    ) -> Self:
        """Instantiate Figure only from the mask.

        Args:
            mask (np.ndarray | None, optional): Mask to apply on data.
            Mask will be set to ones if None. Defaults to None.
            **kwargs: Additional arguments to pass to plotting method.

        Returns:
            Self: Instantiated plot.
        """


AxesManager = TypeVar("AxesManager", bound=BaseAxes)
