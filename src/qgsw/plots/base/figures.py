"""Base plots."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic

import matplotlib.pyplot as plt
from typing_extensions import ParamSpec, Self

from qgsw.plots.base.axes import AxesManager

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from matplotlib.figure import Figure

    from qgsw.models.base import Model

P = ParamSpec("P")


class BaseFigure:
    """Base class for figures."""

    @property
    def figure(self) -> Figure:
        """Figure containing the plot."""
        return self._figure

    def savefig(self, output_file: Path) -> None:
        """Save figure in the given output file.

        Args:
            output_file (Path): File to save figure at.
        """
        self.figure.savefig(fname=output_file)

    def show(self) -> None:
        """Show the figure."""
        plt.pause(0.05)


class BaseSingleFigure(Generic[AxesManager], BaseFigure, metaclass=ABCMeta):
    """Base class for a plot rendering a single Axes."""

    def __init__(
        self,
        axes_manager: AxesManager,
    ) -> None:
        """Instantiate the plot.

        Args:
            axes_manager (AxesManager): Axes Manager.
            figure (Figure | None, optional): Figure to render plot to.
            The figure will be created if None. Defaults to None.
        """
        plt.ion()
        self._figure = self._create_figure()
        self._ax = axes_manager
        self._ax.set_ax(self._ax.create_axes(figure=self.figure))

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

    def _update(self, data: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update and render the plot content.

        Args:
            data (np.ndarray): Data to use.
            **kwargs: Additional arguments to give to the plotting function.
        """
        self._ax.update(data=data, **kwargs)

    def update(self, data: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update and render the plot content.

        Args:
            data (np.ndarray): Data to use.
            **kwargs: Additional arguments to give to the plotting function.
        """
        self._update(data=data, **kwargs)

    def update_with_model(
        self,
        model: Model,
        **kwargs: P.kwargs,
    ) -> None:
        """Update the plot content with a model.

        Args:
            model (Model): Model to use for plot update.
            **kwargs: Additional arguments to give to the plotting function.
        """
        self._update(self._ax.retrieve_data_from_model(model=model), **kwargs)

    def update_with_files(
        self,
        file: Path,
        **kwargs: P.kwargs,
    ) -> None:
        """Update the plot content with a file.

        Args:
            file (Path): File path to use for plot update.
            **kwargs: Additional arguments to give to the plotting function.
        """
        self._update(self._ax.retrieve_data_from_file(file=file), **kwargs)

    @classmethod
    @abstractmethod
    def from_mask(
        cls, mask: np.ndarray | None = None, **kwargs: P.kwargs
    ) -> Self:
        """Instantiate Figure only from the mask.

        Args:
            mask (np.ndarray | None, optional): Mask to apply on data.
            Mask will be set to ones if None. Defaults to None.
            **kwargs: Additional arguments to pass to plotting method.

        Returns:
            Self: Instantiated plot.
        """
        return cls()
