"""Vorticity plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import ParamSpec, Self

from qgsw.models.sw import SW
from qgsw.plots.base import (
    BaseAxes,
    BaseAxesContent,
    BaseAxesContext,
    BaseSinglePlot,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qgsw.models.sw import SW

P = ParamSpec("P")


class SurfaceVortictyAxesContext(BaseAxesContext):
    """Axes Context Manager for surface vorticity plot."""

    _default_title = r"$\omega$"

    def _create_axes(self, figure: Figure) -> Axes:
        """Create Axes on a given figure.

        Args:
            figure (Figure): Figure to add Axes to.

        Returns:
            Axes: Added Axes.
        """
        return super()._create_axes(figure)

    def _style_axes(self, ax: Axes) -> Axes:
        """Style Axes.

        Args:
            ax (Axes): Axes to style.

        Returns:
            Axes: Styled Axes.
        """
        ax.set_xticks([])
        ax.set_yticks([])
        return super()._style_axes(ax)


class SurfaceVorticityAxesContent(BaseAxesContent):
    """Axes Content Manager for surface vorticity plot."""

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format input data.

        Args:
            data (np.ndarray): Input data (1,nl,nx,ny).

        Returns:
            np.ndarray: Formatted data (nx,ny).
        """
        return data[0, 0]

    def _update(self, ax: Axes, data: np.ndarray, mask: np.ndarray) -> Axes:
        """Update the Axes content.

        Args:
            ax (Axes): Axes to update.
            data (np.ndarray): Data to use for update (nx,ny).
            mask (np.ndarray): Mask to apply on data.

        Returns:
            Axes: Updated Axes.
        """
        masked_data = np.ma.masked_where(mask, data).T
        ax.imshow(masked_data, **self._kwargs)
        return ax

    def _retrieve_data_from_model(self, model: SW) -> np.ndarray:
        """Retrieve data from a SW model.

        Args:
            model (SW): SW model.

        Returns:
            np.ndarray: Retrieved data (1,nl,nx,ny).
        """
        omega = model.omega
        area = model.area
        f0 = model.f0
        return (omega / area / f0).cpu().numpy()

    def update(self, ax: Axes, data: np.ndarray, **kwargs: P.kwargs) -> Axes:
        """Update Axes content.

        Args:
            ax (Axes): Axes to update.
            data (np.ndarray): Data to use (1,nl,nx,ny).
            **kwargs: Additional arguments to give to the plotting function.

        Returns:
            Axes: Updated Axes.
        """
        return super().update(ax, data, **kwargs)


class SurfaceVorticityAxes(
    BaseAxes[SurfaceVortictyAxesContext, SurfaceVorticityAxesContent]
):
    """Surface Vorticity Axes."""

    def update(self, data: np.ndarray, **kwargs: P.kwargs) -> Axes:
        """Update Axes content.

        Args:
            data (np.ndarray): Data to use (1,nl,nx,ny).
            **kwargs: Additional arguments to give to the plotting function.

        Returns:
            Axes: Updated Axes.
        """
        return super().update(data, **kwargs)


class SurfaceVorticityPlot(BaseSinglePlot[SurfaceVorticityAxes]):
    """Surface Vorticity Plot."""

    def _create_figure(self) -> Figure:
        return super()._create_figure()

    def update_with(self, data: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update Axes content.

        Args:
            data (np.ndarray): Data to use (1,nl,nx,ny).
            **kwargs: Additional arguments to give to the plotting function.
        """
        return super().update_with(data, **kwargs)

    @classmethod
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
        return cls(
            axes_manager=SurfaceVorticityAxes(
                context=SurfaceVortictyAxesContext(),
                content=SurfaceVorticityAxesContent(mask=mask, **kwargs),
            )
        )
