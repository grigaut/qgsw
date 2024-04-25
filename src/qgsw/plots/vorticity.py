"""Vorticity plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import ParamSpec

from qgsw.plots.base import BaseAxes, BaseSinglePlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qgsw.models.sw import SW

P = ParamSpec("P")


class SurfaceVorticityAxes(BaseAxes):
    """Surface Vorticity Axes."""

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format vorticity input.

        Args:
            data (np.ndarray): (1,nl,nx,ny) Vorticity grid.

        Returns:
            np.ndarray: Surface Vorticity layer (nx,ny).
        """
        return data[0, 0]

    @staticmethod
    def create_axes(figure: Figure) -> Axes:
        """Create axes on the given figure.

        Args:
            figure (Figure): Figure to add axes to.

        Returns:
            Axes: Added Axes.
        """
        return figure.add_subplot()

    def _style_axes(self, ax: Axes) -> Axes:
        """Style Axes.

        Args:
            ax (Axes): Axes to style.

        Returns:
            Axes: Styled axes.
        """
        ax.set_title(r"$\omega$")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def _set_content(self, ax: Axes) -> Axes:
        """Set Axes content.

        Args:
            ax (Axes): Axes to set the content of.

        Returns:
            Axes: Axes with the content set.
        """
        masked_data = np.ma.masked_where(self._mask, self._data).T
        ax.imshow(masked_data, **self._kwargs)
        return ax


class SurfaceVorticityPlot(BaseSinglePlot):
    """Surface Vorticity Plot."""

    def _create_axes(self, figure: Figure) -> Axes:
        """Create Axes on the given Figure.

        Args:
            figure (Figure): Figure to add Axes to.

        Returns:
            Axes: Created Axes.
        """
        return SurfaceVorticityAxes.create_axes(figure=figure)

    def _retrieve_data_from_model(self, model: SW) -> np.ndarray:
        """Retrieve vorticity data from a SW model.

        Args:
            model (SW): Model to retrieve vorticity from.

        Returns:
            np.ndarray: Vorticity fiedl (1,nl,nx,ny)-shaped.
        """
        omega = model.omega
        area = model.area
        f0 = model.f0
        return (omega / area / f0).cpu().numpy()

    def _update(
        self, data: np.ndarray, mask: np.ndarray, **kwargs: P.kwargs
    ) -> Axes:
        """Update the content of the vorticity plot.

        Args:
            data (np.ndarray): Vorticity data (1,nl,nx,ny)-shaped.
            mask (np.ndarray): Mask to apply on data.
            **kwargs: additional argument to pass to ax.imshow

        Returns:
            Axes: Updated Axes.
        """
        content = SurfaceVorticityAxes(data, mask, **kwargs)
        return content.add_to_axes(ax=self.ax)
