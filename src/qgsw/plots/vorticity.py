"""Vorticity plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import ParamSpec, Self

from qgsw.models.sw import SW
from qgsw.plots.base.axes import (
    BaseAxes,
    BaseAxesContent,
    BaseAxesContext,
)
from qgsw.plots.base.comparison import ComparisonFigure
from qgsw.plots.base.figures import BaseSingleFigure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

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

    def __init__(
        self, mask: np.ndarray | None = None, **kwargs: P.kwargs
    ) -> None:
        """Instantiate the AxesContent.

        Args:
            mask (np.ndarray | None, optional): Mask to apply on data.
            Mask will be set to ones if None. Defaults to None.
            **kwargs: Additional arguments to pass to plotting method.
        """
        palette = plt.cm.bwr
        kwargs["cmap"] = kwargs.get("cmap", palette)
        kwargs["origin"] = kwargs.get("origin", "lower")
        kwargs["animated"] = kwargs.get("animated", True)
        super().__init__(mask, **kwargs)
        self._axesim = None

    @property
    def axes_image(self) -> AxesImage:
        """Axes Image."""
        return self._axesim

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
        axesim = ax.imshow(masked_data, **self._kwargs)
        self._axesim = axesim
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

    @classmethod
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
        return cls(
            context=SurfaceVortictyAxesContext(),
            content=SurfaceVorticityAxesContent(mask=mask, **kwargs),
        )


class SurfaceVorticityFigure(BaseSingleFigure[SurfaceVorticityAxes]):
    """Surface Vorticity Figure."""

    def __init__(self, axes_manager: SurfaceVorticityAxes) -> None:
        """Instantiate the Surface Vorticity Figure.

        Args:
            axes_manager (SurfaceVorticityAxes): Axes Manager.
        """
        super().__init__(axes_manager)
        self._cbar_axes = None

    def _show_colorbar(self) -> None:
        """Show the colorbar."""
        if self._cbar_axes is None:
            # Create the colorbar Axes.
            self.figure.subplots_adjust(right=0.85)
            self._cbar_axes: Axes = self.figure.add_axes(
                [0.88, 0.15, 0.04, 0.7]
            )
        if self._ax.content.axes_image is not None:
            # Update the colorbar.
            self.figure.colorbar(
                self._ax.content.axes_image,
                cax=self._cbar_axes,
                label=r"$\omega / f_0$",
            )

    def _update(self, data: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update the Figure.

        Args:
            data (np.ndarray): Data tuse for update (1,nl,nx,ny).
            **kwargs: Additional arguments to give to the plotting function.
        """
        super()._update(data, **kwargs)
        self._show_colorbar()

    def update(self, data: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update Axes content.

        Args:
            data (np.ndarray): Data to use (1,nl,nx,ny).
            **kwargs: Additional arguments to give to the plotting function.
        """
        super().update(data, **kwargs)

    @classmethod
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
        return cls(
            axes_manager=SurfaceVorticityAxes.from_mask(mask=mask, **kwargs),
        )


class SurfaceVorticityComparisonFigure(ComparisonFigure[SurfaceVorticityAxes]):
    """Comparison between Surface Vorticity Axes."""

    def __init__(self, *axes_managers: SurfaceVorticityAxes) -> None:
        """Instantiate the surface vorticity comparison."""
        super().__init__(*axes_managers)
        self._cbar_axes = None

    def _show_colorbar(self) -> None:
        """Show the colorbar."""
        if self._cbar_axes is None:
            # create colorbar Axes.
            self.figure.subplots_adjust(right=0.85)
            self._cbar_axes: Axes = self.figure.add_axes(
                [0.88, 0.15, 0.04, 0.7]
            )
        if self._axes_ms[0].content.axes_image is not None:
            # Add colorbar to the Axes.
            self.figure.colorbar(
                self._axes_ms[0].content.axes_image,
                cax=self._cbar_axes,
                label=r"$\omega / f_0$",
            )

    def _update(self, *datas: np.ndarray | None, **kwargs: P.kwargs) -> None:
        """Update the Axes."""
        super()._update(*datas, **kwargs)
        self._show_colorbar()