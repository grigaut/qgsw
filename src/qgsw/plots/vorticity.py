"""Vorticity plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import ParamSpec, Self

from qgsw.plots.base.axes import (
    BaseAxes,
    BaseAxesContent,
    BaseAxesContext,
)
from qgsw.plots.base.comparison import ComparisonFigure
from qgsw.plots.base.figures import BaseSingleFigure
from qgsw.spatial.exceptions import InvalidLayerNumberError

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

    from qgsw.models.base import Model

P = ParamSpec("P")


class VorticityAxesContext(BaseAxesContext):
    """Axes Context Manager for vorticity plots."""

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


class VorticityAxesContent(BaseAxesContent):
    """Axes Content Manager for vorticity plots."""

    _layer_nb: int = 0

    def __init__(
        self,
        **kwargs: P.kwargs,
    ) -> None:
        """Instantiate the AxesContent.

        Args:
            **kwargs: Additional arguments to pass to plotting method.
        """
        palette = plt.cm.bwr
        kwargs["cmap"] = kwargs.get("cmap", palette)
        kwargs["origin"] = kwargs.get("origin", "lower")
        kwargs["animated"] = kwargs.get("animated", True)
        super().__init__(**kwargs)
        self._axesim = None
        self._cbar: Colorbar | None = None
        self._has_cbar: bool = False

    @property
    def axes_image(self) -> AxesImage:
        """Axes Image."""
        return self._axesim

    def is_array_valid(self, array: np.ndarray) -> bool:
        """Whether the array is valid ofr the plot.

        Args:
            array (np.ndarray): Array to check.

        Returns:
            bool: True if the array is valid, False otherwise.
        """
        return array.shape[1] > self._layer_nb

    def _empty(self, ax: Axes) -> Axes:
        """Set an empty Axes.

        Args:
            ax (Axes): Axes to set.

        Returns:
            Axes: Empty axes.
        """
        ax.clear()
        return ax.text(0.25, 0.5, f"NO DATA FOR LAYER {self._layer_nb}")

    def _center_cbar(self, data: np.ndarray) -> dict[str, Any]:
        """Center the colorbar values.

        Args:
            data (np.ndarray): Data to plot.

        Returns:
            dict[str, Any]: New imshow kwargs.
        """
        if ("vmin" in self._kwargs) or ("vmax" in self._kwargs):
            return self._kwargs
        max_value = np.abs(data).max()
        cbar_extrems = {
            "vmax": max_value,
            "vmin": -max_value,
        }
        return self._kwargs | cbar_extrems

    def _update(self, ax: Axes, data: np.ndarray) -> Axes:
        """Update the Axes content.

        Args:
            ax (Axes): Axes to update.
            data (np.ndarray): Data to use for update (nx,ny).

        Returns:
            Axes: Updated Axes.
        """
        axesim = ax.imshow(data, **self._center_cbar(data))
        self.remove_colorbar()
        self.add_colorbar(ax=ax, axesim=axesim)
        return ax

    def retrieve_data_from_array(self, array: np.ndarray) -> np.ndarray:
        """Format input array.

        Args:
            array (np.ndarray): Input array (1,nl,nx,ny).

        Returns:
            np.ndarray: Data (nx,ny).
        """
        if array.shape[1] <= self._layer_nb:
            msg = (
                f"Impossible to display layer {self._layer_nb} "
                f"with {array.shape}-shaped array."
            )
            raise InvalidLayerNumberError(msg)
        return array[0, self._layer_nb]

    def retrieve_array_from_model(self, model: Model) -> np.ndarray:
        """Retrieve array from a model.

        Args:
            model (Model): Model model.

        Returns:
            np.ndarray: Retrieved array (1,nl,nx,ny).
        """
        return model.get_physical_omega_as_ndarray()

    def retrieve_array_from_file(self, filepath: Path) -> np.ndarray:
        """Retrieve relevant array from a given file.

        Args:
            filepath (Path): File to retrieve data from.

        Returns:
            np.ndarray: Retrieved data (1,nl,nx,ny).
        """
        return np.load(file=filepath)["omega"]

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

    def remove_colorbar(self) -> None:
        """Remove the colorbar."""
        if self._has_cbar:
            self._cbar.remove()
            self._has_cbar = False

    def add_colorbar(self, ax: Axes, axesim: AxesImage) -> None:
        """Remove the colorbar."""
        self._cbar = ax.figure.colorbar(axesim, label=r"$\omega / f_0$")
        self._has_cbar = True
        self._axesim = axesim


class SurfaceVorticityAxesContent(VorticityAxesContent):
    """Axes Content Manager for surface vorticity plot."""

    _layer_nb = 0


class SecondLayerVorticityAxesContent(VorticityAxesContent):
    """Axes Content Manager for second layer vorticity plot."""

    _layer_nb = 1


class SurfaceVorticityAxes(
    BaseAxes[VorticityAxesContext, SurfaceVorticityAxesContent],
):
    """Vorticity axes for Surface vorticity plots."""

    @classmethod
    def from_kwargs(
        cls,
        **kwargs: P.kwargs,
    ) -> Self:
        """Instantiate Figure only from the kwargs.

        Args:
            **kwargs: Additional arguments to pass to plotting method.

        Returns:
            Self: Instantiated plot.
        """
        return cls(
            context=VorticityAxesContext(),
            content=SurfaceVorticityAxesContent(**kwargs),
        )


class SecondLayerVorticityAxes(
    BaseAxes[VorticityAxesContext, SecondLayerVorticityAxesContent],
):
    """Vorticity axes for second layer vorticity plots."""

    @classmethod
    def from_kwargs(
        cls,
        **kwargs: P.kwargs,
    ) -> Self:
        """Instantiate Figure only from the kwargs.

        Args:
            **kwargs: Additional arguments to pass to plotting method.

        Returns:
            Self: Instantiated plot.
        """
        return cls(
            context=VorticityAxesContext(),
            content=SecondLayerVorticityAxesContent(**kwargs),
        )


VorticityAxes = Union[SurfaceVorticityAxes, SecondLayerVorticityAxes]


class VorticityFigure(BaseSingleFigure[VorticityAxes]):
    """Vorticity Figure."""

    def __init__(self, axes_manager: VorticityAxes) -> None:
        """Instantiate the Surface Vorticity Figure.

        Args:
            axes_manager (SurfaceVorticityAxes): Axes Manager.
        """
        super().__init__(axes_manager)
        self._cbar_axes = None

    def _create_figure(self) -> Figure:
        return super()._create_figure()

    def update(self, data: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update Axes content.

        Args:
            data (np.ndarray): Data to use (1,nl,nx,ny).
            **kwargs: Additional arguments to give to the plotting function.
        """
        super().update(data, **kwargs)

    @classmethod
    def from_kwargs(
        cls,
        **kwargs: P.kwargs,
    ) -> Self:
        """Instantiate Figure only from the kwargs.

        Args:
            **kwargs: Additional arguments to pass to plotting method.

        Returns:
            Self: Instantiated plot.
        """
        return cls(
            axes_manager=SurfaceVorticityAxes.from_kwargs(**kwargs),
        )


class VorticityComparisonFigure(ComparisonFigure[VorticityAxes]):
    """Comparison between Surface Vorticity Axes."""

    def __init__(
        self,
        *axes_managers: VorticityAxes,
        common_cbar: bool = True,
    ) -> None:
        """Instantiate the surface vorticity comparison."""
        super().__init__(*axes_managers)
        self._cbar_axes = None
        self._common_cbar = common_cbar

    def _set_cbar_extrems(
        self,
        *datas: np.ndarray,
        **kwargs: P.kwargs,
    ) -> dict[str, Any]:
        """Set the colorbar extrem values if needed.

        Returns:
            dict[str, Any]: Updated kwargs.
        """
        if ("vmin" in kwargs) or ("vmax" in kwargs):
            return kwargs
        max_value = max(np.abs(data).max() for data in datas)
        kwargs["vmax"] = max_value
        kwargs["vmin"] = -max_value
        return kwargs

    def _show_colorbar(self) -> None:
        """Show the colorbar."""
        if self._cbar_axes is None:
            # create colorbar Axes.
            self.figure.subplots_adjust(right=0.85)
            self._cbar_axes: Axes = self.figure.add_axes(
                [0.88, 0.15, 0.04, 0.7],
            )
        if self._axes_ms[0].content.axes_image is not None:
            # Add colorbar to the Axes.
            self.figure.colorbar(
                self._axes_ms[0].content.axes_image,
                cax=self._cbar_axes,
                label=r"$\omega / f_0$",
            )

    def _update(self, *datas: np.ndarray, **kwargs: P.kwargs) -> None:
        """Update the Axes."""
        if self._common_cbar:
            kwargs = self._set_cbar_extrems(*datas, **kwargs)

        super()._update(*datas, **kwargs)

        if self._common_cbar:
            for ax_ms in self._axes_ms:
                ax_ms.content.remove_colorbar()
            self._show_colorbar()
