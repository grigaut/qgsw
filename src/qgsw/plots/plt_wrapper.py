"""Wrapper for some of matplotlib.pyplot's functions."""

from __future__ import annotations

import matplotlib.colorbar as mpl_cbar
import numpy as np
import torch
from matplotlib import figure
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing_extensions import ParamSpec

Param = ParamSpec("Param")

DEFAULT_CMAP = "RdBu_r"


def retrieve_imshow_data(data: torch.Tensor | np.ndarray) -> np.ndarray:
    """Retrieve data for imshow plot.

    Args:
        data (torch.Tensor | np.ndarray): Original data.

    Returns:
        np.ndarray: Data as a numpy array, transposed for imshow compatibility.
    """
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.bool:
            data = data.to(torch.int8)
        data = data.detach().cpu().numpy()
    return data.T


def default_clim(data: np.ndarray) -> tuple[float, float]:
    """Compute default colorbar limit values.

    Args:
        data (np.ndarray): Data for which to compute colorbar limits.

    Returns:
        tuple[float, float]: Colorbar limits as (vmin, vmax).
    """
    vmax = np.max(np.abs(data))
    return -vmax, vmax


def retrieve_colorbar(im: plt.AxesImage, ax: plt.Axes) -> mpl_cbar.Colorbar:
    """Retrieve colorbar axes from axes.

    Args:
        im (plt.AxesImage): Image to associate colorbar with.
        ax (plt.Axes): Axes to retrieve colorbar from.

    Returns:
        plt.Axes: Colorbar axes.
    """
    try:
        return ax.cbar
    except AttributeError:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad="3%")

        ax.cbar = ax.figure.colorbar(im, cax=cax)
        return ax.cbar


def imshow(
    data: torch.Tensor | np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **kwargs: Param.kwargs,
) -> plt.AxesImage:
    """Wrapper for plt.imshow.

    Args:
        data (torch.Tensor | np.ndarray): 2D array to plot.
        ax (plt.Axes | None, optional): Axes to plot on. Defaults to None.
        title (str | None, optional): Title. Defaults to None.
        **kwargs: optional arguments to pass to plt.imshow.
    """
    data = retrieve_imshow_data(data)
    vmin, vmax = default_clim(data)
    kwargs.setdefault("vmax", vmax)
    kwargs.setdefault("vmin", vmin)
    kwargs.setdefault("cmap", DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")
    if ax is None:
        ax = plt.subplot()

    im = ax.imshow(data, **kwargs)
    cbar = retrieve_colorbar(im, ax)
    cbar.update_normal(im)
    if title is not None:
        ax.set_title(title)
    return im


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Param.kwargs,
) -> tuple[figure.Figure, np.ndarray]:
    """Wrapper for plt.subplots.

    Args:
        nrows (int, optional): Number of rows. Defaults to 1.
        ncols (int, optional): Number of columns. Defaults to 1.
        **kwargs: optional arguments to pass to plt.subplots.

    Returns:
        tuple[mpl.figure.Figure, np.ndarray]: Figure, Axes array.
    """
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", ((4 * ncols, 4 * nrows + 1)))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    fig.subplots_adjust(wspace=0.3)
    return fig, axs


def show(**kwargs: Param.kwargs) -> None:
    """Wrapper for plt.show."""
    return plt.show(**kwargs)


def set_coltitles(
    colnames: list[str],
    axs: np.ndarray,
    *,
    pad: int = 5,
    **kwargs: Param.kwargs,
) -> None:
    """Set column titles.

    Args:
        colnames (list[str]): Column names.
        axs (np.ndarray): Axes array.
        pad (int, optional): Padding below text. Defaults to 5.
        **kwargs (Param.kwargs): Keywords arguments to pass to ax.annotate

    Raises:
        ValueError: If length mismatch between colnames and axs.
    """
    if len(colnames) != (n := axs.shape[1]):
        msg = f"There must be exactly {n} column names."
        raise ValueError(msg)
    kwargs.setdefault("xy", (0.5, 1))
    kwargs.setdefault("xycoords", "axes fraction")
    kwargs.setdefault("textcoords", "offset points")
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "baseline")

    for ax, col in zip(axs[0], colnames):
        ax.annotate(col, xytext=(0, pad), **kwargs)


def set_rowtitles(
    rownames: list[str],
    axs: np.ndarray,
    *,
    pad: int = 5,
    **kwargs: Param.kwargs,
) -> None:
    """Set row titles.

    Args:
        rownames (list[str]): Row names.
        axs (np.ndarray): Axes array.
        pad (int, optional): Padding below text. Defaults to 5.
        **kwargs (Param.kwargs): Keywords arguments to pass to ax.annotate

    Raises:
        ValueError: If length mismatch between rownames and axs.
    """
    if len(rownames) != (n := axs.shape[0]):
        msg = f"There must be exactly {n} row names."
        raise ValueError(msg)

    kwargs.setdefault("xy", (0, 0.5))
    kwargs.setdefault("textcoords", "offset points")
    kwargs.setdefault("ha", "right")
    kwargs.setdefault("va", "center")
    kwargs.setdefault("rotation", 90)

    for ax, row in zip(axs[:, 0], rownames):
        ax.annotate(
            row,
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            **kwargs,
        )


def clamp_ylims(bottom: float, top: float, ax: plt.Axes) -> None:
    """Clamp y lims.

    Args:
        bottom (float): Bottom value.
        top (float): Top value.
        ax (plt.Axes): Axes.
    """
    ax.relim()
    ax.autoscale_view()

    _, my = ax.margins()

    y0 = max(bottom, ax.dataLim.ymin)
    y1 = min(top, ax.dataLim.ymax)

    dy = y1 - y0
    pad = my * dy

    ax.set_ylim(y0 - pad, y1 + pad)
