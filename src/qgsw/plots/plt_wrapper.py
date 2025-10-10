"""Wrapper for some of matplotlib.pyplot's functions."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import figure
from matplotlib import pyplot as plt
from typing_extensions import ParamSpec

Param = ParamSpec("Param")

DEFAULT_CMAP = "RdBu_r"


def imshow(
    data: torch.Tensor | np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **kwargs: Param.kwargs,
) -> None:
    """Wrapper for plt.imshow.

    Args:
        data (torch.Tensor | np.ndarray): 2D array to plot.
        ax (plt.Axes | None, optional): Axes to plot on. Defaults to None.
        title (str | None, optional): Title. Defaults to None.
        **kwargs: optional arguments to pass to plt.imshow.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    kwargs.setdefault("vmax", np.max(np.abs(data)))
    kwargs.setdefault("vmin", -kwargs["vmax"])
    kwargs.setdefault("cmap", DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")
    if ax is None:
        ax = plt.subplot()

    cbar = ax.imshow(data.T, **kwargs)
    ax.figure.colorbar(cbar, ax=ax)
    if title is not None:
        ax.set_title(title)


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
    kwargs.setdefault("constrained_layout", True)
    kwargs.setdefault("figsize", ((4 * ncols, 4 * nrows + 1)))
    return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)


def show(**kwargs: Param.kwargs) -> None:
    """Wrapper for plt.show."""
    return plt.show(**kwargs)
