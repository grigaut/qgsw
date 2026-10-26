"""Reshaping utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import numpy as np
    import torch
    import xarray as xr


@overload
def crop(
    t: torch.Tensor, offset_both_or_x: int, offset_y: int | None
) -> torch.Tensor: ...


@overload
def crop(
    t: np.ndarray, offset_both_or_x: int, offset_y: int | None
) -> np.ndarray: ...


def crop(
    t: torch.Tensor | np.ndarray,
    offset_both_or_x: int = 0,
    offset_y: int | None = None,
) -> torch.Tensor | np.ndarray:
    """Crop the last two dimensions of a tensor.

    Args:
        t (torch.Tensor | np.ndarray): Array / Tensor to crop.
        offset_both_or_x (int, optional): Crop width for x and y of just x,
            if offset_y is provided. Defaults to 0.
        offset_y (int | None, optional): Crop width for y. If None, cropping
            will be performed on both dimensions with
            the value of offset_both_or_x. Defaults to None.

    Returns:
        torch.Tensor | np.ndarray: Cropped array / tensor.
    """
    if offset_y is None:
        if offset_both_or_x == 0:
            return t
        slice_x = slice(offset_both_or_x, -offset_both_or_x, 1)
        slice_y = slice(offset_both_or_x, -offset_both_or_x, 1)
    else:
        if offset_both_or_x == 0:
            slice_x = slice(None, None, None)
        else:
            slice_x = slice(offset_both_or_x, -offset_both_or_x, 1)
        if offset_y == 0:
            slice_y = slice(None, None, None)
        else:
            slice_y = slice(offset_y, -offset_y, 1)
    return t[..., slice_x, slice_y]


@overload
def crop_xr(
    t: xr.DataArray,
    offset_both_or_x: int,
    offset_y: int | None,
    *,
    x: str = "i",
    y: str = "j",
) -> xr.DataArray: ...


@overload
def crop_xr(
    t: xr.Dataset,
    offset_both_or_x: int,
    offset_y: int | None,
    *,
    x: str = "i",
    y: str = "j",
) -> xr.Dataset: ...


def crop_xr(
    data: xr.DataArray | xr.Dataset,
    offset_both_or_x: int,
    offset_y: int | None = None,
    *,
    x: str = "i",
    y: str = "j",
) -> xr.DataArray | xr.Dataset:
    """Crop the last two dimensions of a xarray's DataArray or Dataset.

    Args:
        data (xr.DataArray | xr.Dataset): DataArray / Dataset
        offset_both_or_x (int, optional): Crop width for x and y of just x,
            if offset_y is provided. Defaults to 0.
        offset_y (int | None, optional): Crop width for y. If None, cropping
            will be performed on both dimensions with
            the value of offset_both_or_x. Defaults to None.
        x (str, optional): x dimension. Defaults to "i".
        y (str, optional): y dimension. Defaults to "j".

    Returns:
        xr.DataArray | xr.Dataset: Cropped DataArray / Dataset
    """
    if offset_y is None:
        if offset_both_or_x == 0:
            return data
        slice_x = slice(offset_both_or_x, -offset_both_or_x, 1)
        slice_y = slice(offset_both_or_x, -offset_both_or_x, 1)
    else:
        if offset_both_or_x == 0:
            slice_x = slice(None, None, None)
        else:
            slice_x = slice(offset_both_or_x, -offset_both_or_x, 1)
        if offset_y == 0:
            slice_y = slice(None, None, None)
        else:
            slice_y = slice(offset_y, -offset_y, 1)
    return data.isel({x: slice_x, y: slice_y})
