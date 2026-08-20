"""Reshaping utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import numpy as np
    import torch


@overload
def crop(t: torch.Tensor, offset: int) -> torch.Tensor: ...


@overload
def crop(t: np.ndarray, offset: int) -> np.ndarray: ...


def crop(
    t: torch.Tensor | np.ndarray, offset: int = 0
) -> torch.Tensor | np.ndarray:
    """Crop the last two dimensions of a tensor.

    Args:
        t (torch.Tensor | np.ndarray): Array / Tensor to crop.
        offset (int, optional): Crop width. Defaults to 0.

    Returns:
        torch.Tensor | np.ndarray: Cropped array / tensor.
    """
    if offset < 0:
        msg = "Cropping offset must be greater or equal to 0."
        raise ValueError(msg)
    if offset == 0:
        return t
    return t[..., offset:-offset, offset:-offset]
