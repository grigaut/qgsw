"""Default specs."""

from __future__ import annotations

from typing import TypedDict

import torch

from qgsw.specs import DEVICE


def get_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    """Get default dtype.

    Args:
        dtype (torch.dtype | None, optional): Dtype. Defaults to None.

    Returns:
        torch.dtype: Dtype.
    """
    if dtype is None:
        return torch.float64
    return dtype


def get_device(device: torch.device | None = None) -> torch.device:
    """Get default device.

    Args:
        device (torch.device | None, optional): Device. Defaults to None.

    Returns:
        torch.device: Device.
    """
    if device is None:
        return DEVICE.get()
    return device


class TensorSpecs(TypedDict):
    """Defaults specs."""

    dtype: torch.dtype
    device: torch.device


def get(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> TensorSpecs:
    """Get defaults.

    Args:
        dtype (torch.dtype | None, optional): Dtype. Defaults to None.
        device (torch.device | None, optional): Device. Defaults to None.

    Returns:
        TensorSpecs: Default specs.
    """
    return TensorSpecs(
        dtype=get_dtype(dtype),
        device=get_device(device),
    )


def get_save_specs(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> TensorSpecs:
    """Get defaults.

    Args:
        dtype (torch.dtype | None, optional): Dtype. Defaults to None.
        device (torch.device | None, optional): Device. Defaults to None.

    Returns:
        TensorSpecs: Default specs.
    """
    return TensorSpecs(
        dtype=dtype or torch.float32,
        device=device or torch.device("cpu"),
    )
