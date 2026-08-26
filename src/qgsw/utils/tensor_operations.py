"""Tensor operations and conversion utils."""

from __future__ import annotations

import torch

from qgsw.specs import defaults


def as_singe_value_tensor(
    value: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Transform an input into a single valued tensor.

    Args:
        value (float | torch.Tensor): Input.
        dtype (torch.dtype | None, optional): Tensor dtype, only used if value
            is not a tensor. Defaults to None.
        device (torch.device | None, optional): Tensor device, only used if
            value is not a tensor. Defaults to None.

    Raises:
        ValueError: If the tensor has more than one element.

    Returns:
        torch.Tensor: Tensor of shape Size([]).
    """
    if not isinstance(value, torch.Tensor):
        return torch.tensor(value, **defaults.get(dtype=dtype, device=device))
    if value.numel() != 1:
        msg = "There should be a single element in the tensor."
        raise ValueError(msg)
    return value.squeeze()
